from __future__ import annotations

import fnmatch
import io
import itertools
import json
import math
import os
import re
import sqlite3
import tarfile
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

from .providers import http_json
from .types import Grade, Task, dump

MATH_SOURCE = "https://github.com/google-deepmind/alphaevolve_results"


def finite_numbers(values, count: int) -> list[float]:
    if not isinstance(values, list) or len(values) != count:
        raise ValueError(f"Expected {count} numbers")
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in values):
        raise ValueError("Expected numeric values")
    result = [float(x) for x in values]
    if not all(math.isfinite(x) for x in result):
        raise ValueError("Nonfinite coordinates")
    return result


def math_grade(kind: str, solution: dict, n: int) -> Grade:
    tolerance = 1e-9
    if kind == "erdos":
        a = finite_numbers(solution["heights"], n)
        if any(x < 0 or x > 1 for x in a) or abs(math.fsum(a) - n / 2) > tolerance:
            raise ValueError("Heights must lie in [0,1] and sum to n/2")
        # Correlation at all bin shifts, including negative shifts. Between shifts
        # the integral is linear, so the maximum occurs at a bin shift.
        overlap = max(
            math.fsum(a[i] * (1 - a[i + k]) for i in range(n) if 0 <= i + k < n) * 2 / n
            for k in range(-(n - 1), n)
        )
        return Grade(True, overlap, {"mass": 2 * math.fsum(a) / n, "bins": n})
    key = "circles" if kind == "circles" else "points"
    raw = solution[key]
    if not isinstance(raw, list) or len(raw) != n:
        raise ValueError(f"Expected exactly {n} {key}")
    points = [finite_numbers(p, 3 if kind == "circles" else 2) for p in raw]
    if kind == "circles":
        for x, y, r in points:
            if r < 0 or min(x - r, y - r, 1 - x - r, 1 - y - r) < -tolerance:
                raise ValueError("Circle outside the unit square or negative radius")
        for a, b in itertools.combinations(points, 2):
            if math.dist(a[:2], b[:2]) + tolerance < a[2] + b[2]:
                raise ValueError("Overlapping circles")
        return Grade(True, math.fsum(p[2] for p in points))
    if kind == "ratio":
        distances = [math.dist(a, b) for a, b in itertools.combinations(points, 2)]
        if min(distances) <= 1e-12:
            raise ValueError("Coincident points")
        return Grade(
            True,
            max(distances) / min(distances),
            {"squared_ratio": (max(distances) / min(distances)) ** 2},
        )
    if kind == "heilbronn":
        # Triangle (0,0),(2,0),(0,1) has unit area. Affine equivalent
        # to any other unit-area enclosing triangle.
        if any(min(x, y) < -tolerance or x / 2 + y > 1 + tolerance for x, y in points):
            raise ValueError("Point outside the unit-area triangle")
        area = min(
            abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) / 2
            for a, b, c in itertools.combinations(points, 3)
        )
        return Grade(True, area)
    raise ValueError(f"Unknown math problem: {kind}")


def prepare_math(output: Path, image: str = "python:3.12-slim"):
    import random

    output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(1729)
    heil = []
    for _ in range(11):
        x, y = rng.random(), rng.random()
        if x + y > 1:
            x, y = 1 - x, 1 - y
        heil.append([2 * x, y])
    definitions = [
        (
            "circles",
            26,
            "maximize",
            [2.25, 2.4, 2.5],
            2.6,
            "Pack 26 nonoverlapping circles in [0,1]^2, maximizing SUM OF RADII "
            '(radii may differ). solution.json: {"circles": [[x,y,r], ...]}.',
            {"circles": [[(i % 6 + 0.5) / 6, (i // 6 + 0.5) / 5, 1 / 12] for i in range(26)]},
        ),
        (
            "erdos",
            200,
            "minimize",
            [0.46, 0.42, 0.40],
            0.39,
            "Minimize the maximum cross-correlation integral of h and 1-h, zero-extended "
            "outside [0,2]. Use 200 equal-width bins, heights in [0,1], sum=100. "
            "Objective: (2/200)*max_k sum_i h[i]*(1-h[i+k]), over valid i and all shifts. "
            'solution.json: {"heights": [200 numbers]}.',
            {"heights": [0.5] * 200},
        ),
        (
            "ratio",
            16,
            "minimize",
            [4.15, 4.0, 3.8],
            3.6,
            "Place 16 distinct points in the plane to MINIMIZE max pairwise Euclidean "
            "distance / min pairwise Euclidean distance (not its square). "
            'solution.json: {"points": [[x,y], ...]}.',
            {"points": [[i % 4, i // 4] for i in range(16)]},
        ),
        (
            "heilbronn",
            11,
            "maximize",
            [0.005, 0.015, 0.025],
            0.035,
            "Place 11 points inside/on the unit-area triangle with vertices (0,0),(2,0),(0,1), "
            "maximizing the SMALLEST area among all triples. "
            'solution.json: {"points": [[x,y], ...]}.',
            {"points": heil},
        ),
    ]
    tasks = []
    for kind, n, direction, stages, target, prompt, seed in definitions:
        seed_dir = output / kind
        seed_dir.mkdir(exist_ok=True)
        dump(seed_dir / "solution.json", seed)
        task = Task(
            f"math-{kind}",
            "math",
            "test",
            f"math-{kind}",
            prompt,
            stages,
            target,
            direction,
            kind,
            image=image,
            evaluator={"kind": "math", "problem": kind, "n": n},
            source={"url": MATH_SOURCE, "definition": "B.5/B.8/B.9/B.12", "parameters": n},
        )
        tasks.append(task.__dict__)
    (output / "tasks.jsonl").write_text("".join(json.dumps(t) + "\n" for t in tasks))


def build_corpus(corpus: Path, db: Path):
    if db.exists():
        raise ValueError("Refusing to overwrite an existing retrieval index")
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE VIRTUAL TABLE documents USING fts5(docid UNINDEXED, title, text)")
        count = 0
        with corpus.open() as f:
            for line in f:
                row = json.loads(line)
                docid = row.get("docid", row.get("id"))
                if docid is None or "text" not in row:
                    raise ValueError("Corpus needs docid (or id), text, and optional title")
                conn.execute(
                    "INSERT INTO documents VALUES (?,?,?)",
                    (str(docid), row.get("title", ""), row["text"]),
                )
                count += 1
        if not count:
            raise ValueError("Empty corpus")
        conn.execute("INSERT INTO documents(documents) VALUES ('optimize')")
    return count


def search(db: str, query: str, limit: int = 5) -> list[dict]:
    # Literal bag-of-words OR query. The agent cannot issue SQL or mutate the index.
    words = re.findall(r"\w+", query)[:50]
    if not words:
        return []
    fts = " OR ".join('"' + w + '"' for w in words)
    with sqlite3.connect(Path(db).resolve().as_uri() + "?mode=ro", uri=True) as conn:
        rows = conn.execute(
            "SELECT docid,title,text FROM documents WHERE documents MATCH ? "
            "ORDER BY bm25(documents), docid LIMIT ?",
            (fts, min(limit, 5)),
        )
        # Fixed truncation for every model/condition; not the official retriever/token cutoff.
        return [{"docid": row[0], "title": row[1], "text": row[2][:2400]} for row in rows]


def normalize(answer: str) -> str:
    return " ".join(re.findall(r"\w+", answer.casefold()))


def sforge_grade(task: Task, work: Path) -> Grade:
    spec = task.evaluator
    url = os.environ.get(spec.get("url_env", "SFORGE_JUDGE_URL"), "http://127.0.0.1:8080")
    secret = os.environ.get(spec.get("secret_env", "SFORGE_ADMIN_SECRET"))
    if not secret:
        raise RuntimeError("Set SFORGE_ADMIN_SECRET for host-side SForge evaluation")
    registration = http_json(
        url + "/api/v1/register",
        {
            "task_id": spec["task_id"],
            "run_id": "clm-probe-" + uuid.uuid4().hex,
            "admin_secret": secret,
            "submission_cooldown": 0,
        },
        {},
    )
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w:gz") as tar:
        for p in sorted(work.rglob("*")):
            if p.is_symlink():
                raise ValueError("Symlink in submission")
            if p.is_file():
                name = p.relative_to(work).as_posix()
                submit_paths = spec.get("submit_paths", ["."])
                excludes = [".git", *spec.get("submit_exclude", [])]
                allowed = any(
                    x == "." or name == x.rstrip("/") or name.startswith(x.rstrip("/") + "/")
                    for x in submit_paths
                )
                excluded = any(
                    name == x.rstrip("/")
                    or name.startswith(x.rstrip("/") + "/")
                    or fnmatch.fnmatchcase(name, x)
                    for x in excludes
                )
                if allowed and not excluded:
                    tar.add(p, arcname=name, recursive=False)
    boundary = uuid.uuid4().hex
    body = bytearray()
    for key, value in {
        "token": registration["token"],
        "kind": "auto",
        "admin_secret": secret,
    }.items():
        body.extend(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n'.encode()
            + value.encode()
            + b"\r\n"
        )
    body.extend(
        f'--{boundary}\r\nContent-Disposition: form-data; name="archive"; '
        'filename="work.tar.gz"\r\nContent-Type: application/gzip\r\n\r\n'.encode()
    )
    body.extend(archive.getvalue() + f"\r\n--{boundary}--\r\n".encode())
    request = urllib.request.Request(
        url + "/api/v1/submit",
        data=bytes(body),
        headers={"Content-Type": "multipart/form-data; boundary=" + boundary},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        sid = json.load(response)["submission_id"]
    deadline = time.monotonic() + spec.get("timeout", 1800)
    while time.monotonic() < deadline:
        with urllib.request.urlopen(
            url + "/api/v1/result/" + urllib.parse.quote(sid), timeout=60
        ) as r:
            result = json.load(r)
        if result.get("report") is not None:
            report = result["report"]
            score = report.get(spec.get("score_field", "score_0_100"))
            if score is None:
                raise RuntimeError("SForge omitted the configured score field")
            return Grade(
                report.get("valid", False) and not report.get("timed_out", False),
                float(score),
                {"submission_id": sid, "report": report},
            )
        if result.get("error") or result.get("status") in {"failed", "error"}:
            raise RuntimeError("SForge judge failed; inspect its host-side logs")
        time.sleep(1)
    raise RuntimeError("SForge judge timeout")


def evaluate(task: Task, work: Path) -> Grade:
    kind = task.evaluator["kind"]
    if kind == "sforge":
        return sforge_grade(task, work)
    p = work / "solution.json"
    if not p.exists():
        return Grade(False, 0, {"error": "solution.json missing"})
    try:
        if p.is_symlink() or p.stat().st_size > 8_000_000:
            raise ValueError("Invalid solution file")
        solution = json.loads(p.read_text())
        if kind == "math":
            return math_grade(task.evaluator["problem"], solution, task.evaluator["n"])
        if kind == "fixture":
            value = finite_numbers([solution["value"]], 1)[0]
            return Grade(0 <= value <= 10, value)
        if kind == "browse":
            aliases = task.evaluator["answers"]
            answer = solution.get("answer", "")
            if not isinstance(answer, str):
                raise ValueError("Answer must be a string")
            correct = normalize(answer) in {normalize(a) for a in aliases}
            gold = set(map(str, task.evaluator.get("evidence_docids", [])))
            citations = set(map(str, solution.get("citations", [])))
            recall = len(citations & gold) / len(gold) if gold else 0
            return Grade(
                True,
                1.0 if correct else 0.5 * recall,
                {
                    "exact_match": correct,
                    "evidence_recall": recall,
                    "grading": "local exact-match proxy, not official LLM judge",
                },
            )
        raise ValueError(f"Unknown evaluator: {kind}")
    except (ValueError, KeyError, TypeError, OverflowError) as exc:
        return Grade(False, 0, {"error": str(exc)})
