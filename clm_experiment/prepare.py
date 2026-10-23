from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from .sandbox import docker, file_manifest
from .types import Task, dump


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_browse(
    queries: Path,
    qrels: Path,
    corpus_db: Path,
    output: Path,
    train_fraction: float,
    seed: int,
    limit: int | None,
):
    if not 0 < train_fraction < 1:
        raise ValueError("Use a nonempty train/test split")
    if output.exists():
        raise ValueError("Refusing to overwrite prepared benchmark data")
    # Imported inputs and labels remain host-side, never mounted in the work container.
    relevance: dict[str, list[str]] = {}
    for line in qrels.read_text().splitlines():
        qid, _, docid, rel = line.split()
        if float(rel) > 0:
            relevance.setdefault(qid, []).append(docid)
    records = [json.loads(line) for line in queries.read_text().splitlines() if line.strip()]
    records.sort(key=lambda row: str(row["query_id"]))
    random.Random(seed).shuffle(records)
    if limit:
        records = records[:limit]
    if len(records) < 2:
        raise ValueError("At least two queries are required for disjoint learning/evaluation")
    train_size = max(1, min(len(records) - 1, round(len(records) * train_fraction)))
    output.mkdir(parents=True)
    corpus_db = corpus_db.resolve()
    if not corpus_db.is_file():
        raise ValueError("Build the fixed-corpus index first")
    source = {
        "url": "https://github.com/texttron/BrowseComp-Plus",
        "queries_sha256": file_sha(queries),
        "qrels_sha256": file_sha(qrels),
        "corpus_db_sha256": file_sha(corpus_db),
        "retriever": "sqlite-fts5-bm25-top5-2400chars",
        "seed": seed,
        "grading": "exact-match + evidence-recall proxy",
    }
    tasks = []
    for index, row in enumerate(records):
        qid = str(row["query_id"])
        answers = row.get("answers", row.get("answer"))
        if isinstance(answers, str):
            answers = [answers]
        if not answers or not all(isinstance(a, str) and a.strip() for a in answers):
            raise ValueError(f"Missing answer/answers for query {qid}")
        if not relevance.get(qid):
            raise ValueError(f"Missing evidence qrels for query {qid}")
        from .experiment import slug

        folder = "seed-" + slug(qid)
        (output / folder).mkdir()
        dump(output / folder / "solution.json", {"answer": "", "citations": []})
        task = Task(
            "browse-" + qid,
            "browsecomp_plus",
            "train" if index < train_size else "test",
            "browse-query-" + qid,
            row["query"] + "\nUse the fixed-corpus search tool. Write solution.json with "
            '{"answer":"short answer","citations":["docid",...]}. '
            "Update citations as evidence is found, keeping answer empty until ready to answer.",
            [0.15],
            1.0,
            seed_dir=folder,
            evaluator={
                "kind": "browse",
                "answers": answers,
                "evidence_docids": relevance[qid],
                "corpus_db": str(corpus_db),
                "progress_interval": 1,
                "grade_every": 1,
            },
            source={**source, "query_id": qid},
        )
        tasks.append(task.__dict__)
    (output / "tasks.jsonl").write_text("".join(json.dumps(t) + "\n" for t in tasks))
    return {"train": train_size, "test": len(records) - train_size}


def prepare_edge(selection: Path, output: Path):
    """Import explicit SForge task/image selections without inventing the paper's ten tasks."""
    if output.exists():
        raise ValueError("Refusing to overwrite prepared EdgeBench data")
    rows = json.loads(selection.read_text())
    if not rows:
        raise ValueError("Empty EdgeBench selection")
    output.mkdir(parents=True)
    tasks = []
    from .experiment import slug

    for item in rows:
        task_file = (selection.parent / item["task_json"]).resolve()
        data = json.loads(task_file.read_text())
        if data.get("game_mode"):
            raise ValueError(
                "Stateful EdgeBench games need a custom snapshot backend; not supported"
            )
        tid = data["task_id"]
        image = item["work_image"]
        # Must already be pulled via SForge; pin the installed image by immutable ID.
        image_id = docker("image", "inspect", "--format", "{{.Id}}", image).strip()
        folder = "seed-" + slug(tid)
        seed_dir = output / folder
        seed_dir.mkdir()
        container = docker("create", "--entrypoint", "/bin/sh", image_id, "-c", "true").strip()
        try:
            docker("cp", container + ":" + data["cwd"] + "/.", str(seed_dir))
        finally:
            docker("rm", container)
        # Disallow symlinks; they need an explicit materialization strategy before import.
        manifest = file_manifest(seed_dir)
        score_field = item.get("score_field", "score_0_100")
        direction = item.get(
            "direction",
            "maximize"
            if score_field == "score_0_100"
            else data["judge"].get("score_direction", "maximize"),
        )
        task = Task(
            "edge-" + tid,
            "edgebench",
            item["split"],
            item.get("group", tid),
            data["work"]["agent_query"],
            item["stages"],
            item["target"],
            direction,
            folder,
            image_id,
            data["cwd"],
            evaluator={
                "kind": "sforge",
                "task_id": tid,
                "score_field": score_field,
                "timeout": data["judge"].get("eval_timeout", 1800),
                "submit_paths": data["submit_paths"],
                "submit_exclude": data.get("submit_exclude", ["tests/"]),
                "quality_tolerance": item.get("quality_tolerance", 1e-8),
            },
            allowed_artifacts=item.get(
                "allowed_artifacts",
                [
                    "*.py",
                    "*.json",
                    "*.c",
                    "*.cc",
                    "*.cpp",
                    "*.h",
                    "*.hpp",
                    "*.rs",
                    "*.toml",
                    "*.lock",
                    "*.go",
                    "*.mod",
                    "*.sum",
                    "*.sh",
                    "Makefile",
                    "*.yaml",
                    "*.yml",
                ],
            ),
            source={
                "url": "https://github.com/ByteDance-Seed/EdgeBench",
                "task_json_sha256": file_sha(task_file),
                "image_id": image_id,
                "initial_files": len(manifest),
                "task_id": tid,
                "internet_in_original_task": data.get("internet", True),
                "network_in_this_experiment": "none",
            },
        )
        tasks.append(task.__dict__)
    (output / "tasks.jsonl").write_text("".join(json.dumps(t) + "\n" for t in tasks))
    # Verify all split and milestone constraints before claiming preparation succeeded.
    Task.load_many(output / "tasks.jsonl")
    return {"tasks": len(tasks)}


def export_browse(runs: Path, output: Path):
    """Write official BrowseComp-Plus result format, grouped by model and condition."""
    for path in runs.rglob("run.json"):
        run = json.loads(path.read_text())
        if run["suite"] != "browsecomp_plus" or run["synthetic"]:
            continue
        solution_path = path.parent / "final_artifacts" / "solution.json"
        solution = json.loads(solution_path.read_text()) if solution_path.exists() else {}
        transcript = json.loads((path.parent / "transcript.json").read_text())
        retrieved = set()
        counts = {}
        for entry in transcript:
            try:
                content = json.loads(entry["content"])
            except ValueError:
                continue
            if entry["origin"] == "model" and isinstance(content, dict) and "tool" in content:
                counts[content["tool"]] = counts.get(content["tool"], 0) + 1
            if entry["origin"] == "tool" and isinstance(content, dict):
                retrieved.update(d["docid"] for d in content.get("documents", []))
        from .experiment import slug

        target = output / slug(run["model_label"]) / run["condition"]
        qid = run["task_source"]["query_id"]
        dump(
            target / (slug(qid) + ".json"),
            {
                "query_id": qid,
                "tool_call_counts": counts,
                "status": "completed" if solution.get("answer") else "incomplete",
                "retrieved_docids": sorted(retrieved),
                "result": [{"type": "output_text", "output": solution.get("answer", "")}],
            },
        )
