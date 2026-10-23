from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def fingerprint(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


@dataclass
class Cost:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    cache_write_tokens: int = 0
    tool_calls: int = 0
    inspection_calls: int = 0
    inspection_bytes: int = 0
    tool_seconds: float = 0.0
    wall_seconds: float = 0.0
    usd: float = 0.0

    def add(self, other: Cost) -> None:
        for f in fields(self):
            setattr(self, f.name, getattr(self, f.name) + getattr(other, f.name))

    def units(self, weights: dict) -> float:
        # Fixed, preregistered cost units. USD is optional, never guessed.
        return sum(getattr(self, k) * v for k, v in weights.items())


@dataclass
class Grade:
    valid: bool
    score: float
    details: dict = field(default_factory=dict)

    def __post_init__(self):
        if not math.isfinite(self.score):
            raise ValueError("Evaluator returned a nonfinite score")


@dataclass
class Task:
    id: str
    suite: str
    split: str
    group: str
    prompt: str
    stages: list[float]
    target: float
    direction: str = "maximize"
    seed_dir: str | None = None
    image: str = "python:3.12-slim"
    cwd: str = "/work"
    evaluator: dict = field(default_factory=dict)
    allowed_artifacts: list[str] = field(default_factory=lambda: ["*.py", "*.json"])
    source: dict = field(default_factory=dict)

    def reached(self, grade: Grade, threshold: float) -> bool:
        return grade.valid and (
            grade.score >= threshold if self.direction == "maximize" else grade.score <= threshold
        )

    def regressed(self, before: Grade, after: Grade, tolerance: float) -> bool:
        if not after.valid:
            return True
        if not before.valid:
            return False
        delta = after.score - before.score
        return delta < -tolerance if self.direction == "maximize" else delta > tolerance

    @classmethod
    def load_many(cls, path: Path) -> list[Task]:
        tasks = [cls(**json.loads(line)) for line in path.read_text().splitlines() if line.strip()]
        ids: set[str] = set()
        groups: dict[str, str] = {}
        for task in tasks:
            if task.id in ids or not task.id or task.split not in {"train", "test"}:
                raise ValueError(f"Duplicate/invalid task: {task.id}")
            ids.add(task.id)
            if task.direction not in {"maximize", "minimize"}:
                raise ValueError(f"Invalid direction for {task.id}")
            tolerance = task.evaluator.get("quality_tolerance", 0)
            if not math.isfinite(tolerance) or tolerance < 0:
                raise ValueError(f"Invalid quality tolerance for {task.id}")
            values = task.stages + [task.target]
            if not task.stages or not all(math.isfinite(v) for v in values):
                raise ValueError(f"Missing/nonfinite milestones for {task.id}")
            if values != sorted(set(values), reverse=task.direction == "minimize"):
                raise ValueError(f"Stages and target must strictly improve: {task.id}")
            if task.group in groups and groups[task.group] != task.split:
                raise ValueError(f"Train/test group leakage: {task.group}")
            groups[task.group] = task.split
            if task.suite == "math" and task.split != "test":
                raise ValueError("The four primary math instances are evaluation-only")
            if task.seed_dir:
                task.seed_dir = str((path.parent / task.seed_dir).resolve())
        return tasks


def task_identity(task: Task) -> str:
    # Exclude labels/groups so renaming an exact training instance cannot evade the guard.
    from .sandbox import file_manifest

    evaluator = dict(task.evaluator)
    evaluator.pop("corpus_db", None)  # Corpus identity is supplied by its source SHA instead.
    return fingerprint(
        {
            "suite": task.suite,
            "prompt": task.prompt,
            "evaluator": evaluator,
            "corpus_sha256": task.source.get("corpus_db_sha256"),
            "seed_files": file_manifest(Path(task.seed_dir)) if task.seed_dir else {},
            "image": task.image,
            "cwd": task.cwd,
            "allowed_artifacts": task.allowed_artifacts,
        }
    )


def as_json(value) -> dict:
    return asdict(value)
