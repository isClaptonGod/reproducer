from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

from .analysis import analyze
from .experiment import DEFAULT_AGENT, Experiment, load_config
from .policy import CONDITIONS, ActionPolicy
from .prepare import export_browse, prepare_browse, prepare_edge
from .providers import DemoProvider, Provider
from .sandbox import docker
from .tasks import build_corpus, prepare_math
from .types import Task, dump


def load_tasks(paths: list[Path]) -> list[Task]:
    tasks = [t for path in paths for t in Task.load_many(path)]
    if len({t.id for t in tasks}) != len(tasks):
        raise ValueError("Task IDs must be unique across all manifests")
    groups = {}
    for t in tasks:
        if t.group in groups and groups[t.group] != t.split:
            raise ValueError(f"Train/test group leakage across manifests: {t.group}")
        groups[t.group] = t.split
    return tasks


def pin_images(tasks):
    images = {t.image for t in tasks}
    pinned = {
        image: docker("image", "inspect", "--format", "{{.Id}}", image).strip() for image in images
    }
    for task in tasks:
        task.image = pinned[task.image]


def evaluate_grid(config, tasks, output, policies, model_names, conditions, fixture=False):
    if not fixture:
        pin_images(tasks)
    output.mkdir(parents=True, exist_ok=False)
    seed = config.get("seed", 1729)
    todo = [
        (name, task, condition)
        for name in model_names
        for task in tasks
        if task.split == "test"
        for condition in conditions
    ]
    if not todo:
        raise ValueError("No held-out evaluation tasks selected")
    random.Random(seed).shuffle(todo)  # Reduce systematic method-order and service-time effects.
    dump(
        output / "plan.json",
        {
            "runs": [[name, task.id, condition] for name, task, condition in todo],
            "primary_runs_per_cell": 1,
            "seed": seed,
        },
    )
    errors = []
    providers = {
        name: DemoProvider() if fixture else Provider(config["models"][name])
        for name in model_names
    }
    # Preflight every learned policy before making any paid calls.
    for name in model_names:
        if "learned_clm" in conditions and name not in policies:
            raise ValueError(f"Missing trained policy for {name}")
    for name, task, condition in todo:
        print(f"{name} / {task.id} / {condition}", flush=True)
        experiment = Experiment(config, providers[name], output, fixture)
        try:
            policy = policies.get(name) if condition == "learned_clm" else None
            experiment.run(task, condition, policy, seed, name)
        except (RuntimeError, ValueError, OSError, subprocess.SubprocessError) as exc:
            # Other independent cells remain useful. Failed cells stay in the expected grid.
            errors.append(
                {"model": name, "task": task.id, "condition": condition, "error": str(exc)[:2000]}
            )
            dump(output / "errors.json", errors)
    if errors:
        raise RuntimeError(f"{len(errors)} primary runs failed; inspect {output / 'errors.json'}")


def demo(output: Path):
    if output.exists():
        raise ValueError("Choose a new demo output directory")
    output.mkdir(parents=True)
    tasks = []
    for split in ("train", "test"):
        for i in range(2):
            seed = output / "data" / f"{split}-{i}"
            dump(seed / "solution.json", {"value": 0})
            tasks.append(
                Task(
                    f"fixture-{split}-{i}",
                    "fixture",
                    split,
                    f"fixture-{split}-{i}",
                    f"Fixture instance {split}-{i}: increase solution.json value to 8, "
                    "staying within [0,10].",
                    [2, 4, 6],
                    8,
                    seed_dir=str(seed.resolve()),
                    evaluator={"kind": "fixture"},
                )
            )
    config = {
        "seed": 1729,
        "agent": {
            **DEFAULT_AGENT,
            "engineering_calls": 1,
            "grade_every": 1,
            "progress_interval": 2,
            "max_primary_calls": 24,
            "probe_calls": 16,
            "cost_weights": {"calls": 1.0, "input_tokens": 0.0001},
        },
        "models": {"fixture": {"provider": "fixture"}},
    }
    dump(output / "config.json", config)
    policy = Experiment(config, DemoProvider(), output / "training", fixture=True).train(
        tasks, "fixture", episodes=4, seed=1729
    )
    evaluate_grid(
        config,
        tasks,
        output / "evaluation",
        {"fixture": policy},
        ["fixture"],
        CONDITIONS,
        fixture=True,
    )
    report = analyze(output / "evaluation")
    dump(output / "report.json", report)
    print(f"Offline synthetic fixture finished: {output / 'report.json'}")


def parser():
    p = argparse.ArgumentParser(description="Measure and learn reduced CLM history dependence")
    sub = p.add_subparsers(dest="command", required=True)
    d = sub.add_parser("demo", help="Offline scripted plumbing test; no real-model results")
    d.add_argument("--out", type=Path, default=Path("runs/demo"))
    m = sub.add_parser("prepare-math")
    m.add_argument("--out", type=Path, default=Path("data/math"))
    m.add_argument("--image", default="python:3.12-slim")
    c = sub.add_parser("index-corpus")
    c.add_argument("--corpus", type=Path, required=True)
    c.add_argument("--out", type=Path, required=True)
    b = sub.add_parser("prepare-browse")
    for flag in ("queries", "qrels", "corpus-db", "out"):
        b.add_argument("--" + flag, type=Path, required=True)
    b.add_argument("--train-fraction", type=float, default=0.3)
    b.add_argument("--seed", type=int, default=1729)
    b.add_argument("--limit", type=int)
    e = sub.add_parser("prepare-edge")
    e.add_argument("--selection", type=Path, required=True)
    e.add_argument("--out", type=Path, default=Path("data/edge"))
    for command in ("train", "run"):
        x = sub.add_parser(command)
        x.add_argument("--config", type=Path, default=Path("configs/experiment.json"))
        x.add_argument("--tasks", type=Path, action="append", required=True)
        x.add_argument("--out", type=Path, required=True)
        if command == "train":
            x.add_argument("--model", required=True)
            x.add_argument("--episodes", type=int, default=32)
        else:
            x.add_argument("--models", nargs="+")
            x.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
            x.add_argument("--policy-dir", type=Path, default=Path("policies"))
    a = sub.add_parser("analyze")
    a.add_argument("--runs", type=Path, required=True)
    a.add_argument("--out", type=Path, required=True)
    a.add_argument("--artifact-review", type=Path)
    a.add_argument("--minimum-effect", type=float, default=1.0)
    a.add_argument("--quality-margin", type=float, default=0.05)
    a.add_argument("--min-groups", type=int, default=8)
    a.add_argument("--max-censoring", type=float, default=0.1)
    a.add_argument("--seed", type=int, default=1729)
    ex = sub.add_parser("export-browse")
    ex.add_argument("--runs", type=Path, required=True)
    ex.add_argument("--out", type=Path, required=True)
    doctor = sub.add_parser("doctor")
    doctor.add_argument("--config", type=Path, default=Path("configs/experiment.json"))
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "demo":
            demo(args.out)
        elif args.command == "prepare-math":
            if args.out.exists():
                raise ValueError("Choose a new math output directory")
            prepare_math(args.out, args.image)
        elif args.command == "index-corpus":
            args.out.parent.mkdir(parents=True, exist_ok=True)
            print(json.dumps({"documents": build_corpus(args.corpus, args.out)}))
        elif args.command == "prepare-browse":
            print(
                json.dumps(
                    prepare_browse(
                        args.queries,
                        args.qrels,
                        args.corpus_db,
                        args.out,
                        args.train_fraction,
                        args.seed,
                        args.limit,
                    )
                )
            )
        elif args.command == "prepare-edge":
            print(json.dumps(prepare_edge(args.selection, args.out)))
        elif args.command == "train":
            config = load_config(args.config)
            if args.episodes < 1 or args.out.exists():
                raise ValueError("Use positive episodes and a new training output directory")
            provider = Provider(config["models"][args.model])
            tasks = load_tasks(args.tasks)
            pin_images(tasks)
            Experiment(config, provider, args.out).train(
                tasks, args.model, args.episodes, config.get("seed", 1729)
            )
        elif args.command == "run":
            config = load_config(args.config)
            names = args.models or list(config["models"])
            policies = {}
            if "learned_clm" in args.conditions:
                for name in names:
                    path = args.policy_dir / name / "policy.json"
                    policies[name] = ActionPolicy(**json.loads(path.read_text()))
            evaluate_grid(
                config, load_tasks(args.tasks), args.out, policies, names, args.conditions
            )
        elif args.command == "analyze":
            report = analyze(
                args.runs,
                seed=args.seed,
                minimum_effect=args.minimum_effect,
                quality_margin=args.quality_margin,
                min_groups=args.min_groups,
                max_censoring=args.max_censoring,
                artifact_review=args.artifact_review,
            )
            dump(args.out, report)
            print(
                json.dumps(
                    {"runs": report["run_count"], "contrasts": report["contrasts"]}, indent=2
                )
            )
        elif args.command == "export-browse":
            export_browse(args.runs, args.out)
        elif args.command == "doctor":
            config = load_config(args.config)
            status = {"docker": shutil.which("docker") is not None, "models": {}}
            for name, model in config["models"].items():
                key = model.get("api_key_env")
                base_env = model.get("base_url_env")
                status["models"][name] = {
                    "model_id": model["model"],
                    "key_present": bool(os.environ.get(key)) if key else None,
                    "base_url_configured": bool(
                        model.get("base_url") or (base_env and os.environ.get(base_env))
                    ),
                }
            print(json.dumps(status, indent=2))
    except (ValueError, KeyError, OSError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
