from __future__ import annotations

import copy
import json
import math
import random
import re
import time
from dataclasses import asdict
from pathlib import Path

from .agent import Session, State
from .artifacts import clean_snapshot
from .policy import (
    ARMS,
    CONDITIONS,
    HANDOFF,
    PRESERVE,
    STANDARD,
    ActionPolicy,
    dependence_reward,
    protocol_hash,
)
from .sandbox import file_manifest, make_sandbox
from .tasks import evaluate
from .types import Cost, Task, dump, fingerprint, task_identity

DEFAULT_AGENT = {
    "context_tokens": 32000,
    "context_reserve": 4096,
    "max_primary_calls": 256,
    "progress_interval": 8,
    "engineering_calls": 8,
    "probe_calls": 64,
    "probe_cost_cap": 100.0,
    "failure_penalty": 100.0,
    "command_timeout": 120,
    "grade_every": 4,
    "quality_tolerance": 1e-8,
    "cost_weights": {
        "calls": 1.0,
        "input_tokens": 0.0001,
        "output_tokens": 0.0005,
        "tool_seconds": 0.01,
    },
}


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text())
    config["agent"] = {**DEFAULT_AGENT, **config.get("agent", {})}
    cfg = config["agent"]
    for key in (
        "context_tokens",
        "context_reserve",
        "max_primary_calls",
        "progress_interval",
        "engineering_calls",
        "probe_calls",
        "command_timeout",
        "grade_every",
    ):
        if not isinstance(cfg[key], int) or cfg[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if cfg["context_reserve"] >= cfg["context_tokens"]:
        raise ValueError("Context reserve exceeds budget")
    for key in ("probe_cost_cap", "failure_penalty", "quality_tolerance"):
        if not math.isfinite(cfg[key]) or cfg[key] < 0:
            raise ValueError(f"{key} must be nonnegative")
    if not cfg["cost_weights"] or any(
        k not in Cost.__dataclass_fields__ or not math.isfinite(v) or v < 0
        for k, v in cfg["cost_weights"].items()
    ):
        raise ValueError("Invalid cost weights")
    if not any(cfg["cost_weights"].values()):
        raise ValueError("At least one cost weight must be positive")
    if config.get("training_amortization_tasks", 1) <= 0:
        raise ValueError("Training amortization horizon must be positive")
    if "models" not in config:
        raise ValueError("Configure at least one model")
    for model in config["models"].values():
        if model.get("max_output_tokens", 4096) > cfg["context_reserve"]:
            raise ValueError("Output tokens must fit inside the configured context reserve")
    return config


def slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", text)[:80] + "-" + fingerprint(text)[:8]


class Experiment:
    def __init__(self, config: dict, provider, output: Path, fixture: bool = False):
        self.config = config
        self.agent = config["agent"]
        self.provider = provider
        self.output = output
        self.fixture = fixture
        if fixture and provider.model != "scripted-fixture":
            raise ValueError("Host fixture sandbox is restricted to the scripted fixture")
        self.output.mkdir(parents=True, exist_ok=True)

    def sandbox(self, task, source, memory=None):
        return make_sandbox(
            task, source, memory, fixture=self.fixture, resources=self.config.get("resources", {})
        )

    def checkpoint(self, task, session, directory, initial):
        directory.mkdir(parents=True, exist_ok=False)
        session.sandbox.snapshot(directory / "raw", directory / "memory")
        audit = clean_snapshot(
            directory / "raw",
            directory / "functional",
            task.allowed_artifacts,
            initial,
            session.state.transcript,
        )
        raw_grade = evaluate(task, directory / "raw")
        clean_grade = (
            raw_grade
            if file_manifest(directory / "raw") == audit["manifest"]
            else evaluate(task, directory / "functional")
        )
        preserved = raw_grade.valid == clean_grade.valid and (
            abs(raw_grade.score - clean_grade.score) <= self.tolerance(task)
        )
        audit["functional_outcome_preserved"] = preserved
        audit["valid"] = preserved and not audit["violations"]
        audit["raw_grade"] = asdict(raw_grade)
        audit["clean_grade"] = asdict(clean_grade)
        dump(directory / "audit.json", audit)
        return audit

    def tolerance(self, task):
        return task.evaluator.get("quality_tolerance", self.agent["quality_tolerance"])

    def pair(self, task, checkpoint, state, target, directory, seed):
        result = {}
        branches = ["fresh", "retained"]
        random.Random(seed).shuffle(branches)
        for branch in branches:
            started = time.monotonic()
            sb = self.sandbox(
                task,
                checkpoint / "functional",
                checkpoint / "memory" if branch == "retained" else None,
            )
            try:
                branch_state = copy.deepcopy(state) if branch == "retained" else State()
                branch_state.message(
                    "user",
                    f"Continue the original task. The next independently "
                    f"checked milestone is {target} ({task.direction}).",
                    "harness",
                )
                session = Session(task, sb, self.provider, self.agent, branch_state)
                value = session.segment(
                    self.agent["probe_calls"], seed, target, self.agent["probe_cost_cap"]
                )
                units = session.cost.units(self.agent["cost_weights"])
                success = value["status"] == "success"
                value.update(
                    success=success,
                    censored=not success,
                    observed_cost=units,
                    charged_cost=units
                    if success
                    else (max(units, self.agent["probe_cost_cap"]) + self.agent["failure_penalty"]),
                    branch=branch,
                    seed=seed,
                    seed_supported=self.provider.config.get("send_seed", False),
                )
                session.save(directory / branch, value)
                result[branch] = value
            finally:
                sb.close()
            result[branch]["lifecycle_seconds"] = time.monotonic() - started
        return result

    def engineering(self, task, session, condition, arm, calls, seed):
        instruction = (
            ARMS[arm]
            if condition == "learned_clm"
            else HANDOFF
            if condition == "handoff_clm"
            else STANDARD
        )
        session.state.message("user", instruction + PRESERVE, "harness")
        eng = Session(task, session.sandbox, self.provider, self.agent, session.state)
        result = eng.segment(calls, seed)
        result["action"] = arm if condition == "learned_clm" else condition
        result["cost_units"] = eng.cost.units(self.agent["cost_weights"])
        session.state = eng.state
        return eng, result

    def train(self, tasks: list[Task], model_name: str, episodes: int, seed: int):
        selected = [t for t in tasks if t.split == "train"]
        if not selected:
            raise ValueError("Training needs disjoint train tasks; math instances are test-only")
        policy = ActionPolicy(self.provider.model, protocol_hash(self.config))
        records = []
        order = sorted(selected, key=lambda t: t.id)
        random.Random(seed).shuffle(order)
        for episode in range(episodes):
            task = order[episode % len(order)]
            if task.group not in policy.train_groups:
                policy.train_groups.append(task.group)
            if task_identity(task) not in policy.train_ids:
                policy.train_ids.append(task_identity(task))
            directory = self.output / f"episode-{episode:04d}"
            directory.mkdir(exist_ok=False)
            initial = file_manifest(Path(task.seed_dir))
            sb = self.sandbox(task, Path(task.seed_dir))
            try:
                prefix_config = {
                    **self.agent,
                    "grade_every": task.evaluator.get("grade_every", self.agent["grade_every"]),
                }
                prefix = Session(task, sb, self.provider, prefix_config)
                prefix_result = prefix.segment(
                    self.agent["max_primary_calls"], seed + episode * 100000, task.stages[0]
                )
                prefix.save(directory / "prefix", prefix_result)
                record = {"task": task.id, "group": task.group, "prefix": prefix_result}
                if prefix_result["status"] not in {"success", "target_already_reached"}:
                    record["status"] = "stage_unreached"
                    records.append(record)
                    dump(directory / "episode.json", record)
                    continue
                target = (task.stages + [task.target])[1]
                pre_audit = self.checkpoint(task, prefix, directory / "pre", initial)
                arm = policy.choose(task.suite, seed + episode, training=True)
                if not pre_audit["valid"]:
                    record["status"] = "invalid_artifacts"
                    records.append(record)
                    dump(directory / "episode.json", record)
                    continue
                before = self.pair(
                    task,
                    directory / "pre",
                    prefix.state,
                    target,
                    directory / "before",
                    seed + episode * 100000 + 10000,
                )
                eng, action = self.engineering(
                    task,
                    prefix,
                    "learned_clm",
                    arm,
                    self.agent["engineering_calls"],
                    seed + episode + 20000,
                )
                eng.save(directory / "engineering", action)
                post_audit = self.checkpoint(task, prefix, directory / "post", initial)
                after = self.pair(
                    task,
                    directory / "post",
                    prefix.state,
                    target,
                    directory / "after",
                    seed + episode * 100000 + 10000,
                )
                before_grade = pre_audit["clean_grade"]
                after_grade = post_audit["clean_grade"]
                valid = (
                    post_audit["valid"]
                    and after_grade["valid"]
                    and before_grade["valid"]
                    and abs(after_grade["score"] - before_grade["score"]) <= self.tolerance(task)
                    and all(
                        b["status"] != "target_already_reached"
                        for pair in (before, after)
                        for b in pair.values()
                    )
                )
                reward = dependence_reward(
                    before, after, action["cost_units"], valid, self.agent["failure_penalty"]
                )
                policy.update(task.suite, arm, reward, task.group, task_identity(task))
                record.update(
                    status="trained",
                    valid=valid,
                    arm=arm,
                    before=before,
                    after=after,
                    action=action,
                    reward=reward,
                    pre_audit=pre_audit,
                    post_audit=post_audit,
                )
                records.append(record)
                dump(directory / "episode.json", record)
            finally:
                sb.close()
        if not policy.episodes:
            raise ValueError(
                "No trainable checkpoints reached; inspect logs or adjust train milestones"
            )
        policy.training_cost_units = sum(
            Cost(**r["prefix"]["cost"]).units(self.agent["cost_weights"])
            + r.get("action", {}).get("cost_units", 0)
            + sum(
                b["observed_cost"] for key in ("before", "after") for b in r.get(key, {}).values()
            )
            for r in records
        )
        dump(self.output / "policy.json", asdict(policy))
        dump(
            self.output / "training.json",
            {
                "model_label": model_name,
                "episodes": records,
                "synthetic": self.fixture or self.provider.model == "scripted-fixture",
            },
        )
        return policy

    def run(
        self, task: Task, condition: str, policy: ActionPolicy | None, seed: int, model_name: str
    ):
        if task.split != "test" or condition not in CONDITIONS:
            raise ValueError("Evaluation requires held-out tasks and a known condition")
        if condition != "learned_clm":
            # Controls neither use nor pay for the learned engineering-action policy.
            policy = None
        if condition == "learned_clm":
            if policy is None or policy.model != self.provider.model:
                raise ValueError("Policy must have been trained on this exact model ID")
            if policy.protocol != protocol_hash(self.config):
                raise ValueError("Policy training/evaluation protocols differ")
            if task.group in policy.train_groups or task_identity(task) in policy.train_ids:
                raise ValueError("Held-out task overlaps policy training data")
        directory = self.output / slug(model_name) / slug(task.id) / condition
        directory.mkdir(parents=True, exist_ok=False)  # No overwrite or hidden best-of-N selection.
        initial = file_manifest(Path(task.seed_dir))
        sb = self.sandbox(task, Path(task.seed_dir))
        started = time.monotonic()
        try:
            session = Session(task, sb, self.provider, self.agent)
            if condition == "handoff_clm":
                session.state.message("user", HANDOFF, "harness")
            stage_records, done = [], set()
            engineering_cost = Cost()
            status = "budget_exhausted"
            progress_interval = task.evaluator.get(
                "progress_interval", self.agent["progress_interval"]
            )
            while session.cost.calls + engineering_cost.calls < self.agent["max_primary_calls"]:
                remaining = (
                    self.agent["max_primary_calls"] - session.cost.calls - engineering_cost.calls
                )
                primary = session.segment(min(progress_interval, remaining), seed)
                grade = session.grade()
                for i, stage in enumerate(task.stages):
                    if i in done or not task.reached(grade, stage):
                        continue
                    done.add(i)
                    checkpoint_dir = directory / f"stage-{i}"
                    pre_audit = self.checkpoint(task, session, checkpoint_dir / "pre", initial)
                    record = {"stage": stage, "stage_index": i, "pre_audit": pre_audit}
                    remaining = (
                        self.agent["max_primary_calls"]
                        - session.cost.calls
                        - engineering_cost.calls
                    )
                    if not pre_audit["valid"] or remaining < self.agent["engineering_calls"]:
                        record["status"] = (
                            "invalid_artifacts" if not pre_audit["valid"] else "no_action_budget"
                        )
                        stage_records.append(record)
                        continue
                    target = (task.stages + [task.target])[i + 1]
                    probe_seed = seed + i * 10000 + 1000
                    before = self.pair(
                        task,
                        checkpoint_dir / "pre",
                        session.state,
                        target,
                        checkpoint_dir / "before",
                        probe_seed,
                    )
                    arm = (
                        policy.choose(task.suite, seed + i, training=False)
                        if condition == "learned_clm"
                        else None
                    )
                    eng, action = self.engineering(
                        task,
                        session,
                        condition,
                        arm,
                        self.agent["engineering_calls"],
                        seed + i + 20000,
                    )
                    engineering_cost.add(eng.cost)
                    eng.save(checkpoint_dir / "engineering", action)
                    post_audit = self.checkpoint(task, session, checkpoint_dir / "post", initial)
                    after = self.pair(
                        task,
                        checkpoint_dir / "post",
                        session.state,
                        target,
                        checkpoint_dir / "after",
                        probe_seed,
                    )
                    valid = (
                        pre_audit["clean_grade"]["valid"]
                        and post_audit["valid"]
                        and post_audit["clean_grade"]["valid"]
                        and abs(
                            pre_audit["clean_grade"]["score"] - post_audit["clean_grade"]["score"]
                        )
                        <= self.tolerance(task)
                        and all(
                            b["status"] != "target_already_reached"
                            for pair in (before, after)
                            for b in pair.values()
                        )
                    )
                    record.update(
                        status="measured" if valid else "invalid_measurement",
                        valid=valid,
                        target=target,
                        arm=arm,
                        action=action,
                        before=before,
                        after=after,
                        post_audit=post_audit,
                        primary_cost_units_at_checkpoint=(
                            session.cost.units(self.agent["cost_weights"])
                            + engineering_cost.units(self.agent["cost_weights"])
                        ),
                    )
                    stage_records.append(record)
                if primary["status"] in {"context_exhausted", "agent_finished"}:
                    status = primary["status"]
                    break
                if task.reached(session.grade(), task.target) and len(done) == len(task.stages):
                    status = "target_reached"
                    break
                seed += progress_interval
            final = session.grade()
            primary_cost = copy.deepcopy(session.cost)
            primary_cost.add(engineering_cost)
            for i, stage in enumerate(task.stages):
                if i not in done:
                    stage_records.append({"stage_index": i, "stage": stage, "status": "unreached"})
            result = {
                "schema_version": 1,
                "synthetic": self.fixture or self.provider.model == "scripted-fixture",
                "model_label": model_name,
                "model": self.provider.model,
                "condition": condition,
                "task": task.id,
                "group": task.group,
                "suite": task.suite,
                "task_identity": task_identity(task),
                "task_source": task.source,
                "work_image": task.image,
                "seed": self.config.get("seed", 1729),
                "primary_run_count": 1,
                "status": status,
                "final_grade": asdict(final),
                "target_reached": task.reached(final, task.target),
                "stages": stage_records,
                "primary_cost": asdict(primary_cost),
                "primary_cost_units": primary_cost.units(self.agent["cost_weights"]),
                "protocol": protocol_hash(self.config),
                "agent_config": self.agent,
                "provider_config": self.provider.config,
                "reported_model_ids": sorted(getattr(self.provider, "observed_models", [])),
                "measurement_definition": {
                    "stages": task.stages,
                    "target": task.target,
                    "direction": task.direction,
                },
                "policy_hash": fingerprint(asdict(policy)) if policy else None,
                "policy_training_cost_units": policy.training_cost_units if policy else 0,
                "training_amortization_tasks": self.config.get("training_amortization_tasks", 1),
                "diagnostic_probe_cost_units": sum(
                    b["observed_cost"]
                    for s in stage_records
                    for key in ("before", "after")
                    for b in s.get(key, {}).values()
                ),
                "manual_artifact_review_required": True,
                "lifecycle_seconds": time.monotonic() - started,
            }
            session.save(directory, result)
            sb.snapshot(directory / "final_artifacts")
            dump(directory / "run.json", result)
            return result
        finally:
            sb.close()
