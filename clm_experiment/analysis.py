from __future__ import annotations

import json
import random
import statistics
from pathlib import Path

from .policy import CONDITIONS
from .prepare import file_sha
from .types import fingerprint


def mean(values):
    return statistics.fmean(values) if values else None


def interval(values: list[float], seed: int, alpha: float, samples: int = 4000):
    # Input units are original task groups; stages and continuation forks are NOT replicates.
    rng = random.Random(seed)
    draws = sorted(statistics.fmean(rng.choices(values, k=len(values))) for _ in range(samples))
    return [
        draws[int(alpha / 2 * samples)],
        draws[min(samples - 1, int((1 - alpha / 2) * samples))],
    ]


def burden(run: dict) -> dict:
    cap = run["agent_config"]["probe_cost_cap"] + run["agent_config"]["failure_penalty"]
    fresh, gaps, failures, totals, invalid = [], [], [], [], 0
    for stage in run["stages"]:
        if stage.get("status") != "measured":
            fresh.append(cap)
            failures.append(1)
            invalid += 1
            totals.append(run["primary_cost_units"] + cap)
            continue
        pair = stage["after"]
        fresh.append(pair["fresh"]["charged_cost"])
        prefix = stage.get("primary_cost_units_at_checkpoint", run["primary_cost_units"])
        totals.append(prefix + pair["fresh"]["charged_cost"])
        gaps.append(pair["fresh"]["charged_cost"] - pair["retained"]["charged_cost"])
        failures.append(int(pair["fresh"]["censored"] or pair["retained"]["censored"]))
    # Older run files attached the learned policy metadata to control conditions.
    # Correct their summaries without changing raw artifacts or rerunning probes.
    training = (
        run.get("policy_training_cost_units", 0) / run.get("training_amortization_tasks", 1)
        if run["condition"] == "learned_clm"
        else 0
    )
    return {
        "fresh_cost": mean(fresh),
        "amortized_training_cost": training,
        "net_fresh_cost": mean(totals) + training,
        "restricted_dependency_gap": mean(gaps),
        "probe_failure_fraction": mean(failures),
        "invalid_or_unreached_stages": invalid,
        "task_target_success": int(run["target_reached"]),
    }


def analyze(
    directory: Path,
    *,
    seed: int = 1729,
    minimum_effect: float = 1.0,
    quality_margin: float = 0.05,
    min_groups: int = 8,
    max_censoring: float = 0.1,
    artifact_review: Path | None = None,
) -> dict:
    paths = list(directory.rglob("run.json"))
    runs = [json.loads(p.read_text()) for p in paths]
    if not runs:
        raise ValueError("No run.json results found")
    keys = [(r["model_label"], r["task"], r["condition"]) for r in runs]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate primary runs; refusing hidden best-of-N selection")
    if len({r["protocol"] for r in runs}) != 1:
        raise ValueError("Results use different agent protocols")
    index = dict(zip(keys, runs, strict=True))
    plans = list(directory.rglob("plan.json"))
    expected = set()
    for path in plans:
        expected.update(tuple(x) for x in json.loads(path.read_text())["runs"])
    missing = sorted(expected - set(keys))
    reviewed = False
    if artifact_review:
        review = json.loads(artifact_review.read_text())
        reviewed = bool(review.get("reviewer")) and review.get("functional_artifacts_only") is True
        reviewed = reviewed and all(
            review.get("reviewed_runs", {}).get(str(path.relative_to(directory))) == file_sha(path)
            for path in paths
        )
    summaries = [
        {
            "model": r["model_label"],
            "task": r["task"],
            "suite": r["suite"],
            "condition": r["condition"],
            **burden(r),
        }
        for r in runs
    ]
    contrasts = []
    model_suites = sorted({(r["model_label"], r["suite"]) for r in runs})
    # Bonferroni across both controls, every model, and every inferential suite.
    comparison_count = max(1, 2 * sum(suite != "math" for _, suite in model_suites))
    for model, suite in model_suites:
        task_ids = sorted(
            {r["task"] for r in runs if r["model_label"] == model and r["suite"] == suite}
        )
        for baseline in CONDITIONS[:2]:
            groups: dict[str, list[dict]] = {}
            unmatched = []
            for task in task_ids:
                control = index.get((model, task, baseline))
                learned = index.get((model, task, "learned_clm"))
                if not control or not learned:
                    unmatched.append(task)
                    continue
                if control["task_identity"] != learned["task_identity"]:
                    raise ValueError("Conditions do not use identical task definitions")
                if control.get("measurement_definition") != learned.get("measurement_definition"):
                    raise ValueError("Conditions use different targets or objective directions")
                # Fixed absolute milestones must be identical across conditions.
                if [
                    (x["stage_index"], x["stage"])
                    for x in sorted(control["stages"], key=lambda x: x["stage_index"])
                ] != [
                    (x["stage_index"], x["stage"])
                    for x in sorted(learned["stages"], key=lambda x: x["stage_index"])
                ]:
                    raise ValueError("Conditions use different progress milestones")
                a, b = burden(control), burden(learned)
                matched_gaps = []
                cb = {s["stage_index"]: s for s in control["stages"]}
                for s in learned["stages"]:
                    t = cb[s["stage_index"]]
                    if s["status"] == t["status"] == "measured":

                        def gap(v):
                            return (
                                v["after"]["fresh"]["charged_cost"]
                                - v["after"]["retained"]["charged_cost"]
                            )

                        matched_gaps.append(gap(t) - gap(s))
                groups.setdefault(control["group"], []).append(
                    {
                        "fresh_saving": a["net_fresh_cost"] - b["net_fresh_cost"],
                        "gap_reduction": mean(matched_gaps),
                        "quality_delta": b["task_target_success"] - a["task_target_success"],
                        "censoring": max(a["probe_failure_fraction"], b["probe_failure_fraction"]),
                        "invalid": a["invalid_or_unreached_stages"]
                        + b["invalid_or_unreached_stages"],
                    }
                )
            values = [
                {
                    key: mean([v[key] for v in vs if v[key] is not None])
                    for key in (
                        "fresh_saving",
                        "gap_reduction",
                        "quality_delta",
                        "censoring",
                        "invalid",
                    )
                }
                for vs in groups.values()
            ]
            contrast = {
                "model": model,
                "suite": suite,
                "control": baseline,
                "task_groups": len(groups),
                "unmatched_tasks": unmatched,
                "means": {
                    key: mean([v[key] for v in values if v[key] is not None])
                    for key in ("fresh_saving", "gap_reduction", "quality_delta", "censoring")
                },
                "decision": "inconclusive",
            }
            reasons = []
            if suite == "math":
                reasons.append("one run per method/problem; math results are descriptive")
            if len(groups) < min_groups:
                reasons.append("too few independent task groups")
            if unmatched or missing:
                reasons.append("planned primary runs are missing")
            if any(r["synthetic"] for r in runs):
                reasons.append("synthetic fixture results")
            if not reviewed:
                reasons.append("functional-artifact review pending")
            if any(v["gap_reduction"] is None or v["invalid"] for v in values):
                reasons.append("invalid, unmatched, or unreached progress checkpoints")
            if values and mean([v["censoring"] for v in values]) > max_censoring:
                reasons.append("continuation censoring exceeds preregistered limit")
            if values and suite != "math" and len(groups) >= min_groups:
                alpha = 0.05 / (3 * comparison_count)
                cis = {}
                for key in ("fresh_saving", "gap_reduction", "quality_delta"):
                    data = [v[key] for v in values if v[key] is not None]
                    if len(data) == len(values):
                        cis[key] = interval(data, seed, alpha)
                contrast["familywise_intervals"] = cis
                if not reasons and len(cis) == 3:
                    if (
                        cis["fresh_saving"][1] < minimum_effect
                        or cis["gap_reduction"][1] < minimum_effect
                        or cis["quality_delta"][1] < -quality_margin
                    ):
                        contrast["decision"] = "reject_preregistered_operational_hypothesis"
                    elif (
                        cis["fresh_saving"][0] >= minimum_effect
                        and cis["gap_reduction"][0] >= minimum_effect
                        and cis["quality_delta"][0] >= -quality_margin
                    ):
                        contrast["decision"] = "supports_preregistered_operational_hypothesis"
            contrast["limitations"] = reasons
            contrasts.append(contrast)
    return {
        "synthetic": any(r["synthetic"] for r in runs),
        "run_count": len(runs),
        "missing_planned_runs": missing,
        "artifact_review_verified": reviewed,
        "accounting_corrections": [
            {
                "model": r["model_label"],
                "task": r["task"],
                "condition": r["condition"],
                "reason": "ignored learned-policy training cost attached to a control condition",
                "ignored_training_cost_units": r["policy_training_cost_units"],
            }
            for r in runs
            if r["condition"] != "learned_clm" and r.get("policy_training_cost_units", 0) != 0
        ],
        "analysis_parameters": {
            "seed": seed,
            "minimum_effect_units": minimum_effect,
            "quality_margin": quality_margin,
            "min_groups": min_groups,
            "max_censoring": max_censoring,
            "multiple_comparison_count": comparison_count,
        },
        "run_summaries": summaries,
        "contrasts": contrasts,
        "result_set_hash": fingerprint(runs),
    }
