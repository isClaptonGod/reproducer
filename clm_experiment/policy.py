from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass, field
from pathlib import Path

from .types import fingerprint

ARMS = {
    "entrypoint": "Create a reproducible entrypoint for resuming the computation from functional "
    "artifacts, using explicit inputs and configuration. Verify that it runs.",
    "verification": "Create executable checks for invariants and artifact validity, and run them. "
    "Make failures informative enough for a new engineer to diagnose.",
    "structured_state": "Replace implicit task state with a typed, machine-readable functional "
    "representation consumed by the program. Validate it on the current artifacts.",
    "checkpoint": "Make the computation restartable using functional checkpoints and idempotent "
    "commands. Verify restart behavior. Save parameters and results, not reasoning notes.",
}

CONDITIONS = ("standard_clm", "handoff_clm", "learned_clm")
HANDOFF = (
    "Optimize this environment for a clear handoff. Keep useful persistent state, good memory "
    "instructions, and concise handoff notes in /memory. You may also improve functional "
    "artifacts. Use the same tools and budget as every other condition."
)
STANDARD = "Use this engineering window as you judge useful with the standard CLM workflow."
PRESERVE = (
    " Preserve the current task outcome: do not improve or degrade the measured solution "
    "during this engineering window. Do not copy conversation or reasoning into functional "
    "files. Notes belong in /memory. All engineering work is charged."
)


def protocol_hash(config: dict) -> str:
    implementation = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        implementation.update(path.name.encode() + b"\0" + path.read_bytes())
    return fingerprint(
        {
            "agent": config["agent"],
            "models": config.get("models", {}),
            "resources": config.get("resources", {}),
            "arms": ARMS,
            "version": 2,
            "implementation_sha256": implementation.hexdigest(),
            "reward": "min(retained_adjusted_gain,fresh_gain)-engineering_cost",
        }
    )


def dependence_reward(
    before: dict, after: dict, action_cost: float, valid: bool, invalid_penalty: float
) -> float:
    if not valid:
        return -invalid_penalty - action_cost
    before_gap = before["fresh"]["charged_cost"] - before["retained"]["charged_cost"]
    after_gap = after["fresh"]["charged_cost"] - after["retained"]["charged_cost"]
    fresh_gain = before["fresh"]["charged_cost"] - after["fresh"]["charged_cost"]
    # Inflating retained history must not earn reward without cheaper fresh work.
    return min(before_gap - after_gap, fresh_gain) - action_cost


@dataclass
class ActionPolicy:
    model: str
    protocol: str
    counts: dict = field(default_factory=dict)
    means: dict = field(default_factory=dict)
    train_groups: list[str] = field(default_factory=list)
    train_ids: list[str] = field(default_factory=list)
    episodes: int = 0
    training_cost_units: float = 0.0

    def choose(self, suite: str, seed: int, training: bool) -> str:
        counts = self.counts.get(suite, {})
        means = self.means.get(suite, {})
        arms = list(ARMS)
        if training:
            missing = [a for a in arms if counts.get(a, 0) == 0]
            if missing:
                return random.Random(seed).choice(missing)
            total = sum(counts.values())
            return max(
                arms, key=lambda a: means[a] + math.sqrt(2 * math.log(total + 1) / counts[a])
            )
        if not self.episodes:
            raise ValueError("Learned condition requires an actually trained policy")
        # Unseen suite: pool training rewards across suites, without inspecting held-out outcomes.
        if not counts:
            counts = {a: sum(c.get(a, 0) for c in self.counts.values()) for a in arms}
            means = {
                a: sum(self.means[s].get(a, 0) * c.get(a, 0) for s, c in self.counts.items())
                / max(counts[a], 1)
                for a in arms
            }
        seen = [a for a in arms if counts.get(a, 0)]
        return max(seen, key=lambda a: means[a])

    def update(self, suite: str, arm: str, reward: float, group: str, identity: str):
        if not math.isfinite(reward):
            raise ValueError("Nonfinite policy reward")
        counts = self.counts.setdefault(suite, {})
        means = self.means.setdefault(suite, {})
        counts[arm] = counts.get(arm, 0) + 1
        means[arm] = means.get(arm, 0) + (reward - means.get(arm, 0)) / counts[arm]
        self.episodes += 1
        if group not in self.train_groups:
            self.train_groups.append(group)
        if identity not in self.train_ids:
            self.train_ids.append(identity)
