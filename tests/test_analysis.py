import json
import tempfile
import unittest
from pathlib import Path

from clm_experiment.analysis import analyze
from clm_experiment.policy import CONDITIONS
from clm_experiment.prepare import file_sha
from clm_experiment.types import dump


class AnalysisTests(unittest.TestCase):
    def records(self, root, learned_fresh, learned_retained):
        # Fabricated records exercise the decision rule only; not experimental findings.
        for task in range(8):
            for condition in CONDITIONS:
                learned = condition == "learned_clm"
                pair = {
                    "fresh": {"charged_cost": learned_fresh if learned else 10, "censored": False},
                    "retained": {
                        "charged_cost": learned_retained if learned else 5,
                        "censored": False,
                    },
                }
                value = {
                    "model_label": "model",
                    "model": "model",
                    "task": f"task-{task}",
                    "group": f"group-{task}",
                    "suite": "edgebench",
                    "condition": condition,
                    "task_identity": f"instance-{task}",
                    "protocol": "test-protocol",
                    "synthetic": False,
                    "primary_cost_units": 10,
                    "target_reached": True,
                    "agent_config": {"probe_cost_cap": 100, "failure_penalty": 100},
                    "stages": [{"stage_index": 0, "stage": 1, "status": "measured", "after": pair}],
                }
                dump(root / f"task-{task}" / condition / "run.json", value)
        review = root / "review.json"
        dump(
            review,
            {
                "reviewer": "unit-test fixture, not real research review",
                "functional_artifacts_only": True,
                "reviewed_runs": {
                    str(p.relative_to(root)): file_sha(p) for p in root.rglob("run.json")
                },
            },
        )
        return review

    def test_can_reject_a_meaningful_gain_when_the_trained_policy_is_worse(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            review = self.records(root, learned_fresh=20, learned_retained=10)
            report = analyze(root, artifact_review=review)
            self.assertEqual(len(report["contrasts"]), 2)
            for contrast in report["contrasts"]:
                self.assertEqual(
                    contrast["decision"], "reject_preregistered_operational_hypothesis"
                )
                self.assertEqual(contrast["task_groups"], 8)

    def test_can_support_real_fresh_savings_and_smaller_history_gap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            review = self.records(root, learned_fresh=4, learned_retained=3)
            report = analyze(root, artifact_review=review)
            for contrast in report["contrasts"]:
                self.assertEqual(
                    contrast["decision"], "supports_preregistered_operational_hypothesis"
                )

    def test_review_and_missing_stages_cannot_be_silently_bypassed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.records(root, learned_fresh=20, learned_retained=10)
            report = analyze(root)
            self.assertTrue(all(c["decision"] == "inconclusive" for c in report["contrasts"]))
            path = next(root.rglob("run.json"))
            data = json.loads(path.read_text())
            data["stages"][0]["status"] = "unreached"
            dump(path, data)
            report = analyze(root)
            self.assertTrue(all(c["decision"] == "inconclusive" for c in report["contrasts"]))
