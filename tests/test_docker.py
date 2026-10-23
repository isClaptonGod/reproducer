import os
import tempfile
import unittest
from pathlib import Path

from clm_experiment.experiment import DEFAULT_AGENT, Experiment
from clm_experiment.providers import DemoProvider
from clm_experiment.sandbox import DockerSandbox
from clm_experiment.types import Task, dump


@unittest.skipUnless(
    os.environ.get("CLM_TEST_DOCKER") == "1", "set CLM_TEST_DOCKER=1 for Docker checks"
)
class DockerTests(unittest.TestCase):
    def test_whole_experiment_uses_real_docker_continuation_forks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dump(root / "seed/solution.json", {"value": 0})
            task = Task(
                "docker-pipeline",
                "fixture",
                "test",
                "docker-pipeline",
                "Increase value",
                [1],
                2,
                seed_dir=str(root / "seed"),
                evaluator={"kind": "fixture"},
            )
            config = {
                "agent": {
                    **DEFAULT_AGENT,
                    "engineering_calls": 1,
                    "grade_every": 1,
                    "probe_calls": 2,
                    "max_primary_calls": 8,
                    "progress_interval": 2,
                }
            }
            result = Experiment(config, DemoProvider(), root / "runs").run(
                task, "standard_clm", None, 1729, "scripted-docker-fixture"
            )
            self.assertTrue(result["synthetic"])
            self.assertTrue(result["target_reached"])
            self.assertEqual(result["stages"][0]["status"], "measured")
            self.assertTrue(result["stages"][0]["after"]["fresh"]["success"])
            self.assertTrue(result["stages"][0]["after"]["retained"]["success"])

    def test_tools_isolation_snapshot_and_fresh_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dump(root / "seed/solution.json", {"value": 1})
            task = Task("docker", "fixture", "test", "docker", "Test", [1], 2)
            sb = DockerSandbox(task, root / "seed")
            try:
                result = sb.run("python3 -c 'print(6*7)'", 5)
                self.assertEqual(result["exit_code"], 0)
                self.assertEqual(result["output"].strip(), "42")
                sb.write("handoff.txt", "memory sentinel", memory=True)
                # The root image is read-only; model API credentials aren't injected.
                result = sb.run("touch /outside-work", 5)
                self.assertNotEqual(result["exit_code"], 0)
                result = sb.run(
                    "python3 -c 'import os; print(\"ANTHROPIC_API_KEY\" in os.environ)'", 5
                )
                self.assertIn("False", result["output"])
                timeout = sb.run("python3 -c 'import time; time.sleep(30)'", 1)
                self.assertEqual(timeout["exit_code"], 124)
                sb.snapshot(root / "snapshot", root / "memory")
            finally:
                sb.close()
            fresh = DockerSandbox(task, root / "snapshot")
            retained = DockerSandbox(task, root / "snapshot", root / "memory")
            try:
                self.assertIn("solution.json", fresh.list_files())
                self.assertEqual(fresh.read("solution.json"), retained.read("solution.json"))
                with self.assertRaises(RuntimeError):
                    fresh.read("handoff.txt", memory=True)
                self.assertEqual(retained.read("handoff.txt", memory=True), "memory sentinel")
            finally:
                fresh.close()
                retained.close()
