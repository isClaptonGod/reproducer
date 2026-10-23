import json
import math
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from clm_experiment.agent import Session, State
from clm_experiment.analysis import analyze
from clm_experiment.artifacts import clean_snapshot
from clm_experiment.cli import demo
from clm_experiment.experiment import DEFAULT_AGENT, Experiment
from clm_experiment.policy import ActionPolicy, dependence_reward, protocol_hash
from clm_experiment.prepare import prepare_browse
from clm_experiment.providers import DemoProvider, Reply
from clm_experiment.sandbox import FixtureSandbox, file_manifest, relative
from clm_experiment.tasks import build_corpus, evaluate, math_grade, prepare_math, search
from clm_experiment.types import Cost, Task, dump, task_identity


class MathTests(unittest.TestCase):
    def test_circle_objective_is_sum_of_unequal_radii(self):
        grade = math_grade("circles", {"circles": [[0.2, 0.2, 0.2], [0.8, 0.8, 0.1]]}, 2)
        self.assertAlmostEqual(grade.score, 0.3)
        with self.assertRaisesRegex(ValueError, "Overlapping"):
            math_grade("circles", {"circles": [[0.2, 0.2, 0.2], [0.3, 0.2, 0.2]]}, 2)

    def test_erdos_all_shifts_and_mass_constraint(self):
        self.assertEqual(math_grade("erdos", {"heights": [0.5] * 4}, 4).score, 0.5)
        self.assertEqual(math_grade("erdos", {"heights": [1, 0]}, 2).score, 1)
        self.assertEqual(math_grade("erdos", {"heights": [0, 1]}, 2).score, 1)
        with self.assertRaises(ValueError):
            math_grade("erdos", {"heights": [1, 1]}, 2)

    def test_distance_ratio_is_not_squared(self):
        grade = math_grade("ratio", {"points": [[0, 0], [1, 0], [0, 1]]}, 3)
        self.assertAlmostEqual(grade.score, math.sqrt(2))
        self.assertAlmostEqual(grade.details["squared_ratio"], 2)
        with self.assertRaisesRegex(ValueError, "Coincident"):
            math_grade("ratio", {"points": [[0, 0], [0, 0]]}, 2)

    def test_heilbronn_enclosing_triangle_has_unit_area(self):
        self.assertEqual(math_grade("heilbronn", {"points": [[0, 0], [2, 0], [0, 1]]}, 3).score, 1)
        with self.assertRaisesRegex(ValueError, "outside"):
            math_grade("heilbronn", {"points": [[0, 0], [2, 1], [0, 1]]}, 3)

    def test_nonfinite_and_boolean_coordinates_rejected(self):
        for invalid in (float("nan"), float("inf"), True):
            with self.assertRaises(ValueError):
                math_grade("ratio", {"points": [[0, 0], [invalid, 1]]}, 2)

    def test_all_four_prepared_instances_are_feasible(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            prepare_math(p)
            tasks = Task.load_many(p / "tasks.jsonl")
            self.assertEqual(len(tasks), 4)
            for task in tasks:
                grade = evaluate(task, Path(task.seed_dir))
                self.assertTrue(grade.valid, (task.id, grade))
                self.assertFalse(task.reached(grade, task.stages[0]))


class ArtifactTests(unittest.TestCase):
    def test_notes_removed_functional_files_retained(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            source = p / "source"
            source.mkdir()
            (source / "solver.py").write_text("print(42)\n")
            (source / "handoff.json").write_text('{"note":"remember this"}')
            (source / "summary.md").write_text("A dedicated memory note")
            audit = clean_snapshot(source, p / "clean", ["*.py", "*.json"], {}, [])
            self.assertEqual(audit["kept"], ["solver.py"])
            self.assertEqual(set(audit["removed"]), {"handoff.json", "summary.md"})

    def test_disguised_transcript_is_invalidated(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            source = p / "source"
            source.mkdir()
            text = " ".join(f"word{i}" for i in range(40))
            (source / "state.json").write_text(json.dumps({"state": text}))
            audit = clean_snapshot(
                source, p / "clean", ["*.json"], {}, [{"origin": "model", "content": text}]
            )
            self.assertTrue(audit["violations"])
            self.assertFalse((p / "clean/state.json").exists())

    def test_conversation_envelopes_rejected_without_exact_ngrams(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            source = p / "source"
            source.mkdir()
            dump(
                source / "solution.json",
                [{"role": "assistant", "content": str(i)} for i in range(3)],
            )
            audit = clean_snapshot(source, p / "clean", ["*.json"], {}, [])
            self.assertTrue(audit["violations"])

    def test_legitimate_source_is_not_rejected_for_matching_write_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            source = p / "source"
            source.mkdir()
            code = "\n".join(f"variable_{i} = {i}" for i in range(40))
            (source / "solver.py").write_text(code)
            message = {"tool": "write", "path": "solver.py", "content": code}
            audit = clean_snapshot(
                source,
                p / "clean",
                ["*.py"],
                {},
                [{"origin": "model", "content": json.dumps(message)}],
            )
            self.assertFalse(audit["violations"])

    def test_symlinks_and_parent_traversal_rejected(self):
        for name in ("/etc/passwd", "../outside", "a/../../outside"):
            with self.assertRaises(ValueError):
                relative(name)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p / "link").symlink_to("/etc/passwd")
            with self.assertRaisesRegex(ValueError, "Symlink"):
                file_manifest(p)


class PolicyTests(unittest.TestCase):
    def test_reward_subtracts_retained_change_and_action_cost(self):
        def pair(fresh, retained):
            return {"fresh": {"charged_cost": fresh}, "retained": {"charged_cost": retained}}

        # Improving both branches equally provides no history-dependence reward.
        self.assertEqual(dependence_reward(pair(20, 10), pair(15, 5), 2, True, 100), -2)
        self.assertEqual(dependence_reward(pair(20, 10), pair(12, 10), 2, True, 100), 6)
        # Making retained continuation worse cannot manufacture a positive learning signal.
        self.assertEqual(dependence_reward(pair(20, 10), pair(20, 20), 2, True, 100), -2)
        self.assertEqual(dependence_reward(pair(20, 10), pair(12, 10), 2, False, 100), -102)

    def test_learning_changes_action_selection_and_frozen_choice_does_not_update(self):
        policy = ActionPolicy("model", "protocol")
        for _ in range(4):
            policy.update("edgebench", "entrypoint", -10, "group-a", "a")
            policy.update("edgebench", "verification", 5, "group-b", "b")
        frozen = asdict(policy)
        self.assertEqual(policy.choose("edgebench", 1, False), "verification")
        self.assertEqual(policy.choose("math", 1, False), "verification")
        self.assertEqual(frozen, asdict(policy))

    def test_untrained_policy_cannot_masquerade_as_learned(self):
        with self.assertRaises(ValueError):
            ActionPolicy("model", "protocol").choose("math", 0, False)


class RetrievalTests(unittest.TestCase):
    def test_browse_import_splits_labels_and_checks_answers_independently_of_citations(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p / "corpus.jsonl").write_text(
                json.dumps({"docid": "d", "text": "fixture text"}) + "\n"
            )
            build_corpus(p / "corpus.jsonl", p / "index.db")
            (p / "queries.jsonl").write_text(
                "".join(
                    json.dumps(
                        {
                            "query_id": str(i),
                            "query": f"Question {i}?",
                            "answer": "gold-secret-answer",
                        }
                    )
                    + "\n"
                    for i in range(2)
                )
            )
            (p / "qrels").write_text("0 0 d 1\n1 0 d 1\n")
            counts = prepare_browse(
                p / "queries.jsonl", p / "qrels", p / "index.db", p / "prepared", 0.5, 1729, None
            )
            self.assertEqual(counts, {"train": 1, "test": 1})
            tasks = Task.load_many(p / "prepared/tasks.jsonl")
            for task in tasks:
                self.assertNotIn("gold-secret-answer", task.prompt)
                seed = Path(task.seed_dir)
                dump(seed / "solution.json", {"answer": "", "citations": ["d"]})
                self.assertEqual(evaluate(task, seed).score, 0.5)
                dump(seed / "solution.json", {"answer": "gold-secret-answer", "citations": []})
                self.assertEqual(evaluate(task, seed).score, 1.0)

    def test_fixed_corpus_search_uses_actual_documents(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p / "corpus.jsonl").write_text(
                json.dumps({"docid": "a", "text": "The winner is Ada."})
                + "\n"
                + json.dumps({"docid": "b", "text": "Irrelevant text."})
                + "\n"
            )
            self.assertEqual(build_corpus(p / "corpus.jsonl", p / "index.db"), 2)
            docs = search(str(p / "index.db"), "winner Ada")
            self.assertEqual(docs[0]["docid"], "a")
            self.assertIn("Ada", docs[0]["text"])
            with self.assertRaises(ValueError):
                build_corpus(p / "corpus.jsonl", p / "index.db")


class ContinuationTests(unittest.TestCase):
    def setup_task(self, root):
        seed = root / "seed"
        dump(seed / "solution.json", {"value": 0})
        return Task(
            "test",
            "fixture",
            "test",
            "test",
            "Increase value",
            [2],
            4,
            seed_dir=str(seed),
            evaluator={"kind": "fixture"},
        )

    def test_fresh_branch_gets_no_history_or_dedicated_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task = self.setup_task(root)
            config = {"agent": {**DEFAULT_AGENT, "grade_every": 1}, "models": {}}
            exp = Experiment(config, DemoProvider(), root / "run", fixture=True)
            sb = FixtureSandbox(task, Path(task.seed_dir))
            try:
                state = State()
                state.message("user", "UNIQUE_SECRET_HISTORY_SENTINEL", "harness")
                sb.write("handoff.txt", "MEMORY_SENTINEL", memory=True)
                s = Session(task, sb, DemoProvider(), config["agent"], state)
                initial = file_manifest(Path(task.seed_dir))
                audit = exp.checkpoint(task, s, root / "checkpoint", initial)
                self.assertTrue(audit["valid"])
                exp.pair(task, root / "checkpoint", state, 2, root / "pair", 1729)
                fresh = (root / "pair/fresh/transcript.json").read_text()
                retained = (root / "pair/retained/transcript.json").read_text()
                self.assertNotIn("UNIQUE_SECRET_HISTORY_SENTINEL", fresh)
                self.assertIn("UNIQUE_SECRET_HISTORY_SENTINEL", retained)
                clean = FixtureSandbox(task, root / "checkpoint/functional")
                try:
                    with self.assertRaises(FileNotFoundError):
                        clean.read("handoff.txt", memory=True)
                finally:
                    clean.close()
            finally:
                sb.close()

    def test_failed_probes_are_censored_and_not_rewarded_as_cheap(self):
        class NeverProgress(DemoProvider):
            def complete(self, messages, seed):
                return Reply('{"tool":"read","path":"solution.json"}', Cost(calls=1))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task = self.setup_task(root)
            config = {"agent": {**DEFAULT_AGENT, "probe_calls": 1}}
            exp = Experiment(config, NeverProgress(), root / "run", fixture=True)
            sb = FixtureSandbox(task, Path(task.seed_dir))
            try:
                s = Session(task, sb, exp.provider, config["agent"])
                exp.checkpoint(task, s, root / "checkpoint", file_manifest(Path(task.seed_dir)))
                pair = exp.pair(task, root / "checkpoint", s.state, 2, root / "pair", 1)
                self.assertTrue(pair["fresh"]["censored"])
                self.assertGreaterEqual(pair["fresh"]["charged_cost"], 200)
                self.assertEqual(pair["fresh"]["cost"]["calls"], 1)
            finally:
                sb.close()

    def test_exact_training_task_cannot_be_relabelled_for_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task = self.setup_task(root)
            config = {"agent": DEFAULT_AGENT}
            p = ActionPolicy(
                "scripted-fixture",
                protocol_hash(config),
                train_ids=[task_identity(task)],
                episodes=1,
            )
            task.id, task.group = "renamed", "renamed-group"
            exp = Experiment(config, DemoProvider(), root / "out", fixture=True)
            with self.assertRaisesRegex(ValueError, "overlaps"):
                exp.run(task, "learned_clm", p, 1, "fixture")

    def test_context_edit_is_read_back_and_pinned_task_survives(self):
        class Editor(DemoProvider):
            def complete(self, messages, seed):
                return Reply(
                    json.dumps(
                        {
                            "tool": "context",
                            "messages": [{"role": "user", "content": "compact state"}],
                        }
                    ),
                    Cost(calls=1),
                )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task = self.setup_task(root)
            sb = FixtureSandbox(task, Path(task.seed_dir))
            try:
                state = State(history=[{"role": "user", "content": "old history"}])
                s = Session(task, sb, Editor(), DEFAULT_AGENT, state)
                self.assertEqual(s.step(0), "continue")
                self.assertEqual(s.state.history[0]["content"], "compact state")
                self.assertNotIn("old history", json.dumps(s.state.history))
                from clm_experiment.agent import messages

                self.assertIn(task.prompt, messages(task, s.state, 32000)[1]["content"])
            finally:
                sb.close()


class EndToEndTests(unittest.TestCase):
    def test_offline_experiment_runs_actual_learning_and_all_three_conditions(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "demo"
            demo(path)
            report = json.loads((path / "report.json").read_text())
            policy = json.loads((path / "training/policy.json").read_text())
            self.assertEqual(policy["episodes"], 4)
            self.assertEqual(report["run_count"], 6)
            self.assertTrue(report["synthetic"])
            self.assertFalse(report["missing_planned_runs"])
            for run in report["run_summaries"]:
                self.assertEqual(run["invalid_or_unreached_stages"], 0)
                self.assertEqual(run["task_target_success"], 1)
            for contrast in report["contrasts"]:
                self.assertEqual(contrast["decision"], "inconclusive")
            # Repeated checkpoints don't turn two held-out tasks into six independent samples.
            self.assertTrue(all(c["task_groups"] == 2 for c in report["contrasts"]))
            path_run = next((path / "evaluation").rglob("run.json"))
            path_run.unlink()
            self.assertEqual(len(analyze(path / "evaluation")["missing_planned_runs"]), 1)


if __name__ == "__main__":
    unittest.main()
