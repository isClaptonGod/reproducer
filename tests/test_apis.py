import email
import io
import json
import os
import tarfile
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from clm_experiment.providers import Provider, http_json
from clm_experiment.tasks import sforge_grade
from clm_experiment.types import Task


class Handler(BaseHTTPRequestHandler):
    calls = []

    def log_message(self, *args):
        pass

    def send_json(self, body, code=200):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        data = self.rfile.read(int(self.headers["Content-Length"]))
        self.calls.append((self.path, data, dict(self.headers)))
        if self.path == "/api/v1/submit":
            self.send_json({"submission_id": "smoke", "status": "queued"})
            return
        body = json.loads(data)
        if self.path == "/v1/chat/completions":
            self.send_json(
                {
                    "model": body["model"],
                    "choices": [{"message": {"content": '{"tool":"finish"}'}}],
                    "usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 10,
                        "prompt_tokens_details": {"cached_tokens": 40},
                    },
                }
            )
        elif self.path == "/v1/messages":
            self.send_json(
                {
                    "model": body["model"],
                    "content": [{"type": "text", "text": '{"tool":"finish"}'}],
                    "usage": {
                        "input_tokens": 30,
                        "output_tokens": 10,
                        "cache_read_input_tokens": 50,
                        "cache_creation_input_tokens": 20,
                    },
                }
            )
        elif self.path == "/tokenize":
            self.send_json({"count": 123})
        elif self.path == "/v1/messages/count_tokens":
            self.send_json({"input_tokens": 124})
        elif self.path == "/api/v1/register":
            self.send_json({"token": "fixture-session-token"})
        else:
            self.send_json({"error": "echoed-secret-must-not-appear"}, 403)

    def do_GET(self):
        self.send_json(
            {
                "status": "completed",
                "report": {
                    "valid": True,
                    "timed_out": False,
                    "score_0_100": 42.5,
                    "total_tests": 1,
                    "passed": 1,
                },
                "error": None,
            }
        )


class APIContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def setUp(self):
        Handler.calls.clear()

    def provider(self, kind):
        return Provider(
            {
                "provider": kind,
                "model": "fixture-model",
                "base_url": self.url + "/v1",
                "require_key": False,
                "token_counter": "provider",
                "send_seed": True,
                "usd_per_million_tokens": {
                    "input": 2,
                    "output": 4,
                    "cache_read": 0.2,
                    "cache_write": 2.5,
                },
            }
        )

    def test_openai_tokenization_usage_and_seed(self):
        p = self.provider("openai")
        messages = [{"role": "system", "content": "system"}, {"role": "user", "content": "task"}]
        self.assertEqual(p.count_tokens(messages), 123)
        reply = p.complete(messages, 1729)
        self.assertEqual(reply.cost.input_tokens, 100)
        self.assertEqual(reply.cost.cached_tokens, 40)
        self.assertAlmostEqual(reply.cost.usd, 168 / 1e6)
        self.assertEqual(json.loads(Handler.calls[-1][1])["seed"], 1729)
        self.assertEqual(p.observed_models, {"fixture-model"})

    def test_anthropic_separates_system_counts_cache_and_does_not_send_seed(self):
        p = self.provider("anthropic")
        messages = [{"role": "system", "content": "system"}, {"role": "user", "content": "task"}]
        self.assertEqual(p.count_tokens(messages), 124)
        reply = p.complete(messages, 1729)
        body = json.loads(Handler.calls[-1][1])
        self.assertEqual(body["system"], "system")
        self.assertNotIn("seed", body)
        self.assertEqual(len(body["messages"]), 1)
        self.assertEqual(reply.cost.input_tokens, 100)
        self.assertEqual(reply.cost.cache_write_tokens, 20)
        self.assertAlmostEqual(reply.cost.usd, 160 / 1e6)

    def test_error_does_not_echo_response_secrets(self):
        with self.assertRaises(RuntimeError) as ctx:
            http_json(self.url + "/denied", {}, {})
        self.assertIn("403", str(ctx.exception))
        self.assertNotIn("echoed-secret", str(ctx.exception))

    def test_model_override_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "override"):
            Provider(
                {
                    "provider": "openai",
                    "model": "actual",
                    "base_url": self.url,
                    "require_key": False,
                    "extra_body": {"model": "different"},
                }
            )

    def test_sforge_multipart_archive_and_hidden_judge_response(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "solver.py").write_text("print(42)")
            (root / "private-test.txt").write_text("not a submission")
            task = Task(
                "edge",
                "edgebench",
                "test",
                "edge",
                "Task",
                [20],
                80,
                evaluator={
                    "kind": "sforge",
                    "task_id": "fixture-task",
                    "submit_paths": ["solver.py"],
                },
            )
            with patch.dict(
                os.environ, {"SFORGE_ADMIN_SECRET": "test-only-admin", "SFORGE_JUDGE_URL": self.url}
            ):
                result = sforge_grade(task, root)
            self.assertTrue(result.valid)
            self.assertEqual(result.score, 42.5)
            path, body, headers = Handler.calls[-1]
            self.assertEqual(path, "/api/v1/submit")
            mime = email.message_from_bytes(
                (
                    "Content-Type: " + headers["Content-Type"] + "\r\nMIME-Version: 1.0\r\n\r\n"
                ).encode()
                + body
            )
            archives = [p.get_payload(decode=True) for p in mime.walk() if p.get_filename()]
            self.assertEqual(len(archives), 1)
            with tarfile.open(fileobj=io.BytesIO(archives[0]), mode="r:gz") as tar:
                self.assertEqual(tar.getnames(), ["solver.py"])
