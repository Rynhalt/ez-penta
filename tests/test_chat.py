import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch
from urllib.error import HTTPError

import chat


class ClientTests(unittest.TestCase):
    def setUp(self):
        self.requests = []
        self.health = {"status": "ok"}
        self.health_code = 200
        self.completion_code = 200
        self.malformed = False
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                pass

            def respond(self, status, payload):
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                body = b"invalid json" if owner.malformed else json.dumps(
                    payload, ensure_ascii=False
                ).encode("utf-8")
                self.wfile.write(body)

            def do_GET(self):
                if self.path == "/health":
                    self.respond(owner.health_code, owner.health)
                elif self.path == "/v1/models":
                    self.respond(200, {"data": [{"id": chat.DEFAULT_MODEL}]})
                else:
                    self.respond(404, {})

            def do_POST(self):
                body = self.rfile.read(int(self.headers["Content-Length"]))
                owner.requests.append((self.path, self.headers["Content-Type"], json.loads(body)))
                self.respond(owner.completion_code, {
                    "model": chat.DEFAULT_MODEL,
                    "choices": [{"message": {"content": "こんにちは、世界！"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 12, "completion_tokens": 5},
                    "timings": {"predicted_per_second": 20.5},
                })

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def test_completion_round_trip_preserves_japanese_and_settings(self):
        messages = [{"role": "user", "content": "日本語で挨拶して"}]
        result = chat.complete(self.base_url, messages, 42)
        self.assertEqual(result["choices"][0]["message"]["content"], "こんにちは、世界！")
        self.assertEqual(self.requests, [(
            "/v1/chat/completions", "application/json",
            {"messages": messages, "temperature": 0.2, "max_tokens": 42},
        )])

    def test_health_and_model_discovery(self):
        self.assertEqual(chat.server_status(self.base_url), "ready")
        self.assertEqual(chat.loaded_model(self.base_url), chat.DEFAULT_MODEL)
        self.health_code = 503
        self.assertEqual(chat.server_status(self.base_url), "loading")
        self.health_code = 200
        for payload in ([], {"status": "unexpected"}):
            self.health = payload
            self.assertEqual(chat.server_status(self.base_url), "unknown")

    def test_http_and_json_errors_are_reported(self):
        self.completion_code = 500
        with self.assertRaises(HTTPError):
            chat.complete(self.base_url, [], 42)
        self.completion_code = 200
        self.malformed = True
        with self.assertRaises(json.JSONDecodeError):
            chat.complete(self.base_url, [], 42)
        self.assertEqual(chat.server_status(self.base_url), "unknown")

    def test_ready_server_is_reused(self):
        with patch("chat.subprocess.Popen") as start:
            self.assertIsNone(chat.ensure_server(self.base_url, chat.DEFAULT_MODEL, 8080, 2048))
        start.assert_not_called()

    def test_cli_history_reset_stats_and_utf8_log(self):
        output = io.StringIO()
        original_directory = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                with patch("sys.argv", ["chat.py", "--port", str(self.server.server_port), "--save", "--stats"]), patch(
                    "builtins.input", side_effect=["こんにちは", "What did I say?", "/reset", "Hello", "/exit"]
                ), contextlib.redirect_stdout(output):
                    chat.main()
                logs = list(Path("runs").glob("session-*.jsonl"))
                self.assertEqual(len(logs), 1)
                raw = logs[0].read_text(encoding="utf-8")
                records = [json.loads(line) for line in raw.splitlines()]
            finally:
                os.chdir(original_directory)
        self.assertEqual([len(request[2]["messages"]) for request in self.requests], [1, 3, 1])
        self.assertEqual(len(records), 3)
        self.assertIn("こんにちは、世界！", raw)
        self.assertEqual(records[1]["messages"], self.requests[1][2]["messages"])
        self.assertEqual(records[0]["finish_reason"], "stop")
        self.assertEqual(records[0]["timings"]["predicted_per_second"], 20.5)
        self.assertIn("12 prompt tokens, 5 response tokens, 20.5 tokens/sec", output.getvalue())
        self.assertIn("Conversation cleared.", output.getvalue())


if __name__ == "__main__":
    unittest.main()
