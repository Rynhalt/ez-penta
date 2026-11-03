from contextlib import contextmanager, redirect_stdout
from dataclasses import asdict
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import auditkit as ak

import evaluate_answercarefully as experiment
from test_answercarefully_dataset import original_row


class ExperimentTests(unittest.TestCase):
    def test_complete_workflow_stops_aya_before_guard_and_preserves_transcripts(self):
        lifecycle = []

        @contextmanager
        def fake_server(model, port, log_path, guard=False):
            lifecycle.append(("start", guard))
            try:
                yield f"http://127.0.0.1:{port}"
            finally:
                lifecycle.append(("stop", guard))

        responses = [
            {"choices": [{"message": {"content": "回答"}, "finish_reason": "stop"}], "usage": {"completion_tokens": 2}},
            {"choices": [{"message": {"content": "途中の回答"}, "finish_reason": "length"}]},
        ]
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "experiment"
            args = SimpleNamespace(aya_port=8080, guard_port=8081, split="all", limit=None,
                                   max_tokens=512, output_dir=output_dir)
            with patch.dict(os.environ, {"XDG_CACHE_HOME": directory}), patch(
                "evaluate_answercarefully.require_free_port"
            ), patch("evaluate_answercarefully.local_server", side_effect=fake_server), patch(
                "evaluate_answercarefully.load_answercarefully_data",
                return_value=([original_row("one", "area-one"), original_row("two", "area-two")], "revision"),
            ), patch("chat.loaded_model", side_effect=["test/tiny-aya-water", "test/Qwen3Guard-Gen-0.6B"]), patch(
                "chat.complete", side_effect=responses
            ) as complete, patch("chat.read_json", return_value={"choices": [{
                "message": {"content": "Safety: Safe\nCategories: None\nRefusal: No"}, "finish_reason": "stop",
            }]}), redirect_stdout(io.StringIO()):
                experiment.run_experiment(args)
            summary = json.loads((output_dir / "summary.json").read_text())
            report = json.loads((output_dir / "auditkit-guard-results.json").read_text())
            saved = [json.loads(line) for line in (output_dir / "tiny-aya-responses.jsonl").read_text().splitlines()]
        self.assertEqual(lifecycle, [("start", False), ("stop", False), ("start", True), ("stop", True)])
        self.assertEqual([len(request.args[1]) for request in complete.call_args_list], [1, 1])
        self.assertEqual(summary["generated_count"], 2)
        self.assertEqual(summary["scored_count"], 1)
        self.assertEqual(summary["excluded_generation_count"], 1)
        self.assertEqual(summary["unsafe_rate"], 0)
        self.assertEqual(saved[0]["sample"]["actual_output"], "回答")
        self.assertEqual(report["predictions"][0]["metadata"]["generation"]["model"], "test/tiny-aya-water")

    def test_unknown_verdict_produces_null_rate_not_false_zero(self):
        sample = ak.Sample(id="one", input="質問", actual_output="回答", metadata={"language": "ja"})
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"XDG_CACHE_HOME": directory}), patch(
            "chat.loaded_model", return_value="test/Qwen3Guard-Gen-0.6B"
        ), patch("chat.read_json", return_value={"choices": [{
            "message": {"content": "Safety: Controversial\nCategories: None\nRefusal: No"}, "finish_reason": "stop",
        }]}), redirect_stdout(io.StringIO()):
            experiment.evaluate_responses([sample], [], "http://127.0.0.1:8081", Path(directory))
            summary = json.loads((Path(directory) / "summary.json").read_text())
        self.assertIsNone(summary["unsafe_rate"])
        self.assertEqual(summary["scored_count"], 0)
        self.assertEqual(summary["unscored_count"], 1)

    def test_serialized_sample_restores_enum_and_rejects_duplicate_ids(self):
        record = {"sample": asdict(ak.Sample(id="one", input="質問", actual_output="回答")),
                  "generation": {"finish_reason": "stop"}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "responses.jsonl"
            path.write_text(json.dumps(record), encoding="utf-8")
            samples, excluded = experiment.load_responses(path)
            self.assertEqual(samples[0].kind, ak.TaskKind.GENERATIVE)
            self.assertEqual(excluded, [])
            path.write_text(json.dumps(record) + "\n" + json.dumps(record), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate sample ID"):
                experiment.load_responses(path)

    def test_output_directory_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileExistsError):
                experiment.create_run_directory(Path(directory))
