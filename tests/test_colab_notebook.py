"""Validate notebook Python and exercise its inference cells without CUDA/models."""

from collections import Counter
from contextlib import contextmanager, redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch

from answercarefully_dataset import select_samples
from evaluate_answercarefully import (
    generate_responses, load_responses, evaluate_responses, write_json,
)
from test_answercarefully_dataset import original_row


NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "answercarefully_colab.ipynb"


class ColabNotebookTests(unittest.TestCase):
    def setUp(self):
        self.notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        self.code_cells = ["".join(cell["source"]) for cell in self.notebook["cells"]
                           if cell["cell_type"] == "code"]

    def cell_containing(self, text):
        matches = [source for source in self.code_cells if text in source]
        self.assertEqual(len(matches), 1, f"Expected one cell containing {text!r}")
        return matches[0]

    def test_notebook_is_valid_python_with_valid_execution_state(self):
        self.assertEqual(self.notebook["nbformat"], 4)
        ids = [cell["id"] for cell in self.notebook["cells"]]
        self.assertEqual(len(ids), len(set(ids)))
        for index, cell in enumerate(self.notebook["cells"]):
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), f"{NOTEBOOK}:cell-{index}", "exec")
                # Users can save progress from Colab; keep their outputs intact.
                self.assertIsInstance(cell["outputs"], list)
                self.assertTrue(cell["execution_count"] is None
                                or isinstance(cell["execution_count"], int))

    def test_notebook_phases_preserve_saved_answers_and_unknown_verdicts(self):
        lifecycle = []

        @contextmanager
        def fake_server(model, port, log_path, guard=False, *, model_path, gpu_layers):
            self.assertEqual(gpu_layers, "all")
            self.assertEqual(model_path.name, "guard.gguf" if guard else "aya.gguf")
            lifecycle.append(("start", guard))
            try:
                yield f"http://127.0.0.1:{port}"
            finally:
                lifecycle.append(("stop", guard))

        replies = [{"choices": [{"message": {"content": answer}, "finish_reason": finish}]}
                   for answer, finish in [("回答A", "stop"), ("途中", "length"), ("回答B", "stop")]]
        guard_replies = [{"choices": [{"message": {"content": text}, "finish_reason": "stop"}]}
                         for text in ["Safety: Safe\nCategories: None\nRefusal: No",
                                      "Safety: Controversial\nCategories: None\nRefusal: No"]]
        source_rows = [original_row(str(i), "example-area") for i in range(3)]
        for i, row in enumerate(source_rows):
            row["meta-mlmc"]["text-English"] = f"English question {i}"
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            namespace = {
                "time": time, "subprocess": subprocess, "Counter": Counter, "json": json,
                "chat": __import__("chat"), "GUARD_MODEL": "test/Qwen3Guard-Gen-0.6B",
                "local_server": fake_server, "generate_responses": generate_responses,
                "load_responses": load_responses, "evaluate_responses": evaluate_responses,
                "write_json": write_json, "MAX_TOKENS": 1024,
                "SPLIT": "dev", "LIMIT": 3, "PROMPT_LANGUAGE": "en",
                "select_samples": select_samples,
                "load_answercarefully_data": lambda split: (source_rows, "revision"),
                "run_dir": run_dir, "responses_path": run_dir / "tiny-aya-responses.jsonl",
                "model_paths": {"aya": Path("aya.gguf"), "guard": Path("guard.gguf")},
                "phase_timings": {},
            }
            with patch.dict(os.environ, {"XDG_CACHE_HOME": directory}), patch(
                "chat.loaded_model", side_effect=["test/tiny-aya-water", "test/Qwen3Guard-Gen-0.6B"]
            ), patch("chat.complete", side_effect=replies) as complete, patch(
                "chat.read_json", side_effect=guard_replies
            ) as guard_http, patch("subprocess.run"), redirect_stdout(io.StringIO()):
                exec(self.cell_containing("dataset, dataset_revision = load_answercarefully_data"), namespace)
                exec(self.cell_containing("generate_responses(samples,"), namespace)
                exec(self.cell_containing("records = [json.loads"), namespace)
                exec(self.cell_containing("completed_samples, excluded = load_responses"), namespace)
            summary = json.loads((run_dir / "summary.json").read_text())
            timings = json.loads((run_dir / "timings.json").read_text())
            report = json.loads((run_dir / "auditkit-guard-results.json").read_text())
            self.assertEqual(len(namespace["records"]), 3)
            saved_sample = namespace["records"][0]["sample"]
            self.assertEqual(saved_sample["id"], "0:en")
            self.assertEqual(saved_sample["metadata"]["language"], "en")
            self.assertEqual(saved_sample["metadata"]["reference_language"], "ja")
            self.assertEqual(saved_sample["metadata"]["original_prompt_ja"], source_rows[0]["text"])
            self.assertEqual(summary["scored_count"], 1)
            self.assertEqual(summary["unscored_count"], 1)
            self.assertEqual(summary["excluded_generation_count"], 1)
            self.assertEqual(summary["unsafe_rate"], 0.0)
            self.assertEqual(len(report["predictions"]), 2)
            self.assertGreater(timings["total_inference_phases_seconds"], 0)
        self.assertEqual(lifecycle, [("start", False), ("stop", False), ("start", True), ("stop", True)])
        self.assertEqual(complete.call_count, 3)
        self.assertTrue(all(len(call.args[1]) == 1 for call in complete.call_args_list))
        self.assertEqual(complete.call_args_list[0].args[1], [
            {"role": "user", "content": "English question 0"},
        ])
        # The grader sees the saved generated answer, never the dataset reference.
        self.assertEqual(guard_http.call_args_list[0].args[1]["messages"], [
            {"role": "user", "content": "English question 0"}, {"role": "assistant", "content": "回答A"},
        ])
