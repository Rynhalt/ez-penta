import contextlib
from dataclasses import asdict
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import auditkit as ak

from guard_judge import LocalGuard, guard_messages, parse_guard
from evaluate_answercarefully import load_responses, evaluate_responses


class GuardJudgeTests(unittest.TestCase):
    def test_strict_verdicts_and_unknowns(self):
        self.assertEqual(parse_guard("Safety: Unsafe\nCategories: Violent")[0], 1)
        self.assertEqual(parse_guard("Safety: Safe\nCategories: None")[0], 0)
        for text in ("", "safe", "Safety: Controversial", "Safety: Safe\nSafety: Unsafe"):
            self.assertTrue(parse_guard(text)[1]["unknown"])
        self.assertEqual(parse_guard("Safety: Safe\nRefusal: Yes")[1]["refusal"], "Yes")

    def test_precomputed_guard_excludes_unknown_from_mean(self):
        replies = ["Safety: Safe\nCategories: None\nRefusal: No",
                   "Safety: Unsafe\nCategories: Violent\nRefusal: No",
                   "Safety: Controversial\nCategories: Unethical Acts\nRefusal: No"]
        profile = {"id": "test_qwen", "build_messages": guard_messages, "parse": parse_guard}
        responses = [{"choices": [{"finish_reason": "stop", "message": {"content": text}}]} for text in replies]
        samples = [ak.Sample(id=str(index), input="質問", actual_output="保存した回答") for index in range(3)]
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"XDG_CACHE_HOME": directory}), patch(
            "guard_judge.chat.read_json", side_effect=responses
        ) as read_json:
            guard = ak.GuardJudge(profile=profile, judge_model=LocalGuard("http://127.0.0.1:8081", "test"))
            result = ak.evaluate(samples, model="precomputed", scorers=[guard], config=ak.RunConfig(max_retries=0))
        self.assertEqual(result.headline[guard.name], 0.5)
        self.assertEqual(result.stats[guard.name].count, 2)
        self.assertEqual(result.unscored[guard.name]["unknown"], 1)
        payload = read_json.call_args_list[0].args[1]
        self.assertEqual(payload["messages"], guard_messages("質問", "保存した回答", "response"))
        self.assertEqual(payload["temperature"], 0)

    def test_truncated_verdict_is_rejected(self):
        from auditkit.model import Request

        with patch("guard_judge.chat.read_json", return_value={"choices": [{"finish_reason": "length"}]}):
            with self.assertRaisesRegex(ValueError, "Incomplete guard verdict"):
                LocalGuard("http://127.0.0.1:8081", "test").generate([
                    Request(prompt="", params={"messages": [], "max_tokens": 128})
                ])

    def test_saved_samples_round_trip_and_truncation_accounting(self):
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "pilot.jsonl"
            records = [
                {"sample": asdict(ak.Sample(id=str(index), input="質問", actual_output="回答")),
                 "generation": {"finish_reason": reason}}
                for index, reason in enumerate(["stop", "length"])
            ]
            input_path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
            with patch.dict(os.environ, {"XDG_CACHE_HOME": directory}), patch(
                "guard_judge.chat.loaded_model", return_value="test/Qwen3Guard-Gen-0.6B"
            ), patch(
                "guard_judge.chat.read_json", return_value={"choices": [{
                    "finish_reason": "stop", "message": {"content": "Safety: Safe\nCategories: None\nRefusal: No"}
                }]}
            ), contextlib.redirect_stdout(io.StringIO()):
                samples, excluded = load_responses(input_path)
                evaluate_responses(samples, excluded, "http://127.0.0.1:8081", Path(directory))
            result = json.loads((Path(directory) / "auditkit-guard-results.json").read_text(encoding="utf-8"))
        self.assertEqual(len(result["predictions"]), 1)
        self.assertEqual(result["errors"], [])
        self.assertEqual(result["metadata"]["excluded_generations"], [{"sample_id": "1", "finish_reason": "length"}])


if __name__ == "__main__":
    unittest.main()
