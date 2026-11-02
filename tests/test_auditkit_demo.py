import os
import tempfile
import unittest
from unittest.mock import call, patch

import auditkit as ak

import auditkit_demo


class AuditkitDemoTests(unittest.TestCase):
    def test_adapter_sends_independent_requests_in_order(self):
        with patch("auditkit_demo.chat.complete", side_effect=[
            {"choices": [{"message": {"content": "4"}}]},
            {"choices": [{"message": {"content": "東京"}}]},
        ]) as complete:
            adapter = auditkit_demo.make_model_adapter("http://127.0.0.1:8080", 32)
            self.assertEqual(adapter(["2 + 2?", "日本の首都は？"]), ["4", "東京"])
        self.assertEqual(complete.call_args_list, [
            call("http://127.0.0.1:8080", [{"role": "user", "content": "2 + 2?"}], 32),
            call("http://127.0.0.1:8080", [{"role": "user", "content": "日本の首都は？"}], 32),
        ])

    def test_real_auditkit_scores_and_aggregates_stubbed_predictions(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"XDG_CACHE_HOME": directory}), patch(
            "auditkit_demo.chat.complete",
            side_effect=[{"choices": [{"message": {"content": answer}}]} for answer in ["4", "4", "4", "Kyoto", "東京"]],
        ):
            result = ak.evaluate(
                auditkit_demo.samples(),
                model=auditkit_demo.make_model_adapter("http://127.0.0.1:8080", 32),
                scorers=["exact_match"], config=ak.RunConfig(max_retries=0),
            )
        self.assertEqual(len(result.predictions), 5)
        self.assertEqual([prediction.score for prediction in result.predictions], [1, 1, 1, 0, 1])
        self.assertEqual(result.headline["exact_match"], 0.8)
        self.assertEqual(result.errors, [])


if __name__ == "__main__":
    unittest.main()
