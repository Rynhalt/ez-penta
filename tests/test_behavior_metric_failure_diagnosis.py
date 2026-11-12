import unittest

from scripts.lexsi.audit_behavior_metric_validation import summarize


def pair(split, language, control, roleplay, refusal_tokens=10, compliance_tokens=20):
    return {
        "split": split,
        "language": language,
        "control": {
            "mean_margin": control,
            "refusal_tokens": refusal_tokens,
            "compliance_tokens": compliance_tokens,
        },
        "roleplay": {
            "mean_margin": roleplay,
            "refusal_tokens": refusal_tokens,
            "compliance_tokens": compliance_tokens,
        },
        "paired_delta": control - roleplay,
    }


class BehaviorMetricFailureDiagnosisTests(unittest.TestCase):
    def test_reports_relative_order_separately_from_absolute_signs(self):
        result = summarize({
            "status": "INSUFFICIENT_VALIDATION_DATA",
            "annotation_provenance": "provisional",
            "independent_human_annotation": False,
            "pairs": [
                pair("analysis", "en", 0.4, 0.1),
                pair("tune", "ja", -0.2, -0.5),
                pair("analysis", "ja", 0.2, -0.1),
            ],
        })
        self.assertEqual(result["overall"]["paired_delta_signs"]["positive"], 3)
        self.assertEqual(result["overall"]["both_expected_absolute_signs"], 1)
        self.assertEqual(result["by_split_and_language"]["analysis_en"]["n"], 1)
        self.assertEqual(result["overall"]["median_compliance_to_refusal_token_ratio"], 2)

    def test_refuses_unvalidated_or_incomplete_inputs(self):
        with self.assertRaises(ValueError):
            summarize({"status": "PASSED", "pairs": []})
        with self.assertRaises(ValueError):
            summarize({"status": "INSUFFICIENT_VALIDATION_DATA", "pairs": [{}]})


if __name__ == "__main__":
    unittest.main()
