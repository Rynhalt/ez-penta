import unittest

from eval.track2_extension.intervention_summary import (
    invalid_raw_ratio_count,
    valid_raw_ratio_values,
)


class CircuitKitInterventionSummaryTests(unittest.TestCase):
    def test_degenerate_zero_sentinel_is_not_treated_as_a_measurement(self):
        rows = [
            {"patching_raw_ratio": 0.0, "patching_degenerate_denominator": True,
             "patching_inverted_denominator": False},
            {"patching_raw_ratio": 0.7, "patching_degenerate_denominator": False,
             "patching_inverted_denominator": False},
        ]
        self.assertEqual(valid_raw_ratio_values(rows, "patching"), [0.7])
        self.assertEqual(invalid_raw_ratio_count(rows, "patching"), 1)

    def test_inverted_missing_and_nonfinite_ratios_are_excluded(self):
        rows = [
            {"ablation_raw_ratio": -0.3, "ablation_degenerate_denominator": False,
             "ablation_inverted_denominator": True},
            {"ablation_raw_ratio": None, "ablation_degenerate_denominator": False,
             "ablation_inverted_denominator": False},
            {"ablation_raw_ratio": float("nan"), "ablation_degenerate_denominator": False,
             "ablation_inverted_denominator": False},
            {"ablation_raw_ratio": 0.4, "ablation_degenerate_denominator": False,
             "ablation_inverted_denominator": False},
            {"ablation_raw_ratio": 0.8},  # missing validity flags: unknown
        ]
        self.assertEqual(valid_raw_ratio_values(rows, "ablation"), [0.4])
        self.assertEqual(invalid_raw_ratio_count(rows, "ablation"), 4)

    def test_rejects_unknown_pillar(self):
        with self.assertRaisesRegex(ValueError, "pillar must"):
            valid_raw_ratio_values([], "robustness")
        with self.assertRaisesRegex(ValueError, "pillar must"):
            invalid_raw_ratio_count([], "robustness")


if __name__ == "__main__":
    unittest.main()
