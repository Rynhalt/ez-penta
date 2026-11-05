"""Check the independent audit's paired tests and unknown-value denominators."""

import unittest

from inspect_evaluation import mcnemar, numeric_equal, paired_counts, rate


class EvaluationAuditTests(unittest.TestCase):
    def test_exact_test_matches_hand_calculated_cases_and_is_symmetric(self):
        for a, b, expected in [(0, 0, 1.0), (3, 0, 0.25), (6, 1, 0.125), (1, 1, 1.0)]:
            self.assertAlmostEqual(mcnemar(a, b), expected)
            self.assertAlmostEqual(mcnemar(b, a), expected)

    def test_only_pairs_with_two_known_verdicts_enter_the_denominator(self):
        pairs = [({"unsafe": a}, {"unsafe": b})
                 for a, b in [(1, 0), (1, 0), (0, 1), (1, 1), (None, 1), (0, None)]]
        counts = paired_counts(pairs, "unsafe", "unsafe")
        self.assertEqual(counts["pairs"], 4)
        self.assertEqual(counts["first_only"], 2)
        self.assertEqual(counts["second_only"], 1)
        self.assertEqual(counts["mismatch_rate"], 0.75)

    def test_unknown_verdicts_are_not_silently_safe(self):
        self.assertEqual(rate([{"unsafe": 0}, {"unsafe": 1}, {"unsafe": None}], "unsafe"), 0.5)
        self.assertIsNone(rate([{"unsafe": None}], "unsafe"))
        self.assertFalse(numeric_equal(None, 0))
        self.assertTrue(numeric_equal(None, None))
