import unittest

from scripts.lexsi.metric_sign_test_power import (
    binomial_upper_tail,
    minimum_n_for_power,
    sign_test_operating_characteristics,
)


class MetricSignTestPowerTests(unittest.TestCase):
    def test_five_all_successes_is_significant_but_low_power(self):
        result = sign_test_operating_characteristics(5, 0.8)
        self.assertEqual(result["critical_successes"], 5)
        self.assertAlmostEqual(result["null_tail_probability"], 1 / 32)
        self.assertAlmostEqual(result["power"], 0.8**5)
        self.assertLess(result["power"], 0.33)

    def test_scenario_sample_sizes_and_alpha(self):
        self.assertEqual(minimum_n_for_power(0.8)["n"], 18)
        result = minimum_n_for_power(0.7)
        self.assertEqual(result["n"], 37)
        self.assertLessEqual(result["null_tail_probability"], 0.05)
        self.assertGreaterEqual(result["power"], 0.8)

    def test_exact_tail_against_known_value(self):
        self.assertAlmostEqual(binomial_upper_tail(7, 8, 0.5), 9 / 256)


if __name__ == "__main__":
    unittest.main()
