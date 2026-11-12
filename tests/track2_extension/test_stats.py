import unittest

from eval.track2_extension.stats import paired_seed_differences, seed_cluster_bootstrap


class SeedPairedStatsTests(unittest.TestCase):
    def test_pairs_within_seed_and_matching_stratum_and_ignores_unknown(self):
        rows = [
            {"seed": "a", "culture": "us", "language": "en", "safety": "Unsafe"},
            {"seed": "a", "culture": "us", "language": "en", "safety": "Safe"},
            {"seed": "a", "culture": "neutral", "language": "en", "safety": "Safe"},
            {"seed": "a", "culture": "neutral", "language": "ja", "safety": "Safe"},
            {"seed": "a", "culture": "us", "language": "ja", "safety": "Controversial"},
            {"seed": "b", "culture": "us", "language": "en", "safety": "Unsafe"},
            {"seed": "b", "culture": "neutral", "language": "en", "safety": "Unsafe"},
        ]
        result = paired_seed_differences(
            rows,
            seed_column="seed",
            condition_column="culture",
            treatment_value="us",
            control_value="neutral",
            outcome_column="safety",
            matching_columns=["language"],
            outcome_mapping={"Safe": 0, "Unsafe": 1},
        )
        self.assertEqual([row["seed"] for row in result], ["a", "b"])
        self.assertAlmostEqual(result[0]["paired_difference"], 0.5)
        self.assertEqual(result[0]["matched_strata"], 1)
        self.assertEqual(result[1]["paired_difference"], 0.0)

    def test_does_not_pair_across_seeds_or_unmatched_strata(self):
        rows = [
            {"seed": "a", "culture": "us", "language": "en", "safety": "Unsafe"},
            {"seed": "b", "culture": "neutral", "language": "en", "safety": "Safe"},
            {"seed": "c", "culture": "us", "language": "ja", "safety": "Unsafe"},
            {"seed": "c", "culture": "neutral", "language": "en", "safety": "Safe"},
        ]
        result = paired_seed_differences(
            rows,
            seed_column="seed",
            condition_column="culture",
            treatment_value="us",
            control_value="neutral",
            outcome_column="safety",
            matching_columns=["language"],
            outcome_mapping={"Safe": 0, "Unsafe": 1},
        )
        self.assertEqual(result, [])

    def test_bootstrap_is_reproducible_and_seed_weighted(self):
        rows = [
            {"paired_difference": 1.0},
            {"paired_difference": 0.0},
            {"paired_difference": 0.0},
        ]
        first = seed_cluster_bootstrap(rows, replicates=1000, seed=17)
        second = seed_cluster_bootstrap(rows, replicates=1000, seed=17)
        self.assertEqual(first, second)
        self.assertEqual(first["n_seeds"], 3)
        self.assertAlmostEqual(first["point_estimate"], 1 / 3)
        self.assertEqual(first["interval_method"], "percentile_seed_cluster_bootstrap")

    def test_rejects_implicit_or_invalid_outcome_mapping(self):
        row = {"seed": "a", "condition": "us", "stratum": "x", "outcome": "Unsafe"}
        kwargs = dict(
            seed_column="seed",
            condition_column="condition",
            treatment_value="us",
            control_value="neutral",
            outcome_column="outcome",
            matching_columns=["stratum"],
        )
        with self.assertRaises(ValueError):
            paired_seed_differences([row], outcome_mapping={}, **kwargs)
        with self.assertRaises(ValueError):
            paired_seed_differences([row], outcome_mapping={"Unsafe": 2}, **kwargs)


if __name__ == "__main__":
    unittest.main()
