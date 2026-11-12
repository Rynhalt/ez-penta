import unittest

from eval.track2_extension.discovery import select_discovery_pairs_by_language


def _pair(language, index, **overrides):
    row = {
        "language": language,
        "seed_id": f"{language}-{index:03}",
        "split": "analysis",
        "review_status": "human_approved",
        "reviewer_ids": ["reviewer-a", "reviewer-b"],
        "adjudicator_id": "adjudicator",
    }
    row.update(overrides)
    return row


class SelectDiscoveryPairsTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            *[_pair("en", index) for index in range(20)],
            *[_pair("ja", index) for index in range(11)],
        ]

    def test_selects_languages_separately_and_caps_each_cohort(self):
        result = select_discovery_pairs_by_language(self.rows, seed=13, sample_size_per_language=8)
        self.assertEqual({key: len(value) for key, value in result.items()}, {"en": 8, "ja": 8})
        self.assertTrue(all(row["language"] == "en" for row in result["en"]))
        self.assertTrue(all(row["language"] == "ja" for row in result["ja"]))
        self.assertEqual(len({row["seed_id"] for row in result["en"]}), 8)
        self.assertEqual(len({row["seed_id"] for row in result["ja"]}), 8)

    def test_selection_is_reproducible_and_seed_changes_sample(self):
        first = select_discovery_pairs_by_language(self.rows, seed=13, sample_size_per_language=8)
        again = select_discovery_pairs_by_language(self.rows, seed=13, sample_size_per_language=8)
        other = select_discovery_pairs_by_language(self.rows, seed=14, sample_size_per_language=8)
        ids = lambda result, language: [row["seed_id"] for row in result[language]]
        self.assertEqual(ids(first, "en"), ids(again, "en"))
        self.assertEqual(ids(first, "ja"), ids(again, "ja"))
        self.assertNotEqual(ids(first, "en"), ids(other, "en"))

    def test_requires_minimum_independent_seeds_per_language(self):
        with self.assertRaisesRegex(ValueError, "at least 8 adjudicated analysis seeds in ja"):
            select_discovery_pairs_by_language(self.rows[:20] + [_pair("ja", 0)], seed=1)

    def test_rejects_non_analysis_or_unreviewed_data(self):
        bad = [*self.rows]
        bad[0] = _pair("en", 0, split="tune")
        with self.assertRaisesRegex(ValueError, "analysis-split"):
            select_discovery_pairs_by_language(bad, seed=1)
        bad = [*self.rows]
        bad[0] = _pair("en", 0, reviewer_ids=["same", "SAME"])
        with self.assertRaisesRegex(ValueError, "two distinct human reviewers"):
            select_discovery_pairs_by_language(bad, seed=1)

    def test_rejects_duplicate_seed_in_same_language(self):
        bad = [*self.rows, _pair("en", 0, row_id="duplicate")]
        with self.assertRaisesRegex(ValueError, "Duplicate discovery seed in en"):
            select_discovery_pairs_by_language(bad, seed=1)


if __name__ == "__main__":
    unittest.main()
