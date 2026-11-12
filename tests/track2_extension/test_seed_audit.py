import unittest

from eval.track2_extension.seed_audit import validate_fresh_eval_seed_audit


def pairs():
    return [
        {"language": lang, "seed_id": f"{lang}-{i}"}
        for lang in ("en", "ja") for i in range(8)
    ]


class FreshEvalSeedAuditTests(unittest.TestCase):
    def test_accepts_complete_disjoint_inventory(self):
        report = validate_fresh_eval_seed_audit(
            pairs(), discovery_seed_ids=["disc"], phase1_seed_ids=["metric"],
            prior_registry={"complete": True, "inventory_sources": ["historical runs"], "seed_ids": ["trained"]},
        )
        self.assertEqual(report["n_seeds_by_language"], {"en": 8, "ja": 8})

    def test_incomplete_inventory_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "marked complete"):
            validate_fresh_eval_seed_audit(
                pairs(), discovery_seed_ids=[], phase1_seed_ids=[],
                prior_registry={"complete": False, "inventory_sources": [], "seed_ids": []},
            )

    def test_requires_inventory_source(self):
        with self.assertRaisesRegex(ValueError, "identify the inventories"):
            validate_fresh_eval_seed_audit(
                pairs(), discovery_seed_ids=[], phase1_seed_ids=[],
                prior_registry={"complete": True, "inventory_sources": [], "seed_ids": []},
            )

    def test_rejects_overlap_with_training_seed(self):
        with self.assertRaisesRegex(ValueError, "prior training/tuning"):
            validate_fresh_eval_seed_audit(
                pairs(), discovery_seed_ids=[], phase1_seed_ids=[],
                prior_registry={"complete": True, "inventory_sources": ["SFT"], "seed_ids": ["en-3"]},
            )

    def test_rejects_duplicate_seed_and_low_count(self):
        rows = pairs()
        rows.append(rows[0].copy())
        with self.assertRaisesRegex(ValueError, "Duplicate fresh evaluation"):
            validate_fresh_eval_seed_audit(
                rows, discovery_seed_ids=[], phase1_seed_ids=[],
                prior_registry={"complete": True, "inventory_sources": ["runs"], "seed_ids": []},
            )
        with self.assertRaisesRegex(ValueError, "at least 8 fresh ja"):
            validate_fresh_eval_seed_audit(
                pairs()[:15], discovery_seed_ids=[], phase1_seed_ids=[],
                prior_registry={"complete": True, "inventory_sources": ["runs"], "seed_ids": []},
            )


if __name__ == "__main__":
    unittest.main()
