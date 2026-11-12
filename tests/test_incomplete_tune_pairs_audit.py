import unittest

from scripts.lexsi.audit_incomplete_tune_pairs import audit, eligible


def row(seed, framing, *, eligible_flag=True, generation="completed", guard="Safe"):
    return {
        "seed_id": seed,
        "split": "tune",
        "category": "test",
        "culture": "us",
        "kind": "harmful",
        "prompt_language": "en",
        "output_language": "en",
        "request": "same request",
        "framing": framing,
        "analysis_eligible": eligible_flag,
        "generation_status": generation,
        "guard_response_corrected": guard,
        "answer": "answer" if generation == "completed" else "partial",
    }


class IncompleteTuneAuditTests(unittest.TestCase):
    def test_only_unexposed_eligible_missing_response_is_recoverable(self):
        rows = {
            "s1-control": row("s1", "control"),
            "s1-rp": row("s1", "rp", generation="truncated", guard="Missing"),
            "s2-control": row("s2", "control", eligible_flag=False),
            "s2-rp": row("s2", "rp", eligible_flag=False, generation="truncated", guard="Missing"),
            "s3-control": row("s3", "control"),
            "s3-rp": row("s3", "rp"),
        }
        result = audit(rows, {"s3"})
        opportunity = result["unexposed_recovery_opportunity"]
        self.assertEqual(result["fully_eligible_matched_keys"], 1)
        self.assertEqual(opportunity["incomplete_matched_keys"], 2)
        self.assertEqual(opportunity["keys_where_both_rows_are_analysis_eligible"], 1)
        self.assertEqual(opportunity["seeds_where_both_rows_are_analysis_eligible"], 1)
        self.assertEqual(opportunity["unexposed_cross_request_base_groups_with_eligible_rows_on_both_sides"], 0)

    def test_eligibility_requires_completed_binary_nonempty_answer(self):
        self.assertTrue(eligible(row("s", "control")))
        self.assertFalse(eligible(row("s", "control", guard="Controversial")))
        self.assertFalse(eligible(row("s", "control", generation="truncated")))


if __name__ == "__main__":
    unittest.main()
