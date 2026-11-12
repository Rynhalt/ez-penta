import os
from pathlib import Path
import unittest

from scripts.lexsi.recover_guard_response_labels import observations, recover_guard
from scripts.lexsi.review_dataset2 import VERSION, approved_pairs, build, match


class LexsiReviewTests(unittest.TestCase):
    def test_guard_verdicts_stay_distinct(self):
        for text, expected in [
            ("Safety: Controversial", "Controversial"),
            ("Safety: Safe\nSafety: Unsafe", "Unparsed"),
            ("Safety: Unsafe\r\nRefusal: No\r\n", "Unsafe"),
        ]:
            self.assertEqual(recover_guard({"status": "completed", "answer": text}), expected)
        self.assertEqual(recover_guard({"status": "truncated", "answer": "Safety: Safe"}), "Missing")

    def test_duplicate_conditions_fail(self):
        row = {"framing": "control", "seed_id": "seed"}
        with self.assertRaisesRegex(ValueError, "Duplicate condition"):
            match([row, row], "framing", "control", "rp", ("seed_id",))

    def test_approval_gate(self):
        row = {
            "row_id": "row", "seed_id": "seed", "split": "analysis", "category": "category",
            "language": "en", "culture": "neutral", "prompt_id_control": "1", "prompt_id_rp": "2",
            "prompt_control": "control", "prompt_rp": "roleplay",
            "answer_control": "Refusal continuation.", "answer_rp": "Compliance continuation.",
        }
        approval = {
            "review_status": "human_approved", "reviewer_ids": ["a", "b"], "adjudicator_id": "c",
            "review_version": VERSION, "prompt_semantic_equivalence": "equivalent",
            "request_intent_control": "harmful", "request_intent_rp": "harmful",
            "refusal_control": "clear_refusal", "response_safety_control": "safe",
            "refusal_rp": "complied", "response_safety_rp": "unsafe",
            "clean_answer": "Refusal", "corrupted_answer": "Compliance", "rationale": "Reviewed.",
        }
        self.assertEqual(approved_pairs([row], {}), [])
        self.assertEqual(approved_pairs([row], {"row": approval})[0]["clean_prompt"], "control")
        for updated in (
            {"review_status": "provisional"},
            {"reviewer_ids": ["a", " A "]},
            {"refusal_rp": "partial_refusal"},
            {"corrupted_answer": "Absent evidence"},
            {"corrupted_answer": "continuation."},
        ):
            with self.assertRaises(ValueError):
                approved_pairs([row], {"row": {**approval, **updated}})
        with self.assertRaisesRegex(ValueError, "Heldout"):
            approved_pairs([{**row, "split": "heldout"}], {"row": approval})
        with self.assertRaisesRegex(ValueError, "duplicate seed"):
            approved_pairs([row, {**row, "row_id": "row2"}], {"row": approval, "row2": approval})

    @unittest.skipUnless(os.environ.get("LEXSI_DATASET2_INPUT"), "Set LEXSI_DATASET2_INPUT for the real shard audit")
    def test_saved_shards(self):
        folder = Path(os.environ["LEXSI_DATASET2_INPUT"])
        rows, _ = observations([folder / "part1.sqlite3", folder / "part2.sqlite3"])
        pairs, disagreements, calibration, candidates, controls = build(rows.values())
        self.assertEqual((len(pairs), len(disagreements), len(calibration)), (6256, 401, 80))
        self.assertEqual((len(candidates), len(controls)), (87, 16))
        self.assertTrue(all(r["split"] != "heldout" for r in candidates + controls))


if __name__ == "__main__":
    unittest.main()
