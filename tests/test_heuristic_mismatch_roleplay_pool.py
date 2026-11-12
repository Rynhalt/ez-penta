import unittest

from scripts.lexsi.build_heuristic_mismatch_roleplay_pool import build_pool


def source_row(pid, framing, *, refused, guard="Safe", split="analysis", request="request"):
    return {
        "prompt_id": pid,
        "seed_id": "seed-1",
        "split": split,
        "category": "privacy",
        "culture": "neutral",
        "kind": "harmful",
        "framing": framing,
        "prompt_language": "en",
        "output_language": "en",
        "request": request,
        "generation_status": "completed",
        "analysis_eligible": 1,
        "guard_response_corrected": guard,
        "refused": refused,
        "answer": f"answer {pid}",
        "prompt": f"prompt {pid}",
    }


class HeuristicMismatchRoleplayPoolTests(unittest.TestCase):
    def test_selects_matching_guard_same_heuristic_different_pair(self):
        rows = {
            "c": source_row("c", "control", refused=0),
            "r": source_row("r", "rp", refused=1),
        }
        result = build_pool(rows)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["prompt_id_control"], "c")
        self.assertEqual(result[0]["prompt_id_rp"], "r")
        self.assertEqual(result[0]["triage_reason"], "refusal_heuristic_disagrees; recovered_binary_guard_agrees")

    def test_excludes_guard_disagreement(self):
        rows = {
            "c": source_row("c", "control", refused=0, guard="Safe"),
            "r": source_row("r", "rp", refused=1, guard="Unsafe"),
        }
        self.assertEqual(build_pool(rows), [])

    def test_all_matched_mode_includes_pairs_missed_by_both_proxy_mismatches(self):
        rows = {
            "c": source_row("c", "control", refused=0, guard="Safe"),
            "r": source_row("r", "rp", refused=0, guard="Safe"),
        }
        self.assertEqual(build_pool(rows), [])
        result = build_pool(rows, include_all_matched=True)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["cohort"], "roleplay_all_matched_triage")
        self.assertEqual(result[0]["triage_reason"], "guard_disagrees=False; refusal_heuristic_disagrees=False")

    def test_excludes_heldout_and_mixed_language(self):
        heldout = {
            "c": source_row("c", "control", refused=0, split="heldout"),
            "r": source_row("r", "rp", refused=1, split="heldout"),
        }
        mixed = {
            "c": source_row("c", "control", refused=0),
            "r": source_row("r", "rp", refused=1),
        }
        mixed["r"]["prompt_language"] = "mix"
        self.assertEqual(build_pool(heldout), [])
        self.assertEqual(build_pool(mixed), [])

    def test_requires_exact_request_and_complete_answers(self):
        different_request = {
            "c": source_row("c", "control", refused=0, request="one"),
            "r": source_row("r", "rp", refused=1, request="two"),
        }
        empty_answer = {
            "c": source_row("c", "control", refused=0),
            "r": source_row("r", "rp", refused=1),
        }
        empty_answer["r"]["answer"] = ""
        self.assertEqual(build_pool(different_request), [])
        self.assertEqual(build_pool(empty_answer), [])

    def test_duplicate_condition_fails_closed(self):
        rows = {
            "c1": source_row("c1", "control", refused=0),
            "c2": source_row("c2", "control", refused=1),
            "r": source_row("r", "rp", refused=1),
        }
        with self.assertRaisesRegex(ValueError, "Duplicate control row"):
            build_pool(rows)


if __name__ == "__main__":
    unittest.main()
