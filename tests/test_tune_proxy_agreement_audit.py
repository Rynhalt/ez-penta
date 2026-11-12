import unittest

from scripts.lexsi.build_tune_proxy_agreement_audit import select


def row(seed, row_id, *, split="tune", lang="en", guard_c="Safe", guard_r="Safe", refused_c=1, refused_r=1):
    return {
        "seed_id": seed, "row_id": row_id, "split": split, "language": lang,
        "guard_response_corrected_control": guard_c,
        "guard_response_corrected_rp": guard_r,
        "refused_control": refused_c, "refused_rp": refused_r,
        "prompt_id_control": f"p-{row_id}", "prompt_control": "c", "request_control": "x",
        "answer_control": "a", "prompt_id_rp": f"r-{row_id}", "prompt_rp": "r",
        "request_rp": "x", "answer_rp": "b", "category": "privacy",
    }


class TuneProxyAgreementAuditTests(unittest.TestCase):
    def test_selects_one_pair_per_unseen_tune_seed_and_hides_proxy_labels(self):
        rows = [row("s1", "b"), row("s1", "a", lang="ja"), row("s2", "c"),
                row("analysis", "d", split="analysis"),
                row("guard-mismatch", "e", guard_r="Unsafe"),
                row("heuristic-mismatch", "f", refused_r=0)]
        result = select(rows, {"s2"})
        self.assertEqual([r["seed_id"] for r in result], ["s1"])
        self.assertEqual(result[0]["row_id"], "b")
        self.assertFalse(any("guard_response" in key or key.startswith("refused_") for key in result[0]))

    def test_nonrefusal_mode_selects_pairs_where_both_proxy_signals_agree(self):
        rows = [row("s1", "a", refused_c=0, refused_r=0),
                row("s2", "b", refused_c=1, refused_r=1)]
        result = select(rows, set(), require_refusal_heuristic=False)
        self.assertEqual([r["seed_id"] for r in result], ["s1"])


if __name__ == "__main__":
    unittest.main()
