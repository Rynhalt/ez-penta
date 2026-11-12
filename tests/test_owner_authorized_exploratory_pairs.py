from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.lexsi.prepare_owner_authorized_exploratory_pairs import build


def row(i: int, split: str) -> dict[str, str]:
    return {
        "row_id": f"r{i}", "seed_id": f"s{i}", "split": split, "language": "en",
        "clean_prompt": f"control {i}", "corrupt_prompt": f"roleplay {i}",
        "clean_answer": f"refusal {i}", "corrupted_answer": f"compliance {i}",
        "prompt_id_control": f"pc{i}", "prompt_id_rp": f"pr{i}",
        "review_status": "provisional", "reviewer_type": "codex_assisted",
    }


class OwnerAuthorizedPairTests(unittest.TestCase):
    def test_exports_seven_unique_seeds_without_fabricating_human_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.jsonl"
            source.write_text("".join(json.dumps(row(i, "analysis" if i < 6 else "tune")) + "\n" for i in range(7)))
            out, manifest = root / "out.jsonl", root / "manifest.json"
            result = build(source, out, manifest)
            exported = [json.loads(line) for line in out.read_text().splitlines()]
            self.assertEqual(result["unique_seeds"], 7)
            self.assertEqual(result["by_split"], {"analysis": 6, "tune": 1})
            self.assertTrue(all(r["approval_status"] == "owner_authorized_for_exploratory_execution" for r in exported))
            self.assertTrue(all(r["review_status"] == "provisional" for r in exported))
            self.assertFalse(result["independent_human_annotation"])
            self.assertFalse(result["official_human_approved_export_modified"])

    def test_rejects_seed_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = [row(i, "analysis" if i < 6 else "tune") for i in range(7)]
            records[-1]["seed_id"] = records[0]["seed_id"]
            source = root / "source.jsonl"
            source.write_text("".join(json.dumps(r) + "\n" for r in records))
            with self.assertRaisesRegex(ValueError, "Seed-level leakage"):
                build(source, root / "out.jsonl", root / "manifest.json")

    def test_rejects_heldout_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = [row(i, "analysis" if i < 6 else "tune") for i in range(7)]
            records[0]["split"] = "heldout"
            source = root / "source.jsonl"
            source.write_text("".join(json.dumps(r) + "\n" for r in records))
            with self.assertRaisesRegex(ValueError, "Only analysis/tune"):
                build(source, root / "out.jsonl", root / "manifest.json")


if __name__ == "__main__":
    unittest.main()
