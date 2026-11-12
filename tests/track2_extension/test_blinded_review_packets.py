import csv
import tempfile
import unittest
from pathlib import Path

from scripts.lexsi.build_blinded_review_packets import build_packets


class BlindedReviewPacketTests(unittest.TestCase):
    def _input(self, folder):
        path = Path(folder) / "candidates.csv"
        fields = [
            "row_id", "seed_id", "split", "language", "prompt_control", "answer_control",
            "prompt_rp", "answer_rp", "provisional_rationale", "safety_control", "safety_rp",
        ]
        rows = [
            {
                "row_id": f"row-{i}", "seed_id": f"seed-{i}",
                "split": "analysis" if i < 2 else "tune", "language": "en" if i % 2 else "ja",
                "prompt_control": f"control prompt {i}", "answer_control": f"control answer {i}",
                "prompt_rp": f"roleplay prompt {i}", "answer_rp": f"roleplay answer {i}",
                "provisional_rationale": "must not leak", "safety_control": "Unsafe", "safety_rp": "Safe",
            }
            for i in range(3)
        ]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_packets_are_blinded_randomized_and_have_no_prefilled_labels(self):
        with tempfile.TemporaryDirectory() as temp:
            source = self._input(temp)
            output = Path(temp) / "packets"
            manifest = build_packets(source, output, seed=71)
            with (output / "reviewer_1.csv").open(encoding="utf-8", newline="") as handle:
                one = list(csv.DictReader(handle))
            with (output / "reviewer_2.csv").open(encoding="utf-8", newline="") as handle:
                two = list(csv.DictReader(handle))
            self.assertEqual(len(one), 3)
            self.assertEqual(len(two), 3)
            self.assertEqual(manifest["packet_rows"], {"reviewer_1": 3, "reviewer_2": 3})
            self.assertEqual({row["case_id"] for row in one}, {row["case_id"] for row in two})
            self.assertEqual(len({row["case_id"] for row in one}), 3)
            for row in [*one, *two]:
                self.assertNotIn("provisional_rationale", row)
                self.assertNotIn("safety_control", row)
                self.assertNotIn("safety_rp", row)
                self.assertTrue(row["prompt_a"] and row["response_a"])
                self.assertTrue(row["prompt_b"] and row["response_b"])
                self.assertEqual(row["review_status"], "")
                self.assertEqual(row["reviewer_id"], "")
            with (output / "PRIVATE_adjudication_key.csv").open(encoding="utf-8", newline="") as handle:
                key = list(csv.DictReader(handle))
            self.assertEqual(len(key), 6)
            self.assertEqual(len({(row["packet"], row["case_id"]) for row in key}), 6)
            self.assertEqual(
                {row["case_id"] for row in key if row["packet"] == "reviewer_1"},
                {row["case_id"] for row in key if row["packet"] == "reviewer_2"},
            )
            self.assertTrue((output / ".gitignore").exists())

    def test_reproducible_and_rejects_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as temp:
            source = self._input(temp)
            first = Path(temp) / "one"
            second = Path(temp) / "two"
            a = build_packets(source, first, seed=19)
            b = build_packets(source, second, seed=19)
            self.assertEqual(a["packet_sha256"], b["packet_sha256"])
            with source.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            rows[1]["row_id"] = rows[0]["row_id"]
            with source.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "unique"):
                build_packets(source, Path(temp) / "bad", seed=19)


if __name__ == "__main__":
    unittest.main()
