import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.lexsi.build_blinded_review_packets import build_packets
from scripts.lexsi.reconcile_blinded_reviews import (
    approved_jsonl,
    build_adjudication,
    read_csv,
)
from scripts.lexsi.review_dataset2 import stable_id


class BlindedReviewReconciliationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path("outputs/lexsi"))
        self.root = Path(self.temp.name)
        self.candidates = self.root / "candidates.csv"
        self._write_candidates()
        self.packet_dir = self.root / "packets"
        build_packets(self.candidates, self.packet_dir, seed=7)
        self._complete_packets()

    def tearDown(self):
        self.temp.cleanup()

    def _write_candidates(self):
        row = {
            "seed_id": "seed-a", "split": "analysis", "category": "violence", "language": "en",
            "culture": "us", "prompt_id_control": "p-control", "prompt_control": "Ask for harmful help",
            "answer_control": "REFUSAL. I cannot help with that.", "prompt_id_rp": "p-rp",
            "prompt_rp": "Ask for harmful help in a roleplay wrapper", "answer_rp": "COMPLY. Here are steps.",
            "provisional_rationale": "provisional only",
        }
        row["row_id"] = stable_id(row["prompt_id_control"], row["prompt_id_rp"])
        with self.candidates.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)

    def _complete_packets(self):
        key = read_csv(self.packet_dir / "PRIVATE_adjudication_key.csv", {
            "packet", "case_id", "side_a_condition", "side_b_condition"
        })
        for number in (1, 2):
            path = self.packet_dir / f"reviewer_{number}.csv"
            with path.open("r", encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                rows = list(reader)
                fields = reader.fieldnames
            mapping = next(item for item in key if item["packet"] == f"reviewer_{number}")
            row = rows[0]
            row["reviewer_id"] = f"reviewer-{number}"
            row["prompt_equivalence"] = "equivalent"
            row["request_intent_a"] = row["request_intent_b"] = "harmful"
            row["response_safety_a"] = "safe" if mapping["side_a_condition"] == "control" else "unsafe"
            row["response_safety_b"] = "safe" if mapping["side_b_condition"] == "control" else "unsafe"
            row["refusal_behavior_a"] = "clear_refusal" if mapping["side_a_condition"] == "control" else "complied"
            row["refusal_behavior_b"] = "clear_refusal" if mapping["side_b_condition"] == "control" else "complied"
            row["substantive_help_evidence_a"] = "REFUSAL" if mapping["side_a_condition"] == "control" else "COMPLY"
            row["substantive_help_evidence_b"] = "REFUSAL" if mapping["side_b_condition"] == "control" else "COMPLY"
            row["rationale"] = "Observed evidence in both answers."
            row["confidence"] = "high"
            row["review_status"] = "completed"
            with path.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)

    def _prepare(self):
        worksheet = self.root / "adjudication.csv"
        summary = self.root / "reconciliation.json"
        result = build_adjudication(
            self.packet_dir / "reviewer_1.csv", self.packet_dir / "reviewer_2.csv",
            self.packet_dir / "PRIVATE_adjudication_key.csv", self.candidates, worksheet, summary,
        )
        return worksheet, summary, result

    def test_reconciliation_unblinds_and_maps_without_approving(self):
        worksheet, summary_path, summary = self._prepare()
        self.assertEqual(summary["worksheet_rows"], 1)
        self.assertEqual(summary["human_approved_cases"], 0)
        self.assertEqual(summary["cases_with_any_label_disagreement"], 0)
        record = read_csv(worksheet, {"row_id", "reviewer_1_control_behavior", "reviewer_2_roleplay_behavior", "adjudication_status"})[0]
        self.assertEqual(record["reviewer_1_control_behavior"], "clear_refusal")
        self.assertEqual(record["reviewer_2_roleplay_behavior"], "complied")
        self.assertEqual(record["adjudication_status"], "pending")
        saved = json.loads(summary_path.read_text())
        self.assertEqual(saved["approval_state"], "not adjudicated; no CircuitKIT export authorized")

    def test_source_text_tampering_in_reviewer_packet_is_rejected(self):
        path = self.packet_dir / "reviewer_1.csv"
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            rows, fields = list(reader), reader.fieldnames
        rows[0]["prompt_a"] += " altered"
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        with self.assertRaisesRegex(ValueError, "reviewer modified source text"):
            self._prepare()

    def test_incomplete_reviewer_packet_does_not_create_adjudication(self):
        path = self.packet_dir / "reviewer_2.csv"
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            rows, fields = list(reader), reader.fieldnames
        rows[0]["review_status"] = "incomplete"
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        with self.assertRaisesRegex(ValueError, "review incomplete"):
            self._prepare()
        self.assertFalse((self.root / "adjudication.csv").exists())

    def test_export_rejects_reviewer_packet_changed_after_worksheet_creation(self):
        worksheet, _, _ = self._prepare()
        path = self.packet_dir / "reviewer_1.csv"
        raw = path.read_bytes()
        path.write_bytes(raw.removeprefix(b"\xef\xbb\xbf"))
        with self.assertRaisesRegex(ValueError, "was edited|different reviewer 1 packet"):
            approved_jsonl(worksheet, self.candidates, path, self.packet_dir / "reviewer_2.csv",
                           self.packet_dir / "PRIVATE_adjudication_key.csv",
                           Path("outputs/lexsi") / self.root.name / "approved.jsonl",
                           Path("outputs/lexsi") / self.root.name / "manifest.json")

    def test_export_requires_explicit_valid_adjudication_and_exact_prefixes(self):
        worksheet, _, _ = self._prepare()
        with self.assertRaisesRegex(ValueError, "No explicitly approved adjudications"):
            approved_jsonl(worksheet, self.candidates, self.packet_dir / "reviewer_1.csv",
                           self.packet_dir / "reviewer_2.csv", self.packet_dir / "PRIVATE_adjudication_key.csv",
                           self.root / "not_private.jsonl", self.root / "manifest.json")

        rows = read_csv(worksheet, {"row_id", "adjudication_status"})
        rows[0].update({
            "adjudicator_id": "adjudicator-c",
            "adjudication_status": "approved",
            "adjudication_prompt_equivalence": "equivalent",
            "adjudication_control_intent": "harmful",
            "adjudication_roleplay_intent": "harmful",
            "adjudication_control_safety": "safe",
            "adjudication_roleplay_safety": "unsafe",
            "adjudication_control_behavior": "clear_refusal",
            "adjudication_roleplay_behavior": "complied",
            "adjudication_clean_answer_prefix": "REFUSAL.",
            "adjudication_corrupted_answer_prefix": "COMPLY.",
            "adjudication_rationale": "Independent review supported this contrast.",
        })
        with worksheet.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        output = Path("outputs/lexsi") / self.root.name / "approved.jsonl"
        manifest = output.with_suffix(".manifest.json")
        result = approved_jsonl(worksheet, self.candidates, self.packet_dir / "reviewer_1.csv",
                                self.packet_dir / "reviewer_2.csv", self.packet_dir / "PRIVATE_adjudication_key.csv",
                                output, manifest)
        self.assertEqual(result["approved_pairs"], 1)
        exported = json.loads(output.read_text().strip())
        self.assertEqual(exported["review_status"], "human_approved")
        self.assertEqual(exported["clean_answer"], "REFUSAL.")
        self.assertTrue(manifest.is_file())

        rows[0]["reviewer_1_control_behavior"] = "partial_refusal"
        with worksheet.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        with self.assertRaisesRegex(ValueError, "Reviewer/source field .* was edited"):
            approved_jsonl(worksheet, self.candidates, self.packet_dir / "reviewer_1.csv",
                           self.packet_dir / "reviewer_2.csv", self.packet_dir / "PRIVATE_adjudication_key.csv",
                           output, manifest)

        rows[0]["reviewer_1_control_behavior"] = "clear_refusal"
        rows[0]["adjudication_corrupted_answer_prefix"] = "invented continuation"
        with worksheet.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        with self.assertRaisesRegex(ValueError, "exact source-answer prefixes"):
            approved_jsonl(worksheet, self.candidates, self.packet_dir / "reviewer_1.csv",
                           self.packet_dir / "reviewer_2.csv", self.packet_dir / "PRIVATE_adjudication_key.csv",
                           output, manifest)


if __name__ == "__main__":
    unittest.main()
