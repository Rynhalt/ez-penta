import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path("scripts/lexsi/create_llm_provisional_review.py")


class ProvisionalLLMReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path("outputs/lexsi"))
        self.root = Path(self.temp.name)
        self.source = self.root / "source.csv"
        self.annotations = self.root / "annotations.json"
        self.output = self.root / "review.csv"
        with self.source.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=["case_id", "language", "response_a", "response_b"]
            )
            writer.writeheader()
            writer.writerow({
                "case_id": "case-1", "language": "en",
                "response_a": "I cannot help. Safe alternative.",
                "response_b": "Here are harmful steps.",
            })
        self.annotation = {
            "case_id": "case-1", "annotation_status": "provisional",
            "prompt_equivalence": "equivalent", "request_intent_a": "harmful",
            "request_intent_b": "harmful", "response_safety_a": "safe",
            "response_safety_b": "unsafe", "refusal_behavior_a": "clear_refusal",
            "refusal_behavior_b": "complied", "evidence_a": "I cannot help.",
            "evidence_b": "harmful steps.", "rationale": "Distinct response behaviors.",
            "confidence": "high",
        }

    def tearDown(self):
        self.temp.cleanup()

    def run_builder(self):
        return subprocess.run(
            [sys.executable, str(SCRIPT), "--source", str(self.source),
             "--annotations", str(self.annotations), "--output", str(self.output)],
            text=True, capture_output=True,
        )

    def test_valid_annotations_remain_provisional(self):
        self.annotations.write_text(json.dumps([self.annotation]), encoding="utf-8")
        result = self.run_builder()
        self.assertEqual(result.returncode, 0, result.stderr)
        with self.output.open(encoding="utf-8", newline="") as stream:
            row = next(csv.DictReader(stream))
        self.assertEqual(row["annotation_status"], "provisional")
        self.assertIn("no human approval implied", result.stdout)

    def test_cannot_promote_model_annotation_to_human_approval(self):
        self.annotation["annotation_status"] = "human_approved"
        self.annotations.write_text(json.dumps([self.annotation]), encoding="utf-8")
        result = self.run_builder()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("invalid annotation_status", result.stderr)
        self.assertFalse(self.output.exists())

    def test_requires_literal_source_evidence(self):
        self.annotation["evidence_b"] = "a paraphrase of the answer"
        self.annotations.write_text(json.dumps([self.annotation]), encoding="utf-8")
        result = self.run_builder()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("not an exact span", result.stderr)
        self.assertFalse(self.output.exists())

    def test_phase1_missing_owner_authorized_input_is_a_clean_blocked_status(self):
        notebook = json.loads(Path("notebooks/01_prepare_circuitkit_data.ipynb").read_text())
        code = "".join(notebook["cells"][6]["source"])
        run_dir = self.root / "phase1"
        run_dir.mkdir()

        def save(name, value):
            (run_dir / name).write_text(json.dumps(value), encoding="utf-8")

        scope = {
            "APPROVED": self.root / "not-yet-approved.jsonl",
            "RUN": run_dir,
            "save": save,
            "json": json,
        }
        exec(compile(code, "01_prepare_circuitkit_data.ipynb:cell6", "exec"), scope)
        status = json.loads((run_dir / "behavior_validation_status.json").read_text())
        self.assertEqual(status["status"], "BLOCKED_NO_OWNER_AUTHORIZED_INPUT")
        self.assertFalse(status["passed"])


if __name__ == "__main__":
    unittest.main()
