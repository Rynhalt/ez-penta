import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from scripts.lexsi.reconcile_dataset2_shards import reconcile_shards


def guard(label):
    return {"status": "completed", "answer": f"Safety: {label}\nCategories: None"}


class ShardedDatasetReconciliationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.parts = [self.root / "part1.sqlite3", self.root / "part2.sqlite3"]
        self._make_shard(self.parts[0], "seed-a", "Unsafe", "Safe")
        self._make_shard(self.parts[1], "seed-b", "Unsafe", "Unsafe")
        self.labels = self.root / "expected_labels.jsonl"
        records = [
            {"prompt_id": f"{seed}-{culture}", "seed_id": seed, "split": "analysis",
             "guard_response_corrected": response, "guard_prompt_corrected": "Safe"}
            for seed, us, neutral in (("seed-a", "Unsafe", "Safe"), ("seed-b", "Unsafe", "Unsafe"))
            for culture, response in (("us", us), ("neutral", neutral))
        ]
        self.labels.write_text("".join(json.dumps(r) + "\n" for r in records))

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def _make_shard(path, seed, us, neutral):
        con = sqlite3.connect(path)
        con.execute("""CREATE TABLE observations (
            prompt_id TEXT, seed_id TEXT, split TEXT, kind TEXT, category TEXT,
            culture TEXT, framing TEXT, prompt_language TEXT, mix_direction TEXT,
            output_language TEXT, analysis_eligible INTEGER, generation_status TEXT,
            safety TEXT, prompt_safety TEXT, raw_guard_outputs TEXT
        )""")
        for culture, response in (("us", us), ("neutral", neutral)):
            con.execute("INSERT INTO observations VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                f"{seed}-{culture}", seed, "analysis", "harmful", "violence", culture,
                "control", "en", None, "en", 1, "completed", "Error", "Safe",
                json.dumps([guard(response), guard("Safe")]),
            ))
        con.execute("CREATE TABLE run_metadata(json TEXT)")
        con.execute("INSERT INTO run_metadata VALUES (?)", (json.dumps({"dataset": {
            "dataset_sha256": "toy-dataset", "manifest_id": "toy-manifest", "algorithm": "toy", "prompt_count": 4
        }}),))
        con.commit()
        con.close()

    def test_reparses_both_shards_crosschecks_labels_and_bootstraps_combined_seeds(self):
        result = reconcile_shards(self.parts, self.root / "out", expected_labels=self.labels,
                                  expected_rows=4, bootstrap_replicates=100, rng_seed=3)
        self.assertEqual(result["combined_prompt_rows"], 4)
        self.assertEqual(result["unique_source_seeds"], 2)
        self.assertEqual(result["recovered_binary_response_labels"], 4)
        self.assertEqual(result["expected_label_crosscheck"]["mismatched_rows"], 0)
        analysis_en = next(x for x in result["matched_culture_comparisons"]
                           if x["split"] == "analysis" and x["prompt_language"] == "en")
        self.assertEqual(analysis_en["matched_seed_count"], 2)
        self.assertEqual(analysis_en["seed_paired_risk_difference"]["point_estimate"], 0.5)

    def test_rejects_wrong_expected_row_count(self):
        with self.assertRaisesRegex(ValueError, "expected 5"):
            reconcile_shards(self.parts, self.root / "out", expected_rows=5, bootstrap_replicates=10)

    def test_rejects_corrected_label_disagreement(self):
        self.labels.write_text(self.labels.read_text().replace('"Unsafe"', '"Safe"', 1))
        with self.assertRaisesRegex(ValueError, "disagree"):
            reconcile_shards(self.parts, self.root / "out", expected_labels=self.labels,
                             expected_rows=4, bootstrap_replicates=10)


if __name__ == "__main__":
    unittest.main()
