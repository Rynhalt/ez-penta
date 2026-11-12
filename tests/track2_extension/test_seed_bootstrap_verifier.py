import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.lexsi.verify_seed_bootstrap import verify


class SeedBootstrapVerifierTests(unittest.TestCase):
    def test_recomputes_saved_group_mean_and_reproducible_interval(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = root / "effects.csv"
            with csv_path.open("w", newline="") as f:
                writer = csv.DictWriter(
                    f, fieldnames=["split", "prompt_language", "paired_difference"]
                )
                writer.writeheader()
                for value in (0.0, 0.5, 1.0):
                    writer.writerow(
                        {"split": "tune", "prompt_language": "en", "paired_difference": value}
                    )
            summary_path = root / "summary.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "comparisons": [
                            {
                                "split": "tune",
                                "prompt_language": "en",
                                "matched_seeds": 3,
                                "risk_difference": 0.5,
                                "ci95_seed_bootstrap": [0.0, 1.0],
                            }
                        ]
                    }
                )
            )
            a = verify(csv_path, summary_path, replicates=2_000, seed=17)
            b = verify(csv_path, summary_path, replicates=2_000, seed=17)
            self.assertEqual(a, b)
            self.assertEqual(a["seed_effect_rows"], 3)
            self.assertEqual(a["groups"][0]["mean"], 0.5)

    def test_rejects_mean_or_seed_count_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = root / "effects.csv"
            csv_path.write_text(
                "split,prompt_language,paired_difference\ntune,en,0.2\ntune,en,0.4\n"
            )
            summary_path = root / "summary.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "comparisons": [
                            {
                                "split": "tune",
                                "prompt_language": "en",
                                "matched_seeds": 3,
                                "risk_difference": 0.3,
                                "ci95_seed_bootstrap": [0.0, 1.0],
                            }
                        ]
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "seed count mismatch"):
                verify(csv_path, summary_path, replicates=2_000, seed=17)


if __name__ == "__main__":
    unittest.main()
