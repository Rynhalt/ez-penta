import sqlite3
import tempfile
import unittest
from pathlib import Path

from scripts.lexsi.reproduce_pipeline1_mh import pipeline1_mh


class Pipeline1MhReproductionTests(unittest.TestCase):
    def make_shard(self, root: Path, name: str, rows: list[tuple]) -> Path:
        path = root / name
        connection = sqlite3.connect(path)
        connection.execute("CREATE TABLE observations (culture TEXT, prompt_language TEXT, safety TEXT, analysis_eligible INTEGER, generation_status TEXT, scoring_status TEXT, split TEXT, kind TEXT, framing TEXT)")
        connection.executemany("INSERT INTO observations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
        connection.commit()
        connection.close()
        return path

    def test_reproduces_known_stratified_or_and_corrected_cmh(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = []
            # Three strata, each with a positive exposed association.
            for language in ("en", "ja", "mix"):
                for framing in ("control", "rp"):
                    for culture, unsafe, safe in (("us", 200, 400), ("japan", 150, 450), ("neutral", 100, 500)):
                        rows += [(culture, language, "Unsafe", 1, "completed", "completed", "analysis", "harmful", framing)] * unsafe
                        rows += [(culture, language, "Safe", 1, "completed", "completed", "analysis", "harmful", framing)] * safe
            shard = self.make_shard(root, "part.sqlite3", rows)
            result = pipeline1_mh((shard,), expected_cohort_rows=len(rows))
            self.assertAlmostEqual(result["or_mh"], 2.5)
            self.assertLess(result["ci95_rbg"][0], result["or_mh"])
            self.assertGreater(result["ci95_rbg"][1], result["or_mh"])
            self.assertEqual(result["strata"][0]["rows"], 2400)
            self.assertLess(result["p_value_chi_square_1df"], 0.05)
            self.assertEqual(len(result["comparisons"]), 4)
            self.assertEqual({(x["exposure"], x["framing"]) for x in result["comparisons"]}, {
                ("us versus neutral", "all"), ("us versus neutral", "rp"),
                ("japan versus neutral", "all"), ("japan versus neutral", "rp"),
            })

    def test_rejects_cohort_row_count_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [("us", "en", "Unsafe", 1, "completed", "completed", "analysis", "harmful", "control")]
            shard = self.make_shard(root, "part.sqlite3", rows)
            with self.assertRaisesRegex(ValueError, "cohort rows"):
                pipeline1_mh((shard,), expected_cohort_rows=4069)


if __name__ == "__main__":
    unittest.main()
