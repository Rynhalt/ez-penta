import sqlite3
import tempfile
import unittest
from pathlib import Path

from scripts.lexsi.audit_prompt_contrast_coverage import audit_prompt_contrast_coverage


class PromptContrastCoverageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db = Path(self.temp.name) / "source.sqlite3"
        con = sqlite3.connect(self.db)
        con.executescript("""
            CREATE TABLE seeds(source_id TEXT, complete INTEGER);
            CREATE TABLE prompts(
                prompt_id TEXT, source_id TEXT, split TEXT, culture TEXT,
                kind TEXT, category TEXT, framing TEXT, prompt_language TEXT,
                output_language TEXT, mix_direction TEXT, request TEXT
            );
        """)
        con.executemany("INSERT INTO seeds VALUES (?, ?)", [("s1", 1), ("s2", 0)])
        rows = [
            ("s1-c", "s1", "analysis", "us", "harmful", "violence", "control", "en", "en", None, "same request"),
            ("s1-r", "s1", "analysis", "us", "harmful", "violence", "rp", "en", "en", None, "same request"),
            ("s2-c", "s2", "analysis", "us", "harmful", "violence", "control", "en", "en", None, "other request"),
            ("s2-r", "s2", "analysis", "us", "harmful", "violence", "rp", "en", "en", "forward", "other request"),
            ("s1-jc", "s1", "heldout", "neutral", "harmful", "violence", "control", "ja", "ja", None, "Japanese request"),
            ("s1-jr", "s1", "heldout", "neutral", "harmful", "violence", "rp", "ja", "ja", None, "Japanese request"),
        ]
        con.executemany("INSERT INTO prompts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
        con.commit()
        con.close()

    def tearDown(self):
        self.temp.cleanup()

    def test_counts_exact_structural_pairs_and_complete_source_subset(self):
        result = audit_prompt_contrast_coverage(self.db)
        analysis = next(r for r in result["counts"] if r["split"] == "analysis")
        heldout = next(r for r in result["counts"] if r["split"] == "heldout")
        self.assertEqual((analysis["pair_rows"], analysis["unique_seeds"]), (1, 1))
        self.assertEqual(analysis["complete_source_seeds"], 1)
        self.assertEqual((heldout["pair_rows"], heldout["complete_source_seeds"]), (1, 1))
        self.assertNotIn("same request", str(result))
        self.assertIn("no generated answer", " ".join(result["limitations"]).lower())

    def test_requires_prompt_text_join_field_schema(self):
        con = sqlite3.connect(self.db)
        con.execute("ALTER TABLE prompts RENAME TO prompts_old")
        con.execute("CREATE TABLE prompts(source_id TEXT)")
        con.commit()
        con.close()
        with self.assertRaisesRegex(ValueError, "missing required columns"):
            audit_prompt_contrast_coverage(self.db)


if __name__ == "__main__":
    unittest.main()
