import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from scripts.lexsi.audit_answercarefully_snapshot import audit_snapshot


class AnswerCarefullySnapshotAuditTests(unittest.TestCase):
    def _database(self, path: Path) -> None:
        connection = sqlite3.connect(path)
        connection.executescript("""
            CREATE TABLE records (
                source_id TEXT NOT NULL, source_split TEXT NOT NULL, text TEXT, text_en TEXT,
                output TEXT, output_en TEXT, risk_area TEXT, harm_type TEXT, specific_harm TEXT,
                risk_area_en TEXT, harm_type_en TEXT, specific_harm_en TEXT,
                meta TEXT NOT NULL, meta_mlmc TEXT NOT NULL,
                PRIMARY KEY(source_id, source_split)
            );
            CREATE TABLE snapshot_info(dataset TEXT, configuration TEXT, revision TEXT, fetched_at TEXT);
        """)
        connection.execute(
            "INSERT INTO records VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            ("opaque-id", "test", "私的質問", "Private question", "参照回答", None,
             "risk", "harm", "specific", "risk-en", "harm-en", "specific-en",
             json.dumps({"risk-area": "risk"}), json.dumps({"text-English": "Private question", "adaptation-tag": "0", "translation-notes": ""})),
        )
        connection.execute("INSERT INTO snapshot_info VALUES (?,?,?,?)",
                           ("llm-jp/AnswerCarefully", "v2.2", "rev", "timestamp"))
        connection.commit()
        connection.close()

    def test_audit_contains_counts_and_provenance_but_no_row_text(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "snapshot.sqlite3"
            self._database(path)
            report = audit_snapshot(path)
        serialized = json.dumps(report, ensure_ascii=False)
        self.assertEqual(report["integrity"]["records"], 1)
        self.assertEqual(report["rows_by_split"], {"test": 1})
        self.assertEqual(report["nonempty_field_counts"]["english_reference_answer"], 0)
        self.assertEqual(report["adaptation_tag_counts_by_split"]["test"], {"0": 1})
        self.assertNotIn("opaque-id", serialized)
        self.assertNotIn("私的質問", serialized)
        self.assertNotIn("Private question", serialized)

    def test_empty_database_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "empty.sqlite3"
            path.touch()
            with self.assertRaisesRegex(ValueError, "empty"):
                audit_snapshot(path)


if __name__ == "__main__":
    unittest.main()
