"""Freeze AnswerCarefully into a local SQLite database, so later runs don't need Hugging Face.

Downloads the same way answercarefully_dataset.py does (pinned revision, saved HF login),
then writes every row into data_ready/answercarefully.sqlite3 (or --db-path).

Usage:
    python3 data_ready/fetch_answercarefully_to_sqlite.py [--split all|dev|test] [--db-path PATH]
"""

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from answercarefully_dataset import DATASET_CONFIGURATION, DATASET_ID, load_answercarefully_data

DEFAULT_DB_PATH = Path(__file__).resolve().parent / "answercarefully.sqlite3"

SCHEMA = """
DROP TABLE IF EXISTS records;
DROP TABLE IF EXISTS snapshot_info;

-- English columns come from the dataset itself (meta-mlmc), not machine-translated here.
CREATE TABLE records (
    source_id TEXT NOT NULL,
    source_split TEXT NOT NULL,
    text TEXT NOT NULL,
    text_en TEXT,
    output TEXT NOT NULL,
    -- The dataset ships no English reference answer, so this stays NULL until filled separately.
    output_en TEXT,
    risk_area TEXT,
    harm_type TEXT,
    specific_harm TEXT,
    risk_area_en TEXT,
    harm_type_en TEXT,
    specific_harm_en TEXT,
    meta TEXT NOT NULL,
    meta_mlmc TEXT NOT NULL,
    PRIMARY KEY (source_id, source_split)
);

CREATE TABLE snapshot_info (
    dataset TEXT NOT NULL,
    configuration TEXT NOT NULL,
    revision TEXT NOT NULL,
    fetched_at TEXT NOT NULL
);
"""


def freeze_to_sqlite(db_path, split="all"):
    dataset, revision = load_answercarefully_data(split)
    connection = sqlite3.connect(db_path)
    try:
        # A fresh snapshot replaces the old tables outright, so schema changes apply on refetch.
        connection.executescript(SCHEMA)
        connection.executemany(
            "INSERT INTO records (source_id, source_split, text, text_en, output, "
            "risk_area, harm_type, specific_harm, risk_area_en, harm_type_en, specific_harm_en, "
            "meta, meta_mlmc) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                (
                    row["ID"], row["_source_split"],
                    row["text"], row["meta-mlmc"].get("text-English"),
                    row["output"],
                    row["meta"].get("risk-area"),
                    row["meta"].get("harm-type"),
                    row["meta"].get("specific-harm"),
                    row["meta-mlmc"].get("risk-area-English"),
                    row["meta-mlmc"].get("harm-type-English"),
                    row["meta-mlmc"].get("specific-harm-English"),
                    json.dumps(row["meta"], ensure_ascii=False),
                    json.dumps(row["meta-mlmc"], ensure_ascii=False),
                )
                for row in dataset
            ),
        )
        connection.execute(
            "INSERT INTO snapshot_info (dataset, configuration, revision, fetched_at) VALUES (?, ?, ?, ?)",
            (DATASET_ID, DATASET_CONFIGURATION, revision, datetime.now(timezone.utc).isoformat()),
        )
        connection.commit()
    finally:
        connection.close()
    return len(dataset), revision


def load_rows_from_sqlite(db_path, split="all"):
    """Read the frozen snapshot back as (rows, revision), matching load_answercarefully_data's shape.

    The returned rows are plug-compatible with answercarefully_dataset.select_samples.
    """
    connection = sqlite3.connect(db_path)
    try:
        query = "SELECT source_id, source_split, text, output, meta, meta_mlmc FROM records"
        params = ()
        if split != "all":
            query += " WHERE source_split = ?"
            params = (split,)
        cursor = connection.execute(query, params)
        rows = [
            {
                "ID": source_id, "_source_split": source_split, "text": text, "output": output,
                "meta": json.loads(meta), "meta-mlmc": json.loads(meta_mlmc),
            }
            for source_id, source_split, text, output, meta, meta_mlmc in cursor.fetchall()
        ]
        revision_row = connection.execute("SELECT revision FROM snapshot_info LIMIT 1").fetchone()
    finally:
        connection.close()
    if not rows:
        raise ValueError(f"No records found in {db_path} for split {split!r}. Run the fetch script first.")
    return rows, (revision_row[0] if revision_row else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("dev", "test", "all"), default="all")
    parser.add_argument("--db-path", default=str(DEFAULT_DB_PATH))
    args = parser.parse_args()
    print(f"Downloading {DATASET_ID}, configuration {DATASET_CONFIGURATION}, split {args.split}...", flush=True)
    count, revision = freeze_to_sqlite(args.db_path, args.split)
    print(f"Wrote {count} records to {args.db_path}")
    print(f"Dataset revision: {revision}")


if __name__ == "__main__":
    main()
