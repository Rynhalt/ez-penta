#!/usr/bin/env python3
"""Write a prompt-free provenance and coverage audit of a frozen AnswerCarefully DB."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any


REQUIRED_COLUMNS = {
    "source_id", "source_split", "text", "text_en", "output", "output_en",
    "risk_area", "harm_type", "specific_harm", "risk_area_en", "harm_type_en",
    "specific_harm_en", "meta", "meta_mlmc",
}


def audit_snapshot(database: Path) -> dict[str, Any]:
    if not database.is_file() or database.stat().st_size == 0:
        raise ValueError(f"Snapshot is missing or empty: {database}")
    uri = f"file:{database.resolve()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    try:
        integrity = connection.execute("PRAGMA quick_check").fetchone()[0]
        if integrity != "ok":
            raise ValueError(f"SQLite quick_check failed: {integrity}")
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if not {"records", "snapshot_info"}.issubset(tables):
            raise ValueError("Snapshot must contain records and snapshot_info tables")
        columns = {row[1] for row in connection.execute("PRAGMA table_info(records)")}
        missing = sorted(REQUIRED_COLUMNS - columns)
        if missing:
            raise ValueError(f"Snapshot records schema is missing columns: {missing}")

        row_count = connection.execute("SELECT COUNT(*) FROM records").fetchone()[0]
        if row_count == 0:
            raise ValueError("Snapshot has no records")
        duplicate_keys = connection.execute(
            "SELECT COUNT(*) FROM (SELECT source_id, source_split FROM records "
            "GROUP BY source_id, source_split HAVING COUNT(*) != 1)"
        ).fetchone()[0]
        if duplicate_keys:
            raise ValueError(f"Found {duplicate_keys} duplicate source ID/split keys")

        splits = dict(connection.execute(
            "SELECT source_split, COUNT(*) FROM records GROUP BY source_split ORDER BY source_split"
        ).fetchall())
        coverage = connection.execute(
            "SELECT "
            "SUM(text IS NOT NULL AND LENGTH(TRIM(text)) > 0), "
            "SUM(text_en IS NOT NULL AND LENGTH(TRIM(text_en)) > 0), "
            "SUM(output IS NOT NULL AND LENGTH(TRIM(output)) > 0), "
            "SUM(output_en IS NOT NULL AND LENGTH(TRIM(output_en)) > 0) "
            "FROM records"
        ).fetchone()
        categories = dict(connection.execute(
            "SELECT COALESCE(risk_area, '<missing>'), COUNT(*) FROM records "
            "GROUP BY risk_area ORDER BY risk_area"
        ).fetchall())
        adaptation_tags = {
            split: dict(connection.execute(
                "SELECT COALESCE(json_extract(meta_mlmc, '$.\"adaptation-tag\"'), '<missing>'), COUNT(*) "
                "FROM records WHERE source_split = ? GROUP BY json_extract(meta_mlmc, '$.\"adaptation-tag\"') "
                "ORDER BY json_extract(meta_mlmc, '$.\"adaptation-tag\"')",
                (split,),
            ).fetchall())
            for split in sorted(splits)
        }
        translation_notes = connection.execute(
            "SELECT COUNT(*) FROM records WHERE json_extract(meta_mlmc, '$.\"translation-notes\"') "
            "IS NOT NULL AND LENGTH(TRIM(json_extract(meta_mlmc, '$.\"translation-notes\"'))) > 0"
        ).fetchone()[0]
        snapshot = connection.execute(
            "SELECT dataset, configuration, revision, fetched_at FROM snapshot_info ORDER BY rowid DESC LIMIT 1"
        ).fetchone()
        if not snapshot:
            raise ValueError("Snapshot metadata is missing")
        metadata_keys: dict[str, list[str]] = {}
        for column in ("meta", "meta_mlmc"):
            keys = connection.execute(
                f"SELECT DISTINCT j.key FROM records r, json_each(r.{column}) AS j ORDER BY j.key"
            ).fetchall()
            metadata_keys[column] = [row[0] for row in keys]
    finally:
        connection.close()

    return {
        "schema": "lexsi.answercarefully_snapshot_audit/1",
        "source": {
            "path": str(database),
            "bytes": database.stat().st_size,
            "sha256": hashlib.sha256(database.read_bytes()).hexdigest(),
            "redistribution_allowed": False,
        },
        "dataset": {
            "dataset_id": snapshot[0],
            "configuration": snapshot[1],
            "revision": snapshot[2],
            "fetched_at_utc": snapshot[3],
        },
        "integrity": {"sqlite_quick_check": integrity, "records": row_count, "duplicate_source_split_keys": duplicate_keys},
        "rows_by_split": splits,
        "nonempty_field_counts": {
            "japanese_prompt": coverage[0],
            "english_prompt_translation": coverage[1],
            "japanese_reference_answer": coverage[2],
            "english_reference_answer": coverage[3],
        },
        "risk_area_counts": categories,
        "adaptation_tag_counts_by_split": adaptation_tags,
        "nonempty_translation_notes": translation_notes,
        "metadata_keys_only": metadata_keys,
        "research_limits": [
            "Snapshot contains Japanese reference answers but no English reference answers.",
            "Metadata describes risk categories but does not contain an explicit refusal_required field.",
            "The source test split must not be treated as fresh for confirmatory Tiny Aya evaluation until historical exposure is resolved.",
            "Use only for safety research; do not redistribute the source dataset.",
        ],
        "privacy": "No prompt text, reference text, row IDs, or per-record metadata values are included.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit_snapshot(args.db)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
