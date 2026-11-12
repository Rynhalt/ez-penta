"""Create a private minimal SQLite for Dataset2 guard-label reanalysis.

Run inside the authorized Colab runtime after mounting Drive. The output keeps
only matching/design fields and raw guard records; it excludes prompt and
generated-answer columns. It never changes the source database.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path


COLUMNS = (
    "prompt_id", "seed_id", "split", "kind", "category", "culture",
    "framing", "prompt_language", "mix_direction", "output_language",
    "analysis_eligible", "generation_status", "safety", "prompt_safety",
    "raw_guard_outputs",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "source", type=Path,
        nargs="?",
        default=Path("/content/drive/MyDrive/ez-penta/Pipeline1111/combined/results.sqlite3"),
    )
    parser.add_argument(
        "destination", type=Path,
        nargs="?",
        default=Path("/content/drive/MyDrive/ez-penta/Pipeline1111/codex_extension_20261111/results_reduced.sqlite3"),
    )
    args = parser.parse_args()
    if args.source.resolve() == args.destination.resolve():
        raise SystemExit("Source and destination must differ")
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    if args.destination.exists():
        raise SystemExit(f"Refusing to overwrite {args.destination}")

    src = sqlite3.connect(f"file:{args.source.resolve()}?mode=ro", uri=True)
    dst = sqlite3.connect(args.destination)
    try:
        src.row_factory = sqlite3.Row
        dst.execute("PRAGMA journal_mode=DELETE")
        dst.execute("PRAGMA synchronous=FULL")
        dst.execute("CREATE TABLE observations (" + ", ".join(
            f'"{name}" TEXT' for name in COLUMNS
        ) + ")")
        placeholders = ",".join("?" for _ in COLUMNS)
        query = "SELECT " + ",".join(f'"{name}"' for name in COLUMNS) + " FROM observations"
        count = 0
        for row in src.execute(query):
            dst.execute(
                f"INSERT INTO observations VALUES ({placeholders})",
                tuple(row[name] for name in COLUMNS),
            )
            count += 1
        metadata = src.execute("SELECT json FROM run_metadata LIMIT 1").fetchone()
        dst.execute("CREATE TABLE run_metadata (json TEXT)")
        if metadata:
            dst.execute("INSERT INTO run_metadata VALUES (?)", (metadata[0],))
        dst.commit()
        check = dst.execute("PRAGMA quick_check").fetchone()[0]
        if check != "ok":
            raise RuntimeError(f"Export quick_check failed: {check}")
    finally:
        dst.close()
        src.close()

    digest = hashlib.sha256(args.destination.read_bytes()).hexdigest()
    report = {
        "source_name": args.source.name,
        "output_name": args.destination.name,
        "rows": count,
        "columns": list(COLUMNS),
        "prompt_or_answer_columns_included": False,
        "sqlite_quick_check": check,
        "sha256": digest,
        "size_bytes": args.destination.stat().st_size,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
