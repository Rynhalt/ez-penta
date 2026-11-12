"""Private SQLite and JSONL helpers for LEXSI Track-2.

These functions never print source prompts or responses. Keep data products in
the private, git-ignored output directory or an authorized Drive folder.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterator, Mapping


_MANIFEST_COLUMNS = (
    "prompt_id",
    "seed_id",
    "split",
    "kind",
    "category",
    "culture",
    "framing",
    "prompt_language",
    "mix_direction",
    "output_language",
    "prompt",
    "analysis_eligible",
    "generation_status",
    "scoring_status",
    "safety",
    "identical_twin_flag",
    "duplicate_text_flag",
    "expected_behavior_conflict",
    "expected",
)


def iter_observations(sqlite_path: Path) -> Iterator[dict[str, Any]]:
    """Yield manifest fields from `observations` in a read-only SQLite DB."""
    uri = sqlite_path.resolve().as_uri() + "?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    try:
        if connection.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise ValueError("SQLite quick_check failed")
        available = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(observations)")
        }
        missing = sorted(set(_MANIFEST_COLUMNS) - available)
        if missing:
            raise ValueError(f"observations table is missing required columns: {missing}")
        cursor = connection.execute(
            "SELECT " + ", ".join(f'"{name}"' for name in _MANIFEST_COLUMNS)
            + " FROM observations ORDER BY prompt_id"
        )
        for row in cursor:
            yield dict(row)
    finally:
        connection.close()


def write_jsonl_atomic(path: Path, rows: Iterator[Mapping[str, Any]]) -> int:
    """Write UTF-8 JSONL atomically, replacing only the requested output path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    count = 0
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(dict(row), ensure_ascii=False, separators=(",", ":")))
                stream.write("\n")
                count += 1
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return count


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at line {line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row {line_number} must be an object")
            rows.append(row)
    return rows
