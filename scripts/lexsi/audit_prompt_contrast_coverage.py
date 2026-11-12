#!/usr/bin/env python3
"""Count structurally matched Dataset2 control/roleplay prompt contrasts.

This read-only audit uses prompt construction metadata only. It does not inspect
responses or labels and cannot validate that a contrast changes model behavior.
The aggregate output intentionally contains no prompt text or source IDs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_prompt_contrast_coverage(sqlite_path: Path) -> dict:
    """Count exact-request harmful control/rp pairs, including seed completeness."""
    resolved = sqlite_path.resolve()
    con = sqlite3.connect(resolved.as_uri() + "?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        con.execute("PRAGMA query_only=ON")
        if con.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise ValueError("SQLite quick_check failed")
        expected = {
            "prompts": {"prompt_id", "source_id", "split", "culture", "kind", "category", "framing",
                        "prompt_language", "output_language", "mix_direction", "request"},
            "seeds": {"source_id", "complete"},
        }
        for table, required in expected.items():
            columns = {row[1] for row in con.execute(f'PRAGMA table_info("{table}")')}
            if not required <= columns:
                raise ValueError(f"{table} table is missing required columns")

        query = """
            WITH contrasts AS (
                SELECT a.source_id, a.split, a.culture, a.prompt_language,
                       a.prompt_id AS control_id, b.prompt_id AS rp_id,
                       COALESCE(s.complete, 0) AS source_complete
                FROM prompts AS a
                JOIN prompts AS b
                  ON a.source_id = b.source_id
                 AND a.split = b.split
                 AND a.culture = b.culture
                 AND a.kind = b.kind
                 AND a.category = b.category
                 AND a.prompt_language = b.prompt_language
                 AND a.output_language = b.output_language
                 AND COALESCE(a.mix_direction, '<not-applicable>')
                     = COALESCE(b.mix_direction, '<not-applicable>')
                 AND a.request = b.request
                LEFT JOIN seeds AS s ON a.source_id = s.source_id
                WHERE a.kind = 'harmful'
                  AND a.framing = 'control'
                  AND b.framing = 'rp'
                  AND a.prompt_language IN ('en', 'ja')
            )
            SELECT split, culture, prompt_language,
                   COUNT(*) AS pair_rows,
                   COUNT(DISTINCT source_id) AS unique_seeds,
                   SUM(CASE WHEN source_complete = 1 THEN 1 ELSE 0 END)
                       AS complete_source_pair_rows,
                   COUNT(DISTINCT CASE WHEN source_complete = 1 THEN source_id END)
                       AS complete_source_seeds
            FROM contrasts
            GROUP BY split, culture, prompt_language
            ORDER BY split, culture, prompt_language
        """
        counts = [dict(row) for row in con.execute(query)]
        source_summary = {
            "prompt_rows": con.execute("SELECT COUNT(*) FROM prompts").fetchone()[0],
            "unique_source_seeds": con.execute(
                "SELECT COUNT(DISTINCT source_id) FROM prompts"
            ).fetchone()[0],
            "seed_table_complete_flags": {
                str(flag): count for flag, count in con.execute(
                    "SELECT complete, COUNT(*) FROM seeds GROUP BY complete ORDER BY complete"
                )
            },
        }
    finally:
        con.close()

    return {
        "schema": "lexsi.prompt_contrast_structural_coverage/1",
        "source": {
            "path_basename": resolved.name,
            "bytes": resolved.stat().st_size,
            "sha256": sha256_file(resolved),
            "sqlite_quick_check": "ok",
        },
        "source_summary": source_summary,
        "matching_rule": [
            "harmful prompt rows only; English and Japanese prompt languages",
            "same source seed, split, culture, category, exact request text, prompt language, output language, and mix direction",
            "one side uses control framing and the other roleplay (rp) framing",
            "complete_source counts additionally require seeds.complete == 1",
        ],
        "counts": counts,
        "limitations": [
            "This is prompt-design coverage only; no generated answer or safety label exists in this source file.",
            "Exact request-string equality is a structural screen, not human semantic-equivalence adjudication.",
            "A source seed marked complete is not evidence that a model response was generated or scored.",
            "Heldout split prompts were previously exposed and are descriptive inventory only.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sqlite", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit_prompt_contrast_coverage(args.sqlite)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({
        "output": str(args.output),
        "source_sha256": result["source"]["sha256"],
        "source_summary": result["source_summary"],
        "strata_count": len(result["counts"]),
        "limitations": result["limitations"],
    }, indent=2))


if __name__ == "__main__":
    main()
