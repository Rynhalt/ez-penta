"""Build a private seed-matched US/neutral candidate manifest from source SQLite.

The output is a research candidate list, never a human-approved dataset. It
includes full prompts, so write only to private local storage or Drive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .data_io import iter_observations, write_jsonl_atomic
from .manifest import build_matched_candidates


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sqlite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--splits", default="heldout", help="comma-separated source splits")
    parser.add_argument("--prior-seeds", type=Path, help="private JSON array of exposed seed IDs")
    args = parser.parse_args()

    prior_seeds = set()
    if args.prior_seeds:
        prior_seeds = {str(value) for value in json.loads(args.prior_seeds.read_text())}
    rows, counts = build_matched_candidates(
        iter_observations(args.sqlite),
        splits={value.strip() for value in args.splits.split(",") if value.strip()},
        prior_exposed_seed_ids=prior_seeds,
        previously_exposed_splits={"heldout"},
    )
    output_count = write_jsonl_atomic(args.output, iter(rows))
    if output_count != len(rows):
        raise RuntimeError("manifest write count does not match candidate row count")
    summary = {
        "schema": "lexsi.track2.candidate_manifest/1",
        "source_sqlite_sha256": sha256_file(args.sqlite),
        "source_sqlite_bytes": args.sqlite.stat().st_size,
        "source_integrity": "PRAGMA quick_check=ok",
        "source_splits": sorted({row["split"] for row in rows}),
        "selected_rows": len(rows),
        "manifest_sha256": sha256_file(args.output),
        "manifest_bytes": args.output.stat().st_size,
        "candidate_counts": counts,
        "human_review_state": "all rows unreviewed; human fields null; confirmatory_eligible false",
        "leakage_state": "all heldout rows flagged previously exposed",
        "privacy": "manifest includes full prompt text; store only in private Drive/local ignored outputs",
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.summary.with_suffix(args.summary.suffix + ".tmp")
    temporary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.summary)
    print(json.dumps({"summary": summary}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
