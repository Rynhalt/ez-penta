#!/usr/bin/env python3
"""Select previously unseen tune seeds where both safety proxies agree.

The selected full texts are for owner-authorized exploratory LLM triage only.
One pair per seed prevents condition-row multiplicity from inflating the review n.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def read_csv_seeds(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as f:
        return {row["seed_id"] for row in csv.DictReader(f) if row.get("seed_id")}


def select(rows: list[dict[str, Any]], excluded_seeds: set[str], *,
           require_refusal_heuristic: bool = True) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("split") != "tune" or row.get("language") not in {"en", "ja"}:
            continue
        if row.get("guard_response_corrected_control") != "Safe" or row.get("guard_response_corrected_rp") != "Safe":
            continue
        if bool(row.get("refused_control")) != require_refusal_heuristic or bool(row.get("refused_rp")) != require_refusal_heuristic:
            continue
        if row.get("seed_id") in excluded_seeds:
            continue
        buckets.setdefault(row["seed_id"], []).append(row)

    selected = []
    for seed_id in sorted(buckets):
        options = sorted(buckets[seed_id], key=lambda row: (row.get("language", ""), row["row_id"]))
        row = options[0]
        # Keep the annotation view limited to the exact text, pairing, and split.
        selected.append({key: row.get(key) for key in (
            "row_id", "seed_id", "split", "category", "language", "prompt_id_control",
            "prompt_control", "request_control", "answer_control", "prompt_id_rp",
            "prompt_rp", "request_rp", "answer_rp",
        )})
    return selected


def build(input_path: Path, prior_registry: Path, old_candidates: Path, expansion_candidates: Path,
          output: Path, manifest_path: Path, *, require_refusal_heuristic: bool = True) -> dict[str, Any]:
    registry = json.loads(prior_registry.read_text())
    if registry.get("complete") is not False:
        raise ValueError("Expected the known partial historical seed registry; revise the exclusion audit first")
    excluded = set(registry.get("seed_ids", []))
    excluded |= read_csv_seeds(old_candidates) | read_csv_seeds(expansion_candidates)
    selected = select(read_jsonl(input_path), excluded, require_refusal_heuristic=require_refusal_heuristic)
    if len({row["seed_id"] for row in selected}) != len(selected):
        raise ValueError("Seed overlap in selected tune audit")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in selected))
    manifest = {
        "schema": "lexsi.tune_proxy_agreement_audit/1",
        "input_path": str(input_path),
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "excluded_seed_sources": [str(prior_registry), str(old_candidates), str(expansion_candidates)],
        "selected_pairs": len(selected),
        "unique_tune_seeds": len({row["seed_id"] for row in selected}),
        "by_language": dict(Counter(row["language"] for row in selected)),
        "guard_stratum": "Safe/Safe",
        "refusal_heuristic_stratum": f"both {require_refusal_heuristic}",
        "one_pair_per_seed": True,
        "source_labels_in_annotation_view": False,
        "annotation_status": "unreviewed",
        "approval_status": "owner_authorized_for_exploratory_review_only",
        "interpretation": "screen for proxy false negatives; one deterministic pair per eligible seed; not a random sample or human-validated cohort",
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--prior-registry", type=Path, required=True)
    parser.add_argument("--old-candidates", type=Path, required=True)
    parser.add_argument("--expansion-candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--nonrefusal-heuristic", action="store_true",
                        help="Select Safe/Safe tune pairs where neither response triggers the refusal heuristic.")
    args = parser.parse_args()
    print(json.dumps(build(args.input, args.prior_registry, args.old_candidates,
                           args.expansion_candidates, args.output, args.manifest,
                           require_refusal_heuristic=not args.nonrefusal_heuristic), indent=2))


if __name__ == "__main__":
    main()
