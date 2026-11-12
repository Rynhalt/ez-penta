#!/usr/bin/env python3
"""Audit whether missing Dataset2 Tune rows can restore new paired contrasts.

Only aggregate counts are written. Prompt and answer text never leave memory.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from scripts.lexsi.build_heuristic_mismatch_roleplay_pool import MATCH_FIELDS, file_sha256
from scripts.lexsi.recover_guard_response_labels import observations


BASE_FIELDS = ("seed_id", "category", "culture", "prompt_language", "output_language")
BINARY = {"Safe", "Unsafe"}


def load_excluded_seeds(registry: Path, csv_candidates: list[Path], jsonl_candidates: list[Path]) -> set[str]:
    data = json.loads(registry.read_text())
    if data.get("complete") is not False:
        raise ValueError("Expected partial prior-exposure registry; update audit logic for a complete registry")
    seeds = set(data.get("seed_ids", []))
    for path in csv_candidates:
        with path.open(newline="", encoding="utf-8-sig") as stream:
            seeds.update(row["seed_id"] for row in csv.DictReader(stream) if row.get("seed_id"))
    for path in jsonl_candidates:
        seeds.update(
            json.loads(line)["seed_id"]
            for line in path.read_text().splitlines()
            if line.strip()
        )
    return seeds


def eligible(row: dict[str, Any]) -> bool:
    return (
        row.get("generation_status") == "completed"
        and bool(row.get("analysis_eligible"))
        and row.get("guard_response_corrected") in BINARY
        and bool(str(row.get("answer") or "").strip())
    )


def _reason(row: dict[str, Any], side: str, counts: Counter[str]) -> None:
    if row.get("generation_status") != "completed":
        counts[f"{side}:generation_{row.get('generation_status')}"] += 1
    if not row.get("analysis_eligible"):
        counts[f"{side}:analysis_ineligible"] += 1
    if row.get("guard_response_corrected") not in BINARY:
        counts[f"{side}:guard_{row.get('guard_response_corrected')}"] += 1
    if not str(row.get("answer") or "").strip():
        counts[f"{side}:empty_answer"] += 1


def audit(rows: dict[str, dict[str, Any]], excluded_seeds: set[str]) -> dict[str, Any]:
    buckets: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows.values():
        if (
            row.get("split") != "tune"
            or row.get("kind") != "harmful"
            or row.get("prompt_language") not in {"en", "ja"}
            or row.get("prompt_language") != row.get("output_language")
            or row.get("framing") not in {"control", "rp"}
        ):
            continue
        buckets[tuple(row.get(field) for field in MATCH_FIELDS)][row["framing"]] = row

    matched = [v for v in buckets.values() if set(v) == {"control", "rp"}]
    one_sided = [v for v in buckets.values() if len(v) == 1]
    fully_eligible = [v for v in matched if eligible(v["control"]) and eligible(v["rp"])]
    incomplete = [v for v in matched if v not in fully_eligible]
    unexposed_incomplete = [v for v in incomplete if v["control"]["seed_id"] not in excluded_seeds]
    unexposed_one_sided = [v for v in one_sided if next(iter(v.values()))["seed_id"] not in excluded_seeds]

    recoverable = [
        v for v in unexposed_incomplete
        if all(bool(v[side].get("analysis_eligible")) for side in ("control", "rp"))
    ]
    reasons: Counter[str] = Counter()
    for pair in unexposed_incomplete:
        for side in ("control", "rp"):
            _reason(pair[side], side, reasons)

    crossmatched_base_groups: dict[tuple[Any, ...], dict[str, set[str]]] = defaultdict(
        lambda: {"control": set(), "rp": set()}
    )
    for row in rows.values():
        if row.get("split") != "tune" or row.get("seed_id") in excluded_seeds:
            continue
        if row.get("kind") != "harmful" or row.get("prompt_language") not in {"en", "ja"}:
            continue
        if row.get("prompt_language") != row.get("output_language") or row.get("framing") not in {"control", "rp"}:
            continue
        if eligible(row):
            base_key = tuple(row.get(field) for field in BASE_FIELDS)
            crossmatched_base_groups[base_key][row["framing"]].add(str(row.get("request")))
    semantically_reviewable_groups = sum(
        bool(group["control"] and group["rp"])
        for group in crossmatched_base_groups.values()
    )

    return {
        "schema": "lexsi.incomplete_tune_pair_recovery_audit/1",
        "all_exact_request_keys": len(buckets),
        "keys_with_both_control_and_roleplay_rows": len(matched),
        "one_sided_request_keys": len(one_sided),
        "fully_eligible_matched_keys": len(fully_eligible),
        "incomplete_matched_keys": len(incomplete),
        "unique_seeds": {
            "all_structured": len({key[0] for key in buckets}),
            "fully_eligible": len({v["control"]["seed_id"] for v in fully_eligible}),
            "incomplete": len({v["control"]["seed_id"] for v in incomplete}),
            "unexposed_incomplete": len({v["control"]["seed_id"] for v in unexposed_incomplete}),
            "unexposed_one_sided": len({next(iter(v.values()))["seed_id"] for v in unexposed_one_sided}),
            "unexposed_fully_eligible": len({v["control"]["seed_id"] for v in fully_eligible if v["control"]["seed_id"] not in excluded_seeds}),
        },
        "unexposed_recovery_opportunity": {
            "incomplete_matched_keys": len(unexposed_incomplete),
            "keys_where_both_rows_are_analysis_eligible": len(recoverable),
            "seeds_where_both_rows_are_analysis_eligible": len({v["control"]["seed_id"] for v in recoverable}),
            "incomplete_reasons_by_side": dict(sorted(reasons.items())),
            "unexposed_cross_request_base_groups_with_eligible_rows_on_both_sides": semantically_reviewable_groups,
        },
        "interpretation": (
            "Potential inference/scoring recovery is confined to existing seeds and does not itself create validated labels. "
            "Rows marked analysis-ineligible are excluded by the existing design. Do not spend inference compute unless the "
            "remaining independent-seed opportunity can meet a predeclared gate. Counts omit all prompt and response text."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part1", type=Path, required=True)
    parser.add_argument("--part2", type=Path, required=True)
    parser.add_argument("--prior-registry", type=Path, required=True)
    parser.add_argument("--excluded-csv", type=Path, action="append", default=[])
    parser.add_argument("--excluded-jsonl", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    excluded = load_excluded_seeds(args.prior_registry, args.excluded_csv, args.excluded_jsonl)
    rows, _settings = observations((args.part1, args.part2))
    report = audit(rows, excluded)
    report["source_shards"] = [
        {"basename": path.name, "bytes": path.stat().st_size, "sha256": file_sha256(path)}
        for path in (args.part1, args.part2)
    ]
    report["excluded_seed_count"] = len(excluded)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
