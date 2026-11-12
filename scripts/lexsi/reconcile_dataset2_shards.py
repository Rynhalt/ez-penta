#!/usr/bin/env python3
"""Reconcile two immutable Dataset2 response shards as one seed-level study.

Each shard is parsed read-only by the existing raw-guard recovery routine. The
wrapper verifies shard/model-dataset provenance, prompt-ID disjointness, full
row coverage and (optionally) exact parity with an independently saved
corrected-label JSONL, then recomputes uncertainty over the combined seeds.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from eval.track2_extension.reconcile_dataset2 import reconcile_sqlite
from eval.track2_extension.stats import seed_cluster_bootstrap


def _read_labels(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            record = json.loads(line)
            prompt_id = record.get("prompt_id")
            if not isinstance(prompt_id, str) or not prompt_id or prompt_id in rows:
                raise ValueError(f"Missing or duplicate prompt_id at {path.name}:{line_number}")
            rows[prompt_id] = record
    return rows


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def reconcile_shards(
    shard_paths: list[Path], output_dir: Path, *, expected_labels: Path | None = None,
    expected_rows: int | None = None, bootstrap_replicates: int = 5000,
    rng_seed: int = 20261111,
) -> dict[str, Any]:
    if len(shard_paths) != 2:
        raise ValueError("Exactly two Dataset2 result shards are required")
    if bootstrap_replicates < 1:
        raise ValueError("bootstrap_replicates must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    corrected_rows: dict[str, dict[str, Any]] = {}
    seed_splits: dict[str, str] = {}
    for index, path in enumerate(shard_paths, 1):
        shard_out = output_dir / f"shard_{index}"
        summary = reconcile_sqlite(path, shard_out, bootstrap_replicates=bootstrap_replicates,
                                   rng_seed=rng_seed)
        label_rows = _read_labels(shard_out / "corrected_guard_labels.jsonl")
        overlap = set(corrected_rows) & set(label_rows)
        if overlap:
            raise ValueError("Prompt IDs overlap between result shards")
        corrected_rows.update(label_rows)
        for record in label_rows.values():
            seed_id = record.get("seed_id")
            split = record.get("split")
            if seed_id and split:
                prior = seed_splits.setdefault(str(seed_id), str(split))
                if prior != str(split):
                    raise ValueError("A source seed is assigned to multiple splits")
        summaries.append({
            "shard_index": index,
            "path_basename": path.name,
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
            "source_row_count": summary["source_row_count"],
            "dataset_metadata": summary["dataset_metadata"],
            "published_binary_labels": summary["published_binary_labels"],
            "recovered_binary_response_labels": summary["recovered_binary_response_labels"],
            "published_binary_labels_changed": summary["published_binary_labels_changed"],
            "invalid_raw_record_shapes": summary["invalid_raw_record_shapes"],
        })

    row_count = sum(item["source_row_count"] for item in summaries)
    if expected_rows is not None and row_count != expected_rows:
        raise ValueError(f"Combined shard row count {row_count} != expected {expected_rows}")
    dataset_meta = [item["dataset_metadata"] for item in summaries]
    if dataset_meta[0] != dataset_meta[1]:
        raise ValueError("Dataset revision/manifest metadata differs between shards")
    declared_count = dataset_meta[0].get("prompt_count")
    if declared_count is not None and int(declared_count) != row_count:
        raise ValueError("Combined row count differs from shard dataset metadata")

    expected_label_comparison = None
    if expected_labels is not None:
        expected_rows_by_id = _read_labels(expected_labels)
        if set(expected_rows_by_id) != set(corrected_rows):
            raise ValueError("Reparsed prompt IDs differ from independently saved corrected labels")
        fields = ("seed_id", "split", "guard_response_corrected", "guard_prompt_corrected")
        mismatches = sum(
            any(corrected_rows[pid].get(field) != expected_rows_by_id[pid].get(field) for field in fields)
            for pid in corrected_rows
        )
        if mismatches:
            raise ValueError(f"Reparsed corrected labels disagree on {mismatches} prompt rows")
        expected_label_comparison = {
            "path_basename": expected_labels.name,
            "sha256": _sha256(expected_labels),
            "matched_rows": len(corrected_rows),
            "field_comparison": list(fields),
            "mismatched_rows": 0,
        }

    effects_by_group: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    matched_group_rows: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for index in (1, 2):
        summary = json.loads((output_dir / f"shard_{index}" / "reconciliation_summary.json").read_text())
        with (output_dir / f"shard_{index}" / "seed_effects.csv").open(
            "r", encoding="utf-8", newline=""
        ) as stream:
            for row in csv.DictReader(stream):
                key = (row["split"], row["prompt_language"])
                effects_by_group[key].append({
                    "seed_id": row["seed_id"],
                    "paired_difference": float(row["paired_difference"]),
                    "matched_strata": int(row["matched_strata"]),
                })
        for result in summary["matched_culture_comparisons"]:
            key = (result["split"], result["prompt_language"])
            aggregate = matched_group_rows[key]
            aggregate.append({
                "eligible_harmful_completed_rows": result["eligible_harmful_completed_rows"],
                "eligible_seed_count": result["eligible_seed_count"],
                "matched_strata_count": result["matched_strata_count"],
            })

    combined_results = []
    for (split, language), effects in sorted(effects_by_group.items()):
        seeds = [row["seed_id"] for row in effects]
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"Source seed overlap between shards in {split}/{language}")
        bootstrap = seed_cluster_bootstrap(
            effects, replicates=bootstrap_replicates, seed=rng_seed
        ) if effects else None
        totals = matched_group_rows[(split, language)]
        combined_results.append({
            "split": split,
            "prompt_language": language,
            "eligible_harmful_completed_rows": sum(r["eligible_harmful_completed_rows"] for r in totals),
            "eligible_seed_count": sum(r["eligible_seed_count"] for r in totals),
            "matched_seed_count": len(effects),
            "matched_strata_count": sum(row["matched_strata"] for row in effects),
            "seed_paired_risk_difference": bootstrap,
            "matching_columns": ["kind", "category", "framing", "mix_direction", "output_language"],
            "unknown_labels_excluded": True,
        })

    combined_effects_path = output_dir / "combined_seed_effects.csv"
    tmp_path = combined_effects_path.with_suffix(".csv.tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("split", "prompt_language", "seed_id", "paired_difference", "matched_strata"))
        writer.writeheader()
        for key in sorted(effects_by_group):
            for row in sorted(effects_by_group[key], key=lambda item: item["seed_id"]):
                writer.writerow({"split": key[0], "prompt_language": key[1], **row})
    tmp_path.replace(combined_effects_path)

    merged_label_path = output_dir / "combined_corrected_guard_labels.jsonl"
    tmp_path = merged_label_path.with_suffix(".jsonl.tmp")
    with tmp_path.open("w", encoding="utf-8", newline="\n") as stream:
        for prompt_id in sorted(corrected_rows):
            stream.write(json.dumps(corrected_rows[prompt_id], ensure_ascii=False, separators=(",", ":")) + "\n")
    tmp_path.replace(merged_label_path)

    final = {
        "schema_version": 1,
        "status": "completed_seed_matched_corrected_guard_reanalysis",
        "shards": summaries,
        "combined_prompt_rows": row_count,
        "unique_source_seeds": len(seed_splits),
        "dataset_metadata": dataset_meta[0],
        "published_binary_labels": sum(item["published_binary_labels"] for item in summaries),
        "recovered_binary_response_labels": sum(item["recovered_binary_response_labels"] for item in summaries),
        "published_binary_labels_changed": sum(item["published_binary_labels_changed"] for item in summaries),
        "invalid_raw_record_shapes": sum(item["invalid_raw_record_shapes"] for item in summaries),
        "expected_label_crosscheck": expected_label_comparison,
        "matched_culture_comparisons": combined_results,
        "estimand_note": "Equal-weight mean of within-seed US-minus-neutral unsafe guard-label risk differences, matching harmful rows within kind/category/framing/mix-direction/output-language; language and split are separate strata.",
        "interpretation_limits": [
            "Unsafe is an automated guard proxy, not human-verified unsafe compliance.",
            "Unknown, Controversial, Missing and Unparsed response labels are excluded, not recoded as Safe.",
            "Matching is associational and does not establish semantic equivalence or causal effects.",
            "Heldout was previously exposed; its results are descriptive only.",
            "Seed bootstrap intervals do not correct outcome-label error or selection bias.",
        ],
        "private_outputs": {
            "seed_effects_csv": combined_effects_path.name,
            "corrected_labels_jsonl": merged_label_path.name,
        },
        "bootstrap": {"replicates": bootstrap_replicates, "rng_seed": rng_seed, "confidence": 0.95},
    }
    summary_path = output_dir / "combined_reconciliation_summary.json"
    temp = summary_path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(final, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(summary_path)
    return final


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part1", required=True, type=Path)
    parser.add_argument("--part2", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-corrected-labels", type=Path)
    parser.add_argument("--expected-rows", type=int)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20261111)
    args = parser.parse_args()
    result = reconcile_shards(
        [args.part1, args.part2], args.output_dir,
        expected_labels=args.expected_corrected_labels,
        expected_rows=args.expected_rows,
        bootstrap_replicates=args.bootstrap_replicates,
        rng_seed=args.seed,
    )
    print(json.dumps({
        "combined_prompt_rows": result["combined_prompt_rows"],
        "unique_source_seeds": result["unique_source_seeds"],
        "recovered_binary_response_labels": result["recovered_binary_response_labels"],
        "published_binary_labels_changed": result["published_binary_labels_changed"],
        "matched_comparison_strata": len(result["matched_culture_comparisons"]),
        "summary_path": str(args.output_dir / "combined_reconciliation_summary.json"),
    }, indent=2))


if __name__ == "__main__":
    main()
