"""Independently verify saved seed-effect summaries and percentile intervals.

This checks bootstrap arithmetic conditional on an existing private seed-effects
CSV. It does not rebuild those effects from source observations or validate labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def verify(seed_effects: Path, summary_path: Path, replicates: int, seed: int) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    with seed_effects.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required = {"split", "prompt_language", "paired_difference"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"seed-effects CSV must contain columns: {sorted(required)}")
        for row in reader:
            value = float(row["paired_difference"])
            if not (value == value and abs(value) != float("inf")):
                raise ValueError("paired_difference must be finite")
            grouped[(row["split"], row["prompt_language"])].append(value)

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected = {
        (item["split"], item["prompt_language"]): item
        for item in summary["comparisons"]
    }
    if set(grouped) != set(expected):
        raise ValueError("seed-effect groups do not match summary groups")

    results = []
    for index, key in enumerate(sorted(grouped)):
        values = grouped[key]
        expected_item = expected[key]
        if len(values) != expected_item["matched_seeds"]:
            raise ValueError(f"seed count mismatch for {key}")
        mean = sum(values) / len(values)
        if abs(mean - expected_item["risk_difference"]) > 1e-12:
            raise ValueError(f"mean mismatch for {key}")

        rng = random.Random(seed + index)
        draws = []
        for _ in range(replicates):
            draws.append(sum(rng.choice(values) for _ in values) / len(values))
        draws.sort()
        low = draws[int(0.025 * replicates)]
        high = draws[int(0.975 * replicates)]
        reported = expected_item["ci95_seed_bootstrap"]
        results.append(
            {
                "split": key[0],
                "language": key[1],
                "n_seeds": len(values),
                "mean": mean,
                "reported_ci": reported,
                "independent_ci": [low, high],
                "max_endpoint_delta": max(abs(low - reported[0]), abs(high - reported[1])),
            }
        )

    return {
        "schema": "lexsi.seed_bootstrap_verification/1",
        "seed_effect_rows": sum(map(len, grouped.values())),
        "groups": results,
        "replicates_per_group": replicates,
        "rng": f"Python random.Random; base seed {seed}, plus sorted-group index",
        "limits": [
            "Conditional on saved per-seed effects; does not reproduce raw SQLite extraction.",
            "Guard-proxy outcomes are not human labels; heldout was previously exposed.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-effects", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20261112)
    args = parser.parse_args()
    if args.replicates < 1_000:
        parser.error("--replicates must be at least 1000")
    result = verify(args.seed_effects, args.summary, args.replicates, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        f"verified {result['seed_effect_rows']} seed effects across {len(result['groups'])} groups; "
        f"max interval endpoint delta={max(g['max_endpoint_delta'] for g in result['groups']):.6f}"
    )


if __name__ == "__main__":
    main()
