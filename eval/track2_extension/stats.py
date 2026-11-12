"""Seed-clustered paired summaries for repeated Dataset2 prompt rows.

The estimand is the equal-weight mean of within-seed paired differences. Within
each seed and matching stratum, known outcomes are averaged separately by
condition; only strata observed in both conditions contribute. Unknown labels
are excluded, never recoded as safe.
"""

from __future__ import annotations

from collections import defaultdict
import math
import random
from typing import Any, Hashable, Iterable, Mapping, Sequence


def paired_seed_differences(
    rows: Iterable[Mapping[str, Any]],
    *,
    seed_column: str,
    condition_column: str,
    treatment_value: Hashable,
    control_value: Hashable,
    outcome_column: str,
    matching_columns: Sequence[str],
    outcome_mapping: Mapping[Any, float],
) -> list[dict[str, Any]]:
    """Compute one paired difference per seed after matching within strata.

    `outcome_mapping` must explicitly map known outcomes to numeric values in
    [0, 1], for example ``{"Safe": 0, "Unsafe": 1}``. Any value absent from
    that mapping is unknown and excluded. For each seed/matching stratum, rows
    are averaged within treatment and control before taking their difference;
    strata missing either condition do not contribute.
    """
    if treatment_value == control_value:
        raise ValueError("treatment_value and control_value must differ")
    if not matching_columns:
        raise ValueError("matching_columns must include the intended pair structure")
    if not outcome_mapping:
        raise ValueError("outcome_mapping must explicitly define known outcomes")
    for label, value in outcome_mapping.items():
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"Outcome mapping for {label!r} must be finite numeric")
        if not 0 <= value <= 1:
            raise ValueError(f"Outcome mapping for {label!r} must be in [0, 1]")

    # (seed, matching key, condition) -> known outcomes
    cells: dict[tuple[Hashable, tuple[Hashable, ...], Hashable], list[float]] = defaultdict(list)
    required = [seed_column, condition_column, outcome_column, *matching_columns]
    for index, row in enumerate(rows):
        missing = [name for name in required if name not in row]
        if missing:
            raise KeyError(f"Row {index} missing required columns: {missing}")
        seed = row[seed_column]
        condition = row[condition_column]
        if seed is None:
            continue
        if condition not in (treatment_value, control_value):
            continue
        value = outcome_mapping.get(row[outcome_column])
        if value is None:
            continue
        key = tuple(row[column] for column in matching_columns)
        if any(item is None for item in key):
            continue
        cells[(seed, key, condition)].append(float(value))

    paired: dict[Hashable, list[float]] = defaultdict(list)
    for seed, key, condition in list(cells):
        if condition != treatment_value:
            continue
        treatment = cells[(seed, key, treatment_value)]
        control = cells.get((seed, key, control_value))
        if control:
            paired[seed].append(
                sum(treatment) / len(treatment) - sum(control) / len(control)
            )

    results = []
    for seed in sorted(paired, key=str):
        differences = paired[seed]
        results.append(
            {
                seed_column: seed,
                "paired_difference": sum(differences) / len(differences),
                "matched_strata": len(differences),
            }
        )
    return results


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot compute a quantile from no bootstrap replicates")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def seed_cluster_bootstrap(
    seed_differences: Sequence[Mapping[str, Any]],
    *,
    value_column: str = "paired_difference",
    replicates: int = 5000,
    seed: int = 20261111,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Percentile bootstrap over independent source seeds, with equal seed weight."""
    if replicates < 1:
        raise ValueError("replicates must be positive")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be strictly between 0 and 1")
    values = [float(row[value_column]) for row in seed_differences]
    if not values:
        raise ValueError("At least one seed-level difference is required")
    if any(not math.isfinite(value) for value in values):
        raise ValueError("Seed differences must be finite")
    point = sum(values) / len(values)
    rng = random.Random(seed)
    draws = [
        sum(rng.choices(values, k=len(values))) / len(values)
        for _ in range(replicates)
    ]
    tail = (1 - confidence) / 2
    return {
        "estimand": "equal_weight_mean_within_seed_paired_difference",
        "n_seeds": len(values),
        "point_estimate": point,
        "confidence": confidence,
        "interval_method": "percentile_seed_cluster_bootstrap",
        "interval_low": _quantile(draws, tail),
        "interval_high": _quantile(draws, 1 - tail),
        "replicates": replicates,
        "rng_seed": seed,
    }
