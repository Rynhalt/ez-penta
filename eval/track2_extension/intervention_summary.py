"""Helpers that keep CircuitKIT's undefined faithfulness ratios out of summaries."""

from __future__ import annotations

import math
from numbers import Real
from typing import Any, Iterable


def valid_raw_ratio_values(records: Iterable[dict[str, Any]], pillar: str) -> list[float]:
    """Return only finite, defined CircuitKIT ratios for `patching`/`ablation`.

    CircuitKIT returns a numeric 0.0 sentinel when the clean/corrupt denominator
    is degenerate. Require explicit `False` for both invalidity flags; absent
    flags are unknown and therefore excluded from a ratio estimate.
    """
    if pillar not in {"patching", "ablation"}:
        raise ValueError("pillar must be 'patching' or 'ablation'")
    values = []
    for record in records:
        if record.get(f"{pillar}_degenerate_denominator") is not False:
            continue
        if record.get(f"{pillar}_inverted_denominator") is not False:
            continue
        value = record.get(f"{pillar}_raw_ratio")
        if isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(float(value)):
            values.append(float(value))
    return values


def invalid_raw_ratio_count(records: Iterable[dict[str, Any]], pillar: str) -> int:
    """Count rows whose ratio is undefined, invalid, or missing a validity flag."""
    if pillar not in {"patching", "ablation"}:
        raise ValueError("pillar must be 'patching' or 'ablation'")
    rows = list(records)
    return sum(
        row.get(f"{pillar}_degenerate_denominator") is not False
        or row.get(f"{pillar}_inverted_denominator") is not False
        or not isinstance(row.get(f"{pillar}_raw_ratio"), Real)
        or isinstance(row.get(f"{pillar}_raw_ratio"), bool)
        or not math.isfinite(float(row.get(f"{pillar}_raw_ratio")))
        for row in rows
    )
