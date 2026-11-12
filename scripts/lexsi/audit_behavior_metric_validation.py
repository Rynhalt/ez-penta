#!/usr/bin/env python3
"""Summarize Phase 1 refusal-margin signs and continuation-length diagnostics.

The output intentionally omits prompt text, answers, and seed/row identifiers.
It describes an exploratory metric run; it does not validate the metric.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def _sign(value: float) -> str:
    if value > 0:
        return "positive"
    if value < 0:
        return "negative"
    return "tie"


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0}
    refusal_tokens = [row[side]["refusal_tokens"] for row in rows for side in ("control", "roleplay")]
    compliance_tokens = [row[side]["compliance_tokens"] for row in rows for side in ("control", "roleplay")]
    controls = [row["control"]["mean_margin"] for row in rows]
    roleplays = [row["roleplay"]["mean_margin"] for row in rows]
    deltas = [row["paired_delta"] for row in rows]
    return {
        "n": len(rows),
        "control_margin_signs": {sign: sum(_sign(x) == sign for x in controls) for sign in ("positive", "negative", "tie")},
        "roleplay_margin_signs": {sign: sum(_sign(x) == sign for x in roleplays) for sign in ("positive", "negative", "tie")},
        "paired_delta_signs": {sign: sum(_sign(x) == sign for x in deltas) for sign in ("positive", "negative", "tie")},
        "both_expected_absolute_signs": sum(c > 0 and r < 0 for c, r in zip(controls, roleplays)),
        "median_refusal_continuation_tokens": statistics.median(refusal_tokens),
        "median_compliance_continuation_tokens": statistics.median(compliance_tokens),
        "median_compliance_to_refusal_token_ratio": statistics.median(
            c / r for c, r in zip(compliance_tokens, refusal_tokens)
        ),
    }


def summarize(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("status") != "INSUFFICIENT_VALIDATION_DATA":
        raise ValueError("Expected the recorded Phase 1 insufficient-validation report")
    rows = payload.get("pairs")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Phase 1 report must contain scored pairs")

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for side in ("control", "roleplay"):
            if not isinstance(row.get(side), dict):
                raise ValueError(f"Missing {side} scores")
            for field in ("mean_margin", "refusal_tokens", "compliance_tokens"):
                if field not in row[side]:
                    raise ValueError(f"Missing {side}.{field}")
        if "paired_delta" not in row or not row.get("language") or not row.get("split"):
            raise ValueError("Pair is missing language, split, or paired delta")
        grouped[(row["split"], row["language"])].append(row)

    return {
        "schema": "lexsi.behavior_metric_failure_diagnosis/1",
        "source_status": payload["status"],
        "source_annotation_provenance": payload.get("annotation_provenance"),
        "source_independent_human_annotation": payload.get("independent_human_annotation"),
        "overall": _summarize(rows),
        "by_split_and_language": {
            f"{split}_{language}": _summarize(group)
            for (split, language), group in sorted(grouped.items())
        },
        "interpretation": (
            "Descriptive diagnosis only. Positive paired deltas indicate relative margin ordering, "
            "not correct behavioral classification. The absolute-sign gate failed. Full-response "
            "continuations vary in length, are scored without EOS, and contain different semantic "
            "content; length normalization does not remove those confounds. Do not treat this "
            "analysis as metric validation or use analysis-set results to tune a replacement metric."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(json.loads(args.input.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
