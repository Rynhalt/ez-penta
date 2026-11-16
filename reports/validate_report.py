#!/usr/bin/env python3
"""Validate key report claims and local figure/source references."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "experiment_report.md"
DATA = ROOT / "post_train_3" / "reports" / "source_data"
FIGURES = ROOT / "reports" / "figures"
TABLES = ROOT / "reports" / "tables"


def load(name: str) -> dict:
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def close(actual: float, expected: float, tol: float = 0.0002) -> None:
    if abs(actual - expected) > tol:
        raise AssertionError(f"{actual} differs from expected {expected}")


def pooled_count(results: dict, metric: str, condition: str) -> tuple[int, int]:
    """Sum the saved per-seed counts across culture and prompt-language strata."""
    k_total = n_total = 0
    for row in results["headline_g06"][metric]:
        per_seed = row.get("raw", {}).get(condition, {}).get("per_seed", {})
        for k, n in per_seed.values():
            k_total += int(k)
            n_total += int(n)
    return k_total, n_total


def main() -> None:
    text = REPORT.read_text(encoding="utf-8")
    steering = load("steering_results.json")
    bilingual = load("bilingual_results.json")
    followups = load("followups_results.json")
    xsteer = load("xsteer_d_results.json")

    # Abstract's descriptive pooled rates, recomputed from saved per-seed records.
    expected_abstract = {
        ("safety", "base"): (702, 5282),
        ("safety", "bi"): (1147, 16743),
        ("safety", "ja"): (1224, 16542),
        ("over_refusal", "base"): (144, 3800),
        ("over_refusal", "bi"): (2927, 11400),
        ("over_refusal", "ja"): (1726, 11400),
    }
    for key, expected in expected_abstract.items():
        actual = pooled_count(bilingual, *key)
        if actual != expected:
            raise AssertionError(f"Abstract pooled count {key}: got {actual}, expected {expected}")

    close(bilingual["primary_g06"]["H1|safety|bi-en|en"]["estimate"], -0.0275147074)
    close(bilingual["primary_g06"]["H1|safety|bi-en|ja"]["estimate"], -0.0600106457)
    close(bilingual["primary_g06"]["H3|safety|bi-en2|ja"]["estimate"], -0.0743297119)
    close(steering["primary_g06"]["H1|en"]["estimate"], -0.0578470825)
    close(steering["primary_g06"]["H2|en"]["estimate"], 0.0146324054)
    close(steering["primary_g06"]["H3|en"]["estimate"], -0.0519480520)
    close(followups["tests"]["E4"]["E4|safety|base: uk_m-us_m|en"]["estimate"], -0.0024752475)
    close(followups["tests"]["E4"]["E4|safety|base: us-us_m (cue content)|ja"]["estimate"], 0.1317073171)
    close(followups["tests"]["E3"]["E3|safety|ja2-en2|us|ja"]["estimate"], -0.1358524516)

    expected_figures = sorted(set(re.findall(r"!\[[^]]*\]\((figures/[^)]+\.png)\)", text)))
    if len(expected_figures) != 6:
        raise AssertionError(f"Expected six embedded report figures, found {len(expected_figures)}")
    table_names = {
        "fig_safety_overrefusal_tradeoff": "safety_overrefusal_points.csv",
        "fig_bilingual_safety_contrasts": "bilingual_safety_contrasts.csv",
        "fig_cultural_cues": "cultural_cue_effects.csv",
        "fig_steering_effects": "steering_effects.csv",
        "fig_followup_tradeoffs": "followup_tradeoffs.csv",
        "fig_xsteer_causal_tradeoff": "xsteer_causal_effects.csv",
    }
    for rel in expected_figures:
        path = ROOT / "reports" / rel
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
        svg = path.with_suffix(".svg")
        if not svg.is_file() or svg.stat().st_size == 0:
            raise FileNotFoundError(svg)
        table = TABLES / table_names[path.stem]
        if not table.is_file() or table.stat().st_size == 0:
            raise FileNotFoundError(table)
    for name in ("steering_results.json", "bilingual_results.json", "followups_results.json", "xsteer_d_results.json"):
        if not (DATA / name).is_file():
            raise FileNotFoundError(DATA / name)
    if "not proof" not in text and "not evidence" not in text and "not establish equivalence" not in text:
        raise AssertionError("Report must explicitly avoid treating non-detection as equivalence")
    for required in (
        "SafeTune", "CuratorKIT", "AuditKit", "CircuitKIT",
        "original B5 notebook and a structured decomposition output were not present",
        "increasing Unsafe judgments on harmful prompts by 3.8 and 5.8 pp",
        "B then evaluated all 36 layers",
        "No model × language × layer passed the joint held-out readability",
        "not proof of equivalence",
    ):
        if required not in text:
            raise AssertionError(f"Required audited text missing from report: {required}")

    abstract = (ROOT / "reports" / "submission_abstract.md").read_text(encoding="utf-8")
    if len(abstract.split()) >= 500:
        raise AssertionError("Submission abstract must remain under 500 words")
    if "(add here" in abstract.lower():
        raise AssertionError("Submission abstract still contains an unresolved placeholder")
    for required in ("SafeTune", "CuratorKIT", "AuditKit", "CircuitKIT"):
        if required not in abstract:
            raise AssertionError(f"Submission abstract missing toolkit role: {required}")

    d1 = next(r for r in xsteer["rows"] if r["arm"] == "D1_L18" and r["metric"] == "refused" and r["lang"] == "en")
    d4 = next(r for r in xsteer["rows"] if r["arm"] == "D4_L18" and r["metric"] == "unsafe_su" and r["lang"] == "ja")
    close(d1["delta"], -0.1496598639)
    close(d4["delta"], 0.0583333333)

    print(f"Validated {len(expected_figures)} PNG/SVG figure pairs, 9 paired-statistic claims, abstract length/toolkit roles, and report caveats.")
    for name in ("steering_results.json", "bilingual_results.json", "followups_results.json", "xsteer_d_results.json"):
        digest = hashlib.sha256((DATA / name).read_bytes()).hexdigest()
        print(f"{name} sha256={digest}")


if __name__ == "__main__":
    main()
