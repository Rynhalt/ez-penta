"""Reproduce Pipeline 1's Mantel-Haenszel results from result shards.

This reads only the published binary `safety` column and does not infer/relabel
outcomes. Rows and cell counts are summarized; prompt content is never emitted.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Iterable


def pipeline1_mh(shards: Iterable[Path], expected_cohort_rows: int | None = 4069) -> dict:
    tables_by_comparison: dict[tuple[str, str], dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(lambda: [0, 0, 0, 0])
    )
    cohort_rows = 0
    for shard in shards:
        connection = sqlite3.connect(f"file:{shard.resolve()}?mode=ro", uri=True)
        try:
            query = """SELECT culture, prompt_language, safety, framing FROM observations
                WHERE analysis_eligible = 1
                  AND generation_status = 'completed'
                  AND scoring_status = 'completed'
                  AND split != 'heldout'
                  AND kind = 'harmful'"""
            for culture, language, safety, framing in connection.execute(query):
                cohort_rows += 1
                if culture not in {"us", "japan", "neutral"} or language not in {"en", "ja", "mix"}:
                    continue
                if safety not in {"Safe", "Unsafe"}:
                    continue
                for exposure in ("us", "japan"):
                    if culture not in {exposure, "neutral"}:
                        continue
                    for framing_filter in ("all", "rp"):
                        if framing_filter == "rp" and framing != "rp":
                            continue
                        # a=exposed/Unsafe, b=exposed/Safe,
                        # c=neutral/Unsafe, d=neutral/Safe
                        index = (0 if culture == exposure else 2) + (0 if safety == "Unsafe" else 1)
                        tables_by_comparison[(exposure, framing_filter)][language][index] += 1
        finally:
            connection.close()

    if expected_cohort_rows is not None and cohort_rows != expected_cohort_rows:
        raise ValueError(f"Pipeline 1 cohort rows {cohort_rows} != expected {expected_cohort_rows}")
    def summarize(exposure: str, framing: str, tables: dict[str, list[int]]) -> dict:
        if set(tables) != {"en", "ja", "mix"}:
            raise ValueError(f"Expected en/ja/mix strata for {exposure}/{framing}, got {sorted(tables)}")
        numerator = denominator = observed = expected = variance = 0.0
        sum_p_r = sum_p_s = sum_q_r = sum_q_s = 0.0
        strata = []
        for language in ("en", "ja", "mix"):
            a, b, c, d = tables[language]
            n = a + b + c + d
            if n <= 1 or a * d == 0 or b * c == 0:
                raise ValueError(f"Degenerate 2x2 table in {exposure}/{framing}/{language}")
            numerator += a * d / n
            denominator += b * c / n
            r_term, s_term = a * d / n, b * c / n
            p_term, q_term = (a + d) / n, (b + c) / n
            sum_p_r += p_term * r_term
            sum_p_s += p_term * s_term
            sum_q_r += q_term * r_term
            sum_q_s += q_term * s_term
            observed += a
            expected += (a + b) * (a + c) / n
            variance += (a + b) * (c + d) * (a + c) * (b + d) / (n * n * (n - 1))
            strata.append({"prompt_language": language, "a_exposed_unsafe": a, "b_exposed_safe": b,
                           "c_neutral_unsafe": c, "d_neutral_safe": d, "rows": n,
                           "stratum_or": a * d / (b * c)})
        or_mh = numerator / denominator
        chi2_cc = max(abs(observed - expected) - 0.5, 0.0) ** 2 / variance
        p_value = math.erfc(math.sqrt(chi2_cc / 2.0))
        variance_log_or = (sum_p_r / (2 * numerator**2)) + (
            (sum_p_s + sum_q_r) / (2 * numerator * denominator)
        ) + (sum_q_s / (2 * denominator**2))
        ci_multiplier = 1.96 * math.sqrt(variance_log_or)
        return {
            "exposure": f"{exposure} versus neutral",
            "framing": framing,
            "binary_rows": sum(row["rows"] for row in strata),
            "or_mh": or_mh,
            "ci95_rbg": [math.exp(math.log(or_mh) - ci_multiplier),
                         math.exp(math.log(or_mh) + ci_multiplier)],
            "cmh_chi_square_continuity_corrected": chi2_cc,
            "p_value_chi_square_1df": p_value,
            "strata": strata,
        }

    comparisons = [summarize(exposure, framing, tables)
                   for exposure in ("us", "japan")
                   for framing, tables in ((frame, tables_by_comparison[(exposure, frame)])
                                           for frame in ("all", "rp"))]
    us_all = comparisons[0]
    return {
        "schema": "lexsi.pipeline1_mh_reproduction/1",
        "cohort_rows": cohort_rows,
        "stratification": "prompt_language",
        "outcome_source": "published observations.safety values; no corrected-label substitution",
        "ci_method": "Robins-Breslow-Greenland log-OR variance, Wald 1.96 multiplier",
        "comparisons": comparisons,
        # Backward-compatible aliases for the principal all-framing US comparison.
        **{key: value for key, value in us_all.items() if key not in {"framing", "binary_rows"}},
        "binary_us_neutral_rows": us_all["binary_rows"],
        "limits": ["Automated guard outcomes are not human-validated unsafe-compliance labels.",
                   "This reproduces association statistics, not causal effects."],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part1", type=Path, required=True)
    parser.add_argument("--part2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-cohort-rows", type=int, default=4069)
    args = parser.parse_args()
    result = pipeline1_mh((args.part1, args.part2), args.expected_cohort_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
