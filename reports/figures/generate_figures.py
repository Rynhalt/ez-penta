#!/usr/bin/env python3
"""Regenerate publication-style charts from saved post_train_3 result JSON."""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "post_train_3" / "reports" / "source_data"
OUT = Path(__file__).resolve().parent
TABLES = ROOT / "reports" / "tables"

COLORS = {
    "base": "#6B7280", "en": "#0072B2", "ja": "#D55E00",
    "bi": "#009E73", "en2": "#CC79A7", "bi_h2": "#E69F00",
    "A*bi": "#56B4E9", "original": "#8B5CF6", "us_m": "#0072B2",
    "uk_m": "#D55E00", "g06": "#0072B2", "g4b": "#D55E00",
}
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.titlesize": 12, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "figure.dpi": 140, "savefig.dpi": 220,
    "axes.spines.top": False, "axes.spines.right": False,
})


def read_json(name: str) -> dict:
    path = DATA / name
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Required result source missing or empty: {path}")
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def metric_record(results: dict, metric: str, culture: str, lang: str) -> dict:
    rows = results["headline_g06"][metric]
    found = [r for r in rows if r["culture"] == culture and r["lang"] == lang]
    if len(found) != 1:
        raise ValueError(f"Expected one {metric} row for {culture}/{lang}; found {len(found)}")
    return found[0]


def run_points(row: dict, condition: str) -> list[float]:
    """Return the actual per-training-run rates stored in the analysis output."""
    raw = row.get("raw", {}).get(condition)
    if raw is None:
        raise ValueError(f"No raw summary for {condition} in headline row")
    rates = []
    for _, kn in raw.get("per_seed", {}).items():
        k, n = kn
        if not n:
            raise ValueError(f"Zero denominator for {condition}")
        rates.append(k / n)
    if not rates and "mean" in raw:
        rates = [float(raw["mean"])]
    return rates


def run_points_all_cultures(results: dict, metric: str, condition: str, lang: str) -> list[float]:
    """Pool US/Japan cue counts within each training run before making a rate."""
    rows = [r for r in results["headline_g06"][metric] if r["lang"] == lang]
    if not rows:
        raise ValueError(f"No {metric} rows for prompt language {lang}")
    totals: dict[str, list[int]] = {}
    for row in rows:
        raw = row.get("raw", {}).get(condition)
        if raw is None:
            raise ValueError(f"No raw summary for {condition}/{metric}/{lang}")
        for run, (k, n) in raw.get("per_seed", {}).items():
            pair = totals.setdefault(run, [0, 0])
            pair[0] += int(k)
            pair[1] += int(n)
    if not totals or any(n == 0 for _, n in totals.values()):
        raise ValueError(f"No usable pooled per-run rates for {condition}/{metric}/{lang}")
    return [k / n for k, n in totals.values()]


def fig1_tradeoff(bi: dict) -> None:
    """Descriptive safety/refusal rates pooled over cultural cues within run."""
    conds = ["base", "en", "ja", "bi", "en2"]
    labels = {"base": "Base", "en": "English", "ja": "Japanese", "bi": "Bilingual", "en2": "English ×2"}
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.0), sharex=True, sharey=True)
    rows_for_csv = []
    for ax, lang in zip(axes, ("en", "ja")):
        for cond in conds:
            xs = run_points_all_cultures(bi, "safety", cond, lang)
            ys = run_points_all_cultures(bi, "over_refusal", cond, lang)
            if len(xs) != len(ys):
                raise ValueError(f"Mismatched seed counts for {cond}/{lang}: safety={len(xs)} refusal={len(ys)}")
            c = COLORS[cond]
            for j, (x, y) in enumerate(zip(xs, ys)):
                ax.scatter(x * 100, y * 100, s=44, color=c, alpha=.55, edgecolor="white", linewidth=.5, zorder=2)
                rows_for_csv.append({"eval_language": lang, "condition": cond, "training_run": j,
                                     "unsafe_rate": x, "over_refusal_rate": y})
            xm, ym = float(np.mean(xs)), float(np.mean(ys))
            ax.scatter(xm * 100, ym * 100, marker="D", s=86, color=c, edgecolor="black", linewidth=.7,
                       label=labels[cond], zorder=3)
            ax.annotate(labels[cond], (xm * 100, ym * 100), xytext=(5, 4), textcoords="offset points", fontsize=8)
        ax.set_title(f"{lang.upper()} written prompts · US + Japan cues")
        ax.grid(axis="both", color="#D9DEE5", linewidth=.6, alpha=.7)
        ax.set_xlabel("Unsafe judgments on harmful prompts (%)  → lower is safer")
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel("Refusal on harmless twins (%)")
    fig.suptitle("Training-language effects on Unsafe judgments and harmless refusal", y=.97, fontsize=13)
    fig.text(.5, .015, "Counts pool US- and Japan-framed prompts within each run. Dots are adapter training runs (three per condition); base has one run. Diamonds are run means, not CIs.",
             ha="center", fontsize=8.2)
    fig.tight_layout(rect=[.15, .10, .99, .91])
    save(fig, "fig_safety_overrefusal_tradeoff")
    pd.DataFrame(rows_for_csv).to_csv(TABLES / "safety_overrefusal_points.csv", index=False)


def fig2_bilingual_contrasts(bi: dict) -> None:
    """Formal pooled-culture bi-vs-en and bi-vs-en2 effects from the pre-specified tests."""
    tests = bi["primary_g06"]
    tests4 = bi["primary_g4b"]
    rows = []
    labels = [
        ("English eval", "H1|safety|bi-en|en", "Bilingual − English-only"),
        ("Japanese eval", "H1|safety|bi-en|ja", "Bilingual − English-only"),
        ("English eval", "H3|safety|bi-en2|en", "Bilingual − English ×2"),
        ("Japanese eval", "H3|safety|bi-en2|ja", "Bilingual − English ×2"),
    ]
    fig, ax = plt.subplots(figsize=(10.8, 5.5))
    y_positions = [3.2, 2.7, 1.5, 1.0]
    for (lang, key, comparison), y in zip(labels, y_positions):
        for judge, source, marker, offset in (("0.6B", tests, "o", -.08), ("4B", tests4, "s", .08)):
            rec = source[key]
            x = rec["estimate"] * 100
            lo, hi = [v * 100 for v in rec["ci95"]]
            ax.errorbar(x, y + offset, xerr=[[x - lo], [hi - x]], fmt=marker,
                        color=COLORS["g06" if judge == "0.6B" else "g4b"], capsize=3, markersize=6, linewidth=1.5)
            rows.append({"comparison": comparison, "eval_language": lang, "judge": judge,
                         "estimate_pp": x, "ci_low_pp": lo, "ci_high_pp": hi,
                         "holm_p": rec.get("p_holm"), "n_rows": rec.get("n_rows"), "n_seeds": rec.get("n_seeds")})
    ax.axvline(0, color="#222222", linewidth=1)
    ax.set_yticks([3.2, 2.7, 1.5, 1.0], ["Bi − EN · English", "Bi − EN · Japanese", "Bi − EN×2 · English", "Bi − EN×2 · Japanese"])
    ax.set_xlim(-10, 4)
    ax.set_xlabel("Difference in judge-labeled Unsafe rate (percentage points); negative favors bilingual")
    ax.set_title("Bilingual safety effects, pooled over US and Japan cultural prompts")
    ax.grid(axis="x", color="#D9DEE5", linewidth=.6)
    fig.legend(handles=[plt.Line2D([0], [0], marker="o", color=COLORS["g06"], linestyle="none", label="Qwen3Guard 0.6B"),
                        plt.Line2D([0], [0], marker="s", color=COLORS["g4b"], linestyle="none", label="Qwen3Guard 4B")],
               loc="lower center", bbox_to_anchor=(.5, .06), ncol=2, frameon=False)
    fig.text(.5, .01, "Whiskers are two-level bootstrap 95% CIs (evaluation seeds × training runs); Holm p-values are from the primary judge. Each training condition has 3 runs.",
             ha="center", fontsize=8.5)
    fig.tight_layout(rect=[0, .12, 1, 1])
    save(fig, "fig_bilingual_safety_contrasts")
    pd.DataFrame(rows).to_csv(TABLES / "bilingual_safety_contrasts.csv", index=False)


def fig3_culture(fu: dict) -> None:
    """Paired E4 base unsafe-rate differences for matched cues and original-vs-rewrite."""
    tests = fu["tests"]["E4"]
    specs = [
        ("English", "E4|safety|base: us-us_m (cue content)|en", "Original US − matched US"),
        ("Japanese", "E4|safety|base: us-us_m (cue content)|ja", "Original US − matched US"),
        ("English", "E4|safety|base: uk_m-us_m|en", "Matched UK − matched US"),
        ("Japanese", "E4|safety|base: uk_m-us_m|ja", "Matched UK − matched US"),
    ]
    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    y = [3, 2.5, 1.4, .9]
    rows = []
    for (lang, key, lab), yy in zip(specs, y):
        for judge, rec, marker, off in (("0.6B", tests[key], "o", -.08), ("4B", tests[key]["g4b"], "s", .08)):
            x = rec["estimate"] * 100
            lo, hi = [v * 100 for v in rec["ci95"]]
            ax.errorbar(x, yy + off, xerr=[[x-lo], [hi-x]], fmt=marker, capsize=3, markersize=6,
                        color=COLORS["g06" if judge == "0.6B" else "g4b"], linewidth=1.5)
            rows.append({"comparison": lab, "eval_language": lang, "judge": judge, "estimate_pp": x,
                         "ci_low_pp": lo, "ci_high_pp": hi, "holm_p": rec.get("p_holm"),
                         "n_prompts": rec.get("n"), "n_seeds": rec.get("n_seeds")})
    ax.axvline(0, color="#222222", linewidth=1)
    ax.set_yticks(y, [f"{lab} · {lang}" for lang, _, lab in specs])
    ax.set_xlabel("Difference in Unsafe rate (percentage points); positive means first cue is riskier")
    ax.set_title("Matched-cue comparison separates cue content from the US–UK contrast")
    ax.grid(axis="x", color="#D9DEE5", linewidth=.6)
    ax.legend(handles=[plt.Line2D([0], [0], marker="o", color=COLORS["g06"], linestyle="none", label="Qwen3Guard 0.6B"),
                       plt.Line2D([0], [0], marker="s", color=COLORS["g4b"], linestyle="none", label="Qwen3Guard 4B")],
              loc="lower right", frameon=False)
    fig.text(.5, .01, "Paired seed-cluster bootstrap 95% CIs. Original US prompts versus matched US rewrites changes prompt content; it is not a pure culture manipulation. Matched UK−US intervals include zero.",
             ha="center", fontsize=8.2)
    fig.tight_layout(rect=[0, .07, 1, 1])
    save(fig, "fig_cultural_cues")
    pd.DataFrame(rows).to_csv(TABLES / "cultural_cue_effects.csv", index=False)


def fig4_steering(st: dict) -> None:
    """Forest plot with zero and pre-set non-inferiority margins."""
    p6, p4 = st["primary_g06"], st["primary_g4b"]
    specs = [("English", "H1", "over-refusal", None), ("English", "H2", "unsafe rate", .02),
             ("English", "H3", "fix rate", -.05), ("Japanese", "H1", "over-refusal", None),
             ("Japanese", "H2", "unsafe rate", .02), ("Japanese", "H3", "fix rate", -.05)]
    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    ys = np.arange(len(specs))[::-1]
    rows = []
    for (lang, hid, metric, margin), yy in zip(specs, ys):
        key = f"{hid}|{'en' if lang == 'English' else 'ja'}"
        for judge, source, marker, off in (("0.6B", p6, "o", -.08), ("4B", p4, "s", .08)):
            rec = source[key]
            x, = [rec["estimate"] * 100]
            lo, hi = [v * 100 for v in rec["ci95"]]
            ax.errorbar(x, yy + off, xerr=[[x-lo], [hi-x]], fmt=marker, capsize=3, markersize=6,
                        color=COLORS["g06" if judge == "0.6B" else "g4b"], linewidth=1.4)
            rows.append({"language": lang, "hypothesis": hid, "metric": metric, "judge": judge,
                         "estimate_pp": x, "ci_low_pp": lo, "ci_high_pp": hi,
                         "supported_primary": rec.get("supported"), "confirmed_both": rec.get("confirmed_4b"),
                         "margin_pp": None if margin is None else margin*100,
                         "holm_p": rec.get("p_holm"), "n": rec.get("n"), "n_seeds": rec.get("n_seeds")})
        if margin is not None:
            ax.scatter([margin * 100], [yy], marker="|", s=230, color="#111111", linewidth=2, zorder=1)
    ax.axvline(0, color="#222222", linewidth=1)
    ax.set_yticks(ys, [f"{lang}: {hid} {metric}" for lang, hid, metric, _ in specs])
    ax.set_xlabel("Edited adapter − original adapter (percentage points)")
    ax.set_title("Steering lowered English over-refusal but did not establish safety preservation")
    ax.grid(axis="x", color="#D9DEE5", linewidth=.6)
    fig.legend(handles=[plt.Line2D([0], [0], marker="o", color=COLORS["g06"], linestyle="none", label="Qwen3Guard 0.6B"),
                        plt.Line2D([0], [0], marker="s", color=COLORS["g4b"], linestyle="none", label="Qwen3Guard 4B"),
                        plt.Line2D([0], [0], marker="|", color="#111111", linestyle="none", label="Required safety margin")],
               loc="lower center", bbox_to_anchor=(.5, .055), ncol=3, frameon=False, fontsize=8)
    fig.text(.5, .01, "Whiskers are seed-cluster 95% CIs. The black margin tick is the required bound, not a significance threshold. English H2/H3 fail the prespecified safety criteria.",
             ha="center", fontsize=8.2)
    fig.tight_layout(rect=[0, .12, 1, 1])
    save(fig, "fig_steering_effects")
    pd.DataFrame(rows).to_csv(TABLES / "steering_effects.csv", index=False)


def fig5_followups(fu: dict) -> None:
    """E1/E2 trade-off deltas in US-culture prompts, by language."""
    tests = fu["tests"]
    specs = []
    for exp, keyname, label in (("E1", "A*bi-bi", "E1: steer bi"), ("E2", "bi_h2-bi", "E2: duplicate helpful data")):
        for lang in ("en", "ja"):
            for met, pretty in (("safety", "Unsafe"), ("over_refusal", "Harmless refusal")):
                key = f"{exp}|{met}|{keyname}|us|{lang}"
                specs.append((label, lang, pretty, tests[exp][key]))
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.5), sharex=True, sharey=True)
    rows = []
    ylabels = []
    for ax, lang in zip(axes, ("en", "ja")):
        entries = [s for s in specs if s[1] == lang]
        yy = np.arange(len(entries))[::-1]
        for y, (intervention, _, metric, rec) in zip(yy, entries):
            col = "#D55E00" if metric == "Unsafe" else "#0072B2"
            for judge, r, marker, off in (("0.6B", rec, "o", -.08), ("4B", rec["g4b"], "s", .08)):
                x = r["estimate"] * 100
                lo, hi = [v * 100 for v in r["ci95"]]
                ax.errorbar(x, y + off, xerr=[[x-lo], [hi-x]], fmt=marker, capsize=2.5,
                            markersize=5, color=col, alpha=.95 if judge == "0.6B" else .65,
                            linewidth=1.2)
                rows.append({"intervention": intervention, "eval_language": lang, "metric": metric,
                             "judge": judge, "estimate_pp": x, "ci_low_pp": lo, "ci_high_pp": hi,
                             "holm_p": r.get("p_holm"), "confirmed_both": rec.get("confirmed_4b"),
                             "n": rec.get("n"), "n_seeds": rec.get("n_seeds")})
        ax.axvline(0, color="#222222", linewidth=1)
        ax.set_yticks(yy, [f"{i}: {m}" for i, _, m, _ in entries])
        ax.set_title(f"{lang.upper()} prompts · US cultural cue")
        ax.grid(axis="x", color="#D9DEE5", linewidth=.6)
    axes[0].set_xlabel("")
    axes[1].set_xlabel("")
    axes[0].set_ylabel("Experiment and outcome")
    fig.suptitle("E1/E2 follow-ups: fewer harmless refusals coincided with more Unsafe judgments", y=.97, fontsize=13)
    fig.supxlabel("Change from original bilingual adapter (percentage points)", y=.12, fontsize=10)
    fig.text(.5, .015, "Solid markers: Qwen3Guard 0.6B; lighter markers: 4B. Whiskers are two-level seed/training-run bootstrap 95% CIs. E2 trained for more steps than control.",
             ha="center", fontsize=8.2)
    fig.legend(handles=[plt.Line2D([0], [0], marker="o", color="#555555", linestyle="none", label="Primary judge (0.6B)"),
                        plt.Line2D([0], [0], marker="s", color="#999999", linestyle="none", label="Second judge (4B)"),
                        plt.Line2D([0], [0], color="#D55E00", label="Unsafe rate"),
                        plt.Line2D([0], [0], color="#0072B2", label="Harmless refusal")],
               loc="lower center", bbox_to_anchor=(.5, .04), ncol=4, frameon=False, fontsize=8)
    fig.tight_layout(rect=[.15, .19, 1, .91])
    save(fig, "fig_followup_tradeoffs")
    pd.DataFrame(rows).to_csv(TABLES / "followup_tradeoffs.csv", index=False)


def fig6_xsteer_causal(xd: dict) -> None:
    """Layer-18 causal-hook effects with layer-8 and random-direction controls."""
    specs = [
        ("Harmless twins · English", "D1_L18", "refused", "D1_L8", "D5_L18", "D1 − random (EN)"),
        ("Harmless twins · Japanese", "D2_L18", "refused", "D2_L8", None, None),
        ("Harmful prompts · English", "D4_L18", "unsafe_su", "D4_L8", "D4_L18", "D4 − random (EN)"),
        ("Harmful prompts · Japanese", "D4_L18", "unsafe_su", "D4_L8", None, None),
    ]
    rows_by_key = {(r["arm"], r["metric"], r["lang"]): r for r in xd["rows"]}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8), sharey=False)
    table_rows = []
    for ax, metric, entries in (
        (axes[0], "refused", specs[:2]),
        (axes[1], "unsafe_su", specs[2:]),
    ):
        ys = np.arange(len(entries))[::-1]
        for y, (label, l18arm, rowmetric, l8arm, randomarm, randomlabel) in zip(ys, entries):
            lang = "en" if "English" in label else "ja"
            l18 = rows_by_key[(l18arm, rowmetric, lang)]
            l8 = rows_by_key[(l8arm, rowmetric, lang)]
            for name, rec, color, offset in (("L18 removal", l18, "#0072B2", .10),
                                             ("L8 removal", l8, "#999999", -.10)):
                x, lo, hi = [100 * v for v in (rec["delta"], rec["ci95"][0], rec["ci95"][1])]
                ax.errorbar(x, y + offset, xerr=[[x-lo], [hi-x]], fmt="o", color=color,
                            capsize=3, markersize=6, linewidth=1.4)
                table_rows.append({"outcome": metric, "population": label, "arm": name,
                                   "delta_pp": x, "ci_low_pp": lo, "ci_high_pp": hi,
                                   "n": rec["n"], "mcnemar_p": rec.get("mcnemar_p")})
            if randomarm:
                # The exact intervention-minus-random contrasts are stored separately.
                ckey = "D1_L18 - D5_L18" if "D1" in randomlabel else "D4_L18 - D5_L18"
                contrast = next(c for c in xd["contrasts"] if c["contrast"] == ckey)
                x, lo, hi = [100 * v for v in (contrast["diff"], contrast["ci95"][0], contrast["ci95"][1])]
                ax.errorbar(x, y - .28, xerr=[[x-lo], [hi-x]], fmt="s", color="#D55E00",
                            capsize=3, markersize=5, linewidth=1.3)
                table_rows.append({"outcome": metric, "population": label, "arm": randomlabel,
                                   "delta_pp": x, "ci_low_pp": lo, "ci_high_pp": hi,
                                   "n": contrast["n"], "mcnemar_p": None})
            ax.set_yticks(ys, [item[0] for item in entries])
        ax.axvline(0, color="#222222", linewidth=1)
        ax.grid(axis="x", color="#D9DEE5", linewidth=.6)
    axes[0].set_title("Harmless twin refusal")
    axes[0].set_xlabel("Change vs. no hook (percentage points); negative = fewer refusals")
    axes[1].set_title("Clear-harmful Unsafe judgments")
    axes[1].set_xlabel("Change vs. no hook (percentage points); positive = worse safety")
    fig.suptitle("XSTEER D: layer 18 changes refusal and harmful-prompt safety", y=1.02, fontsize=13)
    fig.legend(handles=[plt.Line2D([0], [0], marker="o", color="#0072B2", linestyle="none", label="Layer 18 removal"),
                        plt.Line2D([0], [0], marker="o", color="#999999", linestyle="none", label="Layer 8 removal"),
                        plt.Line2D([0], [0], marker="s", color="#D55E00", linestyle="none", label="Layer 18 minus random-direction control (English only)")],
               loc="lower center", bbox_to_anchor=(.5, -.03), ncol=1, frameon=False, fontsize=8)
    fig.text(.5, -.14, "Whiskers are seed-cluster bootstrap 95% CIs. Hook direction was selected on a separate fold; layer 8 was a fallback with no qualifying direction. The 4B judge was not used in XSTEER D.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=[0, .08, 1, .96])
    save(fig, "fig_xsteer_causal_tradeoff")
    pd.DataFrame(table_rows).to_csv(TABLES / "xsteer_causal_effects.csv", index=False)


def main() -> None:
    st = read_json("steering_results.json")
    bi = read_json("bilingual_results.json")
    fu = read_json("followups_results.json")
    xd = read_json("xsteer_d_results.json")
    required = [(st, "primary_g06"), (st, "primary_g4b"), (bi, "headline_g06"),
                (bi, "primary_g06"), (bi, "primary_g4b"), (fu, "tests"), (fu, "tables")]
    for obj, key in required:
        if key not in obj:
            raise ValueError(f"Source result missing required key {key}")
    for exp in ("E1", "E2", "E3", "E4"):
        if exp not in fu["tests"] or exp not in fu["tables"]:
            raise ValueError(f"Follow-up result missing {exp}")
    TABLES.mkdir(parents=True, exist_ok=True)
    fig1_tradeoff(bi)
    fig2_bilingual_contrasts(bi)
    fig3_culture(fu)
    fig4_steering(st)
    fig5_followups(fu)
    fig6_xsteer_causal(xd)
    print("Generated 6 figure pairs (PNG + SVG) and 6 CSV figure tables.")
    print(f"Inputs: {DATA / 'steering_results.json'}, {DATA / 'bilingual_results.json'}, {DATA / 'followups_results.json'}, {DATA / 'xsteer_d_results.json'}")


if __name__ == "__main__":
    main()
