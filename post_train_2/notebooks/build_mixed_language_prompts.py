"""Build pipeline6_mixed_language_prompts.json: a stratified sample of Dataset2's code-mixed
prompts for pipeline6_mixed_language_eval.ipynb.

Source: `analysis`-split, analysis-eligible rows with prompt_language == "mix" (2,145 rows from
155 seeds). These seeds share nothing with training (`tune` seeds), the prompts were never in the
pipeline 5 validation set (English/Japanese only), and `heldout` stays untouched.

Default sample (deterministic, seed 42): 400 harmful prompts to decline, 100 borderline
(answer_carefully) and 200 harmless twins. Within each group the sample is allocated across
framing x mix direction x culture in proportion to their sizes, and spread across seeds (each
seed's first pick is taken before any seed's second). `--full` writes all 2,145 instead.

    python post_train_2/notebooks/build_mixed_language_prompts.py [--full]
"""
import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1].parent
COMBINED_DIR = ROOT / "notebooks" / "combined"
OUT_DIR = Path(__file__).parent
RNG_SEED = 42
TARGETS = {"decline": 400, "answer_carefully": 100, "answer": 200}
STRATA = ["framing", "mix_direction", "culture"]


def allocate(sizes: pd.Series, n: int) -> pd.Series:
    """Proportional allocation with largest remainders, never above a stratum's size."""
    exact = sizes / sizes.sum() * n
    alloc = exact.astype(int)
    for key in (exact - alloc).sort_values(ascending=False).index[: n - alloc.sum()]:
        alloc[key] += 1
    return alloc.clip(upper=sizes)


def spread_take(rows: pd.DataFrame, n: int) -> pd.DataFrame:
    """Random rows, preferring seeds not yet picked, so the sample covers as many seeds as possible."""
    rows = rows.sample(frac=1, random_state=RNG_SEED)
    rows = rows.assign(_seed_rank=rows.groupby("seed_id").cumcount())
    return rows.sort_values("_seed_rank", kind="stable").head(n).drop(columns="_seed_rank")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="all 2,145 eligible analysis-split mixed prompts")
    args = ap.parse_args()

    df = pd.read_csv(COMBINED_DIR / "observations.csv", low_memory=False)
    pool = df[(df["analysis_eligible"] == True) & (df["generation_status"] == "completed")
              & (df["scoring_status"] == "completed") & (df["split"] == "analysis")
              & (df["prompt_language"] == "mix")].copy()

    if args.full:
        sample = pool
    else:
        parts = []
        for expected, n in TARGETS.items():
            group = pool[pool["expected"] == expected]
            alloc = allocate(group.groupby(STRATA).size(), n)
            for key, k in alloc.items():
                if k:
                    stratum = group.set_index(STRATA).loc[[key]].reset_index()
                    parts.append(spread_take(stratum, int(k)))
        sample = pd.concat(parts)
    sample = sample.sort_values("prompt_id").reset_index(drop=True)

    # ---- checks
    train_seeds = set()
    for f in ("pipeline3_seed_prompts_v2.json", "pipeline3_helpful_seed_prompts_v2.json"):
        train_seeds |= {r["metadata"]["seed_id"] for r in json.loads((OUT_DIR / f).read_text())}
    val_ids = {r["id"] for r in json.loads((OUT_DIR / "pipeline5_validation_prompts_v2.json").read_text())}
    assert sample["prompt_id"].is_unique
    assert set(sample["split"]) == {"analysis"} and set(sample["prompt_language"]) == {"mix"}
    assert not set(sample["seed_id"]) & train_seeds, "seed shared with training"
    assert not set(sample["prompt_id"]) & val_ids, "prompt already in the pipeline 5 validation set"

    rows = [{
        "prompt": r["prompt"], "id": r["prompt_id"], "seed_id": r["seed_id"], "category": r["category"],
        "language": "mix", "mix_direction": r["mix_direction"], "output_language": r["output_language"],
        "culture": r["culture"], "framing": r["framing"], "kind": r["kind"], "expected": r["expected"],
        "gguf_baseline_label": r["safety"],
        "gguf_baseline_refused": None if pd.isna(r["refused"]) else int(bool(r["refused"])),
    } for _, r in sample.iterrows()]
    name = "pipeline6_mixed_language_prompts_full.json" if args.full else "pipeline6_mixed_language_prompts.json"
    (OUT_DIR / name).write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote {OUT_DIR / name}: {len(rows)} prompts from {sample['seed_id'].nunique()} seeds "
          f"(pool: {len(pool)} prompts, {pool['seed_id'].nunique()} seeds)")
    for col in ["expected", "framing", "mix_direction", "output_language", "culture"]:
        print(f"  {col}: {sample[col].value_counts().to_dict()}")
    print("  max prompts per seed:", int(sample["seed_id"].value_counts().max()))
    print("Checks passed: analysis split, mixed language, 0 training seeds, 0 validation prompts.")


if __name__ == "__main__":
    main()
