"""Build the post_train_2 training seeds and validation prompts, seed-disjoint by construction.

dataset2 renders each base request ("seed") into many prompts (cultures x harmful/twin x
framings x languages) and assigns every seed to exactly one split, stratified by category
(analysis 50% / tune 30% / heldout 20%). Splitting by prompt_id alone let rewordings of the
same seed land in both training and validation, so this script splits by seed instead:

  - training   <- split == "tune"      (seeds used to build the fix)
  - validation <- split == "analysis"  (seeds the fix never saw in any form)
  - heldout    <- untouched, reserved for a final evaluation

Outputs (all en/ja only):
  pipeline3_seed_prompts_v2.json          refusal seeds: culture=us, kind=harmful, expected=decline
  pipeline3_helpful_seed_prompts_v2.json  helpful seeds: the matched harmless twin of each refusal
                                          seed (same seed, culture, language, framing)
  pipeline5_validation_prompts_v2.json    matched us/neutral pairs from matched_observations.csv:
      harmful buckets -- ALL "us unsafe, neutral safe" pairs (target pattern), "both safe" pairs
      at a 20:50 ratio to those, ALL "us safe, neutral unsafe" pairs -- plus 40 harmless-twin
      pairs ("twin_probe") used only to measure over-refusal.
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1].parent
COMBINED_DIR = ROOT / "notebooks" / "combined"
OUT_DIR = Path(__file__).parent
RNG_SEED = 42
TRAIN_SPLIT, VAL_SPLIT = "tune", "analysis"
N_TWIN_PROBE_PAIRS = 40

df = pd.read_csv(COMBINED_DIR / "observations.csv", low_memory=False)
m = pd.read_csv(COMBINED_DIR / "matched_observations.csv", low_memory=False)
assert df.groupby("seed_id")["split"].nunique().max() == 1, "a seed spans several splits"

eligible = df[
    (df["analysis_eligible"] == True) & (df["generation_status"] == "completed")
    & (df["scoring_status"] == "completed") & (df["prompt_language"].isin(["en", "ja"]))
].copy()
eligible["unsafe"] = eligible["safety"].eq("Unsafe").astype(int)
lookup = df.set_index("prompt_id")

# ---------------- validation (analysis seeds) ----------------
cv = m[
    (m["comparison"] == "culture_vs_neutral") & (m["metric"] == "unsafe")
    & (m["level_a"] == "us") & (m["known"] == True) & (m["split"] == VAL_SPLIT)
].copy()
cv = cv[cv["prompt_id_a"].map(lookup["prompt_language"]).isin(["en", "ja"])]


def bucket(row):
    return {(1, 0): "us_unsafe_neutral_safe", (0, 1): "us_safe_neutral_unsafe",
            (0, 0): "both_safe"}.get((row["value_a"], row["value_b"]), "both_unsafe")


cv["bucket"] = cv.apply(bucket, axis=1)
harmful_pairs = cv[cv["kind"] == "harmful"]
b50 = harmful_pairs[harmful_pairs["bucket"] == "us_unsafe_neutral_safe"]
b30 = harmful_pairs[harmful_pairs["bucket"] == "us_safe_neutral_unsafe"]
b20_all = harmful_pairs[harmful_pairs["bucket"] == "both_safe"]
b20 = b20_all.sample(n=min(round(len(b50) * 20 / 50), len(b20_all)), random_state=RNG_SEED)
twin_pairs = cv[cv["kind"] == "twin"]
twin_probe = twin_pairs.sample(n=min(N_TWIN_PROBE_PAIRS, len(twin_pairs)), random_state=RNG_SEED).copy()
twin_probe["bucket"] = "twin_probe"
val_pairs = pd.concat([b50, b20, b30, twin_probe]).reset_index(drop=True)


def val_row(pid, culture, bucket_name, pair_idx):
    r = lookup.loc[pid]
    return {"prompt": r["prompt"], "id": pid, "seed_id": r["seed_id"], "language": r["prompt_language"],
            "culture": culture, "framing": r["framing"], "kind": r["kind"], "expected": r["expected"],
            "pair_id": int(pair_idx), "pair_bucket": bucket_name}


val_rows = []
for i, r in val_pairs.iterrows():
    val_rows.append(val_row(r["prompt_id_a"], "us", r["bucket"], i))
    val_rows.append(val_row(r["prompt_id_b"], "neutral", r["bucket"], i))

# ---------------- training (tune seeds) ----------------
refusal_df = eligible[
    (eligible["split"] == TRAIN_SPLIT) & (eligible["kind"] == "harmful")
    & (eligible["culture"] == "us") & (eligible["expected"] == "decline")
].sort_values("prompt_id").reset_index(drop=True)

twins = df[(df["split"] == TRAIN_SPLIT) & (df["kind"] == "twin") & (df["analysis_eligible"] == True)
           & (df["generation_status"] == "completed")]
key = ["seed_id", "culture", "prompt_language", "framing"]
helpful_df = refusal_df[key].merge(twins[key + ["prompt_id", "prompt", "expected"]], on=key, how="inner")
helpful_df = helpful_df.drop_duplicates("prompt_id").sort_values("prompt_id").reset_index(drop=True)

refusal_seeds = [{
    "instruction": r["prompt"], "id": r["prompt_id"],
    "metadata": {"seed_id": r["seed_id"], "language": r["prompt_language"], "framing": r["framing"],
                 "culture": "us", "expected": r["expected"], "baseline_unsafe": bool(r["unsafe"])},
} for _, r in refusal_df.iterrows()]
helpful_seeds = [{
    "instruction": r["prompt"], "id": r["prompt_id"],
    "metadata": {"seed_id": r["seed_id"], "language": r["prompt_language"], "framing": r["framing"],
                 "culture": "us", "kind": "twin", "expected": r["expected"]},
} for _, r in helpful_df.iterrows()]

# ---------------- checks ----------------
val_ids = {x["id"] for x in val_rows}
val_seeds = {x["seed_id"] for x in val_rows}
train_ids = {x["id"] for x in refusal_seeds + helpful_seeds}
train_seeds = {x["metadata"]["seed_id"] for x in refusal_seeds + helpful_seeds}
assert len(val_ids) == len(val_rows), "duplicate validation prompt"
assert not (train_ids & val_ids), "prompt overlap between training and validation"
assert not (train_seeds & val_seeds), "seed overlap between training and validation"
assert not set(df.loc[df["prompt_id"].isin(train_ids | val_ids), "split"]) & {"heldout"}, "heldout used"

for name, rows in [("pipeline5_validation_prompts_v2.json", val_rows),
                   ("pipeline3_seed_prompts_v2.json", refusal_seeds),
                   ("pipeline3_helpful_seed_prompts_v2.json", helpful_seeds)]:
    with (OUT_DIR / name).open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

print(f"Validation ({VAL_SPLIT} seeds): {len(val_rows)} prompts, {len(val_pairs)} pairs, {len(val_seeds)} seeds")
print("  buckets:", val_pairs["bucket"].value_counts().to_dict())
print("  languages:", pd.Series([x["language"] for x in val_rows]).value_counts().to_dict())
print(f"Training ({TRAIN_SPLIT} seeds): {len(refusal_seeds)} refusal seeds "
      f"({sum(x['metadata']['baseline_unsafe'] for x in refusal_seeds)} real jailbreak cases) + "
      f"{len(helpful_seeds)} matched harmless twins, {len(train_seeds)} seeds")
print("  refusal languages:", refusal_df["prompt_language"].value_counts().to_dict(),
      " twin expected:", helpful_df["expected"].value_counts().to_dict())
print("Overlap with validation -- prompts: 0, seeds: 0 (asserted); heldout untouched (asserted)")
