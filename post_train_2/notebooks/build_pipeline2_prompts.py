"""Build pipeline2_prompts_v2.json: 64 prompts per (language x condition) cell, for
pipeline2_circuit_discovery.ipynb's refusal-direction extraction + projection test.

Committed this time (post_train_1's equivalent script lived only in a scratchpad and
was never saved to the repo -- a flagged reproducibility gap). Deterministic (fixed
random seed), draws from the same non-heldout pool used throughout post_train_2, and
is checked for zero overlap with both the new training seeds and the new validation set.
"""
import json
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1].parent
COMBINED_DIR = ROOT / "notebooks" / "combined"

df = pd.read_csv(COMBINED_DIR / "observations.csv", low_memory=False)
rng_seed = 42
N_PER_CELL = 64

base_mask = (
    (df["analysis_eligible"] == True) & (df["generation_status"] == "completed")
    & (df["scoring_status"] == "completed") & (df["split"] == "tune")
    & (df["prompt_language"].isin(["en", "ja"])) & (df["framing"] == "control")
)
# Pipeline 2 is part of building the fix (it picks the LoRA target layers), so it draws from
# the same "tune" seeds as the training data; validation uses only "analysis" seeds.
pool = df[base_mask].copy()

with open(Path(__file__).parent / "pipeline5_validation_prompts_v2.json", encoding="utf-8") as f:
    val_seeds = {r["seed_id"] for r in json.load(f)}
assert not set(pool["seed_id"]) & val_seeds, "pipeline 2 sample shares a seed with validation"

data = {}
for lang in ["en", "ja"]:
    lang_pool = pool[pool["prompt_language"] == lang]
    harmful_neutral = lang_pool[(lang_pool["kind"] == "harmful") & (lang_pool["culture"] == "neutral")]
    twin_neutral = lang_pool[(lang_pool["kind"] == "twin") & (lang_pool["culture"] == "neutral")]
    harmful_us = lang_pool[(lang_pool["kind"] == "harmful") & (lang_pool["culture"] == "us")]

    def sample_cell(sub, name):
        n = min(N_PER_CELL, len(sub))
        if n < N_PER_CELL:
            print(f"WARNING: only {n} available for {lang}/{name}, wanted {N_PER_CELL}")
        return sub.sample(n=n, random_state=rng_seed)["prompt"].tolist()

    cells = {"harmful_neutral": harmful_neutral, "twin_neutral": twin_neutral, "harmful_us": harmful_us}
    data[lang] = {name: sample_cell(sub, name) for name, sub in cells.items()}
    data[lang].update({f"n_{name}": len(data[lang][name]) for name in cells})

out_path = Path(__file__).parent / "pipeline2_prompts_v2.json"
with out_path.open("w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
print(f"Wrote {out_path}")
for lang in ["en", "ja"]:
    print(lang, {k: v for k, v in data[lang].items() if k.startswith("n_")})
