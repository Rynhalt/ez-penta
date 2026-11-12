"""Export structural validation, review flags, missing coverage and shard balance."""

import argparse
import json
from pathlib import Path
import pandas as pd
from eval.dataset2_data import (
    load_dataset,
    verify_manifest,
    quality_flags,
    missing_variants,
)
from eval.dataset2_runtime import atomic_json


def validate(dataset, manifest, output):
    p, s = load_dataset(dataset)
    m = pd.read_csv(manifest)
    meta = json.loads(Path(manifest).with_suffix(".json").read_text())
    flags = quality_flags(verify_manifest(p, s, m, meta, dataset))
    missing = missing_variants(p, s)
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    flags.to_csv(directory / "validation.csv", index=False)
    missing.to_csv(directory / "missing_variants.csv", index=False)
    for column in [
        "category",
        "framing",
        "culture",
        "prompt_language",
        "output_language",
        "split",
    ]:
        flags.groupby([column, "shard_id"]).size().rename("retained").to_csv(
            directory / f"coverage_by_{column}.csv"
        )
        missing.groupby(column).size().rename("missing").to_csv(
            directory / f"missing_by_{column}.csv"
        )
    report = {
        "dataset": meta,
        "prompts": len(flags),
        "seeds": len(s),
        "missing_design_variants": len(missing),
        "complete_seeds": int(s.complete.sum()),
        "identical_control_pairs": int(
            flags[
                (flags.framing == "control")
                & flags.identical_twin_flag
                & (flags.kind == "harmful")
            ].shape[0]
        ),
        "duplicate_text_extra_rows": int(p.prompt.duplicated().sum()),
        "structural_errors": 0,
        "excluded_from_generation": 0,
        "human_validated_expectations": 0,
        "shards": m.groupby("shard_id")
        .agg(seeds=("seed_id", "size"), prompts=("actual_prompt_count", "sum"))
        .reset_index()
        .to_dict("records"),
    }
    atomic_json(directory / "validation_report.json", report)
    m.groupby(["category", "split", "shard_id"]).agg(
        seeds=("seed_id", "size"), prompts=("actual_prompt_count", "sum")
    ).to_csv(directory / "shard_balance.csv")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="marcus/dataset2_partial.sqlite3")
    parser.add_argument("--manifest", default="eval/manifests/dataset2_shards.csv")
    parser.add_argument("--output", default="/tmp/dataset2_validation")
    print(json.dumps(validate(**vars(parser.parse_args())), indent=2))
