"""Validate two shard exports and recompute observation-level combined results.

python -m eval.merge_dataset2_results --part1 PATH/part1 --part2 PATH/part2 --output PATH/combined
"""

import argparse
import json
from pathlib import Path
import pandas as pd
from eval.checkpoints import config_id
from eval.dataset2_data import load_dataset, verify_manifest, quality_flags
from eval.dataset2_runtime import response_identity, score_identity
from eval.dataset2_analysis import observation_table, export_bundle


def read_current(path, ids, shard):
    result = {}
    for line in Path(path).read_text().splitlines():
        r = json.loads(line)
        pid = r["prompt_id"]
        if pid in result:
            raise ValueError("Duplicate current prompt ID")
        if r["shard_id"] != shard or pid not in ids:
            raise ValueError("Unexpected or foreign prompt ID")
        result[pid] = r
    if set(result) != ids:
        raise ValueError(
            "Missing prompt IDs in export; re-export shard including explicit missing observations"
        )
    return result


def merge(
    part1,
    part2,
    output,
    dataset="marcus/dataset2_partial.sqlite3",
    manifest="eval/manifests/dataset2_shards.csv",
):
    p, s = load_dataset(dataset)
    m = pd.read_csv(manifest)
    dm = json.loads(Path(manifest).with_suffix(".json").read_text())
    full = quality_flags(verify_manifest(p, s, m, dm, dataset))
    metas = [
        json.loads((Path(d) / "run_metadata.json").read_text()) for d in [part1, part2]
    ]
    for shard, meta in enumerate(metas, 1):
        if (
            meta["shard_id"] != shard
            or meta["run_mode"] != "full_shard"
            or meta["dataset"] != dm
        ):
            raise ValueError("Dataset, shard or run mode incompatible")
    for key in ["generation", "judge", "analysis_policy"]:
        if metas[0][key] != metas[1][key]:
            raise ValueError(f"Incompatible {key} configurations")
    answers = {}
    scores = {}
    for shard, directory, meta in zip([1, 2], [part1, part2], metas):
        subset = full[full.shard_id == shard]
        ids = set(subset.prompt_id)
        if set(meta["prompt_ids"]) != ids or len(meta["prompt_ids"]) != len(ids):
            raise ValueError("Manifest prompt inventory mismatch")
        a = read_current(Path(directory) / "answers_current.jsonl", ids, shard)
        b = read_current(Path(directory) / "scores_current.jsonl", ids, shard)
        for row in subset.astype(object).where(subset.notna(), None).to_dict("records"):
            pid = row["prompt_id"]
            ar = a[pid]
            sr = b[pid]
            if ar["status"] != "missing":
                if ar.get("configuration") != meta["generation"] or ar[
                    "identity"
                ] != response_identity(row, meta["generation"]):
                    raise ValueError("Answer identity mismatch")
                for field, value in row.items():
                    if ar.get(field) != value:
                        raise ValueError(f"Answer metadata mismatch {pid}: {field}")
            if sr["status"] != "missing":
                if (
                    sr.get("configuration") != meta["judge"]
                    or ar["status"] != "completed"
                    or sr["identity"] != score_identity(row, ar, meta["judge"])
                ):
                    raise ValueError("Score identity mismatch")
            if pid in answers:
                raise ValueError("Cross-shard duplicate")
            answers[pid] = ar if ar["status"] != "missing" else None
            scores[pid] = sr if sr["status"] != "missing" else None
    if set(answers) != set(full.prompt_id):
        raise ValueError("Union mismatch")
    rows = full.astype(object).where(full.notna(), None).to_dict("records")
    table = observation_table(rows, answers, scores)
    metadata = {
        **metas[0],
        "shard_id": "combined",
        "source_runs": metas,
        "prompt_ids": sorted(answers),
    }
    return export_bundle(output, table, answers, scores, metadata)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["part1", "part2", "output"]:
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--dataset", default="marcus/dataset2_partial.sqlite3")
    parser.add_argument("--manifest", default="eval/manifests/dataset2_shards.csv")
    args = parser.parse_args()
    _, coverage = merge(**vars(args))
    print(json.dumps(coverage, indent=2))
