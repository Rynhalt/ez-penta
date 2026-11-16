"""Upload the folders built by build_hf_release.py to the Hugging Face Hub and verify them.

Reads HF_TOKEN from the environment (never printed). Each repo is created public if missing, uploaded with
upload_folder, then checked: every local file is on the Hub with the same size, and every parquet config has the
row count recorded in manifest.json.

    .venv/bin/python scripts/publish_hf_release.py [--only results] [--collection]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_hf_release import (ARTICLE, DEFAULT_OUT, NAMESPACE, REPO_MODELS, REPO_PROMPTS, REPO_RESULTS,  # noqa: E402
                              REPO_TRAINING)

REPOS = {"prompts": (REPO_PROMPTS, "dataset"), "training": (REPO_TRAINING, "dataset"),
         "results": (REPO_RESULTS, "dataset"), "models": (REPO_MODELS, "model")}


def publish(api: HfApi, folder: Path, repo_id: str, repo_type: str) -> None:
    api.create_repo(repo_id, repo_type=repo_type, private=False, exist_ok=True)
    api.upload_folder(repo_id=repo_id, repo_type=repo_type, folder_path=str(folder),
                      commit_message="Release for the article: " + ARTICLE, delete_patterns=["data/*.parquet"])
    info = api.repo_info(repo_id, repo_type=repo_type, files_metadata=True)
    remote = {s.rfilename: s.size for s in info.siblings}
    local = {str(p.relative_to(folder)): p.stat().st_size for p in folder.rglob("*") if p.is_file()}
    bad = [f for f, n in local.items() if remote.get(f) != n]
    if bad:
        raise SystemExit(f"{repo_id}: size mismatch or missing on the Hub: {bad[:5]}")
    if info.private:
        raise SystemExit(f"{repo_id}: repo is private, expected public")
    man = json.loads((folder / "manifest.json").read_text())
    for f, m in man.get("files", {}).items():
        n = len(pd.read_parquet(hf_hub_download(repo_id, f, repo_type=repo_type)))
        if n != m["rows"]:
            raise SystemExit(f"{repo_id}/{f}: {n} rows on the Hub, manifest says {m['rows']}")
    print(f"OK {repo_type:7s} {repo_id}: {len(local)} files, public, "
          f"{len(man.get('files', {}))} parquet configs re-read from the Hub with matching row counts")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--only", choices=sorted(REPOS), action="append")
    ap.add_argument("--collection", action="store_true", help="also add the four repos to one Hub collection")
    a = ap.parse_args()
    api = HfApi()
    for key in a.only or REPOS:
        name, rtype = REPOS[key]
        publish(api, a.out / name, f"{NAMESPACE}/{name}", rtype)
    if a.collection:
        col = api.create_collection(title="Tiny Aya multilingual safety fine-tuning", namespace=NAMESPACE,
                                    description=ARTICLE[:150], private=False, exists_ok=True)
        for name, rtype in REPOS.values():
            api.add_collection_item(col.slug, item_id=f"{NAMESPACE}/{name}", item_type=rtype, exists_ok=True)
        print("collection:", f"https://huggingface.co/collections/{col.slug}")


if __name__ == "__main__":
    main()
