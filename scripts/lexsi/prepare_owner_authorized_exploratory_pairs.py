#!/usr/bin/env python3
"""Prepare a private, owner-authorized exploratory CircuitKIT input.

This does not claim that an LLM annotation is a human annotation. It records the
project owner's authorization to use the existing LLM-screened examples for an
exploratory model-score diagnostic, with the original review provenance intact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


REQUIRED = (
    "row_id", "seed_id", "split", "language", "clean_prompt", "corrupt_prompt",
    "clean_answer", "corrupted_answer", "prompt_id_control", "prompt_id_rp",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(source_path: Path, output_path: Path, manifest_path: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in source_path.read_text().splitlines() if line.strip()]
    for row in rows:
        missing = [key for key in REQUIRED if not isinstance(row.get(key), str) or not row[key].strip()]
        if missing:
            raise ValueError(f"{row.get('row_id', '<missing row_id>')} missing: {missing}")
        if row["split"] not in {"analysis", "tune"}:
            raise ValueError("Only analysis/tune examples may enter the exploratory contrast set")
        if row["language"] not in {"en", "ja"}:
            raise ValueError("Only monolingual English/Japanese contrasts are eligible")
        if row["prompt_id_control"] == row["prompt_id_rp"]:
            raise ValueError("Control and roleplay prompt IDs must differ")
        if row["clean_prompt"] == row["corrupt_prompt"] or row["clean_answer"] == row["corrupted_answer"]:
            raise ValueError("Paired prompts and continuations must differ")

    if len({row["row_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate row_id")
    if len({row["seed_id"] for row in rows}) != len(rows):
        raise ValueError("Seed-level leakage: each seed must appear once")
    if Counter(row["split"] for row in rows) != Counter({"analysis": 6, "tune": 1}):
        raise ValueError("Expected the frozen seven-row cohort: six analysis seeds and one tune seed")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    exported = []
    for row in rows:
        item = {key: row[key] for key in REQUIRED}
        item.update({
            "approval_status": "owner_authorized_for_exploratory_execution",
            "approval_basis": "User explicitly authorized use of LLM-provisional reviews for this project.",
            "annotation_provenance": row.get("annotation_provenance", "Codex-assisted single-pass; provisional"),
            "review_status": row.get("review_status", "provisional"),
            "reviewer_type": row.get("reviewer_type", "codex_assisted"),
            "validation_level": "exploratory model-score diagnostic; not independent human annotation",
        })
        exported.append(item)
    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in exported)
    output_path.write_text(payload)
    manifest = {
        "schema": "lexsi.owner_authorized_exploratory_pairs/1",
        "rows": len(exported),
        "unique_seeds": len({row["seed_id"] for row in exported}),
        "by_split": dict(Counter(row["split"] for row in exported)),
        "by_language": dict(Counter(row["language"] for row in exported)),
        "source_path": str(source_path),
        "source_sha256": sha256(source_path),
        "output_path": str(output_path),
        "output_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "approval_status": "owner_authorized_for_exploratory_execution",
        "approval_basis": "User explicitly authorized using LLM-provisional reviews for this project.",
        "independent_human_annotation": False,
        "fresh_evaluation_seeds": 0,
        "scientific_use": "diagnostic only; cannot establish behavioral metric validity or causal generalization",
        "official_human_approved_export_modified": False,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.output, args.manifest), indent=2))


if __name__ == "__main__":
    main()
