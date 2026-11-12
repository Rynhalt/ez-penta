#!/usr/bin/env python3
"""Build an exploratory same-language roleplay screen beyond guard disagreements.

The candidate selector is a triage rule only: exact within-seed control/rp
matching, complete eligible harmful prompts, same-language outputs, and a
changed refusal heuristic despite matching recovered binary guard labels.
The heuristic and guard labels are not ground truth. Text-bearing outputs are
private and must not be committed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.lexsi.recover_guard_response_labels import observations


MATCH_FIELDS = (
    "seed_id", "split", "category", "culture", "kind", "prompt_language",
    "output_language", "request",
)
COPY_FIELDS = (
    "prompt_id", "prompt", "request", "answer", "generation_status",
    "scoring_status", "safety", "prompt_safety", "guard_response_corrected",
    "identical_twin_flag", "analysis_eligible", "refused",
)


def stable_id(control_prompt_id: str, rp_prompt_id: str) -> str:
    payload = json.dumps([control_prompt_id, rp_prompt_id], ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_pool(rows: dict[str, dict[str, Any]], *, include_all_matched: bool = False) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = {}
    for row in rows.values():
        if (
            row.get("split") not in {"analysis", "tune"}
            or row.get("kind") != "harmful"
            or row.get("prompt_language") not in {"en", "ja"}
            or row.get("prompt_language") != row.get("output_language")
            or row.get("generation_status") != "completed"
            or not row.get("analysis_eligible")
            or row.get("guard_response_corrected") not in {"Safe", "Unsafe"}
            or not str(row.get("answer") or "").strip()
        ):
            continue
        framing = row.get("framing")
        if framing not in {"control", "rp"}:
            continue
        key = tuple(row.get(field) for field in MATCH_FIELDS)
        levels = buckets.setdefault(key, {})
        if framing in levels:
            raise ValueError(f"Duplicate {framing} row for a matched condition")
        levels[framing] = row

    result = []
    for key, levels in sorted(buckets.items(), key=lambda item: str(item[0])):
        if set(levels) != {"control", "rp"}:
            continue
        control, roleplay = levels["control"], levels["rp"]
        guard_disagrees = control["guard_response_corrected"] != roleplay["guard_response_corrected"]
        heuristic_disagrees = bool(control.get("refused")) != bool(roleplay.get("refused"))
        # The default additive cohort isolates heuristic disagreements the old
        # guard-disagreement screen missed. --all-matched is a broader triage
        # inventory for auditing what both automated signals may miss.
        if not include_all_matched and (guard_disagrees or not heuristic_disagrees):
            continue
        row: dict[str, Any] = {
            "row_id": stable_id(control["prompt_id"], roleplay["prompt_id"]),
            "cohort": "roleplay_all_matched_triage" if include_all_matched else "roleplay_heuristic_mismatch_additional",
            **{field: control.get(field) for field in MATCH_FIELDS},
            "framing": "control/rp",
            "language": control["prompt_language"],
        }
        for side, source in (("control", control), ("rp", roleplay)):
            row.update({f"{field}_{side}": source.get(field) for field in COPY_FIELDS})
        row["triage_reason"] = (
            f"guard_disagrees={guard_disagrees}; refusal_heuristic_disagrees={heuristic_disagrees}"
            if include_all_matched else "refusal_heuristic_disagrees; recovered_binary_guard_agrees"
        )
        result.append(row)

    ids = [row["row_id"] for row in result]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate stable candidate IDs")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part1", type=Path, required=True)
    parser.add_argument("--part2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--all-matched", action="store_true",
                        help="Write all eligible analysis/tune pairs, including same-guard/same-heuristic pairs, for broader triage.")
    args = parser.parse_args()

    rows, _settings = observations((args.part1, args.part2))
    candidates = build_pool(rows, include_all_matched=args.all_matched)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        for row in candidates:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    splits: dict[str, set[str]] = {}
    languages: dict[str, int] = {}
    for row in candidates:
        splits.setdefault(row["split"], set()).add(row["seed_id"])
        languages[row["language"]] = languages.get(row["language"], 0) + 1
    manifest = {
        "schema": "lexsi.roleplay_heuristic_mismatch_pool/1",
        "selector_version": "exact-control-rp-all-matched-v1" if args.all_matched else "exact-control-rp-match-heuristic-mismatch-v1",
        "source_shards": [
            {"basename": path.name, "bytes": path.stat().st_size,
             "sha256": file_sha256(path)}
            for path in (args.part1, args.part2)
        ],
        "candidate_count": len(candidates),
        "unique_seeds_by_split": {split: len(seeds) for split, seeds in sorted(splits.items())},
        "candidate_rows_by_language": dict(sorted(languages.items())),
        "selection_is_ground_truth": False,
        "annotation_status": "unreviewed",
        "exclusions": ["heldout", "mixed prompt language", "incomplete or ineligible rows",
                       "non-binary recovered guard labels", "guard-discordant pairs already screened"],
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: manifest[k] for k in (
        "candidate_count", "unique_seeds_by_split", "candidate_rows_by_language",
        "selection_is_ground_truth", "annotation_status",
    )}, indent=2))


if __name__ == "__main__":
    main()
