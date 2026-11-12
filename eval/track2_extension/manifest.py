"""Build and validate conservative matched evaluation candidates.

Automated metadata filters nominate rows; they do not establish semantic
equivalence, harmlessness, unsafe compliance, or human approval.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any, Iterable, Mapping


MATCH_FIELDS = (
    "seed_id",
    "split",
    "kind",
    "category",
    "framing",
    "prompt_language",
    "mix_direction",
    "output_language",
)


def _flag(value: Any) -> bool:
    return value is True or value == 1 or (
        isinstance(value, str) and value.strip().lower() in {"true", "1", "yes"}
    )


def _identifier(value: Any) -> str:
    return "" if value is None else str(value)


def _pair_id(key: tuple[Any, ...]) -> str:
    encoded = "\x1f".join(_identifier(value) for value in key).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:20]


def build_matched_candidates(
    observations: Iterable[Mapping[str, Any]],
    *,
    splits: set[str] | None = None,
    prior_exposed_seed_ids: set[str] | None = None,
    previously_exposed_splits: set[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Select exact metadata-matched US/neutral harmful prompt candidates.

    Pairs with duplicate/malformed rows or unreviewed prompt validity remain
    excluded or explicitly provisional. This helper never sets a human field
    to a positive value.
    """
    allowed_splits = splits or {"heldout"}
    prior_seeds = prior_exposed_seed_ids or set()
    exposed_splits = previously_exposed_splits or {"heldout"}
    grouped: dict[tuple[Any, ...], dict[str, list[Mapping[str, Any]]]] = defaultdict(
        lambda: {"us": [], "neutral": []}
    )
    counts: dict[str, int] = defaultdict(int)
    seen_ids: set[str] = set()

    for row in observations:
        counts["source_rows"] += 1
        prompt_id = _identifier(row.get("prompt_id"))
        seed_id = _identifier(row.get("seed_id"))
        if not prompt_id or not seed_id:
            counts["missing_id_or_seed"] += 1
            continue
        if prompt_id in seen_ids:
            counts["duplicate_prompt_id"] += 1
            continue
        seen_ids.add(prompt_id)
        if row.get("split") not in allowed_splits:
            continue
        counts["rows_in_requested_splits"] += 1
        if row.get("kind") != "harmful":
            continue
        if row.get("prompt_language") not in {"en", "ja"}:
            continue
        if not _flag(row.get("analysis_eligible")):
            counts["not_analysis_eligible"] += 1
            continue
        if row.get("generation_status") != "completed":
            counts["source_generation_not_completed"] += 1
            continue
        if row.get("scoring_status") != "completed":
            counts["source_scoring_not_completed"] += 1
            continue
        if _flag(row.get("identical_twin_flag")):
            counts["identical_twin_excluded"] += 1
            continue
        if _flag(row.get("duplicate_text_flag")):
            counts["duplicate_text_excluded"] += 1
            continue
        if _flag(row.get("expected_behavior_conflict")):
            counts["expected_behavior_conflict_excluded"] += 1
            continue
        if row.get("culture") not in {"us", "neutral"}:
            continue
        key_values = tuple(row.get(field) for field in MATCH_FIELDS)
        missing_fields = [
            field
            for field, value in zip(MATCH_FIELDS, key_values)
            if (value is None or value == "")
            and not (field == "mix_direction" and row.get("prompt_language") in {"en", "ja"})
        ]
        if missing_fields:
            counts["missing_match_metadata"] += 1
            continue
        grouped[key_values][str(row["culture"])].append(row)

    candidates: list[dict[str, Any]] = []
    for key, conditions in sorted(grouped.items(), key=lambda item: tuple(map(_identifier, item[0]))):
        us_rows, neutral_rows = conditions["us"], conditions["neutral"]
        if len(us_rows) != 1 or len(neutral_rows) != 1:
            counts["non_unique_or_unmatched_strata"] += 1
            continue
        matched_rows = (neutral_rows[0], us_rows[0])
        if any(
            not isinstance(row.get("prompt"), str) or not row["prompt"].strip()
            for row in matched_rows
        ):
            counts["missing_prompt_text"] += 1
            continue
        pair_id = _pair_id(key)
        for row in matched_rows:
            prompt = row.get("prompt")
            assert isinstance(prompt, str)
            split = str(row["split"])
            leakage_reasons = []
            if split in exposed_splits:
                leakage_reasons.append(f"split_{split}_previously_exposed")
            if str(row["seed_id"]) in prior_seeds:
                leakage_reasons.append("seed_previously_used")
            candidates.append(
                {
                    "prompt_id": str(row["prompt_id"]),
                    "pair_id": pair_id,
                    "seed_id": str(row["seed_id"]),
                    "split": split,
                    "kind": row["kind"],
                    "category": row["category"],
                    "framing": row["framing"],
                    "prompt_language": row["prompt_language"],
                    "mix_direction": row["mix_direction"],
                    "output_language": row["output_language"],
                    "culture": row["culture"],
                    "prompt": prompt,
                    "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                    "source_expected": row.get("expected"),
                    "source_safety_proxy": row.get("safety"),
                    "source_generation_status": row.get("generation_status"),
                    "source_scoring_status": row.get("scoring_status"),
                    "human_prompt_valid": None,
                    "human_semantic_equivalence": None,
                    "human_unsafe_compliance": None,
                    "review_status": "unreviewed",
                    "selection_cohort": "matched_candidate_not_validated_population_sample",
                    "leakage_flag": bool(leakage_reasons),
                    "leakage_reason": ";".join(leakage_reasons) or None,
                    "confirmatory_eligible": False,
                }
            )
        counts["matched_pair_strata"] += 1
        counts["candidate_prompt_rows"] += 2

    validate_candidate_manifest(candidates)
    counts["candidate_prompt_rows"] = len(candidates)
    counts["candidate_pairs"] = len(candidates) // 2
    counts["unique_candidate_seeds"] = len({row["seed_id"] for row in candidates})
    counts["rows_leakage_flagged"] = sum(row["leakage_flag"] for row in candidates)
    return candidates, dict(counts)


def validate_candidate_manifest(rows: list[Mapping[str, Any]]) -> None:
    """Fail closed on duplicates, malformed pairings, and synthetic approvals."""
    ids = [row.get("prompt_id") for row in rows]
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("candidate prompt IDs must be present and unique")
    pairs: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        pairs[str(row.get("pair_id"))].append(row)
        if row.get("human_prompt_valid") is True or row.get("human_unsafe_compliance") is not None:
            raise ValueError("human review fields cannot be populated by candidate generation")
        if row.get("confirmatory_eligible") is True:
            raise ValueError("automatically generated candidates cannot be confirmatory eligible")
        if hashlib.sha256(str(row.get("prompt", "")).encode("utf-8")).hexdigest() != row.get(
            "prompt_sha256"
        ):
            raise ValueError("prompt hash mismatch")
    for pair_id, pair in pairs.items():
        if len(pair) != 2 or {row.get("culture") for row in pair} != {"us", "neutral"}:
            raise ValueError(f"invalid matched pair {pair_id}")
        for field in MATCH_FIELDS:
            if len({row.get(field) for row in pair}) != 1:
                raise ValueError(f"pair {pair_id} disagrees on matching field {field}")
