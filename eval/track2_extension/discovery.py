"""Seed-safe, language-stratified candidate selection for CircuitKIT discovery."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Iterable


def select_discovery_pairs_by_language(
    pairs: Iterable[dict[str, Any]],
    *,
    seed: int,
    sample_size_per_language: int = 16,
    minimum_per_language: int = 8,
) -> dict[str, list[dict[str, Any]]]:
    """Select reproducible, separately analyzed EN/JA discovery cohorts.

    Only adjudicated analysis rows with two distinct reviewers are accepted.
    Sampling occurs independently within language and never duplicates a source
    seed within a language. A cross-language source seed may appear once in each
    language cohort; callers should keep all discovery seeds out of evaluation.
    """
    if sample_size_per_language < minimum_per_language or minimum_per_language < 1:
        raise ValueError("sample_size_per_language must be >= minimum_per_language >= 1")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seeds_by_language: dict[str, set[str]] = defaultdict(set)
    for pair in pairs:
        if pair.get("split") != "analysis":
            raise ValueError("Circuit discovery accepts analysis-split examples only")
        if pair.get("review_status") != "human_approved" or not pair.get("adjudicator_id"):
            raise ValueError("Circuit discovery requires human approval and adjudication")
        reviewers = pair.get("reviewer_ids")
        if not isinstance(reviewers, list) or len(
            {str(value).strip().casefold() for value in reviewers if str(value).strip()}
        ) < 2:
            raise ValueError("Each discovery pair needs two distinct human reviewers")
        language = pair.get("language")
        if language not in {"en", "ja"}:
            raise ValueError(f"Unsupported or missing discovery language: {language!r}")
        source_seed = pair.get("seed_id")
        if not isinstance(source_seed, str) or not source_seed.strip():
            raise ValueError("Every discovery pair needs a nonempty source seed ID")
        if source_seed in seeds_by_language[language]:
            raise ValueError(f"Duplicate discovery seed in {language}: {source_seed}")
        seeds_by_language[language].add(source_seed)
        grouped[language].append(pair)

    out: dict[str, list[dict[str, Any]]] = {}
    for language in ("en", "ja"):
        candidates = sorted(grouped.get(language, []), key=lambda row: row["seed_id"])
        if len(candidates) < minimum_per_language:
            raise ValueError(
                f"Need at least {minimum_per_language} adjudicated analysis seeds in {language}; "
                f"found {len(candidates)}"
            )
        rng = random.Random(f"{seed}:{language}")
        selected = rng.sample(candidates, min(sample_size_per_language, len(candidates)))
        out[language] = sorted(selected, key=lambda row: row["seed_id"])
    return out
