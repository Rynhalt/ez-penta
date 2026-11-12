"""Strict source-seed leakage checks for the CircuitKIT fresh-evaluation split."""

from __future__ import annotations

from typing import Any, Iterable


def validate_fresh_eval_seed_audit(
    eval_pairs: Iterable[dict[str, Any]],
    *,
    discovery_seed_ids: Iterable[str],
    phase1_seed_ids: Iterable[str],
    prior_registry: dict[str, Any],
    minimum_per_language: int = 8,
    maximum_rows: int = 32,
) -> dict[str, Any]:
    """Reject evaluation unless all known training/tuning exposure is audited.

    The registry is an explicit human-maintained inventory of historical SFT,
    preference generation, probes, tuning, and prior evaluation seeds. A
    missing or incomplete registry is a hard stop; row-level `previously_exposed`
    annotations alone cannot establish a negative exposure claim.
    """
    if prior_registry.get("complete") is not True:
        raise ValueError("Prior seed registry must be explicitly marked complete")
    if not isinstance(prior_registry.get("inventory_sources"), list) or not prior_registry["inventory_sources"]:
        raise ValueError("Prior seed registry must identify the inventories that were checked")
    prior_ids = prior_registry.get("seed_ids")
    if not isinstance(prior_ids, list) or not all(isinstance(seed, str) and seed.strip() for seed in prior_ids):
        raise ValueError("Prior seed registry must contain a list of nonempty seed IDs")

    rows = list(eval_pairs)
    if not rows:
        raise ValueError("Fresh evaluation export is empty")
    if len(rows) > maximum_rows:
        raise ValueError(f"Fresh evaluation has {len(rows)} rows; predeclared maximum is {maximum_rows}")
    by_language: dict[str, set[str]] = {"en": set(), "ja": set()}
    seen: set[tuple[str, str]] = set()
    for pair in rows:
        lang, seed = pair.get("language"), pair.get("seed_id")
        if lang not in by_language or not isinstance(seed, str) or not seed.strip():
            raise ValueError("Every fresh evaluation row needs language en/ja and a nonempty seed_id")
        key = (lang, seed)
        if key in seen:
            raise ValueError(f"Duplicate fresh evaluation seed within language: {key}")
        seen.add(key)
        by_language[lang].add(seed)
    for language, seeds in by_language.items():
        if len(seeds) < minimum_per_language:
            raise ValueError(f"Need at least {minimum_per_language} fresh {language} seeds; found {len(seeds)}")

    excluded = set(discovery_seed_ids) | set(phase1_seed_ids) | set(prior_ids)
    eval_seeds = {seed for _, seed in seen}
    overlap = sorted(eval_seeds & excluded)
    if overlap:
        raise ValueError(f"Fresh evaluation overlaps discovery, Phase 1, or prior training/tuning seeds: {overlap}")
    return {
        "n_rows": len(rows),
        "n_seeds": len(eval_seeds),
        "n_seeds_by_language": {lang: len(seeds) for lang, seeds in by_language.items()},
        "prior_registry_sources": list(prior_registry["inventory_sources"]),
    }
