"""Custom EAP dataset adapter for reviewed refusal/compliance contrasts.

Aligned discovery requires the version-pinned CircuitKIT core patch documented
in ``docs/lexsi/patches``. Data validation and non-CircuitKIT unit tests remain
runnable without CircuitKIT installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Iterable

import torch
from torch.utils.data import DataLoader, Dataset

from .circuitkit_metric import refusal_compliance_margin


def _render(tokenizer, prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def _encode(tokenizer, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)["input_ids"]
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    return [int(token) for token in encoded]


def _validate_pair(pair: dict[str, Any], *, allow_synthetic_smoke: bool) -> None:
    synthetic = pair.get("data_mode") == "synthetic_smoke"
    if synthetic and not allow_synthetic_smoke:
        raise ValueError("Synthetic rows require explicit allow_synthetic_smoke=True")
    if synthetic and ("review_status" in pair or "reviewer_ids" in pair):
        raise ValueError("Synthetic smoke rows must not carry human-approval metadata")
    if not synthetic:
        if pair.get("review_status") != "human_approved":
            raise ValueError("CircuitKIT inputs must be human-approved; provisional rows are forbidden")
        reviewers = pair.get("reviewer_ids")
        if not isinstance(reviewers, list) or len({str(x).strip().casefold() for x in reviewers if str(x).strip()}) < 2:
            raise ValueError("Each pair needs at least two independent human reviewers")
    for field in ("seed_id", "clean_prompt", "corrupt_prompt", "clean_answer", "corrupted_answer"):
        if not isinstance(pair.get(field), str) or not pair[field].strip():
            raise ValueError(f"Approved pair is missing nonempty field {field!r}")
    if pair["clean_prompt"] == pair["corrupt_prompt"]:
        raise ValueError("Clean and corrupt prompts must differ")
    if pair["clean_answer"] == pair["corrupted_answer"]:
        raise ValueError("Refusal and compliance continuations must differ")


def prepare_eap_rows(
    pairs: Iterable[dict[str, Any]],
    tokenizer,
    *,
    allow_synthetic_smoke: bool = False,
    include_position_maps: bool = False,
) -> list[tuple]:
    """Make two balanced EAP examples per approved pair.

    Both prompts in each EAP row receive the same validated target continuation.
    The target IDs must be identical under both prompt contexts; otherwise the
    example cannot share one label vector and is rejected for explicit review.
    """
    prepared = []
    seen_seeds: set[str] = set()
    for pair in pairs:
        _validate_pair(pair, allow_synthetic_smoke=allow_synthetic_smoke)
        seed = pair["seed_id"]
        if seed in seen_seeds:
            raise ValueError(f"Duplicate seed in this split: {seed}")
        seen_seeds.add(seed)
        clean = _render(tokenizer, pair["clean_prompt"])
        corrupt = _render(tokenizer, pair["corrupt_prompt"])
        for answer, sign, label in (
            (pair["clean_answer"], 1, "refusal"),
            (pair["corrupted_answer"], -1, "compliance"),
        ):
            clean_ids = _encode(tokenizer, clean)
            corrupt_ids = _encode(tokenizer, corrupt)
            clean_full = _encode(tokenizer, clean + answer)
            corrupt_full = _encode(tokenizer, corrupt + answer)
            if clean_full[: len(clean_ids)] != clean_ids or corrupt_full[: len(corrupt_ids)] != corrupt_ids:
                raise ValueError(f"Tokenizer merges across assistant boundary for seed {seed}, {label}")
            clean_target = clean_full[len(clean_ids) :]
            corrupt_target = corrupt_full[len(corrupt_ids) :]
            if not clean_target or clean_target != corrupt_target:
                raise ValueError(
                    f"Continuation token IDs differ across clean/corrupt prompts for seed {seed}, {label}; "
                    "explicit token-alignment review is required"
                )
            labels = [len(clean_target), sign, *clean_target]
            if include_position_maps:
                prefix = 0
                while prefix < min(len(clean_ids), len(corrupt_ids)) and clean_ids[prefix] == corrupt_ids[prefix]:
                    prefix += 1
                if prefix == 0:
                    raise ValueError(f"No exact shared token prefix for seed {seed}, {label}")
                clean_positions = [*range(prefix), *range(len(clean_ids), len(clean_full))]
                corrupt_positions = [*range(prefix), *range(len(corrupt_ids), len(corrupt_full))]
                prepared.append(
                    (clean + answer, corrupt + answer, labels, (clean_positions, corrupt_positions))
                )
            else:
                prepared.append((clean + answer, corrupt + answer, labels))

    if not prepared:
        raise ValueError("No approved pairs were supplied")
    max_target = max(row[2][0] for row in prepared)
    padded = []
    for row in prepared:
        clean, corrupt, labels = row[:3]
        item = (clean, corrupt, labels + [-100] * (max_target - labels[0]))
        padded.append((*item, row[3]) if include_position_maps else item)
    return padded


class _ReviewedContrastDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


@dataclass
class RefusalComplianceTaskSpec:
    """CircuitKIT TaskSpec for the signed full-continuation likelihood margin."""

    pairs: list[dict[str, Any]]
    name: str = "lexsi_refusal_compliance"
    allow_synthetic_smoke: bool = False

    def validate_discovery_config(self, discovery_cfg):
        if str(discovery_cfg.get("algorithm", "")).lower() not in {"eap", "eap-ig"}:
            raise ValueError("RefusalComplianceTaskSpec supports EAP and EAP-IG only")
        if discovery_cfg.get("level", "node") not in {"node", "neuron"}:
            raise ValueError("Discovery level must be 'node' or 'neuron'")

    def build_dataloader(self, model, discovery_cfg, device):
        del device
        if model is None:
            raise ValueError("A loaded CircuitKIT model is required")
        try:
            from circuitkit.backends.eap.eap_utils import collate_EAP_with_position_maps
        except ImportError as exc:
            raise RuntimeError(
                "Apply the pinned CircuitKIT EAP position-map patch from "
                "docs/lexsi/patches before building a discovery dataloader."
            ) from exc

        rows = prepare_eap_rows(
            self.pairs,
            model.tokenizer,
            allow_synthetic_smoke=self.allow_synthetic_smoke,
            include_position_maps=True,
        )
        batch_size = int(discovery_cfg.get("batch_size", 2))
        if batch_size < 2 or batch_size % 2:
            raise ValueError("batch_size must be even and at least 2 to keep refusal/compliance balanced")
        loader = DataLoader(
            _ReviewedContrastDataset(rows),
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=partial(
                collate_EAP_with_position_maps,
                tokenizer=model.tokenizer,
                padding_side="left",
            ),
        )
        loader.pair_padding_side = "left"
        loader.templated = True
        return loader

    @staticmethod
    def metric_fn():
        return refusal_compliance_margin

    def artifact_metadata(self, discovery_cfg):
        return {
            "task_name": self.name,
            "metric": "mean_refusal_logp_minus_mean_compliance_logp",
            "metric_normalization": "per continuation token",
            "n_reviewed_pairs": len(self.pairs),
            "algorithm": discovery_cfg.get("algorithm"),
            "level": discovery_cfg.get("level"),
        }
