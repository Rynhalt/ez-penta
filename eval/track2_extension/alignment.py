"""Conservative token-position maps for unequal-length paired prompts.

This utility is not wired into CircuitKIT. It only identifies positions that
are exactly token-identical: the shared prompt prefix and a validated common
continuation suffix. The divergent prompt span is explicitly left unmapped.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass(frozen=True)
class PositionAlignment:
    """A pair of equal-length activation positions, with region provenance."""

    clean_positions: tuple[int, ...]
    corrupt_positions: tuple[int, ...]
    regions: tuple[str, ...]
    shared_prefix_tokens: int
    clean_prompt_tokens: int
    corrupt_prompt_tokens: int
    continuation_tokens: int

    @property
    def aligned_count(self) -> int:
        return len(self.clean_positions)

    @property
    def divergent_clean_span(self) -> tuple[int, int]:
        return (self.shared_prefix_tokens, self.clean_prompt_tokens)

    @property
    def divergent_corrupt_span(self) -> tuple[int, int]:
        return (self.shared_prefix_tokens, self.corrupt_prompt_tokens)

    @property
    def clean_unmapped_tokens(self) -> int:
        return self.clean_prompt_tokens - sum(r == "shared_prefix" for r in self.regions)

    @property
    def corrupt_unmapped_tokens(self) -> int:
        return self.corrupt_prompt_tokens - sum(r == "shared_prefix" for r in self.regions)

    def padded_positions(
        self,
        *,
        clean_pad_left: int,
        corrupt_pad_left: int,
        clean_sequence_length: int,
        corrupt_sequence_length: int,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Translate raw token indices into separately padded batch coordinates.

        Offsets are explicit because CircuitKIT can left-pad each side before
        cross-aligning them. This function validates that every mapped token
        remains in-bounds; it does not define or perform activation patching.
        """
        if min(clean_pad_left, corrupt_pad_left) < 0:
            raise ValueError("Left-padding offsets cannot be negative")
        clean = tuple(i + clean_pad_left for i in self.clean_positions)
        corrupt = tuple(i + corrupt_pad_left for i in self.corrupt_positions)
        if any(i >= clean_sequence_length for i in clean):
            raise ValueError("Mapped clean position exceeds padded sequence length")
        if any(i >= corrupt_sequence_length for i in corrupt):
            raise ValueError("Mapped corrupt position exceeds padded sequence length")
        return clean, corrupt


def _longest_common_prefix(a: Sequence[int], b: Sequence[int]) -> int:
    i = 0
    while i < min(len(a), len(b)) and a[i] == b[i]:
        i += 1
    return i


def exact_prefix_and_continuation_alignment(
    clean_prompt_ids: Sequence[int],
    corrupt_prompt_ids: Sequence[int],
    clean_continuation_ids: Sequence[int],
    corrupt_continuation_ids: Sequence[int],
) -> PositionAlignment:
    """Map exact common prefix and continuation indices; leave framing unmapped.

    This map is suitable as an input contract for future per-row EAP hooks.
    It deliberately does not infer correspondences inside the changed prompt
    suffix. Callers must reject empty prompts/continuations and verify that the
    continuation tokenization is identical on both sides before using it.
    """
    if not clean_prompt_ids or not corrupt_prompt_ids:
        raise ValueError("Both prompts must contain at least one token")
    if not clean_continuation_ids or not corrupt_continuation_ids:
        raise ValueError("Both continuations must contain at least one token")
    if tuple(clean_continuation_ids) != tuple(corrupt_continuation_ids):
        raise ValueError("Continuation token IDs must match exactly to align answer positions")

    prefix = _longest_common_prefix(clean_prompt_ids, corrupt_prompt_ids)
    if prefix == 0:
        raise ValueError("No exact shared prompt prefix; no conservative map can be built")

    clean_pos = list(range(prefix))
    corrupt_pos = list(range(prefix))
    regions = ["shared_prefix"] * prefix
    clean_answer_start = len(clean_prompt_ids)
    corrupt_answer_start = len(corrupt_prompt_ids)
    clean_pos.extend(range(clean_answer_start, clean_answer_start + len(clean_continuation_ids)))
    corrupt_pos.extend(range(corrupt_answer_start, corrupt_answer_start + len(corrupt_continuation_ids)))
    regions.extend(["continuation"] * len(clean_continuation_ids))

    return PositionAlignment(
        clean_positions=tuple(clean_pos),
        corrupt_positions=tuple(corrupt_pos),
        regions=tuple(regions),
        shared_prefix_tokens=prefix,
        clean_prompt_tokens=len(clean_prompt_ids),
        corrupt_prompt_tokens=len(corrupt_prompt_ids),
        continuation_tokens=len(clean_continuation_ids),
    )


def align_activation_batch(
    corrupt_activations: torch.Tensor,
    maps: Sequence[tuple[Sequence[int], Sequence[int]]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather mapped corrupt activations into clean coordinates.

    Args:
        corrupt_activations: ``[batch, position, ...]`` activation tensor.
        maps: One ``(clean_positions, corrupt_positions)`` pair per batch row,
            already translated into padded tensor coordinates.

    Returns:
        ``(aligned, valid_mask)`` where ``aligned`` has the same shape as the
        input, and ``valid_mask`` is boolean ``[batch, position]``. Positions
        outside each explicit map remain zero and invalid; consumers must mask
        them before attribution so an unmapped position cannot look like a
        meaningful zero activation difference.
    """
    if corrupt_activations.ndim < 3:
        raise ValueError("Activations must have batch, position, and feature dimensions")
    if len(maps) != corrupt_activations.shape[0]:
        raise ValueError("One position map is required for each batch row")

    aligned = torch.zeros_like(corrupt_activations)
    valid = torch.zeros(
        corrupt_activations.shape[:2], dtype=torch.bool, device=corrupt_activations.device
    )
    seq_len = corrupt_activations.shape[1]
    for row, (clean_pos, corrupt_pos) in enumerate(maps):
        if len(clean_pos) != len(corrupt_pos):
            raise ValueError("Clean and corrupt position maps must have equal lengths")
        clean_idx = torch.as_tensor(clean_pos, dtype=torch.long, device=corrupt_activations.device)
        corrupt_idx = torch.as_tensor(corrupt_pos, dtype=torch.long, device=corrupt_activations.device)
        if clean_idx.numel() == 0:
            raise ValueError("Position maps cannot be empty")
        if torch.any(clean_idx < 0) or torch.any(clean_idx >= seq_len):
            raise ValueError("Clean map index exceeds activation sequence bounds")
        if torch.any(corrupt_idx < 0) or torch.any(corrupt_idx >= seq_len):
            raise ValueError("Corrupt map index exceeds activation sequence bounds")
        if torch.unique(clean_idx).numel() != clean_idx.numel() or torch.unique(corrupt_idx).numel() != corrupt_idx.numel():
            raise ValueError("Position maps must be one-to-one")
        aligned[row, clean_idx] = corrupt_activations[row, corrupt_idx]
        valid[row, clean_idx] = True
    return aligned, valid


def mask_unmapped_gradients(gradients: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """Zero attribution gradients outside the explicitly aligned positions."""
    if gradients.ndim < 2 or valid_mask.ndim != 2:
        raise ValueError("Expected gradients [batch, position, ...] and mask [batch, position]")
    if gradients.shape[:2] != valid_mask.shape:
        raise ValueError("Gradient and alignment mask batch/position shapes must match")
    mask = valid_mask.to(device=gradients.device, dtype=gradients.dtype)
    while mask.ndim < gradients.ndim:
        mask = mask.unsqueeze(-1)
    return gradients * mask
