"""Differentiable full-continuation margin metric for CircuitKIT EAP.

Each EAP row must contain a prompt followed by one target continuation. The
paired clean/corrupt strings use the same target continuation, while the
dataset contains one positive (refusal) and one negative (compliance) row per
prompt pair. Labels are ``[target_length, sign, token_0, ..., token_n]``.
Continuation positions are located from CircuitKIT's ``input_lengths`` when
available, supporting both left- and right-padded batches. Without lengths the
metric retains right-edge behavior for standalone reference tests.

The metric returns mean refusal log-probability minus mean compliance
log-probability. A batch must contain a balanced number of each sign; use an
even batch size and keep each two-row contrast group adjacent.
"""

from __future__ import annotations

import torch


def refusal_compliance_margin(logits, clean_logits, input_lengths, labels):
    """Return a differentiable scalar full-sequence likelihood margin.

    Args:
        logits: Float or low-precision logits ``[batch, seq, vocab]`` from
            CircuitKIT's EAP model forward pass.
        clean_logits: Accepted for the CircuitKIT metric API and unused here.
        input_lengths: Real token counts for each clean sequence, if provided.
            CircuitKIT sets these to the batch sequence length for left padding;
            for right padding they identify each row's actual sequence end.
        labels: Long tensor ``[batch, 2 + max_target_len]`` containing target
            lengths, signs (+1 refusal / -1 compliance), and padded token IDs.

    Returns:
        Scalar tensor ``mean(refusal mean-token logP) -
        mean(compliance mean-token logP)`` with gradients to ``logits``.
    """
    del clean_logits
    if logits.ndim != 3 or labels.ndim != 2:
        raise ValueError("Expected logits [batch, seq, vocab] and labels [batch, fields]")
    batch, seq_len, _ = logits.shape
    if labels.shape[0] != batch or labels.shape[1] < 3:
        raise ValueError("Labels must align with logits and include length, sign, and target")
    labels = labels.to(device=logits.device, dtype=torch.long)
    lengths, signs, target_ids = labels[:, 0], labels[:, 1], labels[:, 2:]
    if torch.any(lengths <= 0) or torch.any(lengths > target_ids.shape[1]):
        raise ValueError("Every target length must be positive and fit the padded target span")
    if input_lengths is None:
        sequence_ends = torch.full((batch,), seq_len, device=logits.device, dtype=torch.long)
    else:
        sequence_ends = torch.as_tensor(input_lengths, device=logits.device, dtype=torch.long)
        if sequence_ends.shape != (batch,):
            raise ValueError("input_lengths must contain one sequence length per row")
        if torch.any(sequence_ends <= 0) or torch.any(sequence_ends > seq_len):
            raise ValueError("input_lengths must be positive and no greater than logits sequence length")
    if torch.any(lengths >= sequence_ends):
        raise ValueError("Each sequence must contain at least one prompt token before its target")
    if not torch.all((signs == 1) | (signs == -1)):
        raise ValueError("Target signs must be +1 for refusal or -1 for compliance")
    if int((signs == 1).sum()) != int((signs == -1).sum()):
        raise ValueError("Each metric batch must balance refusal and compliance rows")

    # Target token at absolute position p is predicted by logits at p - 1.
    row = torch.arange(batch, device=logits.device)
    per_row = logits.new_zeros((batch,), dtype=torch.float32)
    for offset in range(target_ids.shape[1]):
        active = offset < lengths
        token_pos = sequence_ends - lengths + offset
        pred_pos = token_pos - 1
        token_id = target_ids[:, offset].clamp(min=0)
        # Normalize only the positions being scored. This avoids allocating a
        # second [batch, sequence, vocabulary] tensor for Tiny Aya's large vocab.
        selected_logits = logits[row, pred_pos].float()
        token_lp = selected_logits.gather(1, token_id[:, None]).squeeze(1)
        token_lp = token_lp - torch.logsumexp(selected_logits, dim=-1)
        per_row = per_row + torch.where(active, token_lp, torch.zeros_like(token_lp))
    per_row = per_row / lengths.float()

    # Balanced signs mean 2 * E[sign * score] equals mean(+) - mean(-).
    return 2.0 * (per_row * signs.float()).mean()
