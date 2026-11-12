import unittest
from types import SimpleNamespace

import torch

from eval.track2_extension.circuitkit_data import prepare_eap_rows
from eval.track2_extension.circuitkit_metric import refusal_compliance_margin
from eval.track2_extension.circuitkit_data import RefusalComplianceTaskSpec


class _CharacterTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize is False and add_generation_prompt is True
        return f"user:{messages[0]['content']} assistant:"

    def __call__(self, text, add_special_tokens):
        assert add_special_tokens is False
        return {"input_ids": [ord(char) for char in text]}


def _synthetic_pair():
    return {
        "data_mode": "synthetic_smoke",
        "seed_id": "seed-1",
        "clean_prompt": "control request",
        "corrupt_prompt": "roleplay request",
        "clean_answer": "I cannot help with that.",
        "corrupted_answer": "Here is the unsafe answer.",
    }


class ContinuationMarginTests(unittest.TestCase):
    def test_builds_balanced_rows_with_shared_suffix_targets(self):
        rows = prepare_eap_rows(
            [_synthetic_pair()], _CharacterTokenizer(), allow_synthetic_smoke=True
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual([row[2][1] for row in rows], [1, -1])
        for clean, corrupt, labels in rows:
            n = labels[0]
            self.assertEqual(clean[-n:], corrupt[-n:])
            self.assertEqual(labels[2 : 2 + n], [ord(x) for x in (
                _synthetic_pair()["clean_answer"] if labels[1] == 1
                else _synthetic_pair()["corrupted_answer"]
            )])

    def test_optional_maps_cover_exact_prefix_and_answer_only(self):
        rows = prepare_eap_rows(
            [_synthetic_pair()],
            _CharacterTokenizer(),
            allow_synthetic_smoke=True,
            include_position_maps=True,
        )
        self.assertEqual(len(rows), 2)
        for clean, corrupt, labels, (clean_map, corrupt_map) in rows:
            n_target = labels[0]
            prompt_clean_len = len(clean) - n_target
            prompt_corrupt_len = len(corrupt) - n_target
            common_prefix = 5  # exact rendered prefix "user:"
            self.assertEqual(clean_map[:common_prefix], list(range(common_prefix)))
            self.assertEqual(corrupt_map[:common_prefix], list(range(common_prefix)))
            self.assertEqual(clean_map[-n_target:], list(range(prompt_clean_len, len(clean))))
            self.assertEqual(corrupt_map[-n_target:], list(range(prompt_corrupt_len, len(corrupt))))
            self.assertEqual(len(clean_map), common_prefix + n_target)
            self.assertEqual(len(corrupt_map), common_prefix + n_target)

    def test_taskspec_emits_left_padded_maps_to_circuitkit_collator(self):
        try:
            from circuitkit.backends.eap.eap_utils import collate_EAP_with_position_maps
        except ImportError:
            self.skipTest("CircuitKIT is not installed in this local test environment")
        self.assertTrue(callable(collate_EAP_with_position_maps))
        tokenizer = _CharacterTokenizer()
        task = RefusalComplianceTaskSpec([_synthetic_pair()], allow_synthetic_smoke=True)
        dataloader = task.build_dataloader(
            SimpleNamespace(tokenizer=tokenizer),
            {"algorithm": "eap-ig", "batch_size": 2},
            "cpu",
        )
        clean, corrupt, labels, maps = next(iter(dataloader))
        self.assertEqual(len(clean), 2)
        self.assertEqual(len(corrupt), 2)
        self.assertEqual(labels.shape[0], 2)
        self.assertEqual(len(maps), 2)
        sequence_length = max(
            len(tokenizer(text, add_special_tokens=False)["input_ids"])
            for text in [*clean, *corrupt]
        )
        for row, (clean_positions, corrupt_positions) in enumerate(maps):
            n_target = int(labels[row, 0])
            self.assertEqual(clean_positions[-n_target:], tuple(range(sequence_length - n_target, sequence_length)))
            self.assertEqual(corrupt_positions[-n_target:], tuple(range(sequence_length - n_target, sequence_length)))

    def test_provisional_real_pair_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "human-approved"):
            pair = _synthetic_pair()
            pair.pop("data_mode")
            pair["review_status"] = "provisional"
            prepare_eap_rows([pair], _CharacterTokenizer())

    def test_real_pair_missing_human_approval_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "human-approved"):
            pair = _synthetic_pair()
            pair.pop("data_mode")
            pair["review_status"] = "provisional"
            prepare_eap_rows([pair], _CharacterTokenizer())

    def test_synthetic_smoke_must_be_explicitly_enabled(self):
        with self.assertRaisesRegex(ValueError, "allow_synthetic_smoke=True"):
            prepare_eap_rows([_synthetic_pair()], _CharacterTokenizer())

    def test_synthetic_rows_cannot_claim_human_review(self):
        pair = _synthetic_pair()
        pair["review_status"] = "human_approved"
        pair["reviewer_ids"] = ["placeholder-a", "placeholder-b"]
        with self.assertRaisesRegex(ValueError, "must not carry human-approval metadata"):
            prepare_eap_rows([pair], _CharacterTokenizer(), allow_synthetic_smoke=True)

    def test_matches_manual_mean_token_logprob_difference_and_backpropagates(self):
        # Two rows, one two-token refusal and one one-token compliance target.
        logits = torch.tensor(
            [[[0.2, 1.0, -0.2], [1.1, 0.0, -0.1], [0.0, 0.3, 0.9]],
             [[0.0, 0.7, 0.1], [0.9, -0.2, 0.1], [0.1, 1.0, -0.3]]],
            dtype=torch.float32,
            requires_grad=True,
        )
        # Right-aligned target IDs: refusal [1,2], compliance [1].
        labels = torch.tensor([[2, 1, 1, 2], [1, -1, 1, -100]])
        actual = refusal_compliance_margin(logits, None, None, labels)
        lp = torch.log_softmax(logits, dim=-1)
        refusal = (lp[0, 0, 1] + lp[0, 1, 2]) / 2
        compliance = lp[1, 1, 1]
        expected = refusal - compliance
        self.assertTrue(torch.allclose(actual, expected, atol=1e-7, rtol=1e-7))
        actual.backward()
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(float(logits.grad.abs().sum()), 0)

    def test_rejects_unbalanced_batch(self):
        logits = torch.zeros((2, 3, 5), requires_grad=True)
        labels = torch.tensor([[1, 1, 2], [1, 1, 3]])
        with self.assertRaisesRegex(ValueError, "balance"):
            refusal_compliance_margin(logits, None, None, labels)

    def test_uses_per_row_lengths_for_right_padded_batches(self):
        logits = torch.tensor(
            [[[0.1, 0.2, 0.3, 0.4], [1.0, 0.0, -0.2, 0.1], [0.0, 0.0, 0.0, 0.0],
              [9.0, -9.0, 0.0, 0.0], [9.0, -9.0, 0.0, 0.0]],
             [[0.0, 0.0, 0.0, 0.0], [0.1, 0.2, 0.3, 0.4], [1.1, -0.1, 0.2, 0.0],
              [0.3, 0.1, 0.7, -0.2], [9.0, -9.0, 0.0, 0.0]]],
            dtype=torch.float32,
            requires_grad=True,
        )
        labels = torch.tensor([[1, 1, 2, -100], [2, -1, 0, 2]])
        actual = refusal_compliance_margin(logits, None, torch.tensor([3, 4]), labels)
        lp = torch.log_softmax(logits, dim=-1)
        refusal = lp[0, 1, 2]  # prompt length 2, target position 2, predicted at position 1
        compliance = (lp[1, 1, 0] + lp[1, 2, 2]) / 2  # target positions 2 and 3
        self.assertTrue(torch.allclose(actual, refusal - compliance, atol=1e-7, rtol=1e-7))
        actual.backward()
        self.assertEqual(float(logits.grad[0, 3:].abs().sum()), 0.0)
        self.assertEqual(float(logits.grad[1, 3:].abs().sum()), 0.0)

    def test_rejects_invalid_input_lengths(self):
        logits = torch.zeros((2, 4, 5), requires_grad=True)
        labels = torch.tensor([[1, 1, 2], [1, -1, 3]])
        with self.assertRaisesRegex(ValueError, "one sequence length per row"):
            refusal_compliance_margin(logits, None, torch.tensor([4]), labels)

    def test_rejects_target_without_prompt_context(self):
        logits = torch.zeros((2, 1, 5), requires_grad=True)
        labels = torch.tensor([[1, 1, 2], [1, -1, 3]])
        with self.assertRaisesRegex(ValueError, "prompt token"):
            refusal_compliance_margin(logits, None, None, labels)


if __name__ == "__main__":
    unittest.main()
