import unittest

import torch

from eval.track2_extension.alignment import (
    align_activation_batch,
    exact_prefix_and_continuation_alignment,
    mask_unmapped_gradients,
)


class PositionAlignmentTests(unittest.TestCase):
    def test_maps_only_exact_prefix_and_continuation(self):
        # Prefix [1,2,3], divergent framing [4,5] vs [8,9,10], shared target [6,7].
        result = exact_prefix_and_continuation_alignment(
            [1, 2, 3, 4, 5], [1, 2, 3, 8, 9, 10], [6, 7], [6, 7]
        )
        self.assertEqual(result.clean_positions, (0, 1, 2, 5, 6))
        self.assertEqual(result.corrupt_positions, (0, 1, 2, 6, 7))
        self.assertEqual(result.regions, ("shared_prefix",) * 3 + ("continuation",) * 2)
        self.assertEqual(result.divergent_clean_span, (3, 5))
        self.assertEqual(result.divergent_corrupt_span, (3, 6))
        self.assertEqual(result.clean_unmapped_tokens, 2)
        self.assertEqual(result.corrupt_unmapped_tokens, 3)

    def test_rejects_different_continuation_tokenization(self):
        with self.assertRaisesRegex(ValueError, "must match exactly"):
            exact_prefix_and_continuation_alignment([1], [1, 2], [5], [6])

    def test_left_padding_aligns_continuation_but_offsets_shared_prefix(self):
        result = exact_prefix_and_continuation_alignment(
            [1, 2, 3, 4, 5], [1, 2, 3, 8, 9, 10], [6, 7], [6, 7]
        )
        clean, corrupt = result.padded_positions(
            clean_pad_left=4,
            corrupt_pad_left=3,
            clean_sequence_length=11,
            corrupt_sequence_length=11,
        )
        self.assertEqual(clean[:3], (4, 5, 6))
        self.assertEqual(corrupt[:3], (3, 4, 5))
        self.assertEqual(clean[3:], corrupt[3:])

    def test_padding_map_rejects_out_of_range_indices(self):
        result = exact_prefix_and_continuation_alignment([1, 2], [1, 2, 3], [4], [4])
        with self.assertRaisesRegex(ValueError, "exceeds padded sequence"):
            result.padded_positions(
                clean_pad_left=0, corrupt_pad_left=0,
                clean_sequence_length=2, corrupt_sequence_length=4
            )

    def test_batched_activation_gather_and_unmapped_gradient_mask(self):
        # Two examples have different clean/corrupt positions for their shared prefix.
        acts = torch.arange(2 * 8 * 3, dtype=torch.float32).reshape(2, 8, 3)
        maps = [((2, 3, 6, 7), (1, 2, 6, 7)), ((0, 4, 5), (0, 5, 6))]
        aligned, valid = align_activation_batch(acts, maps)
        self.assertEqual(tuple(valid.shape), (2, 8))
        self.assertEqual(valid[0].nonzero().flatten().tolist(), [2, 3, 6, 7])
        self.assertTrue(torch.equal(aligned[0, 2], acts[0, 1]))
        self.assertTrue(torch.equal(aligned[0, 3], acts[0, 2]))
        self.assertTrue(torch.equal(aligned[1, 4], acts[1, 5]))
        grads = torch.ones((2, 8, 3))
        masked = mask_unmapped_gradients(grads, valid)
        self.assertEqual(float(masked[~valid].sum()), 0.0)
        self.assertEqual(float(masked[valid].sum()), float(valid.sum() * 3))

    def test_batched_activation_map_rejects_non_bijective_or_bad_shapes(self):
        acts = torch.zeros((1, 4, 2))
        with self.assertRaisesRegex(ValueError, "One position map"):
            align_activation_batch(acts, [])
        with self.assertRaisesRegex(ValueError, "one-to-one"):
            align_activation_batch(acts, [((0, 0), (1, 2))])
        with self.assertRaisesRegex(ValueError, "equal lengths"):
            align_activation_batch(acts, [((0,), (0, 1))])

    def test_rejects_no_shared_prefix(self):
        with self.assertRaisesRegex(ValueError, "No exact shared prompt prefix"):
            exact_prefix_and_continuation_alignment([1, 2], [3, 4], [5], [5])

    def test_rejects_empty_regions(self):
        with self.assertRaisesRegex(ValueError, "Both prompts"):
            exact_prefix_and_continuation_alignment([], [1], [5], [5])
        with self.assertRaisesRegex(ValueError, "Both continuations"):
            exact_prefix_and_continuation_alignment([1], [1, 2], [], [])


if __name__ == "__main__":
    unittest.main()
