# CircuitKIT explicit EAP position maps

This patch targets CircuitKIT commit `e3501bc72b349963db76df324f5dbd41ced2a02f` and is an experimental compatibility patch for our exact-prefix/validated-continuation maps.

Apply to a clean checkout of that commit:

```bash
git checkout e3501bc72b349963db76df324f5dbd41ced2a02f
git apply /path/to/ez-penta/docs/lexsi/patches/circuitkit_eap_position_maps_e3501bc.patch
```

It adds a `collate_EAP_with_position_maps()` collator, accepts `(clean, corrupt, label, position_maps)` batches in EAP and EAP-IG, gathers corrupted activations into clean coordinates, masks gradients outside mapped coordinates, and preserves clean input embeddings at unmapped sites during EAP-IG interpolation. Maps are explicit; the patch does not infer correspondence within divergent prompt framing. It fails explicitly for the separate-activation normalization path rather than silently applying an invalid map.

Run the patch-specific tests from the CircuitKIT checkout after applying the patch:

```bash
PYTHONPATH=src pytest -q /path/to/ez-penta/tests/circuitkit_patches/test_position_map_hooks.py
```

The patch tests were executed against this pinned source on Colab CPU. They cover collator offsets, exact gather behavior, one-to-one/bounds validation, unmapped gradient masking, EAP-IG endpoint preservation, legacy-hook behavior, unsupported normalization rejection, and a real random one-layer TransformerLens EAP-IG smoke including identity-map parity and a partial-map run. They do not validate Tiny Aya, human-reviewed data, GPU behavior, or the scientific interpretation of any resulting circuit. This patch has not been merged upstream and must remain version-pinned.
