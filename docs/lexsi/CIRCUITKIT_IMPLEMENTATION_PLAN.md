# Tiny Aya Water CircuitKIT implementation plan

**Status:** planning complete; a differentiable metric prototype now passes synthetic scalar/gradient tests. Execution remains gated on human-approved same-language contrasts, metric parity on approved continuations, and Colab GPU availability.  
**Updated:** 2026-11-12  
**Owner:** LEXSI multilingual safety project

## Research question

Identify and causally test mechanisms associated with refusal versus unsafe compliance in Tiny Aya Water. The first experiment is same-language, same-intent control-versus-roleplay framing. Japanese-versus-English circuit comparison follows only after per-language experiments have valid data and causal tests.

## Current state and hard gates

- The original Hugging Face Tiny Aya Water checkpoint, tokenizer, chat template, teacher forcing, and CircuitKIT parity passed on a Colab L4. The run used FP32 and checkpoint revision `41a99216d12b3ef4881df762934186e5e4d2bbc1`; see `COLAB_DEBUG_LOG.md` and `outputs/lexsi/colab_phase1/authorized/`.
- This establishes backend parity only. No behavior-metric validation or EAP-IG discovery has run. The approved execution export is empty.
- The present same-language control/roleplay pool has five provisional English candidates: four analysis seeds and one tune seed. They are not human-approved and are not enough for a stable discovery, tuning, and evaluation design. Expand screening across independent analysis/tune seeds and obtain two independent human reviews plus adjudication.
- The 52 bilingual pairs are a separate provisional cohort. They are not same-language control/roleplay pairs and cannot fill the current experiment's data gap.
- The 91 previously exposed heldout bilingual disagreements are contaminated for confirmatory evaluation. Reserve a fresh, seed-disjoint evaluation set.

**Do not execute a real-data discovery or causal claim until:**

1. Human reviewers approve eligible rows and exact refusal/compliance continuation prefixes.
2. At least 32 independent eligible discovery seeds are available for a first stable discovery, with disjoint tune and fresh evaluation seeds. If the pool cannot support these splits, report the true smaller pilot and its limits; do not repurpose heldout or duplicate seeds.
3. The likelihood-margin metric passes on adjudicated behavior and its differentiable CircuitKIT integration is verified against the existing teacher-forced scorer.
4. Prompt/answer token alignment for clean and corrupt examples is validated at the activation positions being attributed or patched.

## Key CircuitKIT-specific design decision

Use explicit `clean` and `corrupt` examples. For a pair, clean is the control/refusal condition and corrupt is the matched roleplay/compliance condition; preserve both prompts and both validated continuations. Do not use automatic corruption for instruction-style safety examples.

CircuitKIT's generic paired CSV path expects one `correct_idx` and one `incorrect_idx` token, so it cannot represent our target. The latest upstream source also has a WinoGrande-specific variable-span suffix-likelihood metric and label layout. This is an implementation precedent, not a drop-in solution: that task scores a shared suffix, while this study must score two alternative validated continuations and retain the refusal-minus-compliance sign. The study's metric is:

`m(prompt) = mean_token_logP(validated_refusal | prompt) - mean_token_logP(validated_compliance | prompt)`

Score both fixed continuations against each prompt. The paired framing effect is `m(control_prompt) - m(roleplay_prompt)`. Prototype: `eval/track2_extension/circuitkit_metric.py` implements a differentiable EAP-style full-continuation margin using padded labels `[target_length, sign, target_token_ids...]`. Its intended adapter creates two rows per prompt pair (same continuation across clean/corrupt sides): refusal `+1`, compliance `-1`. Batches must contain both rows for every contrast; the metric rejects unbalanced batches. The upstream WinoGrande implementation shows EAP's collator preserves integer label vectors, but our custom TaskSpec and continuation-suffix dataset require model-forward and EAP discovery tests against the pinned CircuitKIT version.

Compare the scalar to `eval/dataset2_likelihood.py` on approved examples and verify gradients reach exactly the continuation-prediction positions. Confirm EAP-IG and intervention evaluation use the same signed objective. Target token IDs must come from the actual jointly encoded prompt-plus-continuation sequence, after verifying the tokenizer does not merge across the chat boundary. If CircuitKIT cannot support this metric and aligned patch sites without a fragile fork, pause and report the incompatibility before discovery.

Do not replace this objective with first-token `logit_diff`, nor with KL, just to fit the default loader. The target is continuation likelihood, and the paired score must account for tokenization, continuation lengths, prompt boundaries, and numerical stability.

## Implementation stages

### Stage 0 — Human-approved, seed-disjoint data

- Complete two independent reviews and adjudication for the same-language candidates. Confirm prompt equivalence, harmful intent, refusal versus substantive unsafe help, and verbatim continuation prefixes.
- Expand candidates from analysis and tune data if the pool remains below the minimum. Keep analysis for circuit discovery, tune for metric/configuration decisions, and a fresh seed-disjoint set for evaluation.
- Export only adjudicated rows to `outputs/lexsi/circuitkit_pairs.jsonl`; keep provisional rows in the pending file. Record the split manifest, seeds, row IDs, review version, file hash, and every exclusion reason.
- Add separate both-refuse and both-comply controls when available; never mix them into the primary positive contrast.

**Gate:** approved export validates, is nonempty, has unique seeds within each split, contains exact answer prefixes, and has enough examples for the declared pilot or full design.

### Stage 1 — Behavioral metric and custom metric adapter

- Re-run the exact approved tune rows through HF teacher forcing; score refusal and compliance continuations on both control and roleplay prompts.
- Check the predicted direction per seed, paired margins, bootstrap/sign uncertainty, answer-token counts, tokenizer boundary, and sensitivity to mean versus sum log probability. Do not pick a normalization after looking at discovery results.
- Prototype a custom differentiable CircuitKIT task/metric on harmless synthetic data, then compare its forward scalar and gradients with `eval/dataset2_likelihood.py` on a small approved example.
- Require parity at the likelihood scalar and correct gradient flow through answer-token positions before passing the custom metric into EAP-IG.
- Implement the custom EAP dataset/TaskSpec against the pinned CircuitKIT version. Keep refusal/compliance rows paired in each batch and prove left-padding, prompt-length differences, and suffix token positions are handled correctly.

**Gate:** preregistered tune-set behavior criterion passes; the custom metric passes finite-difference/gradient checks and reproduces the independent teacher-forcing scorer within a fixed tolerance.

### Stage 2 — Tiny Aya and token-alignment smoke test

- Preserve the pinned HF checkpoint and Colab dependency versions from the successful parity run. Verify CircuitKIT's Cohere2 integration after each package change.
- Validate the native chat template, clean/corrupt token sequences, and continuation boundaries. Establish an explicit token-position mapping for prompt-side control/roleplay changes and answer-side completions. Do not assume different prompts align by absolute position or that EAP-IG head expansion proves semantic alignment.
- Run CircuitKIT's data validation command on the approved export and a tiny smoke test with EAP-IG. Save normalized data and the mapping report.
- Start with `ig_steps=3` (current documented default), a low batch size, and the smallest valid approved analysis subset. Record peak allocated and reserved memory. Only increase `n_examples` and integration steps after the smoke test.

**Gate:** no dropped or degenerate pairs, all mappings auditable, no memory/architecture errors, and the discovered scalar matches the validated refusal margin.

### Stage 3 — Discovery and artifact capture

- Run EAP-IG on analysis seeds only. Store config, package lock, row/seed manifest, signed edge scores, component-aggregated scores, masks/circuit artifacts, metric values, seeds, and logs.
- Aggregate attention-head and MLP component scores while retaining direction/sign. Do not interpret visualization edges as observed causal connections: CircuitKIT's graph draws edges among selected nodes by layer; they are a layout, not measured edge attribution.
- Export interactive graph HTML and a score table. Record GPU memory and the exact notebook execution environment.

**Gate:** reruns with fixed settings reproduce score ranks within a prespecified tolerance; sensitivity to sample resampling and `ig_steps` is reported. Attribution alone is a candidate list, not a mechanism claim.

### Stage 4 — Causal validation

- On tune seeds, patch candidate activations between matched clean/corrupt runs and score the same continuation-likelihood margin. Test the token-position mapping established in Stage 2.
- Ablate selected candidate heads/MLPs; compare with random components matched by count, layer, and component type. Include sham/no-op interventions and both-refuse/both-comply controls where available.
- Report per-seed raw signed refusal-margin changes, clean/corrupt baselines, unclipped patching/ablation ratios, denominator-validity flags, and random baselines. CircuitKIT's clipped 0–1 faithfulness scores are secondary for a signed metric. Select candidate components on analysis only; do not tune the final set on evaluation.
- Evaluate the frozen intervention on the fresh seed-disjoint evaluation set. Report negative, null, or unstable results without relabeling examples or changing the metric.

**Gate:** at least one candidate undergoes both patching and ablation with quantitative effect estimates and random baselines. Generalization claims require a clean untouched seed-level evaluation set.

### Stage 5 — Japanese/English comparison (later)

- Build parallel per-language approved sets with matched requests, comparable refusal/compliance continuations, and seed-level split control.
- Discover and causally validate each language circuit separately before comparison.
- Compare signed attribution ranks and component overlap at matched sparsity; test Japanese-to-English and English-to-Japanese interventions against same-language and random baselines. Quantify uncertainty and sensitivity to prompt equivalence, tokenization, alignment, and annotation disagreement.

## Notebook and output plan

1. `notebooks/01_prepare_circuitkit_data.ipynb` — data gate, model/template checks, likelihood metric, custom metric adapter parity, and alignment report.
2. `notebooks/02_circuit_discovery.ipynb` — approved analysis export, EAP-IG smoke/discovery, signed scores, component summaries, visualization.
3. `notebooks/03_circuit_validation.ipynb` — activation patching, ablation, matched random baselines, per-seed effects, fresh evaluation.
4. Save executed notebooks, JSON reports, split manifest, package versions, CircuitKIT artifacts, exported visualizations, GPU memory snapshots, and logs under `outputs/lexsi/circuitkit_runs/<run_id>/`.
5. Append failures, fixes, runtime details, and scientific decisions to `COLAB_DEBUG_LOG.md`, `CIRCUITKIT_ROADMAP.md`, and `DECISIONS.md`.

## Immediate next actions

1. Obtain human review of the five same-language candidates and expand the pool from more independent analysis/tune seeds.
2. Update the approved continuation export through the adjudication pipeline; keep the provisional bilingual file separate.
3. Implement and validate the custom metric adapter and token-position alignment on tune examples.
4. Only after Gates 1 and 2 pass, prepare and run the first EAP-IG discovery smoke test.

## Current CircuitKIT references checked 2026-11-12

- [Custom data and validation](https://circuitkit.lexsi.ai/user-guide/custom-data/)
- [Data and metric requirements](https://circuitkit.lexsi.ai/user-guide/data/)
- [EAP/EAP-IG and tuning guidance](https://circuitkit.lexsi.ai/algorithms/eap/)
- [Safety case studies 23 and 24](https://circuitkit.lexsi.ai/examples/case-studies/)
- [Case study 23 source notebook](https://github.com/Lexsi-Labs/CircuitKIT/blob/main/examples/case-studies/23-jailbreak-refusal-multi-model.ipynb)
- [Case study 24 source script](https://github.com/Lexsi-Labs/CircuitKIT/blob/main/examples/case-studies/24-safety-refusal-custom-data.py)
- [Circuit visualization](https://circuitkit.lexsi.ai/user-guide/visualization/)

## 2026-11-12 — Core-hook integration design audit

> Point-in-time audit, superseded by the position-map prototype result below.

- Re-inspected the pinned upstream source at `/tmp/circuitkit-inspect-20261112` (commit `e3501bc72b349963db76df324f5dbd41ced2a02f`). `get_scores_eap_ig()` unpacks only `(clean, corrupted, label)`, calls `make_hooks_and_matrices()`, and that function adds/subtracts activations at identical `[batch, position]` coordinates. Its backward hook then contracts the same-position activation difference with gradients. This confirms the position-map helper is not yet integrated; passing maps only through the TaskSpec metric cannot repair attribution alignment.
- EAP-IG additionally derives its input-node interpolation endpoints from the activation-difference tensor. Any mapped implementation must retain the full clean input at unmapped framing positions, use mapped corrupted values only at explicitly aligned positions, and mask gradients there. Otherwise zero-filled unmapped positions would become an artificial intervention. This is a required invariant for a future core patch.
- `make_hooks_and_matrices()` has a separate-activation mode for models with normalization before and after; it stores clean and corrupted activations in separate halves. A first implementation must either correctly map both halves or fail explicitly when that mode is active. Do not silently fall back to same-index subtraction.
- At the time of the audit, the pinned source was an unmodified disposable clone and no map-aware EAP-IG result existed. The current prototype is preserved in the later result section. Colab remains CPU-only, so no GPU run was attempted.
- The next implementation experiment at that time was a map-aware EAP-IG hook API. Its synthetic CPU implementation and test results are recorded below; real Tiny Aya evaluation remains pending.

## 2026-11-12 — Position-map core prototype result

- Implemented a disposable patch against pinned CircuitKIT commit `e3501bc72b349963db76df324f5dbd41ced2a02f`. The patch is preserved at `docs/lexsi/patches/circuitkit_eap_position_maps_e3501bc.patch`; clean-checkout `git apply --check` passed.
- Added `collate_EAP_with_position_maps()` to preserve row maps and translate raw coordinates to left-padded batch coordinates. The project TaskSpec now constructs exact common-prefix and identical continuation maps and emits them with each row.
- EAP and EAP-IG accept the optional fourth batch field. Map-aware hooks gather corrupted activations into clean coordinates, mask gradients outside mapped sites, and capture full clean input embeddings. EAP-IG uses clean-to-clean endpoints at unmapped positions, avoiding zero-vector interpolation. Unsupported separate-activation normalization is rejected explicitly. Three-field batches keep the old path.
- Official Colab CPU test run: 41 passed (8 patch-specific checks and 33 project checks; no failures). Patch tests include actual EAP-IG scoring on a random one-layer TransformerLens model, identity-map parity against the legacy path, partial-map EAP-IG execution, unmapped endpoint checks, and TaskSpec-to-collator padding. Environment: CircuitKIT 0.2.2, TransformerLens 3.8.0, PyTorch 2.14.1+cpu; CUDA unavailable.
- This advances only API and synthetic hook integration. The patch is not yet installed by the experiment notebooks, and no Tiny Aya real-map validation, approved-data metric validation, GPU discovery, or causal experiment has occurred. The divergent framing span is intentionally unmapped, so localization of the source framing tokens remains unresolved.

## 2026-11-12 — Position-mapping feasibility audit

- Re-cloned CircuitKIT and verified source at commit `e3501bc72b349963db76df324f5dbd41ced2a02f`. `tokenize_batch_pair()` only provides a global `pair_padding_side`; it returns one `clean_lengths` tensor, and EAP's attribution loop computes positional activation differences at the same tensor index. There is no per-example token-position map or span mask in the TaskSpec metric interface.
- With left/end padding, continuation positions line up, but the 361-token median identical prefix shifts by the control/roleplay length delta. With right/start padding, the prefix lines up, but continuation positions and the metric's current right-edge indexing no longer match for unequal lengths. Neither global padding direction simultaneously aligns both regions.
- Consequence: do not simply switch padding side or accept CircuitKIT's warning as adequate. A valid extension needs an explicit per-example alignment tensor consumed inside the EAP activation-difference/patching hooks, plus target-token indices for both sides. Alternatives are redesigned matched prompts whose saved responses correspond to their exact text, or pausing EAP for these contrasts. Any implementation must test identity mappings, prefix mapping, divergent-span handling, suffix target parity, and patching on a toy model before real examples.
- At the time of this feasibility audit, no repository adapter changes had yet been made because a TaskSpec-only change cannot alter the core hooks. The later prototype consumes the maps in the pinned core.
- Added `eval/track2_extension/alignment.py` as the initial conservative mapping contract: it maps exact shared-prefix tokens and identical continuation tokens and reports differing framing spans as unmapped. The subsequent core patch consumes these maps for EAP/EAP-IG synthetic tests. It still does not localize the divergent framing tokens themselves; no discovery claim follows.

## 2026-11-12 — Latest EAP API and metric prototype audit

- Audited CircuitKIT upstream `main` at `e3501bc72b349963db76df324f5dbd41ced2a02f`. `discover_circuit` obtains the differentiable metric from the registered TaskSpec; the EAP collator preserves integer label vectors. Generic EAP CSV labels remain single-token correct/incorrect IDs.
- The built-in WinoGrande TaskSpec uses variable-length suffix token labels and differentiable mean suffix log-likelihood. It confirms a custom span metric fits the EAP scalar metric protocol, but does not itself implement refusal-minus-compliance scoring over two alternative continuations.
- Added `eval/track2_extension/circuitkit_metric.py` plus `circuitkit_data.py`, a provisional-safe EAP row builder and TaskSpec implementation. Eight metric/data tests cover row construction, explicit synthetic opt-in, human-approval gating, hand-computed scalar parity, gradient flow, and invalid batches/positions.
- Proposed adapter: two rows per prompt pair with the same continuation appended to both clean/corrupt prompts; refusal rows have sign `+1`, compliance rows `-1`. Batches must be balanced. Token IDs must be sliced from joint prompt-plus-continuation encoding. Different prompt lengths and nonmatching prompt interiors remain an alignment concern; score-position masking and patch-site correspondence need integration tests.
- TaskSpec registration and EAP-collator integration passed on Colab CPU in an explicitly synthetic mode with no approval metadata. A deterministic random 1-layer HF GPT-2 scorer parity check also passed exactly (margin `0.0189359188079834`, absolute error `0.0`) with finite nonzero gradients. These are API/toy-LM checks only. No human-approved continuation data, Tiny Aya metric/behavior validation, real-model EAP-IG attribution, or Colab GPU execution has occurred. Colab quota remains 0.00 units.


- The custom metric was compared with `eval/dataset2_likelihood.py` on a deterministic randomly initialized 1-layer HF GPT-2 using two- and one-token targets. Both margins were `0.0189359188079834` (absolute error `0.0`); backpropagation was finite and nonzero. This verifies generic causal-LM token shifts and mean normalization only, not Tiny Aya behavior.

- Approval-integrity audit: the first synthetic TaskSpec notebook used placeholder reviewers and `human_approved` metadata. It is preserved as superseded, not counted as approval. The data adapter now has a distinct `synthetic_smoke` mode that requires explicit opt-in; normal data still requires human review status and two distinct reviewer IDs. Corrected TaskSpec + real EAP collator execution passed on CPU Colab. No model or real data was involved.

## 2026-11-12 — Public discovery-path patch verification

Pinned-source inspection showed the initial position-map patch modified the older `attribute.py` EAP-IG implementation, while `discover_circuit()` dispatches to `attribute_node.py`. The patch and tests now cover the latter path. A rank-aware broadcast fix was required for attention-head gradients; the corrected synthetic regression suite passed 9/9 on official Colab CPU. This verifies wiring and mechanics on a random one-layer TransformerLens model only. The public notebook must import the patched checkout before importing CircuitKIT, assert the exact source commit, and check real data gates before discovery. `CircuitScores.node_scores` are absolute magnitudes; do not report them as signed attribution.

The discovery entrypoint wraps the pinned `discover_circuit` module-level `attribute_node` call only to serialize `graph.edges` signed scores before the API converts them to absolute node magnitudes. It returns the original attribution result unchanged. This wrapper still requires a real execution check when the scientific gates and GPU quota permit. The visualization's edges remain graph layout, not causal findings.

## Dataset2 Tune exhaustion update (2026-11-12)

An aggregate audit of both source response shards and prior training/selection/review exposure found 79 fully eligible matched Tune seeds but **zero unexposed fully eligible Tune seeds**; only two incomplete request keys from one seed could potentially be restored by rerunning inference. This cannot satisfy the five-seed metric gate. Do not recycle reviewed Tune seeds to tune a replacement metric. The next valid design requires genuinely new source seeds/responses or an appropriate external benchmark, split by seed into discovery, metric development, and untouched evaluation before running any metric. Evidence: `outputs/lexsi/tune_pair_recovery_audit_20261112.json`.
