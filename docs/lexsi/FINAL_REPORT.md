# LEXSI Track-2 / CircuitKIT evidence report

**Status date:** 2026-11-12  
**Overall conclusion:** the project has working data-review and CircuitKIT integration infrastructure, including real Tiny Aya HF↔CircuitKIT numerical parity on a Colab L4. It has **not** yet produced a validated refusal/compliance metric, a Tiny Aya EAP-IG circuit, a causal mechanism result, or evidence that tuning improved safety. The next work is gated by human review/adjudication, complete seed provenance, and access to the gated model from a new Colab runtime; GPU quota is available.

Latest local verification: the Track-2 suite ran 84 tests, with 83 passing and one expected skip (CircuitKIT is not installed locally); see `outputs/lexsi/latest_local_tests_20261112_execution_audit.log`. The official Colab CLI last listed no active sessions, zero active assignments, and 99.90 compute units. The current blocker is HF Secret access through the browser UI, not quota.

## Executive summary

The most defensible current result is associational: re-parsing stored guard outputs recovered 17,373 binary response labels from 21,184 Dataset2 rows, without changing any of the 13,730 previously binary labels. A seed-clustered comparison of US-framed versus neutral harmful prompts found higher unsafe-guard-proxy rates in English, Japanese, and mixed-language prompts in the analysis and tune splits. This is a corrected automated-guard analysis, not a human-labeled behavioral result, and it pools across framing and category within seed rather than matching those cells.

The HF Tiny Aya Water checkpoint can be scored through both Hugging Face and CircuitKIT with close numerical agreement on English, Japanese, and long-context probes. This verifies backend compatibility, not the research metric's ability to distinguish validated refusals from unsafe compliance.

A real Colab L4 smoke comparison generated and judged the same eight prompts for baseline, restricted-SFT, and broad-SFT arms. All arms received seven Safe and one Unsafe Qwen3Guard label; paired differences were zero across four seeds. This is an execution smoke test with a degenerate four-cluster interval, previously exposed prompts, and uncertain overlap with an older 40-row pilot. It says nothing reliable about efficacy or equivalence.

No human-reviewed/adjudicated CircuitKIT examples have been received. `outputs/lexsi/circuitkit_pairs.jsonl` is empty by design. CircuitKIT EAP-IG discovery and causal testing on Tiny Aya have not run. Historical SFT checkpoints could not be recovered completely from Drive; the smoke run used checkpoints present in an L4 runtime, whose weights were not durably persisted.

## What has been established

### Source and behavioral association

- The reduced SQLite source contains 21,184 observations and has a recorded SHA-256 and successful SQLite integrity check. The corrected label source is the full raw guard-output reparse.
- There are 17,373 binary response-guard labels: 15,749 Safe and 1,624 Unsafe. There were zero changes to the 13,730 prior binary labels; 3,643 additional binary labels were recovered where the earlier prompt verdict was non-binary or missing.
- The corrected seed-only estimator averages each seed's eligible harmful-row unsafe-proxy rate by culture, takes US minus neutral within seed, then equally averages matched seed differences. It does not match category, framing, or condition-cell composition within a seed.
- Analysis split: en +6.97 percentage points (110 seeds, 95% seed-bootstrap interval +0.38 to +13.79); ja +12.97 points (106, +4.87 to +21.07); mix +9.55 points (96, +2.41 to +16.41).
- Tune split: en +12.61 points (78, +5.77 to +19.76); ja +7.46 points (76, +0.77 to +14.58); mix +13.98 points (72, +7.41 to +21.20).
- Heldout split: en +9.85 points (44, -0.95 to +20.64); ja +17.68 points (41, +8.33 to +27.24); mix +13.61 points (42, +3.77 to +23.93). This split was previously exposed in project analyses and is descriptive, not confirmatory.
- The nine cohort counts and means were independently recomputed from the private seed-effects CSV and match the aggregate summary. A second implementation (`scripts/lexsi/verify_seed_bootstrap.py`) resampled those 665 seed effects 20,000 times per group with a different PRNG: all means matched exactly and intervals were within 0.379 percentage points of the stored intervals. This does not independently reproduce the upstream raw-SQLite extraction.
- An older Pipeline1 report gives a language-stratified Mantel–Haenszel OR of 1.982 (95% CI 1.583–2.482, p≈2.21e-9) for US versus neutral and OR 1.097 (95% CI 0.862–1.397, p=.487) for Japan versus neutral. The cohort, three language-specific tables, MH OR, and continuity-corrected CMH p-value have since been independently reconstructed from the original shards using the published labels; the historical RBG interval was independently reproduced as 1.5832–2.4818. Do not combine its estimate with the distinct corrected-label seed/cell matched estimand as if they were one analysis.

All these labels are outputs of an automated safety guard. They do not establish actual unsafe compliance, semantic equivalence, or causality.

An earlier attempt at stricter within-seed matching reported no English/Japanese matches. That result was an implementation artifact: NULL `mix_direction` is inapplicable for monolingual rows, but the generic matcher had dropped it as missing. After correcting this, the strict estimator ran on both original shards; see the later raw-shard update and `SOURCE_RECONCILIATION.md` for estimates and limits.

### Human review inventory

- The prior bilingual audit found 401 guard-discordant pairs. The initial provisional shortlist contained 50 distinct examples; a second pass promoted two more, yielding 52 current provisional bilingual examples. These change language and are not substitutes for same-language controls.
- The 87 same-language control/roleplay candidates (53 analysis rows, 34 tune rows, 62 seeds) have been packaged for two blinded reviewers. At this report date, all 87 remain unreviewed, adjudication is blank, and **human-approved contrasts = 0**. The approved CircuitKIT input file is empty.
- A separate Codex-assisted first-pass screen now annotates all 87 candidates but remains explicitly provisional. Among 39 pairs screened as equivalent and harmful on both sides, 11 appear to have the target control-refusal/roleplay-compliance pattern and 4 appear reversed. This is triage only, not human ground truth, reviewer agreement, or export eligibility.
- A read-only audit of the current Drive reviewer sheets confirms all 87 rows per reviewer still have blank status. The historical Pipeline3 training export was recovered and exactly mapped to 21 distinct Dataset2 seeds across 22 accepted pairs. The Pipeline2 probe seed list and any other tuning exposures remain unknown; `outputs/lexsi/circuitkit_prior_seed_registry.json` is therefore marked `complete=false`, and notebook 03 refuses to treat evaluation as seed-disjoint.
- The broader provisional discrepancy inventory has 138 ambiguous, 53 English guard-false-positive, 91 Japanese guard-false-positive, 56 non-equivalent-prompt, 7 other, and 56 provisional real-model-difference rows. These are audit dispositions, not human validation.
- Across the original 401 bilingual disagreements, 232 are analysis, 78 tune, and 91 heldout. A 80-row agreement-control sample has 10 provisional reviews and 70 unreviewed. Calibration and provisional labels remain distinct from human adjudication.

### Model/backend and CircuitKIT integration

- Model: `CohereLabs/tiny-aya-water`, pinned revision `41a99216d12b3ef4881df762934186e5e4d2bbc1`, original HF checkpoint, FP32.
- Architecture probe: Cohere2, 36 layers, model width 2,048, 16 query heads / 4 KV heads, vocabulary 262,144; sliding attention window 4,096. Tokenizer backend is `TokenizersBackend`; native chat-template SHA-256 is recorded in `outputs/lexsi/colab_phase1/authorized/checkpoint.json`.
- On a real NVIDIA L4, CircuitKIT 0.2.2 / TransformerLens 3.8.0 parity probes on English, Japanese, and a 4,196-token context agreed on top token. Largest recorded absolute log-probability error was `7.44e-5`; peak PyTorch allocation was `22,248,377,344` bytes. Environment versions are in `outputs/lexsi/colab_phase1/authorized/environment.json`.
- CPU tests verified the custom differentiable continuation-margin prototype, TaskSpec/collator integration, and patched explicit position-map behavior, including synthetic EAP-IG execution. These are integration/synthetic checks, not real-data discovery.
- The proposed metric is refusal-continuation mean log probability minus compliance-continuation mean log probability, with both continuations teacher-forced against each prompt and tokenized/length-normalized consistently. It has not been tested on human-approved Tiny Aya examples and has not passed a behavioral discrimination gate.
- All 87 provisional same-language control/roleplay candidates inspected for tokenizer geometry had unequal prompt token lengths. A position-map approach exists and has passed synthetic tests; real candidate alignment remains to be checked after review, and divergent framing positions must not be falsely aligned.

### Intervention and before/after pilot

- Historical reports describe 22 CuratorKIT preference-shaped records generated from 24 prompts and one-epoch SFT runs (restricted LoRA on MLP down projections in layers 18 and 23; broad LoRA comparison). The generation gate was not an independent human judge, and training consumed chosen responses as SFT rather than optimizing a DPO preference objective. The reported circuit target layers came from a very small SafeTune probe, not CircuitKIT EAP-IG.
- Historical Pipeline5 used 40 heldout rows; guard outcomes were too sparse for defined odds ratios. It was inconclusive.
- The newer real L4 three-arm smoke used eight prompts / four US-neutral pairs / four seeds, greedy decoding, 192 generated-token cap, and Qwen3Guard. Counts were 7 Safe / 1 Unsafe per arm; paired proxy risk difference was 0 with `[0,0]` seed-bootstrap interval. This is a plumbing check only.
- The newer smoke's prompts were previously exposed; their overlap with the historical 40-row set is unknown. Human response-safety labels and reviewed benign twins were unavailable. Do not state that any model improved, regressed, or was equivalent.
- The exact restricted/broad checkpoints used in the smoke remain volatile L4 scratch artifacts. Drive search found only a 25.2 MB restricted `model.safetensors` plus config; it is about 0.7% of the previously observed 3.60 GB runtime file and fails Safetensors header deserialization (`incomplete metadata, file not fully covered`). The broad folder was empty. Neither is a recoverable checkpoint.

## Current gate table

| Gate | State | Evidence required to advance |
|---|---|---|
| Source audit | Partial/pass for present snapshot | Preserve hashes; independently reproduce bootstrap and corrected-label Pipeline1 statistic. |
| Human-validated contrast set | Blocked on reviewers | Two independent blinded judgments plus adjudication; validate prompt equivalence, refusal/compliance, and exact response continuation spans. |
| Behavioral likelihood metric | Not passed | On approved tune seeds, show the metric separates validated refuse/comply continuations in the expected direction; report uncertainty and failures. |
| CircuitKIT EAP-IG discovery | Not run on Tiny Aya | Use approved discovery seeds, validated objective and justified token-position maps; retain per-seed scores and settings. |
| Causal validation | Not run on Tiny Aya | Notebook 03 now captures raw signed margin outcomes, clean/corrupt baselines, unclipped ratios, and denominator flags, but requires a complete prior training/tuning seed registry, reviewed fresh-evaluation examples, and restored GPU access before patching/ablation or generalization can be claimed. |
| Model improvement claim | Unsupported | Preregistered sufficiently powered paired evaluation, valid labels, recovered checkpoints, seed-clustered inference, and safety/utility trade-offs. |
| Cross-language circuit comparison | Future | English/Japanese discovery with matched sparsity, component overlap, attribution differences, and cross-language intervention transfer. |

## Recommended next sequence

1. Have two independent reviewers complete the existing blinded packets and adjudicate disagreements. The Drive folders are prepared but currently visible only to the owner and connected project account; the owner must grant each reviewer access to their separate folder. Never share the owner-only folder or mapping key.
2. Validate reviewed prompt equivalence, actual response behavior, and exact continuation spans. Keep all current automated annotations provisional.
3. Freeze seed-disjoint discovery, tune, and evaluation cohorts. The historical heldout split is contaminated by prior project exposure and is not a fresh confirmatory set. Add reviewed benign controls and prespecify exclusions and seed-clustered inference.
4. Install the pinned CircuitKIT position-map patch reproducibly and validate the continuation-likelihood metric on approved tune examples using actual Tiny Aya. Do not proceed to EAP-IG if the score fails to distinguish the reviewed behaviors.
5. Use the available Colab quota after the owner fetches `HF_TOKEN` through the browser UI. Resolve checkpoint durability before any multi-arm evaluation; save checkpoints immediately to durable private storage and verify a fresh-process reload/hash before evaluating.
6. Run EAP-IG on discovery seeds. Save full attribution tables, settings, component ranks, and visualizations. Treat these as candidate selection only.
7. Run causal activation patching and ablations on separate evaluation seeds, against random matched-component baselines and benign controls. Report null or harmful results as such.
8. Only after causal evidence and sufficient reviewed training data, train circuit-restricted and broad comparison arms; evaluate both against baseline on a fresh seed-disjoint set. The article goal is a measured before/after safety metric with uncertainty and utility checks, not a desired positive outcome.
9. Compare English and Japanese circuits only after each language has independently passed the above gates.

## Artifact locations and privacy

- Roadmap and gate states: [CIRCUITKIT_ROADMAP.md](CIRCUITKIT_ROADMAP.md), [STATUS.md](STATUS.md), [DECISIONS.md](DECISIONS.md).
- Run-by-run evidence: [EXPERIMENTS.md](EXPERIMENTS.md), [COLAB_DEBUG_LOG.md](COLAB_DEBUG_LOG.md), [DATA_PROVENANCE.md](DATA_PROVENANCE.md), [SOURCE_RECONCILIATION.md](SOURCE_RECONCILIATION.md).
- HF/CircuitKIT parity: ignored local `outputs/lexsi/colab_phase1/authorized/`.
- Corrected-label seed-only aggregate: ignored local `outputs/lexsi/corrected_seed_only_summary.json`; per-seed rows remain private.
- L4 smoke generations/judgments/manifest: ignored local `outputs/lexsi/track2_pilot_20261112_nf4_smoke/` and authorized private Drive output.
- Human-review packets and mapping key: ignored local `outputs/lexsi/blind_review_packets_20261112/` and private Drive folders. These contain sensitive prompts and responses; do not commit or broadly share.
- Approved export remains empty: `outputs/lexsi/circuitkit_pairs.jsonl`.

No prompt, response, seed identifier, or raw judgment is reproduced in this report.

## Latest integration audit (2026-11-12)

A pinned-source audit caught that the initial explicit-position-map patch modified CircuitKIT's legacy EAP implementation but not the `attribute_node.py` path called by `discover_circuit()`. The patch now covers that public path, including the EAP-IG input interpolation, and a synthetic one-layer TransformerLens regression suite passed 9/9 on official Colab CPU. The first public-path run failed on attention-head mask broadcasting; the corrected shape-aware mask passed. This verifies only API integration. It does not change the result above: no human-approved input, Tiny Aya behavior-metric validation, GPU EAP-IG, or causal validation has been completed. See `COLAB_DEBUG_LOG.md` and `EXPERIMENTS.md` for exact evidence and artifact paths.
# Raw-shard matched association update (2026-11-12)

The strict corrected-label analysis that was previously pending has now run on the recovered original result shards. All 21,184 prompt rows exactly match the separate corrected-label export, with no changed prior binary labels. Exact within-seed matches by category/framing/mix-direction/output-language yielded nine split/language estimates; analysis and tuning results are listed in `SOURCE_RECONCILIATION.md`. An independent NumPy bootstrap reproduced the means and intervals within 0.52 percentage points.

This improves the reproducibility of the Dataset2 association analysis, not the mechanistic result. The outcomes are automated guard labels, not human labels; matching does not prove semantic equivalence or causality; heldout was previously exposed. All four Pipeline 1 MH comparisons, including their language-specific tables, ORs, RBG CIs, and continuity-corrected tests, are independently reconstructed from the original shards. Human review remains at zero completed rows, the approved CircuitKIT export remains empty, Tiny Aya's behavioral metric has not passed, GPU EAP-IG and causal validation have not run, and no before/after safety improvement is established. Colab currently has 99.90 compute units, no active session, and no accruing usage; fresh-runtime HF token access must be bootstrapped through the Colab UI. Overall objective remains `PARTIAL`.


## Latest execution and data-gate audit (2026-11-12)

The saved Phase 1 Colab notebook passed three real-weight HF↔CircuitKIT parity probes on an NVIDIA L4, then raised in its last cell because `outputs/lexsi/circuitkit_pairs.jsonl` was empty. The CLI reported exit code 0 despite the cell traceback. `scripts/lexsi/audit_notebook_execution.py` now catches this; its private audit of the saved attempt returned `CELL_ERRORS` and exit code 1. Notebook 01 now records the anticipated missing-approval state as `BLOCKED_HUMAN_APPROVAL_REQUIRED` rather than raising, but the corrected notebook has not yet been rerun remotely.

A new L4 check showed that CLI execution cannot fetch Colab Secrets: `google.colab.userdata.get("HF_TOKEN")` must run in the browser-attached runtime. That runtime was stopped before model loading; current account balance is 99.90 CUs, with no active session or usage. The exact UI bootstrap cell is in `REPRODUCE.md`.

The independent Drive reviewer CSVs remain unmodified since their original 2026-11-12 06:16 UTC upload; the local 87-row adjudication template has 87 blank adjudication statuses, and the approved export is zero bytes. The LLM-assisted pass is only provisional. Current local Track-2 suite: 84 tests, 83 pass and one expected skip. No validated Tiny Aya behavior metric, real-data EAP-IG circuit, causal intervention, or model-improvement result exists.
