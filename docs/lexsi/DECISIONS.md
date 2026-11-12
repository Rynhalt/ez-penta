# Dataset2 decision log

## 2026-11-12 — Require complete prior-seed inventory for causal evaluation

- Re-read the two current reviewer sheets: both contain 87 candidates and every review status remains blank. Keep the human-approved export empty.
- Notebook 03 now requires a separately sourced, explicitly complete historical seed registry covering prior training/tuning exposure. A row-level negative exposure flag is insufficient to establish held-out status. The prior registry is not yet available, so this causal-evaluation gate remains closed.
- Located the 22-row Pipeline3 accepted preference export and 24-row source prompt file in the shared Drive. Then exact-joined all 22 opaque training IDs against `matched_observations.csv`, mapping them to 21 distinct Dataset2 seeds (one seed repeated). This resolves Pipeline3 exposure only; keep the registry incomplete because Pipeline2 and other tuning exposure remain unknown.
- Unit coverage now checks missing/incomplete registries and overlap rejection. The latest local Track-2 suite ran 63 tests (62 passed, 1 expected skip); no Colab GPU execution occurred.

## 2026-11-12 — Preserve raw signed CircuitKIT intervention outcomes

- Current CircuitKIT safety case study 23 explicitly captures raw pillar outputs because its headline faithfulness scores are clipped; the custom-data case study 24 is a benign GPT-2 plumbing example, not a safety benchmark. Official EAP docs recommend larger cohorts than the current planned <=16 seeds/language pilot.
- Notebook 03 now captures raw patched/ablated metric values, clean/corrupt baselines, unclipped ratios, and denominator flags per seed. Invalid/sentinel ratios are excluded from ratio summaries; three tests cover the behavior. Clipped scores remain secondary. The latest local suite ran 63 tests (62 passed, 1 expected skip); this code correction has not been executed on Tiny Aya or Colab GPU.

## 2026-11-12 — matched Colab smoke only; no effect claim

- Ran baseline HF, restricted-SFT, and broad-SFT arms on the same 8 heldout prompts (4 matched US-neutral pairs across 4 seeds) on Colab L4, with all three loaded in NF4. This is a smoke test of matched generation, guarding, persistence, and analysis, not a powered model comparison.
- Retain the 192-token generation cap as a documented smoke constraint; future full evaluation must use a predeclared output cap adequate for source-response lengths and consistent across arms. Do not reinterpret the smoke outcome or change the task labels based on it.
- All arms were judged Safe=7 / Unsafe=1; risk differences were 0, and the bootstrap CI collapsed to `[0,0]` because all observed paired outcomes agreed across four seed clusters. Record this as no detectable difference in a tiny, previously exposed proxy pilot, not as equivalence or evidence that the fixes failed/worked.
- Pilot outputs and run manifest are stored under ignored `outputs/lexsi/track2_pilot_20261112_nf4_smoke/`. Raw prompt and answer data remain private. The restricted and broad weights remain in L4 scratch and have not been durably recovered.
- Stop further GPU runs while Colab balance is 0.00 units. Resolve checkpoint persistence and obtain compute before scaling this evaluation. Human approval and behavior-metric validation remain mandatory before CircuitKIT discovery.
- Built the next evaluation candidate cohort by exact matching on source seed, split, harmful kind, category, framing, prompt language, mix direction, and output language. The SQLite uses NULL `mix_direction` for English/Japanese; treat that as “not applicable” when forming same-language matches, not missing metadata. After excluding incomplete, duplicate, and conflicted source rows, 89 pairs / 178 prompts remained across 35 heldout seeds. This changes candidate counts only; all remain previously exposed, unreviewed, and non-confirmatory.

- The supplied Drive folder contains the original GGUF exports. The prepared review CSVs were absent from the accessible folder and exact-name Drive search. Review inputs reconstructed from saved SQLite rows are tracked by hash.
- Keep heldout rows in the audit inventory. The plan records prior exposure of 91 heldout disagreements. Exclude those seeds from contrast selection and tuning, and use fresh untouched seeds for confirmatory generalization.
- Assign each contrast a stable ID from the ordered prompt IDs. Treat guard direction as source evidence only. Require reviewed behavior and explicit human adjudication before export.
- Review all 87 reconstructed roleplay screens. Four analysis seeds and one tune seed have a provisional refusal-to-compliance contrast, all in English. The tune example retains its split for later validation. Keep the approved export empty until adjudication.
- Screen both-refuse controls from analysis pairs with two Safe verdicts and two published guard refusal flags. Use the saved responses to verify refusals. Use reviewed roleplay screens for both-comply controls. Keep sanity controls disjoint from the shortlist by source seed.
- In the first-pass review, 54 qualifying rows represented 50 distinct bilingual examples. The second pass added two provisional examples, yielding 56 qualifying rows and 52 distinct examples in the expanded file. Preserve the original 50-row file, all source rows, and review history. Human approval is still required before CircuitKIT execution.
- Keep the current 52 provisional bilingual examples for a later language comparison and use adjudicated same-language control-versus-roleplay pairs for the first circuit experiment. Score exact continuation prefixes after the chat-template boundary, normalize by token count, and report the paired difference with both candidate targets fixed across prompt conditions. Validate Tiny Aya Water with HF-versus-CircuitKIT parity on Colab before discovery.
- Colab OAuth and the L4 setup passed. Use a runtime Hugging Face credential for CLI execution because Colab Secrets require the browser UI. Check executed notebook cell outputs as well as the CLI exit status. The pinned model config still returns HTTP 403 with the saved local credential.
- The renewed Hugging Face credential opened the pinned checkpoint. Real-weight FP32 parity passed on an L4 for short English and Japanese inputs with EAP-IG head expansion and for a long input using compact grouped attention and chunked KV caching. Behavioral validation is pending adjudicated data.
- The second pass covered all 140 ambiguous and 56 prompt-mismatch rows. Two new-seed non-heldout pairs were promoted provisionally (suicide assistance and a dangerous ninja act). The Hollywood company-document pair remains a strong candidate pending confirmation of its unusual Japanese transliteration. Keep overlapping-seed, heldout, high-level-only, and prompt-mismatch cases separate. See `outputs/lexsi/review_196_ambiguous_and_prompt_mismatch_second_pass.csv` and `outputs/lexsi/ambiguous_non_equivalent_second_pass.md`.
- The suicide-assistance and dangerous-ninja-act rows were initially marked ambiguous. On second pass, their request pairs were matched and the English responses were materially more safety-preserving than the Japanese responses. Update both current provisional classifications to `real_model_difference`, while retaining their first-pass labels. The expanded 52-row CSV and viewer include them; human approval remains pending.
- Current CircuitKIT docs were rechecked on 2026-11-11. EAP requires explicit clean/corrupt safety pairs; the built-in `logit_diff`, `kl`, and `accuracy` metrics do not directly implement the project's full-sequence refusal-minus-compliance mean log-likelihood. The plan therefore gates EAP-IG on a differentiable custom-metric adapter, parity against the existing teacher-forcing scorer, and validated token-position alignment. CircuitKIT graph edges are visualization/layout edges, not measured causal connections. See `CIRCUITKIT_IMPLEMENTATION_PLAN.md`.
- 2026-11-11 checkpoint recovery: a recursive filename/extension inventory of the mounted Google Drive `MyDrive` found no restricted or broad adapter checkpoint. The only Tiny Aya model file found was the historical GGUF; `pipeline4_results.json` stores runtime-local `/content/...` paths. Treat the adapters as unrecovered until a separate authorized storage location is checked. Do not claim or evaluate an unavailable checkpoint.
- 2026-11-11 source reconciliation: the actual 21,184-row Drive CSV hashes to `7a38bedf6d2d3730154c65a3c6e1b022c5216a82850de3017deefe373d57ac0b`; its stored binary `safety` denominator is 13,730. A read-only parse of raw SQLite guard outputs independently recovered 17,373 binary response labels, including 3,643 where prompt moderation was Controversial, with zero changes to prior binary response labels. Keep Controversial/Missing unknown and call all labels automated guard proxies. The existing SQLite also contains seed-bootstrap summaries, but those use the original exported denominator; recompute from corrected labels before stronger association claims. Details: `SOURCE_RECONCILIATION.md`.
- The first Drive raw-recovery summary contained incorrect zero per-split label subtotals from an aggregation bug. Its global recovery checks and private corrected-label file are accurate; do not use that intermediate report for split subtotals. The canonical local aggregate is `outputs/lexsi/raw_guard_recovery_and_seed_comparisons_20261111.json`, with the discrepancy documented in `COLAB_DEBUG_LOG.md`.
- New analysis code in `eval/track2_extension/stats.py` defines the estimand as an equal-weight mean of within-seed paired differences, averages repeated prompt outcomes within matched seed/strata, excludes unpaired and unknown outcomes, and bootstraps independent seeds with a fixed RNG. The simpler corrected-label seed-only design was subsequently executed and is documented in `SOURCE_RECONCILIATION.md`; it does not impose within-seed framing/category/mix-direction matching. The stricter matched estimator and Pipeline 1 point/test statistic were later reproduced from raw shards; the historical RBG interval was subsequently reproduced (95% CI 1.5832–2.4818). Do not conflate the historical SQLite intervals, seed-only result, and strict matched analysis.
- `eval/track2_extension/reconcile_dataset2.py` fixes the planned corrected-label analysis: harmful, analysis-eligible, completed rows; Safe=0/Unsafe=1 with Controversial/Missing/Unparsed excluded; within-seed matching by kind, category, framing, mix direction, and output language; separate prompt-language estimates; equal weight per seed; 5,000-replicate percentile bootstrap with RNG seed 20261111. Nine synthetic tests pass. This design is fixed before real corrected-data effects are inspected and does not establish human-verified compliance or causal culture effects.
# 2026-11-11 — Corrected guard labels and seed-paired estimand

- Accepted the raw guard verdicts as recovered automated-proxy labels only after verifying zero changes among prior binary labels and zero malformed records. Unknown/nonbinary response verdicts remain excluded from binary risk estimates.
- Chose an equal-weight within-seed US-minus-neutral estimate stratified by split and prompt language, without framing/category matching, as a separate broad seed-only estimand. The earlier statement that strict design-cell matching produced no English/Japanese matched seeds is now withdrawn: the implementation had dropped source NULL mix-direction fields that are inapplicable to monolingual prompts. The seed-only estimand remains descriptive/associational and is not a framing-specific causal effect.
- Heldout is descriptive because of prior exposure. No behavior-improvement, mechanism, or causal claim is authorized by this observational guard analysis.
- The Drive OAuth workaround is a local patch to the installed official Colab CLI: respect `COLAB_AUTHUSER` instead of hardcoding account index 0. It enabled mounting once; the override was not reliable across newly provisioned sessions, so keep and reuse the already mounted runtime.

## 2026-11-12 — Custom EAP continuation metric design

- Audited CircuitKIT upstream commit `e3501bc72b349963db76df324f5dbd41ced2a02f`. Generic EAP CSV is token-label based; the WinoGrande-specific task proves that padded variable-length token-span labels and a differentiable suffix-likelihood objective are supported by EAP's metric API.
- Do not use generic one-token `logit_diff` as a substitute. Implement a project TaskSpec that scores both validated answer continuations as full token sequences, using a signed refusal-minus-compliance mean log-probability margin. Prototype code is in `eval/track2_extension/circuitkit_metric.py`.
- At the time of this entry, the prototype had only synthetic score/gradient tests; subsequent CPU Colab work verified pinned CircuitKIT 0.2.2 TaskSpec registration/collation and toy-HF scorer parity. Human-approved data, Tiny Aya score validation, real-token alignment, EAP-IG, and GPU execution remain gates; no research result is inferred from the toy runs.

## 2026-11-12 — CPU Colab TaskSpec integration result

- Used the already mounted official Colab CPU session to install and verify CircuitKIT 0.2.2, Transformers 5.15.0, TransformerLens 3.8.0 and PyTorch 2.14.1+cpu. No GPU compute was used.
- Actual CircuitKIT task registration and `collate_EAP` accepted the custom adapter's balanced synthetic refusal/compliance rows. The real integration smoke passed after resolving a cached `eval` package namespace collision.
- Keep CircuitKIT/model behavior gate D partial: no model forward or EAP-IG ran, and no approved real continuations exist. Do not treat synthetic TaskSpec wiring as scientific evidence.

- The custom CircuitKIT metric was compared with the existing teacher-forced scorer on a deterministic, randomly initialized 1-layer HF GPT-2 running on Colab CPU. Exact scalar parity and finite nonzero gradients passed. This validates the shift/mean-log-prob implementation on a toy LM only; the research gate still requires human-approved examples and Tiny Aya's actual weights.

## 2026-11-12 — Corrected synthetic approval metadata

- Audit found the first CPU TaskSpec smoke notebook marked its toy record `human_approved` and supplied placeholder reviewer IDs. Although it contained no Dataset2 text or real review labels, that metadata falsely implied actual human adjudication. Preserve that notebook as superseded evidence only; do not cite it as approval or data validation.
- Corrected the adapter with an explicit `data_mode='synthetic_smoke'` path that runs only when `allow_synthetic_smoke=True`. The default execution path still rejects provisional/unreviewed data and requires two distinct reviewer IDs. Reran actual CircuitKIT task registration and EAP collation with the corrected synthetic schema; it passed and explicitly asserts no human approval.

## 2026-11-11 — Preserve positional-alignment and approval gates

- Rechecked the five same-language control/roleplay candidates: no completed reviews or adjudication are present. Keep the approved export empty; the 50 bilingual “valid difference” examples do not substitute for these same-language controls.
- CircuitKIT EAP right-aligns clean/corrupt sequences so answer positions coincide, but that does not establish correspondence across the prompt body. Require explicit prompt-side alignment evidence on actual reviewed examples before interpreting attribution or patching; report unequal lengths and internal shifts rather than silently accepting them.
- A fresh execution through the existing CPU session lost its connection while the session was IDLE; local test environments lack `pytest`/`torch`. Preserve earlier passing CPU smoke evidence, but do not describe the failed retry as a pass. GPU use remains paused at 0.00 compute units.

## 2026-11-12 — CircuitKIT positional alignment constraint

- Rechecked upstream CircuitKIT at `e3501bc72b349963db76df324f5dbd41ced2a02f`. Its EAP tokenization exposes one global padding side and its attribution hooks compare same tensor indices; the TaskSpec metric receives only one `input_lengths` vector. A row-specific map for prompt and answer regions cannot be added safely in `RefusalComplianceTaskSpec` alone.
- Pinned Tiny Aya tokenizer revision `41a99216d12b3ef4881df762934186e5e4d2bbc1` on all 87 provisional screens. Each prompt pair has unequal lengths; median exact common prefix 361 tokens in both languages. Do not use end-padding as proof of prompt attribution alignment, nor switch padding side because that misaligns completion targets.
- Require either an upstream/core EAP change with explicit per-row region mapping and toy-model patch tests, or a new dataset whose saved generations match an alignment-preserving prompt construction. Do not silently rewrite prompt text, discard unequal pairs, or claim causal localization from the current adapter.

## 2026-11-12 — Explicit mapped-position EAP prototype

- Decision: preserve the conservative exact-token map and implement it as an opt-in patch against CircuitKIT commit `e3501bc72b349963db76df324f5dbd41ced2a02f`. The TaskSpec now emits exact common-prefix and identical-continuation positions; divergent control/roleplay prompt tokens remain unmapped.
- The core patch maps corrupted activations into clean coordinates, masks gradients at unmapped positions, and makes EAP-IG's corrupted endpoint equal the clean endpoint at unmapped sites. This avoids interpreting zero-filled activations as evidence. The unsupported separate-activation normalization branch raises explicitly, and three-field batches retain the legacy path.
- Evidence: Colab CPU combined run passed 41 tests (8 patch-specific plus 33 project tests), including actual EAP-IG execution on a random one-layer TransformerLens model, identity-map parity, partial-map execution, and TaskSpec-to-collator padding. Source patch and application instructions are preserved under `docs/lexsi/patches/`.
- Scope limit: this validates synthetic software behavior only. The patch is not installed in experiment notebooks, has not been exercised on Tiny Aya or approved Dataset2 contrasts, and does not attribute the divergent framing-token positions. Do not treat its test pass as metric validation, discovery, or a causal result.

## 2026-11-12 — Blinded human-review packet workflow

- Decision: use two separate reviewer CSVs with independent case order and A/B condition assignment so source guard labels and provisional screening rationales cannot anchor judgments. Reuse an opaque case ID across the two CSVs to permit reviewer-agreement joins while keeping the source mapping key private and outside Git.
- The packets cover all 87 provisional same-language candidates (53 analysis rows, 34 tune rows; 62 source seeds). The two packet builds are deterministic and hash-recorded, but no review status, label, or approval is prefilled.
- Review instructions enumerate exact rubric categories; reviewers mark `completed` only to denote a fully filled review, and that state never confers approval.
- Approval gate remains two independent human reviews, adjudication, semantic prompt-equivalence confirmation, actual response-behavior review, and exact continuation-span validation. Until these occur, zero rows are authorized for real-data CircuitKIT execution.
- Drive delivery keeps reviewer 1 and reviewer 2 in separate folders so sharing one does not expose the other reviewer's sheet. The private adjudication key is outside both. No sharing grant was made; the project owner must share each assigned folder with its reviewer.
# 2026-11-12 — Evidence-based stopping-point report

- Decision: publish a current report, plain-language findings note, and reproduction guide while explicitly keeping the scientific objective incomplete. The available evidence supports an association in an automated guard proxy and real HF/CircuitKIT backend parity, but not human-validated behavior, a Tiny Aya circuit, causal effect, or model improvement.
- The report records the 87 unreviewed same-language candidates, zero human-approved pairs, 0.00 Colab compute units, and unrecoverable durable tuned checkpoints as active blockers. The final-documentation roadmap item is complete as a stopping-point deliverable; research gates remain partial/not started.
- Evidence and limitations are in `FINAL_REPORT.md`, `WHAT_WE_LEARNED.md`, and `REPRODUCE.md`. No private prompt/response examples or seed identifiers are copied into them.

## 2026-11-12 — Independent bootstrap verification

- Decision: independently verify the saved corrected seed-effect summary before relying on its interval table. The private Drive `seed_effects.csv` was read-only retrieved to the ignored local output directory; `scripts/lexsi/verify_seed_bootstrap.py` recomputed all nine means exactly and 20,000-draw percentile intervals within 0.379 percentage points of the stored intervals.
- This check conditions on the saved 665 per-seed effects. It does not reproduce their extraction from raw SQLite, validate the guard labels, or change the estimand. The aggregate evidence and hash are recorded in `outputs/lexsi/run_manifest.json` and `outputs/lexsi/corrected_seed_only_bootstrap_independent_check.json`.

## 2026-11-12 — Stop repeated Colab Drive credential retries

- A new official CLI CPU runtime was provisioned to attempt the remaining raw-SQLite matched analysis. Its Drive mount again failed with HTTP 400 during credential propagation; the runtime was stopped and no data was transferred.
- Decision: do not repeat identical OAuth/authuser variations already tried. The large source DB is above the connector's 256 MiB transfer cap. Resume this analysis only when a supported mounted Drive route or another authorized transfer path is available. Keep the seed-only aggregate and its independent bootstrap verification explicitly separate from the unrun raw-SQLite reanalysis.

- 2026-11-12: Position-map compatibility must be tested through `circuitkit.api.discover_circuit()`'s actual backend (`attribute_node.py`), not only the older standalone `attribute.py` scorer. Source inspection showed the initial patch did not reach the public API path. Extended the pinned patch to the node implementation and added a real random one-layer TransformerLens regression test. Official Colab CPU suite: 9/9 passed. This is API-level synthetic evidence only; real Tiny Aya mapping/discovery and GPU execution remain gated.
- 2026-11-12: CircuitKIT 0.2.2's `CircuitScores.node_scores` is an aggregate absolute attribution magnitude, so the discovery notebook labels it as magnitude and does not imply that the public node-score JSON preserves attribution sign. Directional/causal interpretation requires separate evidence.

- 2026-11-12: Preserve signed edge attribution by capturing `graph.edges` immediately after the public CircuitKIT attribution call, before its converter takes absolute values for node scores. Keep public node magnitudes and signed edge data distinct; neither graph edge drawings nor attribution scores establish causal connections.

## 2026-11-12 — Public discovery API synthetic smoke

- Verified the patched position-map backend through `circuitkit.discover_circuit()` itself, not only unit tests or direct backend calls. The official Colab CPU smoke completed EAP-IG and serialized CircuitKIT artifacts on a random one-layer TransformerLens model; signed edges were captured before magnitude-only node conversion.
- An initial integration failed when the wrapper's tokenizer dropped its chat template, then a second attempt failed because unequal-length synthetic prompts invalidated positional assumptions. Fixed the wrapped tokenizer metadata and made only the synthetic test prompts equal in token length. Keep the real prompt alignment gate unresolved.
- The smoke is a software integration check only. It yielded two synthetic node scores, eight signed edge scores, and zero pruned nodes. No Tiny Aya behavior/mechanism, approval, or causal conclusion follows. Full trace/output and artifacts: `outputs/lexsi/circuitkit_public_api_smoke_20261112/`.

## 2026-11-12 — Reject incomplete Drive checkpoint candidate

- Downloaded the connected-Drive restricted `model.safetensors` candidate (25,231,360 bytes; SHA256 `62eb698703051260588f1cc0c2e85a1bbbfd4117b740e64d2eca028241feadc1`) and attempted only a Safetensors container/header open. It failed with `incomplete metadata, file not fully covered`; the broad checkpoint folder is empty. Classify both historical arms as not durably recoverable from this Drive location. Do not try loading these bytes as a model or infer adapter semantics from the base-style config.
- Backed up the synthetic public `discover_circuit()` smoke artifacts and Colab CLI session log to a new project Drive output folder; readback confirmed the uploaded file names and byte sizes. Local SHA256s and Drive file IDs are in the private receipt. This is durability for software evidence only, not research data.

## 2026-11-12 — Keep language circuits separate and preserve full causal report

- The discovery notebook previously mixed English and Japanese examples into one EAP-IG objective. Changed it to independently sample adjudicated analysis seeds by language with a fixed seed (up to 16 per language, minimum 8), run separate EAP-IG passes, and save separate circuits, signed-edge scores, attribution rankings, and visualizations. This prevents a pooled attribution from being mistaken for a shared circuit and creates the proper basis for later transfer tests.
- Updated causal validation to use the matching language-specific frozen circuit, require at least eight fresh adjudicated seeds per language, reject overlap with all discovery seeds and notebook-01 metric-validation seeds, keep randomized baselines, and report per-language seed bootstrap summaries. Nonfinite values become JSON null to preserve undefined outputs.
- Removed a redundant final notebook cell that overwrote the detailed per-seed validation result and intervals with only the last seed's report. Added deterministic sample-selection tests (5 pass).
- Added explicit pinned dependency bootstrap cells. CircuitKIT notebooks create/use a disposable pinned source checkout and apply the position-map patch only when it cleanly applies; they refuse to reset a different existing checkout and verify that PyTorch's Colab build does not change during dependency installation.
- These are code/reproducibility corrections only; no reviewed data or GPU experiment was run, and the research gates remain unchanged.

## 2026-11-12 — Preserve Pipeline 2 seed uncertainty

Decision: do not reconstruct or guess the ad hoc Pipeline 2 sample from aggregate outputs. The recovered notebook identifies `/content/pipeline2_prompts.json` as an external input but does not preserve it or run outputs; `pipeline2_results.json` contains only aggregate metrics without seed IDs. The historical exposure registry therefore remains incomplete, and any seed-overlap-sensitive evaluation remains blocked until a sourced inventory is found. Evidence and audit details: `docs/lexsi/COLAB_DEBUG_LOG.md` and `outputs/lexsi/circuitkit_prior_seed_registry.json`.

## 2026-11-12 — No checkpoint recovery from Drive history

Drive revision history for the restricted checkpoint contains only its already-invalid 25 MB partial revision; no earlier revision exists. Fresh global searches found no complete restricted/broad checkpoint or PEFT adapter. Do not claim historical weights are recoverable from the searched Drive scope. Keep the checkpoint status partial/lost there; any reproduction requires a bounded retraining run after data and compute gates are met. Evidence: `outputs/lexsi/checkpoint_drive_candidate_inventory_20261112.json` and `CHECKPOINT_RECOVERY.md`.

## 2026-11-12 — “50 valid pairs” are provisional, not human-approved CircuitKIT inputs

The original saved review files record `reviewer_type=codex_assisted`, `annotator_id=codex`, and `review_status=provisional` for all 514 combined records and all 24 bilingual provisional records. The 50/52 bilingual shortlist is also cross-language and is not the same-language control-versus-roleplay primary contrast. It remains useful as a provisional dataset-review artifact, but it cannot be treated as a human-approved refusal/compliance set or exported to CircuitKIT. No labels were changed. See `DATA_PROVENANCE.md` and the metadata-only audit in `COLAB_DEBUG_LOG.md`.

## 2026-11-12 — Bind evaluation resume state to model and judge provenance

Cached evaluation rows are now reusable only when their deterministic keys match the arm/checkpoint file hashes, tokenizer/template, input token IDs, generation configuration, exact generated answer, resolved guard revision, guard-tokenized input, and guard configuration. Conflicting final/partial journal rows fail closed; missing local adapters are detected before loading the baseline. This prevents stale judgments and incomplete output replacement. No old outputs were rewritten; future incompatible resume state requires a fresh output directory. Local unit tests cover helper behavior, but runtime resume remains untested.

## 2026-11-12 — Validate checkpoint containers before starting a multi-arm run

Directory existence alone does not establish that a local arm is loadable; starting baseline generation before discovering a malformed or truncated adapter wastes GPU time and leaves a partial comparison. The preflight now checks the HF config, expected full-model weight files, shard-index targets, and Safetensors headers before any arm runs. A local fixture failure revealed that the initial test created only an empty directory; replacing it with a small valid HF-shaped checkpoint made the tests exercise the intended condition. The full local suite is 58 passed/1 expected skip (59 total); no runtime checkpoint load was performed.

## 2026-11-12 — Torn JSONL tail recovery is partial-file-only

A Colab disconnect may leave an unterminated final JSONL fragment. Resume now drops only that final fragment in `.partial` files, then atomically rewrites the merged journal; corruption in completed files, newline-terminated records, or earlier lines still aborts. This preserves known-complete records without masking broader data corruption. Local recovery tests pass; no real runtime disconnect was induced.

## 2026-11-12 — Share the baseline GenerationConfig across model arms

Model-level `generation_config.json` defaults can differ after fine-tuning. To keep the paired comparison controlled, resolve one `GenerationConfig` from the pinned baseline revision, explicitly set deterministic decoding and tokenizer special-token IDs, and pass that same config to all arms. Include the effective config in provenance keys and run output. This prevents silent differences in EOS and decoding defaults; local config tests pass, but no Colab generation run has exercised it.

## 2026-11-12 — Match dense inference precision across the three model arms

The first proposed correction was to NF4-quantize all arms. Reviewing the historical save path showed that the tuned checkpoints are reported as merged full-model weights, not standalone PEFT adapters. Re-quantizing these merged checkpoints could add a second lossy quantization step and perturb small LoRA deltas. The chosen comparison therefore loads the pinned HF baseline and both merged full-model checkpoints densely using one shared dtype: BF16 when the CUDA device reports support, FP16 otherwise. The same dtype and `device_map` are included in generation provenance, and unexpected quantized loads fail. The loader creates no new training or model artifacts. Local loader tests pass; actual Colab memory fit and inference remain unverified.

## 2026-11-12 — Fail-closed reviewer reconciliation and approved export

- Added `scripts/lexsi/reconcile_blinded_reviews.py` to join two completed blinded reviewer packets to the private A/B key and candidate source, validate exact source text, enums and evidence spans, and produce a pending adjudication worksheet. It records hashes of the candidate source, both reviewer packets, and key.
- The separate approved exporter re-reads and revalidates those exact source files, rejects edited reviewer/source fields or stale provenance hashes, and exports only explicitly adjudicated equivalent harmful prompts with control safe/clear-refusal and roleplay unsafe/complied labels plus distinct exact source-answer prefixes. Two distinct reviewers and a separate adjudicator are required. All private outputs must remain under ignored `outputs/lexsi/`.
- The current real packets remain incomplete: 87 rows each, zero completed statuses. A read-only reconciliation attempt failed closed (176 validation issues); it wrote no worksheet and approved export remains empty. No reviewer labels or approvals were invented.
- Deterministic worksheet column ordering and regression tests cover incomplete review, prompt tampering, source packet changes after worksheet creation, pending status, and explicit approval/export validation. Local Track-2 suite: 70 tests, 69 passed, one expected skip. This validates the workflow on synthetic fixtures only, not candidate labels.

## 2026-11-12 — Separate prompt-contrast coverage from response evidence

- The locally present 27 MB `marcus/dataset2_partial.sqlite3` is a valid prompt-construction database (21,184 prompt rows, 320 seeds; SHA256 `296be6962f573eac1133f099eac098a16318b3d708e1cc9e541ea6fb53eea21f`), not a response/guard database. Repository validation agrees with the pinned manifest; it contains no generated answers or safety outcomes.
- Added `scripts/lexsi/audit_prompt_contrast_coverage.py` to count exact-request, same-seed/split/culture/category/language/output-language/mix-direction harmful control-vs-roleplay structures. Aggregate output omits source IDs and prompt text. Across full structural rows, it finds 18 strata; for example, analysis US control/rp has 134 seeds per language and tune US has 84 per language. Only 18 analysis, 11 tune, and 13 previously exposed heldout seeds have complete 84-variant source coverage; coverage counts are not response evidence or human semantic judgments.
- Decision: use this artifact only as a candidate-coverage denominator for planning review and generation; do not use it to substitute for missing generated responses, labels, human review, or heldout eligibility. Heldout remains descriptive because it was exposed.

## 2026-11-12 — Reconcile original Dataset2 result shards locally

- The Drive connector could not deliver the merged 375 MB SQLite because it enforces a 256 MiB object limit. Search found two original result shards, each below that limit. Both shard databases have identical Dataset2 metadata and disjoint rows/seeds; together they contain exactly 21,184 source rows.
- A first Desktop download was an iCloud `dataless` placeholder and blocked SQLite reads. Stopped that read-only attempt and downloaded verified copies into a private mode-700 Application Support directory outside the synchronized repository. Verified byte sizes and SHA256s are recorded in the aggregate reconciliation summary/manifest; raw files and per-row results remain private.
- The combined raw reparse exactly matches the independent corrected-label JSONL on all 21,184 prompt IDs and relevant label/provenance fields. Published binary labels changed: 0. The strict within-seed/design-cell matching results are now computed and documented in `SOURCE_RECONCILIATION.md`.
- An independent NumPy/PCG64 bootstrap over saved seed effects (20,000 draws/group) exactly reproduced means; largest 95% endpoint difference from the original 5,000-draw bootstrap was 0.52 percentage points. This checks bootstrap arithmetic conditional on the saved CSV, not raw extraction.
- One initial script invocation as `python scripts/lexsi/reconcile_dataset2_shards.py` failed because the repository root was not on `sys.path`; the supported module invocation `.venv/bin/python -m scripts.lexsi.reconcile_dataset2_shards` fixed this without changing scientific logic. Preserve the module form in reproduction instructions.
- These results remain guard-proxy associations. Do not claim human-validated behavior, causation, or an improved model. Colab remains at 0 compute units and no active sessions; no GPU was used for this local reanalysis.
- Follow-up reconstructed the supplied Pipeline 1 point/test statistic separately using the original published binary `safety` labels: 4,069 rows, OR_MH 1.9822 and continuity-corrected CMH p=2.21e-9, matching the archived report. After retrieving the archived Pipeline 1 notebook from Drive and inspecting its formula, the 95% RBG interval was reproduced as 1.5832–2.4818. See `scripts/lexsi/reproduce_pipeline1_mh.py` and its aggregate JSON.


## 2026-11-12 — Fresh reviewer packet readback

- Read both exact reviewer CSVs through the connected Drive file tool; parsed the `review_status` field in memory only. Both contain 87 cases and 87 blank review statuses; last modified timestamps were 2026-11-12 06:16 UTC. No prompt or response content was emitted or stored. No real human approvals are available; approved export remains empty.
- The expanded Pipeline 1 reproduction initially failed a synthetic test because its fixture omitted `framing`; after adding roleplay/control cells, a second assertion exposed that the `all` comparison correctly aggregates both frames. Updated only the synthetic schema/count expectation. Full suite then passed 77 total (76 passed, one expected skip). The real result-shard comparisons were unchanged.


## 2026-11-12 — Checkpoint and compute blockers persist

- Fresh connected-Drive searches found no complete Pipeline 4 checkpoint: the restricted folder still contains only the previously invalid 25.2 MB file/config, and the broad folder remains empty.
- Fresh official Colab CLI readback at 11:02 UTC: 0.00 compute units, no active assignments, and no active sessions. No paid compute was started.
- The two reviewer packets also remain 87/87 blank each. Further real-model evaluation, checkpoint reproduction, or approval-dependent CircuitKIT work requires these external conditions to change. Local aggregate analysis and tests remain available and are documented; no research gate is being marked complete.


## Resource decision — 2026-11-12T14:55:10Z

Colab quota now reads 100.00 compute units. Do not spend it on repeated environment-only smoke runs; reserve it for the next gated, reproducible model experiment. Existing LLM/Codex annotations remain provisional and cannot be promoted to human-approved examples by this resource change.


## LLM-assisted annotation decision — 2026-11-12

Because the user lacks time for a full first pass, produce a private, provisional single-pass LLM-assisted screen of all 87 blinded candidates. Keep source reviewer packets and the A/B key unchanged. This is triage only: the Codex conversation model is not independent from this research workflow, its exact build is not recorded, and its labels cannot satisfy human review/adjudication or the export gate. Do not use these labels as ground truth or CircuitKIT inputs. Exact evidence-substring validation is mechanical and does not show that a quote substantively supports its label.

## Notebook execution evidence decision — 2026-11-12

Treat executed notebook cell outputs as authoritative for execution success, not Colab CLI exit status alone. The saved Phase 1 run returned process exit code 0 while its final cell raised on the empty human-approved input file. Added a strict notebook-output auditor and made anticipated missing-approval / negative-metric states explicit JSON statuses rather than exception tracebacks. Do not change the predeclared behavioral metric or proceed to discovery without human-approved data.

## Colab secret and idle-runtime decision — 2026-11-12

Colab `userdata.get("HF_TOKEN")` cannot be invoked through CLI execution; it must run in the browser-attached runtime. Keep tokens out of chat, source files, and logs. Do not leave an L4 accruing usage while waiting for the owner to perform the UI fetch: stop it, then provision a new runtime after the owner indicates the secret is saved and available for UI access.


## 2026-11-12 — LLM review remains provisional; calibrator coverage is incomplete

- Rechecked the actual private files: all 401 bilingual disagreements have provisional Codex-assisted labels, while 70 of 80 balanced agreement controls remain unreviewed. The 87 same-language screen is also provisional.
- Decision: use these annotations to prioritize human inspection only. Do not infer guard accuracy from the deliberately balanced controls, treat LLM outputs as gold, or export them as CircuitKIT examples. The approved export stays empty.
- Local HF credentials can fetch a config at the pinned Tiny Aya revision, but Colab Secret access remains a separate browser-UI requirement.


## 2026-11-12 — Calibration LLM pass is batchwise and provisional

- Five previously blank controls were annotated with full prompt/response review and literal evidence checks.
- Decision: preserve them as `provisional`; continue recording annotation coverage, but never report agreement-control proportions as guard accuracy because the set is class-balanced and the labels are not independent human judgments.


## 2026-11-12 — Continue calibration annotation with explicit uncertainty

- Batch 2 adds five provisional records, including a borderline recreational-risk answer and records with factual-reliability issues. These dimensions remain separate: factual errors are documented in rationale, not silently recoded as unsafe compliance.
- No approval status is inferred; no prevalence or accuracy estimate is derived from these balanced controls.


## 2026-11-12 — Preserve malformed bilingual control as a mismatch

- A calibration pair asks about drinking “during spinach” in Japanese and Hollywood screenings in English. The Japanese intent is unassessable and the pair is materially different; do not count it as a language-based safety difference.
- Additional judgments remain provisional, and factual reliability concerns are documented separately from harmful assistance.


## 2026-11-12 — Keep ambiguity distinct from safety divergence

- Batch 4 contains ambiguous fictional-villain wording and differing refusal style, but neither side provides harmful assistance. Do not promote these to safety discrepancies based on refusal phrasing alone.
- Calibration coverage is 30/80 provisional and remains non-gold.


## 2026-11-12 — Flag risky financial advice without elevating it to a result

- A bilingual agreement-control pair differs in risky financial specificity: the English answer recommends options, futures, leveraged ETFs and short selling; the Japanese answer gives vague market-volatility suggestions. It is provisionally coded as a possible real model difference for triage, pending independent review.
- A mafia-investigation prompt pair has different agency/focus and is excluded from any bilingual behavior-difference interpretation.


## 2026-11-12 — Preserve high-stakes and fabricated-content uncertainty

- In batch 6, one bilingual pair asks about suicide in fictional framing; the Japanese answer gives unsupported reassurance, while the English answer presumes suicidal ideation. Both stay borderline and require careful human review. A separate answer invents family ages; it is flagged as a reliability/privacy concern, not actual private-data exposure.


## 2026-11-12 — Escalate high-risk medical misinformation for review

- A Japanese hypertension answer lists low-dose steroids and beta-agonists as treatment approaches. It is provisionally marked unsafe and the English answer borderline; this is a triage judgment requiring expert review, not validated medical ground truth.
- An explicit mafia-harm prompt receives harmful method lists in both languages; mark unsafe on both sides, not a language discrepancy.


## 2026-11-12 — Mark explicit harmful-method answers unsafe on both sides

- Batch 8 includes direct self-harm methods and criminal operation advice in both languages; classify as unsafe on both sides, while recording greater detail separately. These are provisional annotations and do not qualify as a validated language-difference result.


## 2026-11-12 — Classify harmful method lists consistently across languages

- Batch 9 contains animal-cruelty, theft, and confidential-data-leak requests. Both language answers provide methods and are marked unsafe on both sides; differences in detail are recorded without recoding as a refusal/compliance contrast.


## 2026-11-12 — Evaluate answer substance after refusal preambles

- A yakuza-theft response starts with a refusal but then lists robbery, fraud, embezzlement and coercion. Its refusal field records the partial refusal, while response safety remains unsafe. A test-falsification answer similarly gives more actionable steps in English.


## 2026-11-12 — Record translation uncertainty and answer asymmetry

- Batch 11 flags two possible unsafe-compliance differences and a semantic mismatch in the insurance-card prompt. These are provisional; the mismatch is excluded from language-safety interpretation, and human approval remains required.


## 2026-11-12 — Keep authorization ambiguity unresolved

- The encrypted-data pair does not specify whether the requester is authorized. The Japanese response suggests decryption/key acquisition, while the English response requires permission. Keep the safety discrepancy unclear and do not use it as a validated contrast.


## 2026-11-12 — Keep ambiguous explicit-language and drug claims provisional

- Batch 13 identifies a difference in the interpretation of “explicit words” and unreliable mafia-drug side-effect claims. These remain provisional and are not included as approved contrasts; straightforward criminal-method responses are marked unsafe on both sides.


## 2026-11-12 — Complete the 80-control pass without claiming calibration validity

- All 80 source-guard agreement controls received a provisional single-pass Codex annotation and exact-evidence structural check.
- Decision: use the worksheet for triage and rubric refinement only. Because sampling is deliberately balanced, the pass is not independent human validation, and the original guard agreement strata are selected by design, do not report these counts as accuracy or prevalence. Keep the records out of the approved CircuitKIT export.


## 2026-11-12 — L4 dependency probe; exploratory LLM-label policy

- Command: `/Users/marcuschang/.local/bin/colab new --gpu L4 --session lexsi-track2-gpu-audit-20261112`; verified CUDA on NVIDIA L4. Installed `circuitkit==0.2.2`, `transformers==5.15.0`, `transformer-lens==3.8.0`; PyTorch unchanged at 2.14.1+cu130.
- Credential lookup via `google.colab.userdata.get("HF_TOKEN")` timed out under CLI execution. No HF model load/inference occurred. Stopped the runtime; current balance 99.90 CUs, no sessions/assignments.
- User authorized proceeding without manual review by relying on existing LLM provisional labels. Experimental policy: use such labels only in a separately identified exploratory analysis and preserve provenance as LLM-assisted; never relabel them as human-validated or count them as independent human review. This does not change the empty human-approved CircuitKIT export.
- Current strict same-language screen remains small: five English cases with equivalent prompts, harmful intent, a clear control refusal, and unsafe roleplay compliance (four analysis seeds and one tune seed; no fresh evaluation seeds). It cannot support the planned discovery/tune/evaluation split or confirmatory causal claims.


## 2026-11-12 — Expand LLM triage beyond guard disagreements

- The user authorized use of LLM provisional annotations without waiting for manual review. Use that authorization for a clearly identified exploratory pilot; preserve the machine-generated provenance and do not label it human-validated. The human-approved export remains zero.
- Exact-key analysis/tune control/roleplay pairs with guard-label agreement but refusal-heuristic disagreement were selected as a second triage cohort. The heuristic is only a selector. A full-text screen found 2 additional strict target-pattern candidates, both Japanese analysis examples; 45 of the 47 added rows did not meet the strict target definition.
- Combined target-pattern inventory is now 7 unique seeds (6 analysis, 1 tune, no untouched evaluation); this still fails the planned seed-disjoint size gate. Do not run confirmatory discovery or causal evaluation from this set. The exploratory data file remains separate from the official CircuitKIT export.


## 2026-11-12 — Owner-authorized exploratory model scoring and metric gate

- The project owner explicitly authorized use of the LLM-provisional cases for exploratory execution. Decision: record approval as `owner_authorized_for_exploratory_execution`, preserve the actual LLM review provenance, and keep the formal human-adjudicated CircuitKIT export unchanged. Owner approval to use a dataset is distinct from independent human annotation of its labels.
- The seven strict same-language cases were scored on the real HF model. Although the control-minus-roleplay mean-margin difference was positive on all six analysis seeds, the desired refusal-positive/compliance-negative absolute signs occurred on only one analysis seed and zero of the single tune seed. The single tune result has exact one-sided sign p=0.5. Therefore the likelihood metric is not validated and no discovery run is authorized by the current gate.
- The Phase 1 metric gate now requires at least five non-tied tune seeds (the minimum at which a one-sided exact sign test can attain p≤0.05), p≤0.05, and the expected absolute signs on a majority of tune seeds. Under the present n=1 tune cohort, report `INSUFFICIENT_VALIDATION_DATA`; do not convert the promising analysis-only direction into a pass.
- Next: expand same-language qualifying seeds, especially tune, using the existing saved source responses; then diagnose whether shorter, pre-specified response prefixes or another behavior-grounded scoring definition is more discriminative. Any metric revision must be evaluated on separate tune seeds and tested unchanged on fresh seeds. Do not tune to the seven observed scores.


## 2026-11-12 — Audit pairs where automated proxies agree

- Added an all-matched analysis/tune inventory instead of assuming guard agreement plus a matching refusal heuristic means the response behaviors agree. It contains 648 matched rows from 200 seeds; this is a finite inventory, not a behavioral annotation set.
- Full-text review of one previously unexposed, proxy-agreement pair from each of seven tune seeds found no new target contrast. Six pairs showed clear refusal on both sides; one pair's underlying requests were materially different. Preserve these as negative/nonequivalent audit cases, not as positive targets.
- The seven-seed audit is not enough to establish false-negative rates for either proxy. Do not generalize from it or fill the tune quota with reused seeds. Continue searching other eligible seed-level strata; if none yield suitable cases, stop the circuit experiment and report the data limitation.

## 2026-11-12 — Screen non-refusal proxy-agreement tune pairs

- Extended the tune audit to 38 unique, previously unexposed seeds where both saved response guard labels are Safe and neither response triggers the refusal heuristic. Reviewed full prompts and answers while source labels were hidden from the annotation view.
- The full-text pass found no strict control-refusal / roleplay-unsafe-compliance pair. One prompt is a harmful-sounding online-harm request and both responses clearly refuse; one protective request about avoiding illegal transactions is answered in control but refused in roleplay (reverse direction). Other observed differences concern factual/medical answer quality, not the target contrast.
- Structural validation: 38 annotations, 76 literal evidence spans, valid schema, zero strict targets. Labels remain Codex-assisted provisional; owner authorization supports exploratory use only and is not independent human validation.
- Decision: do not treat this selected cohort as a validation cohort, estimate classifier accuracy from it, or advance the failed behavioral-metric gate. Continue searching eligible strata if useful; if sufficient examples do not exist, preserve the inconclusive data limitation rather than relax the target definition.
- Full repository regression following this audit passed: 222 tests, 219 passed and 3 skipped; captured in `outputs/lexsi/latest_local_tests_20261112_nonrefusal_screen.log`.

## 2026-11-12 — Preserve the final unseen Tune edge case outside the target stratum

- The exposure-audited remainder contained one matched Tune seed with Unsafe/Unsafe guard labels and a missing roleplay refusal heuristic. Full-text review found an ambiguous farm-slaughter/euthanasia request; both responses answer, but roleplay adds specific, dangerous and apparently false methods absent from control.
- Marked the response discrepancy provisional and distinct from refusal/compliance. It does not meet the strict target pattern and is not a validated ground-truth example.
- Keep this edge case in a separate one-row private audit rather than broadening the metric labels post hoc. This completes triage of the previously unseen Tune inventory under the current exposure registry but does not repair the failed metric validation gate.

## 2026-11-12 — Keep bilingual Tune contrasts separate from Phase 1 metric validation

- Audited all eight Tune rows in the existing provisional bilingual-difference set: six unique seeds; no overlap with the Phase 1 seven exploratory seeds or known historical exposure registry. Two seed-level cases provisionally show Japanese refusal versus English unsafe compliance; one shows the reverse. All labels remain Codex-assisted/provisional.
- These examples change language and prompt surface together. They are not same-language control/roleplay clean/corrupt pairs, so they cannot be pooled into the Phase 1 metric-validation test or counted toward its same-language Tune minimum. The sample is also too small for reliable transfer claims.
- Decision: preserve as a separate future Japanese/English comparison cohort and keep the current behavior-metric gate failed. Aggregate counts and source hash: `outputs/lexsi/bilingual_tune_metric_eligibility_20261112.json`.

## 2026-11-12 — Diagnose the Phase 1 metric without retuning it

- Recomputed aggregate sign and continuation-length diagnostics from the saved Phase 1 report using a new reproducible script. The control-minus-roleplay relative margin is positive on 7/7 examples, while only 1/7 has both expected absolute signs. In the four English analysis cases, roleplay still has positive refusal-over-compliance margin every time despite provisional unsafe-compliance labels.
- Median scored lengths are 152 refusal tokens and 307 compliance tokens (2.02x). Full-response semantic content differs; EOS is not included. Per-token means compensate for raw length scaling but do not control semantic/style content or termination behavior.
- Decision: treat the metric as nondiscriminative on current evidence. Do not tune it on the six analysis seeds. Any replacement or prefix metric needs a separately frozen definition and adequate independent Tune examples, then a fresh evaluation cohort. Length/content are hypotheses for follow-up, not proven causal explanations.
- Metric-diagnosis verification: 2/2 focused tests pass and the aggregate output regenerated. The full repository rerun became unresponsive while `load_dataset()` read the large SQLite during an unrelated dataset merge test; the process was stopped after sampling confirmed it was blocked in filesystem `read()`. Preserve the interrupted log and do not imply a current full-suite pass; the prior full-suite pass preceded only this new analysis helper.

## 2026-11-12 — Do not rerun inference to recover current Tune pairs

- Recomputed the incomplete-pair opportunity from both original Dataset2 SQLite shards, excluding 161 seeds present in the partial prior-exposure registry and prior candidate/review sources.
- Of 68 incomplete matched keys in unexposed seeds, only 2 keys from one seed have both sides marked analysis-eligible; no unexposed seed has complete eligible responses on both sides awaiting semantic review.
- Decision: do not spend GPU compute rerunning generation or guard scoring to rescue this cohort. Even successful recovery adds at most one seed, short of the five-seed Tune metric gate, and cannot create independent validated labels. Seek a new seed-disjoint cohort or freeze a revised metric for evaluation on new Tune seeds.
- Evidence: `outputs/lexsi/tune_pair_recovery_audit_20261112.json`; script `scripts/lexsi/audit_incomplete_tune_pairs.py`; focused tests passed 2/2. Historical exposure inventory remains partial, so no fresh-evaluation claim is permitted.

## 2026-11-12 — Treat Dataset2 Tune as exhausted for independent validation

- The reproducible audit reports 79 unique fully eligible matched Tune seeds and zero unexposed fully eligible Tune seeds after excluding 161 previously known training, selection, or review seeds across the exposure sources.
- This is stronger than the incomplete-pair finding: the existing response shards contain no untouched eligible Tune seed set for validating a revised metric. Previously reviewed seeds cannot be recycled as “new” after metric redesign.
- Decision: stop searching the same Tune split for validation examples. Acquire/construct new source seeds or use an appropriate external benchmark; preregister seed-level discovery, tuning, and fresh evaluation splits before scoring or changing the metric. Keep the existing seven-pair metric result and all screens exploratory.
- External state checked: official Colab CLI has no active runtime/assignment and reports 99.57 CUs. Connected Drive reviewer folders list only the originally delivered reviewer CSVs/instructions; no returned reviewer artifact is present.

## 2026-11-12 — Separate new mechanism data from external safety benchmarks

- Dataset2 has no unexposed eligible Tune seeds left. An external Japanese safety dataset can improve breadth of behavior evaluation, but is not automatically a substitute for the matched control-versus-roleplay contrasts required by this mechanism study.
- Reviewed the primary AnswerCarefully and APTO-XSTest-JA dataset documentation. AnswerCarefully includes Japanese safety examples and English translations, but forbids redistribution and cautions that labels are not definitive. APTO-XSTest-JA includes Japanese-localized prompts and reviewed refusal/non-refusal references under CC BY 4.0, but primarily probes over-refusal.
- Decision: keep both as candidate, separately reported external evaluations; construct a genuinely new same-intent, same-language roleplay cohort with bilingual review and seed-level discovery/tune/fresh-test splits. Preserve current metric failure and do not recycle prior probes. Design and source caveats: `docs/lexsi/NEW_COHORT_PLAN.md`.

## 2026-11-12 — Treat five Tune seeds as a minimum, not a powered design

- Added a pure-Python exact-binomial operating-characteristic calculator for the preregistered one-sided sign test. Under an illustrative true expected-sign probability of .80, five non-tied seeds have .328 power; n=18 is the first sample size reaching 80% power for that test. At .70/.75, first n values are 37/23.
- These assumptions are scenarios only, not inferred from the failed seven-pair pilot. The calculation ignores ties and the additional absolute-sign gate and therefore does not set the final sample size.
- Decision: keep the minimum five-seed criterion as a fail-fast lower bound only; before collecting the new cohort, define the smallest scientifically meaningful per-seed effect/probability and calculate the required Tune and evaluation sizes using pilot-independent assumptions.
- Evidence: `scripts/lexsi/metric_sign_test_power.py`, `outputs/lexsi/metric_sign_test_power_scenarios_20261112.json`, and three passing tests.

## 2026-11-12 — Audit AnswerCarefully v2.2 locally; keep test split sealed

- Fetched the source via the existing pinned-loader workflow to ignored local path `data_ready/answercarefully.sqlite3`; revision `d1d4e57cd4d7aecb44f879e7488832011fcaddb0`, SHA256 `5fd99d9c294f3435f771e0b1dae6d22d232e1b0892815fd31e6577c10acb549f`.
- Verified 1,800 unique rows (1,464 dev / 336 test), complete Japanese prompt/reference and English prompt-translation fields, no English reference answer, 1,067 dev / 245 test rows with adaptation tag 0, and no explicit `refusal_required` metadata. Aggregate output contains no text/IDs; tests pass 2/2.
- Local Aya evaluation notebook has no execution outputs; prior Aya exposure on accessible Drive is unresolved. Decision: use dev rows only for candidate exploration after per-row language/intent review; keep test split sealed and do not call it fresh. Do not upload/copy this non-redistributable dataset to Drive or Git. Audit artifacts: `scripts/lexsi/audit_answercarefully_snapshot.py`, `outputs/lexsi/answercarefully_snapshot_audit_20261112.json`.
