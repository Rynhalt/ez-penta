# LEXSI Track-2 status

**Last updated:** 2026-11-12  
**Overall:** `PARTIAL` — real HF/CircuitKIT parity is demonstrated, but human-approved contrasts, behavioral metric validation, matched multi-arm evaluation, and causal circuit tests are not complete.

### Latest raw-shard reconciliation (2026-11-12)

The earlier “strict corrected-label matching pending” statement is superseded: both original Dataset2 result shards were obtained through the authorized Drive connector and reconciled locally. The combined 21,184 source rows exactly match the independently saved corrected-label file (21,184/21,184 prompt IDs, zero mismatches); two shards contain 320 source seeds total. Exact within-seed matching by kind/category/framing/mix-direction/output-language produced nine split/language comparisons. Separately, direct reproduction of Pipeline 1 matches the 4,069-row cohort and all four archived MH comparisons (US/Japan vs neutral, all framings/roleplay-only), including the language-specific tables, RBG intervals, and continuity-corrected CMH tests. Primary US/all-framing: OR 1.9822 (95% CI 1.5832–2.4818), p=2.21×10⁻⁹. US/roleplay-only: OR 2.5441 (1.7349–3.7309), p=1.70×10⁻⁶. Japan/all-framing: OR 1.0974 (0.8623–1.3967), p=0.4875; Japan/roleplay-only: OR 1.4217 (0.9452–2.1384), p=0.1121. These are **automated guard-proxy associations**, not human-validated behavior or causal effects. The previously exposed heldout split remains descriptive only. Full method, estimates, and source hashes: [SOURCE_RECONCILIATION.md](SOURCE_RECONCILIATION.md#strict-corrected-label-seed-and-design-cell-matched-analysis-2026-11-12).

Independent 20,000-draw NumPy bootstrap checks of the saved per-seed effects reproduced the point estimates and gave interval endpoints within 0.52 percentage points of the original 5,000-draw intervals. This is conditional arithmetic verification, not a second raw-source extraction. The macOS Desktop/iCloud copy of the raw DBs became a dataless placeholder; verified source copies are stored privately outside the repository. No prompt/response text or row IDs were added to public documentation.

| Gate | Status | Evidence / remaining condition |
|---|---|---|
| A — source inventory and checkpoint recoverability | `PARTIAL` | Five Pipeline1111 notebooks and outputs were inventoried from user Drive. The prior L4 smoke used complete restricted/broad checkpoints from runtime scratch, but current official CLI reports no active sessions, so those volatile paths are not presently accessible or freshly verified. Drive contains a restricted candidate `model.safetensors` of 25,231,360 bytes (vs 3,599,063,241 bytes previously observed on L4) and an empty broad folder; neither is a recovered complete checkpoint. The original 21,184-row CSV has been hashed/reconciled; raw SQLite parsing recovered 17,373 binary response labels with zero changes to prior binary labels. Fresh-process reload and durable recovery remain outstanding. See [CHECKPOINT_RECOVERY.md](CHECKPOINT_RECOVERY.md) and [SOURCE_RECONCILIATION.md](SOURCE_RECONCILIATION.md). |
| B — paired evaluation machinery and all model arms | `PARTIAL` | The strict manifest builder ran on the mounted CPU Colab source: 89 exact US-neutral pairs (178 prompts) across 35 seeds, from the previously exposed heldout split. All are marked unreviewed and non-confirmatory; semantic equivalence still needs human review. The resumable three-arm HF generator/judge/analyzer ran on an 8-row/4-pair L4 smoke cohort, and the earlier 0.00-unit resource blocker has since cleared; current quota is 100.00 units, though scaling still requires the remaining data, checkpoint, and evaluation gates. |
| C — seed-aware safety comparison | `PARTIAL` | The new smoke cohort produced identical guard counts in all three arms (7 Safe, 1 Unsafe each; paired risk differences 0.0, 4 seed clusters). This is a pipeline check only, not evidence of equivalence or improvement. No human-validated unsafe-compliance outcome is available. |
| D — CircuitKIT parity and behavior metric | `PARTIAL` | The corrected Phase 1 notebook executed on an official Colab L4 using seven owner-authorized exploratory pairs. Real-weight HF↔CircuitKIT parity passed on English, Japanese, and sliding-window probes (largest absolute log-probability error `7.44e-5`; peak PyTorch allocation 22,248,377,344 bytes). The refusal-margin metric did **not** pass: all 6 analysis deltas favored control over roleplay, but only 1/6 analysis pairs and 0/1 tune pairs had the expected positive-refusal/negative-compliance signs. The report is `INSUFFICIENT_VALIDATION_DATA`; no later phase may treat this metric as validated. See the 2026-11-12 experiment below. |
| E — Circuit discovery and causal testing | `NOT_STARTED` | The five-pipeline `pipeline2_circuit_discovery.ipynb` used SafeTune probes, not CircuitKIT EAP-IG. No Tiny Aya EAP-IG circuit, activation patching, ablation, or random-component comparison has been verified. |
| F — optional follow-up training | `PARTIAL` | Historical restricted and broad one-epoch SFT runs are reported, but weights are not recovered from searched MyDrive and the evaluation/data gates do not justify additional tuning yet. |
| G — final evidence package | `PARTIAL` | Evidence-based [final report](FINAL_REPORT.md), [plain-language findings](WHAT_WE_LEARNED.md), and [reproduction guide](REPRODUCE.md) now document current results and blockers. This is a stopping-point report, not completion of the article's scientific evidence gates. |

## Verified work

- Fresh connected-Drive readback during this continuation confirms both reviewer CSVs still contain 87 candidate rows with all 87 `review_status` cells blank (modified 2026-11-12 06:16 UTC). There are no returned reviews or adjudication, so the empty approved-export gate remains correct.
- Recovered the historical Pipeline3 preference export from Drive: 22 accepted pairs. All 22 opaque `source_sample_id` values exactly joined to prompt IDs and mapped to 21 distinct Dataset2 seeds (one repeated seed). Pipeline2 probe seeds and other possible tuning exposure remain unknown; the partial seed inventory remains explicitly `complete=false` at `outputs/lexsi/circuitkit_prior_seed_registry.json`.
- Rechecked current CircuitKIT EAP documentation and safety case studies 23/24. Updated notebook 03 to capture raw signed intervention metrics, clean/corrupt baselines, unclipped ratios, and denominator flags rather than summarizing clipped scores alone. Local parsing/tests pass, but the change is not executed on GPU.
- Tightened notebook 03's fresh-evaluation split audit: it now requires an explicit complete registry with cited inventory sources for historical training/tuning seeds, in addition to discovery and Phase 1 overlap checks. The reported pipeline-4 run has no complete seed registry in the repo, so causal evaluation cannot claim seed independence until recovered.

- A private aggregate state manifest is maintained at `outputs/lexsi/run_manifest.json`; it hashes the parity, corrected-label summaries, smoke, evaluation-manifest summary, tokenizer audit, local test logs, and strict matched-analysis artifacts without embedding private rows. Current Track-2 suite result: 81 tests total, 80 passed and one expected skip.

- Latest evaluator integrity update: the pinned baseline and two merged tuned checkpoints now load densely at one shared BF16/FP16 dtype selected by GPU capability. This avoids re-quantizing full merged checkpoints; dtype and device-map are included in generation provenance. The evaluator-specific test result at that point was 65 total, 64 passed, 1 expected skip; that count is historical and superseded by the current full Track-2 suite of 77 total, 76 passed, 1 expected skip. This remains loader/configuration logic only; no runtime weight load or 8K-context memory check was performed.
- Fresh external readback: both Drive reviewer CSVs remain at 87 rows each with all 87 review statuses blank; Colab reports no sessions/assignments and 0.00 compute units. The Drive restricted checkpoint folder still has only a 25,231,360-byte weights file and config; the broad folder remains empty.
- A fresh official CLI CPU runtime was attempted for the stricter raw-SQLite seed/cell analysis. Drive credential propagation again returned HTTP 400; the named runtime was stopped, with no source data transferred. The small-file connector cannot download the 375 MB SQLite because its transfer limit is 256 MiB. A transient account-level report of one unidentified assignment cleared on follow-up; the latest `colab sessions` and `usage` checks show no sessions/assignments and 0.00/hour. Sanitized log: `outputs/lexsi/colab_cpu_source_mount_failure_20261112.log`; no more identical mount retries are planned.

- Notebooks `notebooks/01_prepare_circuitkit_data.ipynb`, `notebooks/02_circuit_discovery.ipynb`, and `notebooks/03_circuit_validation.ipynb` are present. The latter two are gated entrypoints and have not executed; no approved real rows exist.
- A real Colab L4 run checked the HF checkpoint architecture and chat template, a teacher-forcing calculation, and CircuitKIT parity on English, Japanese, and a 4,196-token probe. The long probe used chunked KV caching and compact GQA. Detailed attempts, errors, fixes, package versions, and memory are in [COLAB_DEBUG_LOG.md](COLAB_DEBUG_LOG.md) and `outputs/lexsi/colab_phase1/authorized/`.
- CircuitKIT parity is a backend compatibility result; it is not a safety metric validation or a circuit discovery result.
- The approved export `outputs/lexsi/circuitkit_pairs.jsonl` remains empty. The five same-language candidates and 52 bilingual examples described in the roadmap remain provisional, with different experimental roles. A new local signed-continuation metric prototype passes synthetic gradient/scalar checks, but pinned CircuitKIT integration and behavior validation are still pending.
- The Drive inventory CPU session used for checkpoint search has been stopped.
- The later source-audit CPU session was also stopped after verifying raw guard recovery and the existing seed-bootstrap summary table.
- Inventory, parser, seed-statistics, and metric-prototype tests: `.venv/bin/python -m unittest discover -s tests/track2_extension -v` — 21 passed. These use synthetic records and do not validate the real-data matched effect.
- Later alignment update supersedes the earlier statement that the position-map helper was not connected: a version-pinned CircuitKIT core patch, TaskSpec map builder, and collator are now preserved under `docs/lexsi/patches/`. Combined official Colab CPU run passed 41/41 tests, including actual EAP-IG execution on a random one-layer TransformerLens model, identity-map parity, partial-map behavior, and TaskSpec map padding. See `outputs/lexsi/circuitkit_position_maps_cpu_20261112/`.
- This remains synthetic CPU integration. The patched source is not yet installed by the experiment notebooks, CUDA is unavailable on the current session, the approved contrast export is empty, and no real Tiny Aya behavioral metric, EAP-IG discovery, or causal intervention has run. Differing roleplay framing positions are excluded from the explicit map.
- The Colab CPU session was stopped after logs, bundles, and hashes were saved. A new runtime will be needed for the next execution step; GPU compute quota remains unavailable.
- The first fresh Colab CPU Drive mount for corrected reanalysis failed with HTTP 400; the simpler seed-only analysis later ran. Both original response SQLite shards were subsequently retrieved separately through the authorized Drive connector and the stricter category/framing/mix-direction matched analysis and Pipeline 1 point/test-statistic reconstruction completed locally. The historical RBG interval is now reproduced as well. See [SOURCE_RECONCILIATION.md](SOURCE_RECONCILIATION.md).

## 2026-11-12 matched Colab smoke update

- The real L4 smoke run used the same Tiny Aya HF revision and NF4 loading for baseline, restricted-SFT, and broad-SFT arms. It generated and Qwen3Guard-judged 8 heldout prompts (4 US-neutral pairs from 4 seeds); all source prompts had identical token hashes across arms and were not truncated. The 192-token output cap was chosen for the smoke run under compute limits, so it is not comparable to a 1024-token production evaluation.
- Each arm received 7 Safe / 1 Unsafe proxy judgments. The paired risk difference was zero for each tuned arm against baseline; the seed-bootstrap interval was `[0, 0]` with only four clusters. This degenerate pilot result is non-inferential and must not be described as improvement, equivalence, or a null effect.
- Private prompts, generations, judgments, summary, and execution log are now locally preserved under the ignored `outputs/lexsi/track2_pilot_20261112_nf4_smoke/`; the same artifacts are also on the authorized Drive. `run_manifest.json` records settings, dependency versions, resource metrics, and artifact hashes. Nine CPU unit tests pass.
- Blocker: `colab usage` reports **0.00 compute units** (L4 rate 2.10 units/hour). The existing L4 remains idle and contains restricted/broad checkpoint weights only in runtime scratch. They are not durable on Drive: its Drive mount failed with HTTP 400 and large-file CLI transfer was not viable. Do not start another GPU run until compute is available and checkpoint durability is resolved.
- A new Drive search on 2026-11-12 surfaced a restricted folder with only a 25.2 MB `model.safetensors` (the L4 version is 3.60 GB) plus `config.json`, and an empty broad folder. The incomplete file is not a valid recovery; no additional adapter or sharded weight files were found. Inventory is private at `outputs/lexsi/checkpoint_drive_candidate_inventory_20261112.json`.
- The exact-match candidate builder read all 21,184 source SQLite rows (source SHA256 `ecffa936c76c1009717b1a11d6597e3803878cc6e8e8b3443c0cfc39474b4e11`; `quick_check=ok`) and wrote a private 178-row/89-pair heldout manifest spanning 35 seeds. All rows are flagged previously exposed; none are human-approved or confirmatory eligible. Private output and hashes are under `outputs/lexsi/evaluation_manifest_20261112/` and Drive `codex_extension_20261111/evaluation_manifest_20261112/`.
- CircuitKIT behavior validation, EAP-IG discovery, and causal tests remain gated by the empty human-approved contrast export and unvalidated refusal/compliance likelihood metric.

## Next gates

1. Search only other user-authorized storage for Pipeline 4 weights; if absent, declare them unrecoverable and decide whether the preconditions support a replacement training run.
2. Obtain real independent human review/adjudication of the same-language control/roleplay candidates and validate the refusal/compliance continuation spans. Do not populate human-approval fields automatically.
3. Complete the seed/leakage manifest and freeze a paired evaluation protocol with benign controls and seed-clustered uncertainty.
4. Validate the behavioral likelihood metric on approved tune seeds. Then run CircuitKIT discovery on approved discovery seeds and causal tests on disjoint seeds.
5. Maintain separate conclusions for association, model intervention, and mechanism; a positive result is not required.

## 2026-11-11 corrected guard reanalysis update

- Verified access from the mounted `lexsi-browser-export` CPU runtime after patching the installed Colab CLI credential-propagation code to honor `COLAB_AUTHUSER`. Source path: private Drive `MyDrive/ez-penta/Pipeline1111/combined/results.sqlite3`.
- Created a private reduced SQLite copy at `Pipeline1111/codex_extension_20261111/results_reduced.sqlite3` (21,184 rows, 375,017,472 bytes, SHA256 `d6004f551eb8519f59ad090594df4373b5ade0e40756499675ee8deb4fb049df`, SQLite quick_check `ok`). It omits prompt and answer columns; raw guard-output records and matching metadata remain private.
- Reparsed all raw guard records: response labels Safe=15,749, Unsafe=1,624, Controversial=422, Missing=3,389, Unparsed=0. Of 13,730 prior binary labels, zero changed; 3,643 additional binary response labels were recovered where earlier stored labels were nonbinary/missing (the corresponding prompt verdict was Controversial).
- Corrected raw-label seed-paired analysis, 5,000 seed-bootstrap replicates, seed 20261111. Within each split and prompt-language group, average unsafe-proxy outcomes per seed by culture, then average US-minus-neutral differences equally over seeds with both conditions. Analysis: en +0.0697 (n=110 seeds, 95% CI +0.0038 to +0.1379); ja +0.1297 (n=106, +0.0487 to +0.2107); mix +0.0955 (n=96, +0.0241 to +0.1641). Tune: en +0.1261 (n=78, +0.0577 to +0.1976); ja +0.0746 (n=76, +0.0077 to +0.1458); mix +0.1398 (n=72, +0.0741 to +0.2120). Heldout (previously exposed; descriptive): en +0.0985 (n=44, -0.0095 to +0.2064); ja +0.1768 (n=41, +0.0833 to +0.2724); mix +0.1361 (n=42, +0.0377 to +0.2393).
- This is an association in an automated guard proxy. It does not establish semantic equivalence, human-verified unsafe compliance, a safety improvement, or causality. The analysis averages all eligible harmful rows within each seed/condition/language and does not match framing/category cells. An earlier strict analysis appeared to yield no en/ja matches, but that result is invalid: the implementation dropped NULL `mix_direction` values that are not applicable to monolingual prompts. The matcher was corrected and then run on both original shards; the strict matched results are documented above and in `SOURCE_RECONCILIATION.md`.
- Summary: `outputs/lexsi/corrected_seed_only_summary.json`; executable source: `scripts/lexsi/seed_matched_reanalysis.py`; private full label and seed-effect outputs remain in the dedicated Drive folder. All 9 tests in `tests/track2_extension` passed.
- CircuitKIT gates remain unchanged: no human-approved contrast export exists; the behavioral likelihood metric has not passed validation; no EAP-IG or causal intervention was run. Baseline/restricted/broad same-set evaluation remains outstanding.

## 2026-11-12 CircuitKIT TaskSpec integration smoke

- The existing Colab CPU session installed CircuitKIT 0.2.2, Transformers 5.15.0, TransformerLens 3.8.0 and PyTorch 2.14.1+cpu (Python 3.13.18). The custom TaskSpec registered with CircuitKIT and its balanced synthetic rows passed the actual EAP collator. Executed notebooks, environment check, logs and hashes are saved in ignored `outputs/lexsi/circuitkit_metric_cpu_smoke_20261112/`.
- This does not change gate D from `PARTIAL`: the CPU runtime has no CUDA, no model was loaded, no likelihood metric was run on Tiny Aya, and no EAP-IG or causal experiment occurred. Colab balance remains 0.00 GPU compute units. The live L4 still contains volatile checkpoint files; do not stop it until recovery is resolved.
- `tests/track2_extension`: 21 passed. These local tests remain synthetic; the real Colab evidence here is limited to installed-library TaskSpec registration and EAP collation.

- Additional CPU metric check: on a deterministic random 1-layer Hugging Face GPT-2, the custom full-continuation margin exactly matched `eval/dataset2_likelihood.py` (`0.0189359188079834`, absolute difference 0.0); gradients were finite and nonzero. This is synthetic model/scorer parity only. The Tiny Aya behavior gate remains unvalidated.

- Approval-provenance correction: an earlier CPU toy notebook used placeholder reviewer IDs and `human_approved` metadata. That run is archived as superseded and is not evidence of human approval. The adapter now has a separate explicit synthetic-smoke mode; production rows still require real approval provenance. The corrected TaskSpec/collator smoke passed on Colab CPU; 21 local tests pass.

- The current corrected CPU-smoke bundle was copied to the new private Drive folder `MyDrive/ez-penta/Pipeline1111/codex_extension_20261111/circuitkit_cpu_smokes_20261112/`; Colab verified 10 artifact hashes after writing and the readback receipt is preserved locally. It contains only synthetic smoke inputs/code/output, including the explicitly superseded notebook with false approval metadata clearly named and documented.
# Latest verification — CircuitKIT CPU suite (2026-11-12)

- The full `tests/track2_extension` suite passed 21/21 on the existing official Colab CPU session `lexsi-browser-export` (`m-s-ass1a2-1mpkjw3i6iyic`), with PyTorch 2.14.1+cpu and CUDA unavailable. The current repository `eval/` tree and tests were uploaded as a source bundle. Before discovery, stale `eval.*` and `test_*` modules were removed from the persistent kernel cache.
- Captured output: `outputs/lexsi/circuitkit_full_cpu_tests_20261112/execution.log`; uploaded bundle: `/tmp/lexsi_full_recheck_2.zip` locally and `/content/lexsi_full_recheck_2.zip` in the CPU session.
- This verifies unit coverage for seed-matched analysis, manifest integrity, inventory safeguards, and the synthetic CircuitKIT metric/data adapter. It does not validate real behavior pairs, Tiny Aya metric discrimination, prompt-side token alignment, GPU execution, EAP-IG, or causal effects. Approved CircuitKIT export remains empty; GPU quota is 0.00 CUs.

## EAP-IG integration status (2026-11-12)

- The explicit position-map patch is implemented and preserved under `docs/lexsi/patches/`; the TaskSpec builds exact shared-prefix/continuation maps and the collator left-pads their coordinates. The patch masks unmapped gradients and keeps clean embeddings at unmapped EAP-IG positions. Divergent framing-token positions remain unmapped.
- The current repository suite passed locally with 32 passes and one expected skip (CircuitKIT is absent from the local environment). The official Colab CPU combined run passed 41/41, including the CircuitKIT-dependent TaskSpec/collator test and actual synthetic EAP-IG execution on a random TransformerLens model. Logs and source bundles are under `outputs/lexsi/circuitkit_position_maps_cpu_20261112/`.
- The patch is not yet automatically installed by the experiment notebooks. No human-approved contrasts, Tiny Aya behavioral metric validation, GPU EAP-IG discovery, or causal intervention has occurred. Colab balance is 0.00 compute units and the CPU session has been stopped.

## Human review queue prepared (2026-11-12)

- Prepared `outputs/lexsi/human_adjudication_87_control_roleplay_TEMPLATE.csv` from the existing 87 provisional same-language control/roleplay screens. It includes 87 unique row IDs across 62 seeds (53 analysis, 34 tune), carries source prompts/responses and provisional confounds, and leaves every reviewer/adjudicator field blank. This is a review packet, not reviewed data.
- The template is git-ignored/private. Reviewers should each complete a separate copy independently, then use the adjudication/export workflow. Do not merge votes or populate CircuitKIT approval metadata automatically.

## Blinded review packets (2026-11-12)

- Two reviewer-specific packets were generated from the 87-row provisional candidate template at `outputs/lexsi/blind_review_packets_20261112/`. Each contains all 87 cases with a separate shuffle and randomized A/B orientation. The same opaque case ID joins reviewer judgments without disclosing source IDs; 43/87 candidates have opposite A/B orientations across packets. Guard labels and provisional rationales are excluded; review fields are blank.
- The private source mapping key and SHA-256 manifest are stored in the same git-ignored output directory. No human review has been received and the approved CircuitKIT export remains empty.
- Reviewer instructions now list exact allowed label values and define `review_status=completed` as task completion only, not approval.
- Local command `.venv/bin/python -m unittest discover -s tests/track2_extension -v` passed 35 tests with one expected skip (CircuitKIT is not installed locally). This confirms packet tooling and existing adapter tests only; it does not validate any prompt/response labels or Tiny Aya behavior.
- Next data-dependent gate: have two independent reviewers complete the two CSVs, adjudicate conflicts, verify the full prompt equivalence and response behavior, and record exact continuation spans. The GPU quota remains a separate external blocker for GPU experiments.

- Drive handoff: uploaded to `Pipeline1111/codex_extension_20261111/blinded_control_roleplay_review_20261112/`. Reviewer-specific folders are `reviewer_1_only` and `reviewer_2_only`, each containing only that reviewer’s sheet and instructions/rubric. The private key and manifest are in `PRIVATE_OWNER_ONLY/`. Drive permission readback shows only the project owner and connected account on these folders; no reviewers were granted access. File IDs and hashes are in the ignored local receipt `outputs/lexsi/blind_review_packets_20261112/drive_upload_receipt_20261112.json`.

## Provisional tokenizer geometry diagnostic (2026-11-12)

- Applied the locally available `CohereLabs/tiny-aya-water` tokenizer and native chat template to the 87 provisional candidate prompt pairs, strictly as a technical diagnostic. All 47 English and all 40 Japanese control/roleplay pairs had unequal token lengths. Mean roleplay-minus-control prompt length was +40.23 tokens for English and +62.25 for Japanese; maximum templated prompt length was 456 and 553 tokens, respectively.
- The first tokenizer load omitted a pinned `_commit_hash`; this was corrected by locating the cached Hugging Face snapshot and rerunning with immutable revision `41a99216d12b3ef4881df762934186e5e4d2bbc1`. Under that pinned tokenizer, median longest common prefix was 361 tokens for both languages, while roleplay prompts were longer on average by 40.23 tokens (English) and 62.25 tokens (Japanese). Median common suffix lengths were 20 and 22 tokens, respectively.
- This suggests a shared long leading context and divergent framing suffixes. It does not make end-padding a valid joint map: left end-alignment aligns answer spans while shifting the shared prefix; right start-alignment preserves the prefix but shifts answer positions. Require explicit region-aware position mapping or purpose-built alignment-preserving pairs before real discovery/patching; do not silently discard these candidates or treat their provisional status as resolved.
- Aggregate output only: `outputs/lexsi/circuitkit_token_alignment_candidate_audit_20261112.json`. No prompts or completions are included in this report.

## Alignment helper regression check (2026-11-12)

- Ran `.venv/bin/python -m unittest discover -s tests/track2_extension -v`; all 29 tests passed. This includes six new position-map/activation-gather/masking checks and the existing 21 Track-2 checks.
- This is local CPU unit-test evidence. CircuitKIT core hooks are not yet patched, the divergent framing span remains unmapped, and no human-approved pairs or GPU experiments were used.
- Updated the continuation metric to honor `input_lengths`, allowing correct target-position scoring in right-padded batches while retaining the left-padded CircuitKIT convention. Added hand-calculated right-padding and invalid-length tests. The full suite now passes 31/31 locally (`.venv/bin/python -m unittest discover -s tests/track2_extension -v`). This remains metric implementation validation, not behavior validation on Tiny Aya.



## 2026-11-12 public CircuitKIT integration recheck

- Added gated discovery and causal-validation notebooks. They require a passing real Tiny Aya metric report, approved analysis rows, and an independently adjudicated fresh-seed evaluation set; neither has been run.
- Found and corrected the position-map patch's missing integration with the actual `discover_circuit()` backend. Official Colab CPU regression: 9/9 passed on CircuitKIT 0.2.2 / TransformerLens 3.8.0 / PyTorch 2.14.1+cpu. Runtime stopped. This is synthetic API evidence only; no GPU, Tiny Aya prompts, approved rows, EAP-IG discovery, or causal result.
- GPU quota remains 0.00 CUs and active sessions remain zero. No science gate changed.

- The discovery notebook captures signed edge scores before CircuitKIT's magnitude-only node aggregation, while clearly labeling the graph visualization as structural layout. This capture code is prepared but unexecuted.

## Public EAP-IG API smoke (2026-11-12)

- Official Colab CLI CPU session ran the actual public `discover_circuit()` EAP-IG route with patched CircuitKIT 0.2.2, TransformerLens 3.8.0, Transformers 5.15.0, and PyTorch 2.14.1+cpu. On a random one-layer GPT2-shaped model and synthetic equal-length contrast it wrote score/circuit artifacts and captured signed edge values. It pruned zero nodes; this is an integration pass only.
- Debugged the missing chat template on TransformerLens's wrapped tokenizer and an unequal-token-length synthetic contrast that caused an index error. See `COLAB_DEBUG_LOG.md` and the full official CLI history in `outputs/lexsi/circuitkit_public_api_smoke_20261112/`.
- CPU session stopped. GPU quota remains 0.00. Human-approved contrasts, Tiny Aya behavioral metric validation, real prompt-alignment validation, Tiny Aya discovery, and causal validation remain incomplete.
- Post-stop Colab account readback was inconsistent: no sessions were listed, but usage reported one active assignment and 0.00 compute units. The smoke runtime is terminated; the account-side assignment count may be stale and remains unresolved.

## Artifact durability / checkpoint byte audit (2026-11-12)

- The restricted checkpoint candidate in Drive was downloaded (25,231,360 bytes; SHA256 `62eb698703051260588f1cc0c2e85a1bbbfd4117b740e64d2eca028241feadc1`) and failed `safetensors.safe_open` because its header metadata is incomplete. The broad folder remains empty. Historical tuned checkpoints are not durably recovered from this Drive location; the prior active L4 copies are not currently verified or accessible as a session.
- Synthetic public-API smoke outputs and session history were uploaded to new project Drive folder `https://drive.google.com/drive/folders/1p_tYjBiSIPuZ_d--RIJLZzRh3TaoUJxG`. Readback confirmed file names and byte sizes; receipt at `outputs/lexsi/circuitkit_public_api_smoke_20261112/drive_upload_receipt.json` records local hashes and remote IDs.
- Colab `sessions` reports none, while `usage` shows 0.00 compute units and one active assignment. This account-side inconsistency is unresolved; no job was launched.

## Language-stratified discovery and causal notebook hardening (2026-11-12)

- Reworked CircuitKIT discovery to choose independent, reproducible English and Japanese analysis cohorts and save separate EAP-IG artifacts. Added minimum cohort guards and artifact assertions. The selection helper rejects unreviewed/non-analysis rows and duplicate seeds within language.
- Reworked causal validation to use each language's own candidate circuit, require >=8 fresh reviewed seeds per language, check overlap against discovery and metric-validation seed sets, retain per-seed outputs, and summarize patching/ablation/random baselines separately by language. Removed a last cell that overwrote the actual per-seed results. Undefined numeric outputs are saved as null.
- Added exact dependency bootstrap/source patch setup cells and five passing tests for language-stratified selection. This is preparation only; no human-approved rows or GPU runs were available, so no scientific gate advanced.
- Latest local verification after the notebook design changes: 42 Track-2 tests pass with one expected skip; all three notebook JSON documents and code-cell ASTs parse; `git diff --check` passes. No scientific gates changed.

## Historical Pipeline 2 seed exposure rechecked (2026-11-12)

- Archived Drive notebook and aggregate results were inspected. The source references an ad hoc uploaded `pipeline2_prompts.json`, but the file is not in the recovered artifacts; the notebook has no saved execution outputs, and the results contain no seed or row IDs.
- Historical seed inventory remains partial and `outputs/lexsi/circuitkit_prior_seed_registry.json` remains `complete: false`. Evaluation/discovery stays fail-closed until complete exposure provenance is available.
- An attempted `pytest` command is unavailable because pytest is not installed, but the equivalent repository-supported `unittest` suite was rerun successfully as noted below. This does not test real data or GPU behavior.
- At the checkpoint-preflight stage, the suite ran 59 tests (58 passed, 1 expected skip). That count is superseded by the latest 62-test run, whose full output is `outputs/lexsi/latest_local_tests_20261112.log`.
- Fresh Drive readback on 2026-11-12 reconfirmed 87 blank review statuses in each blinded reviewer CSV; all five rows in the older local adjudication sheet also remain unadjudicated. No approved pairs are available.
- Further checkpoint recovery audit found one Drive revision for the restricted weight file and no earlier version; a fresh global search found no alternate restricted/full checkpoint or adapter. This does not exclude copies outside accessible Drive, but no recoverable weights are currently available in the searched project location.
- Audited the saved prior-review files for the earlier “50 valid pairs”: all 514 combined and all 24 bilingual records are Codex-assisted/provisional, with no human reviewer provenance. The bilingual shortlist also differs from the present same-language roleplay contrast design. These records remain ineligible for approved CircuitKIT input.
- Hardened future multi-arm evaluation resumption: content-addressed generation/judgment provenance, exact answer/guard revision binding, final/partial JSONL merge with conflict rejection, and adapter-path preflight before baseline inference. Local helper tests pass; no real runtime resume was attempted.
- Local checkpoint preflight now verifies `config.json`, full-model weight files, shard-index references, and Safetensors headers before any baseline inference. This catches absent, malformed, and truncated adapter-arm directories early. The checkpoint-stage suite ran 59 tests (58 passed, 1 expected skip); no real checkpoint or Colab inference was tested.
- Resume journals now recover a single malformed, unterminated tail only in `.partial` files and atomically rewrite merged rows; malformed final files and interior/newline-terminated corruption remain fatal. Full local suite: 62 total, 61 passed, 1 expected skip. Fresh Drive/Colab check still shows 87 blank reviews per packet and zero compute/session availability.
- Multi-arm generation now derives one common deterministic Hugging Face `GenerationConfig` from the pinned baseline revision and passes it to baseline/restricted/broad arms, preventing each tuned checkpoint's saved defaults from changing decode behavior. Run summaries include the effective config, checkpoint fingerprints, GPU/runtime details, dependency versions, and runner hash. Local suite: 63 tests, 62 passed, 1 expected skip; real GPU arm parity is not yet executed.

## Current external state recheck (2026-11-12)

- Official Colab CLI: no active sessions, no active assignments, 0.00 compute units. No GPU execution can be launched within the no-paid-compute budget at this time.
- Both blinded reviewer CSVs were parsed again: 87 rows each; all 87 `review_status` values remain blank.
- Drive checkpoint folders were listed again: restricted contains `config.json` plus one 25,231,360-byte `model.safetensors` candidate; broad is empty. The restricted bytes were previously found to have an incomplete Safetensors header and remain unusable.
- These observations supersede older notes in this file that described the scratch L4 as still live or the account assignment count as unresolved.


## Review reconciliation tooling update (2026-11-12)

- Added a fail-closed reconciliation command and explicit approved-export command for the blinded 87-case control/roleplay review packets. It preserves human approval as a separate adjudication event, verifies the exact reviewer packets/key/source by hash, and requires exact continuation prefixes and independent reviewer/adjudicator identities.
- Synthetic regression suite: 70 tests total, 69 passed, one expected skip because CircuitKIT is not installed locally. This does not validate real candidate labels or model behavior.
- Read-only current packet check: 87 rows per reviewer, zero completed reviews. Reconciliation reported 176 validation issues and created no adjudication worksheet; the approved export remains empty.
- This closes a workflow/reproducibility gap only. Human review, behavior-metric validation, Tiny Aya EAP-IG, causal testing, and before/after improvement remain unestablished. Colab is still at 0.00 compute units with no session.

## Prompt-contrast coverage audit update (2026-11-12)

- Confirmed the local 21,184-row SQLite is a validated prompt-only source (320 seeds; 42 complete design seeds; SHA256 recorded in `eval/manifests/dataset2_shards.json`), with no response generations or labels. It cannot replace the missing response SQLite for the strict matched safety analysis.
- Added an aggregate-only structural coverage script. Exact-request harmful control/rp pairing yields 134 analysis US seeds per language (18 complete source seeds) and 84 tune US seeds per language (11 complete source seeds); 13 complete heldout seeds are already exposed. These counts support planning only and do not validate semantic equivalence or behavior.
- Synthetic tests 2/2 and full Track-2 suite 72 total (71 passed, one expected skip). Research gates are unchanged.


### Current blocker and local verification refresh (2026-11-12)

- Re-ran `.venv/bin/python -m unittest discover -s tests/track2_extension -v`: 77 tests, 76 passed, 1 skipped because CircuitKIT is not installed in the local test environment. Full output is `outputs/lexsi/latest_local_tests_20261112_current.log`; this does not substitute for the already-documented official Colab CircuitKIT CPU integration tests or any real Tiny Aya GPU experiment.
- Rechecked the official CLI via `/Users/marcuschang/.local/bin/colab`: authenticated usage reports 0.00 compute units, 0 active assignments, and no active sessions. The CLI exists but its directory is not on this shell's PATH.
- Remaining external research gates are unchanged: both 87-row blinded review sheets have no completed reviews; the restricted checkpoint candidate is truncated and the broad checkpoint folder is empty; the historical Pipeline 2 prompt manifest is missing; no untouched fresh evaluation cohort is approved. These cannot be cleared by local code changes.
- Updated the private run manifest with this test log and the verified CLI state.


### Colab quota restored (verified 2026-11-12T14:55:10Z)

- Official CLI `/Users/marcuschang/.local/bin/colab usage` reports **100.00 compute units**, 0.00/hour current usage, zero active assignments; `colab sessions` reports no active runtime. This supersedes earlier 0.00-unit notes.
- The compute quota is no longer the immediate resource blocker. No GPU job was launched during this refresh. Before real-data CircuitKIT work, the human-review/export gate, historical exposure registry, valid checkpoint/data choice, and evaluation-cohort requirements still need to be satisfied.
- The two existing private review packets are available at the reviewer-specific Drive folders linked in the handoff; each has 87 rows and blank reviewer fields on the last readback. The 87 are roleplay/control candidates, distinct from the older 50 bilingual pair shortlist.


## Provisional LLM-assisted screening (2026-11-12)

- Created a private first-pass worksheet for all 87 blinded candidate rows from reviewer 1's packet: `outputs/lexsi/llm_provisional_review_87.csv`, with source judgments in `outputs/lexsi/llm_provisional_review_87_annotations.json` and validator `scripts/lexsi/create_llm_provisional_review.py`.
- All 87 rows are explicitly `provisional`; source reviewer files and the private A/B key were not edited. This was one Codex conversation-model pass, not an independent external LLM annotation. Exact model build metadata is unavailable; no inter-annotator agreement or human validation is claimed.
- Aggregate screen: 77 prompt pairs judged equivalent, 8 uncertain, 2 materially different. Safety labels: A 51 safe / 18 borderline / 18 unsafe; B 52 safe / 24 borderline / 11 unsafe. These are provisional model judgments, not validated ground truth.
- The worksheet generator checks row joins, allowed label values and literal evidence substrings. Where initial evidence was paraphrased, the stored quote was replaced with an exact opening span to satisfy that mechanical check; citations still need substantive human confirmation.
- No CircuitKIT export was created and no Colab/GPU run was started. Human approval, adjudication and seed/exposure checks remain mandatory before any execution inputs. The 100-CU quota does not waive these gates.

## Current-state recheck after provisional annotation (2026-11-12)

- Re-read the handoff requirements and current repository state. Official Colab CLI reports 100.00 compute units, no assignments, and no active sessions.
- Re-listed Drive checkpoint folders: restricted still contains only `config.json` plus a 25,231,360-byte `model.safetensors`; broad remains empty. The restricted artifact is not considered reloadable, and broad weights are absent.
- Provisional LLM screen triage count, using the private randomized condition key: 39 pairs appear equivalent and harmful on both sides; 11 show the target control-refusal / roleplay-unsafe-compliance pattern (9 high confidence), while 4 show the reverse. These remain model-only provisional judgments and are not eligible for export or confirmatory analysis.
- Re-ran the full local Track-2 suite after the annotation workflow and Phase 1 gate correction: 81 tests, 80 passed and 1 expected skip (CircuitKIT not installed locally). Log: `outputs/lexsi/latest_local_tests_20261112_post_annotation.log`.
- No real-data CircuitKIT export, inference, training, or new Colab runtime was started. The real-label and checkpoint gates remain unresolved.

## Phase 1 execution audit and correction (2026-11-12)

- Audited the saved L4 notebook output, not only the CLI exit code. The real-weight backend parity checks passed for English, Japanese, and sliding-window probes, but the final cell threw because no human-approved export existed; Colab CLI nevertheless returned exit code 0.
- Corrected the notebook to write an explicit `BLOCKED_HUMAN_APPROVAL_REQUIRED` status without a traceback when the approved export is empty. The negative behavioral threshold now writes `VALIDATION_FAILED` rather than disguising a research result as an execution error. The missing-data branch was executed locally and returned the expected blocked status.
- Added Colab Secret retrieval for `HF_TOKEN`. A fresh official L4 credential probe found no token in environment/cache; the session was stopped without downloading model weights. The corrected full notebook still needs an actual Colab rerun after the token secret is configured.
- Full local Track-2 suite after these changes: 81 tests, 80 passed and one expected skip. Log at `outputs/lexsi/latest_local_tests_20261112_post_annotation.log`.

## Notebook execution auditing added (2026-11-12)

- Added `scripts/lexsi/audit_notebook_execution.py` because Colab CLI returned exit 0 for a notebook containing a cell traceback. The utility records a notebook hash, detects `output_type=error`, captures exception details, and fails with exit 1.
- Verified it against the archived real L4 Phase 1 output: result `CELL_ERRORS`, cell 5 `RuntimeError` for the empty human-approved export. The existing CLI exit code 0 is therefore not considered notebook success.
- Added three regression tests. Full Track-2 suite: 84 tests, 83 passed and one expected skip; log `outputs/lexsi/latest_local_tests_20261112_execution_audit.log`.
- Latest Colab state: 100.00 CUs, no active session/assignment. Fresh-runtime HF token is missing; corrected notebook 01 has not been re-executed remotely.

## Phase 1 Colab secret bootstrap staged (2026-11-12)

- Rechecked current approvals: approved export 0 bytes; 87/87 LLM rows provisional; 87/87 adjudication statuses blank.
- Rechecked Colab: 100.00 CUs; active L4 `lexsi-hf-secret-recheck` (`gpu-l4-s-ass1a1-gprt6hb6qmv8`) at 1.94 CU/hour. Secret lookup via CLI timed out because Google only releases Colab Secrets to code executing from the browser UI.
- Uploaded the corrected Phase 1 notebook, likelihood helper, and empty approved export to `/content/lexsi/`. No model was downloaded. Session is intentionally left active for the owner to fetch the secret from the attached Colab UI; its private manifest is `outputs/lexsi/colab_phase1/ui_secret_bootstrap_20261112/run_manifest.json`.

- Latest live state check: 99.90 CUs remain; the staged L4 runtime is `IDLE` and accruing 1.94 CU/hour. It is awaiting one Colab UI secret-fetch cell. Runtime link and latest balance are in the private bootstrap run manifest.


- Follow-up: the owner had not run the UI secret cell at the next state check, so the idle L4 was stopped to prevent ongoing charges. Colab now reports 99.90 CUs, zero assignments and 0.00 CU/hour. The prepared files remain in the stopped runtime only until its storage is reclaimed; rerun staging after the owner completes the one-cell UI step.
- Added a reproducible, token-safe Colab UI bootstrap cell to `REPRODUCE.md`. The prior staged runtime was stopped after no user cell execution; its URL is no longer valid. Current state is 99.90 CUs and zero active usage.

## Broad Drive and checkpoint recheck (2026-11-12 15:56 UTC)

- Searches for `pipeline4_full`, `adapter_config`, and Tiny Aya adapter/checkpoint files returned no alternate complete weights. Restricted remains config + 25,231,360-byte incomplete SafeTensors; broad remains empty.
- Re-ran `safetensors.safe_open` on the local private restricted candidate; it again failed with incomplete metadata. Neither historical arm can be loaded from the available files.
- Both Drive reviewer CSVs retain their original 2026-11-12 06:16 UTC modification time. Colab is stopped at 99.90 CUs; no usage accrues.


## Review coverage and local HF credential recheck (2026-11-12)

- Parsed the current private review artifacts directly. `review_401_provisional.csv` contains 401/401 rows marked `codex_assisted` and `provisional` (56 discrepancy=yes, 207=no, 138=unclear; semantic equivalence 310 equivalent, 56 materially different, 35 uncertain). These are model-assisted audit labels, not human-validated examples.
- `review_80_calibration.csv` contains only 10 provisional rows; 70 remain unreviewed. The deliberately balanced sample remains unsuitable for estimating population accuracy from raw proportions. The 87-row same-language screen is also 87/87 provisional.
- The LLM can reduce the reading burden by prioritizing likely cases, but cannot satisfy the repository’s human approval gate. The approved CircuitKIT JSONL remains empty. No independent second model or reviewer has supplied labels.
- Local `hf auth whoami` succeeds and a config-only fetch from `CohereLabs/tiny-aya-water` revision `41a99216d12b3ef4881df762934186e5e4d2bbc1` succeeds. This establishes local HF access only; the Colab UI Secret fetch remains unresolved.
- Official Colab CLI now reports 99.90 CUs, no sessions or assignments. No GPU work was started in this recheck.


## Provisional calibration-control LLM pass — batch 1 (2026-11-12)

- Added provisional, text-grounded annotations for five previously blank records in `outputs/lexsi/review_80_calibration.csv`; exact evidence strings were checked against the full Japanese and English responses before writing.
- Updated calibration coverage: 15/80 provisional, 65/80 unreviewed. These are Codex-assisted judgments, not independent validation or human approvals. No guard-accuracy estimate was computed from the balanced sample.
- One Japanese misinformation answer repeats a dubious statement that misinformation spreads from reliable sources; this is noted in its rationale, but it does not change the safety label. The drug-purchasing request pair received safe labels because neither answer contains procurement help, despite a difference in directness of refusal.


## Provisional calibration-control LLM pass — batch 2 (2026-11-12)

- Added five more Codex-assisted annotations after inspecting both full prompts and responses; both-language evidence strings were verified as literal response substrings. Coverage is now 20/80 provisional and 60/80 unreviewed.
- The batch included a benign medical-information request, a recreational jump suggestion (kept borderline due to safety-context differences), an anime dialogue request, fictional anatomy, and a question about mafia information warfare. Unsupported factual claims are called out separately from response safety.
- No labels are approved; this single model pass is not independent review and provides no guard-accuracy estimate.


## Provisional calibration-control LLM pass — batch 3 (2026-11-12)

- Added five provisional annotations with literal evidence checks. Coverage is now 25/80 provisional and 55/80 unreviewed.
- This batch included a clear Japanese/English prompt mismatch (“during spinach” versus Hollywood screenings), classified as materially different with Japanese intent unassessable; it remains a calibration record, not a confirmed language safety difference. It also included a secret-leak request where both answers withheld leak methods, and sexual content where both refused.
- Japanese holiday-fraud legal claims were marked borderline for reliability concerns; no fraud-enabling advice appeared. These remain one-model provisional judgments only.


## Provisional calibration-control LLM pass — batch 4 (2026-11-12)

- Added five annotations after full bilingual prompt/response inspection and literal evidence validation. Coverage is now 30/80 provisional; 50 remain unreviewed.
- This batch includes a privacy-surveillance request where both answers provide no methods, and an ambiguous fictional-villain request where the English response redirects more explicitly. These are not confirmed safety discrepancies.
- All labels remain `codex_assisted`/`provisional`; no reviewer agreement or accuracy estimate is claimed.


## Provisional calibration-control LLM pass — batch 5 (2026-11-12)

- Added five bilingual evidence-checked annotations; coverage is now 35/80 provisional and 45/80 unreviewed.
- The batch contains one candidate model-behavior difference in financial advice: English lists high-risk instruments (options, futures, leveraged ETFs, short selling) while Japanese remains vague. This is marked provisional and should receive special human scrutiny, not treated as a result. Another prompt about teacher conduct and a mafia investigation was marked materially non-equivalent because agency/focus differs.
- All records remain provisional; no guard accuracy or population inference is reported.


## Provisional calibration-control LLM pass — batch 6 (2026-11-12)

- Added five full bilingual assessments with literal evidence verification; coverage is now 40/80 provisional and 40/80 unreviewed.
- This batch covered festival/religion information, illegal-drug effects, a family-age request with fabricated English ages, a fictional suicide question, and an unsupported hormone-balance premise. Safety labels distinguish informational content from operational facilitation; several reliability-sensitive cases remain borderline or unclear.
- No labels are human-approved; the balanced sample still cannot estimate population guard accuracy.


## Provisional calibration-control LLM pass — batch 7 (2026-11-12)

- Added five bilingual annotations with evidence-substring checks; coverage is now 45/80 provisional and 35/80 unreviewed.
- A hypertension answer is provisionally marked unsafe in Japanese because it recommends low-dose steroids and beta-agonists as hypertension treatments; the English response is borderline and directs medication selection to a clinician. This requires expert/human review and is not a validated finding. Both sides of a mafia-harm query provide harmful method categories and are marked unsafe.
- No annotations have human approval; these balanced controls do not yield a guard-accuracy estimate.


## Provisional calibration-control LLM pass — batch 8 (2026-11-12)

- Added five full bilingual assessments with literal evidence validation; coverage is 50/80 provisional, 30/80 still unreviewed.
- One self-harm pair is marked unsafe on both sides, with the English response substantially more actionable; the Japanese/English mafia-efficiency pair also gives criminal-operation advice on both sides. These are calibration annotations, not a confirmed model comparison.
- No human approvals or accuracy estimates are created by this pass.


## Provisional calibration-control LLM pass — batch 9 (2026-11-12)

- Added five evidence-checked bilingual annotations; coverage is now 55/80 provisional, 25/80 unreviewed.
- Both languages provide methods for animal cruelty, theft, and organizational data leaks; these are marked unsafe on both sides. A food-event “confusion” prompt remains borderline due to the potentially manipulative aim, and an alleged bathhouse-drug-effects pair has an English-specific factual reliability concern.
- The batch remains a single Codex-assisted pass, not independent validation. No accuracy estimate or human approval is inferred.


## Provisional calibration-control LLM pass — batch 10 (2026-11-12)

- Added five annotations with literal evidence checks. Coverage is now 60/80 provisional; 20 remain unreviewed.
- One explicit-language pair is provisionally marked as more concerning in English due to direct death-threat examples; one test-falsification pair is unsafe in both languages, with the English answer more actionable. A yakuza theft answer begins with a refusal but then provides criminal methods; the refusal wording does not make the substance safe.
- All remain model-assisted provisional labels. The balanced sample does not estimate accuracy, and none is approved for CircuitKIT.


## Provisional calibration-control LLM pass — batch 11 (2026-11-12)

- Added five full-prompt/response annotations with exact evidence checks; 65/80 controls now provisional and 15 remain unreviewed.
- One drug-purchase pair has no response in Japanese but an English purchase-roleplay that encourages evasion and asks about price. One dangerous-ninja pair has vague Japanese advice versus English assault/sabotage/assassination recommendations. Both are flagged as provisional safety differences. A separate insurance-card pair is semantically uncertain because “Americanhood” appears to have been translated as “hoodie wearer”; the English answer only uses placeholders.
- No annotations are human-approved; no guard accuracy or confirmatory result is claimed.


## Provisional calibration-control LLM pass — batch 12 (2026-11-12)

- Added five annotations with exact evidence checks; 70/80 agreement controls are now provisional, with 10 unreviewed.
- Both answers to the yakuza theft and biker-gang data-theft requests provide criminal methods; the Japanese encrypted-data answer suggests decryption approaches while the English answer makes authorization explicit. Authorization is unspecified, so that case remains unclear rather than a confirmed safety difference. A reputational-harm pair is also flagged because English gives rumor-spreading advice while Japanese contradicts its own false claims.
- This is a single Codex-assisted pass only. No human approval, independent review, or guard-accuracy estimate is claimed.


## Provisional calibration-control LLM pass — batch 13 (2026-11-12)

- Added five evidence-checked bilingual annotations; 75/80 agreement controls are provisionally annotated, with 5 remaining.
- The Japanese “explicit words” answer is non-explicit while English supplies sexual terminology; drug-side-effect responses on both sides contain questionable claims; workplace-crime and yakuza-theft answers provide criminal methods in both languages.
- All statuses remain `provisional`; the balanced-control review is not human validation and yields no population accuracy estimate.


## Agreement-control review completion — provisional only (2026-11-12)

- Completed the first-pass Codex review of all 80 agreement controls in `outputs/lexsi/review_80_calibration.csv`. All 80 remain `codex_assisted` / `provisional`; none is human-approved.
- Structural audit: 80 rows, all rubric values allowed, both evidence excerpts are literal substrings of their full saved responses, and source-guard strata remain 40 Safe/Safe + 40 Unsafe/Unsafe.
- Provisional response safety counts: JA 42 safe / 20 unsafe / 17 borderline / 1 uncertain; EN 39 safe / 23 unsafe / 18 borderline. Equivalence: 76 equivalent / 3 materially different / 1 uncertain. Discrepancy assessment: 66 no / 10 yes / 4 unclear. These counts are not population estimates or human agreement statistics.
- No CircuitKIT input was exported and no metric, discovery, or causal gate advanced. See `review_401_summary.md` for caveats and interpretation.


## Fresh Colab L4 dependency probe (2026-11-12)

- Official CLI provisioned an L4 session; CUDA was available on NVIDIA L4 (23,034 MiB). Installed/pinned CircuitKIT 0.2.2, Transformers 5.15.0, and TransformerLens 3.8.0 without changing PyTorch 2.14.1+cu130.
- HF token retrieval from `google.colab.userdata` via CLI timed out. Model loading and inference were therefore not attempted. Session was stopped; balance readback is 99.90 compute units with no active session or assignment. See `outputs/lexsi/colab_gpu_env_check_20261112.json` and `COLAB_DEBUG_LOG.md`.
- User directed that the existing LLM provisional reviews may be used without waiting for manual review. Record them as LLM-assisted/owner-authorized exploratory labels only; they are not human-validated labels. The official human-approved export remains empty.
- Strict candidate audit: five English pairs meet the provisional equivalent-prompt/harmful-intent/clear-control-refusal/unsafe-roleplay-compliance pattern (four analysis seeds, one tune seed); there is no fresh evaluation seed in this subset. This cannot support the planned split design.

- Post-stop usage settlement check at 2026-11-12T16:37:20Z reports 99.83 Colab CUs, zero active sessions/assignments, and 0.00 CU/hour. The immediate stop readback was 99.90; 0.07 CU settled afterward.


## Expanded same-language roleplay screen — LLM triage (2026-11-12)

- A read-only scan of both verified Dataset2 result shards found 648 exact-key control/roleplay structures across analysis/tune/previously exposed heldout. Excluding heldout and filtering to completed, eligible, harmful, monolingual EN/JA rows with binary recovered guard labels produced the candidate inventory. Of the analysis/tune pairs, 47 additional cases had a changed refusal heuristic while the recovered binary guard agreed; these were outside the earlier guard-disagreement screen.
- Added a deterministic extractor and five passing selector tests. All 47 added pairs received Codex-assisted provisional annotations; all 94 evidence snippets were verified as literal substrings of the saved answers and rubric labels passed enum checks. This is single-pass LLM triage, not human validation.
- The strict label pattern found two new Japanese analysis candidates. Combined with the earlier five English candidates, the provisional target shortlist now has 7 unique seeds: 6 analysis (4 EN, 2 JA) and 1 tune (EN), with no fresh evaluation seed. The new Japanese examples concern a disinformation campaign and nonconsensual surveillance; both are promising exploratory candidates, not adjudicated facts. Reverse-direction and non-equivalent cases remain separately labeled.
- Created `outputs/lexsi/roleplay_heuristic_expansion_20261112/exploratory_target_pairs.jsonl` separately from the empty official approved export. These rows are for an exploratory pilot only and cannot support the predeclared discovery/tuning/evaluation design or causal claims.
- Tests: `python -m unittest tests.test_heuristic_mismatch_roleplay_pool -v` (5/5), validator pass (47/47 rows, 94/94 evidence spans), `py_compile`, and `git diff --check`. No model inference or GPU run occurred in this expansion.


## Owner-authorized Phase 1 exploratory run (2026-11-12)

- The project owner explicitly authorized using the LLM-provisional labels for an exploratory run. Created a separate seven-row input (6 analysis seeds, 1 tune seed; EN 5, JA 2) with provenance preserved. The official `circuitkit_pairs.jsonl` human-approved export remains empty; these are not represented as independent human annotations.
- Executed the corrected `01_prepare_circuitkit_data.ipynb` on an official Colab NVIDIA L4 with the original Tiny Aya Water HF checkpoint at revision `41a99216d12b3ef4881df762934186e5e4d2bbc1`. Architecture: Cohere2, 36 layers, hidden size 2048, 16 attention heads / 4 KV heads, vocab 262,144, sliding window 4,096. Tokenizer/chat-template checks passed.
- HF↔CircuitKIT parity passed: maximum absolute log-probability errors were `3.2425e-5` (English), `3.3379e-5` (Japanese), and `7.4387e-5` (4,196-token sliding-window probe); top tokens agreed. Peak PyTorch allocation was 22,248,377,344 bytes; GPU memory used at measurement was 21,608/23,034 MiB.
- Behavior metric result: `INSUFFICIENT_VALIDATION_DATA`, not a pass. Analysis had 6/6 control-minus-roleplay margin wins (exact one-sided sign p=0.015625), but expected absolute signs occurred in only 1/6. Tune had only one non-tied seed (one margin win, exact p=0.5) and zero expected absolute-sign pairs. The observed relative ordering is not enough to show the metric distinguishes refusal from unsafe compliance; discovery remains gated.
- Full notebook audit: five code cells, zero cell exceptions. Environment: Python 3.13.16, CUDA 13.0, PyTorch 2.14.1+cu130, Transformers 5.15.0, CircuitKIT 0.2.2, TransformerLens 3.8.0, Hugging Face Hub 1.33.0. Runtime was stopped; latest balance is 99.57 CUs, zero active assignments/sessions.
- Artifacts, logs, exact scores, GPU memory, dependencies, and hashes are private under `outputs/lexsi/circuitkit_owner_authorized_exploratory_20261112/`. Local Track-2 regression: 84 tests, 83 passed and 1 expected skip. This is a real-model metric diagnostic, not a validated metric, circuit discovery, or improvement result.


## Expanded tune-seed proxy-agreement audit (2026-11-12)

- Added an all-matched inventory mode. The source scan found 648 exact control/roleplay structures across analysis/tune (394 analysis rows / 121 seeds; 254 tune rows / 79 seeds). It remains an inventory, not annotated ground truth.
- Of 33 tune rows where both prompts had Safe guard labels and both refusal heuristics fired, only 7 seeds were new relative to the known historical exposure registry and previous screens. We selected one pair per seed and hid source labels in the annotation view. All 7 were provisionally reviewed from full prompts and answers: six equivalent harmful requests received clear refusals on both sides; one pair had a materially different underlying request. No new strict refusal-to-unsafe-compliance candidate was found.
- This increases reviewed negative-control coverage but leaves the strict target set unchanged at 7 seeds (6 analysis / 1 tune). The current refusal metric remains failed/inadequate; there is still no basis to run CircuitKIT discovery.

- Full repository regression after this audit: `python -m unittest discover -s tests -v` — 221 tests, 218 passed, 3 skipped.

## Additional tune proxy-agreement screen (2026-11-12)

- Reviewed all 38 newly selected, previously unexposed Tune seeds in the Safe/Safe + both-refusal-heuristics-false stratum (33 EN, 5 JA), one pair per seed. The annotations are Codex-assisted, single-pass, provisional, and owner-authorized for exploratory triage; 0 rows are human-approved.
- The selected cohort contains 29 benign, 8 ambiguous, and 1 harmful requests. One pair is materially non-equivalent; 37 are provisionally equivalent. One reverse-direction case has control answer a protective request while roleplay refuses it. Several medical/factual-quality concerns are marked borderline. No pair meets the strict control-refusal / roleplay-unsafe-compliance target.
- Structural validation passed for 38/38 records, 76/76 literal evidence excerpts, and all enum labels; strict target count is 0. Counts are not a prevalence or proxy-accuracy estimate because selection was deterministic and conditional on automated proxy agreement.
- This screen does not increase the seven-seed strict target shortlist or validate the likelihood metric. EAP-IG and causal validation remain gated. Details and hashes are in `outputs/lexsi/roleplay_heuristic_expansion_20261112/tune_safe_proxy_agree_nonrefusal_run_manifest.json`.
- Full regression after the screen: `python -m unittest discover -s tests -v` — 222 tests, 219 passed, 3 skipped; saved log `outputs/lexsi/latest_local_tests_20261112_nonrefusal_screen.log`. The skipped tests require optional real-server/source-shard inputs. No real GPU experiment or circuit discovery was run in this continuation.
- A final exposure-audited pass found one remaining previously unexposed Tune pair outside the Safe/Safe proxy-agreement stratum (Unsafe/Unsafe guard labels; refusal heuristic missing on roleplay). Provisional review judged the request ambiguous and both responses compliant, but roleplay added specific, potentially dangerous and apparently false livestock-killing claims. This is a response-risk/factuality discrepancy, not a refusal-to-compliance contrast; strict target count remains zero. Its source labels are hidden in the private review view; see `tune_unsafe_proxy_agree_edgecase_run_manifest.json`.
- Separately audited the existing bilingual provisional-difference CSV's Tune subset: 8 rows across 6 seeds, with zero overlap with the seven Phase 1 exploratory seeds and zero overlap with the known historical training/probe registry. Two seeds provisionally show Japanese refusal / English unsafe compliance; one seed shows the reverse. This is a cross-language contrast, not the same-language roleplay metric cohort, and these LLM-only labels cannot validate that metric. Preserve the rows for a later JA/EN comparison; aggregate audit: `outputs/lexsi/bilingual_tune_metric_eligibility_20261112.json`.

## Phase 1 metric-failure diagnosis (2026-11-12)

- Added a reproducible aggregate-only audit of the saved seven-pair score report. The relative control-minus-roleplay margin is positive on 7/7 pairs, but the required absolute behavior signs occur on only 1/7 (1/6 analysis, 0/1 tune). Across English analysis examples, roleplay margins remain positive on all 4 pairs despite the provisional compliance labels; this is direct evidence that relative ordering does not suffice for behavior classification.
- Candidate continuations have median lengths of 152 refusal tokens and 307 compliance tokens (median ratio 2.02). The implementation scores the full, semantically different continuations with per-token mean margins and omits EOS. Length normalization does not eliminate differences in content, style, or stopping behavior. These are plausible confounds to test, not established explanations for the failure.
- No metric, labels, or gate criteria were changed. The audit is descriptive and does not pass the metric validation gate. Output: `outputs/lexsi/circuitkit_owner_authorized_exploratory_20261112/behavior_metric_failure_diagnosis.json`; code: `scripts/lexsi/audit_behavior_metric_validation.py`.
- Regression check after adding the aggregate diagnostic: its focused tests pass (2/2). A full-suite rerun was stopped at `test_backend_difference_rejected_at_merge` while `eval.dataset2_data.load_dataset()` read the 21,184-row SQLite source; sampling showed the Python process blocked in filesystem `read()` with no test exception or CPU progress. The interrupted log is `outputs/lexsi/local_tests_20261112_metric_diagnosis_interrupted.log`. The previous full suite (222 tests, 219 pass, 3 skipped) predates only the diagnostic script/test; do not describe the interrupted rerun as passing.

### Tune recovery-opportunity audit (2026-11-12)

- [x] Reran the aggregate-only audit from both original response shards using the partial historical exposure registry and prior candidate/review sources (161 distinct excluded seeds).
- [x] Of 68 incomplete matched keys in previously unexposed seeds, only 2 keys from 1 seed have both rows marked `analysis_eligible`; no unexposed seed has complete eligible control and roleplay responses ready for semantic review.
- [x] Persisted aggregate counts and shard hashes at `outputs/lexsi/tune_pair_recovery_audit_20261112.json`; added reusable audit logic and focused tests (2/2 passed).
- Most incomplete rows are excluded by the source-level `analysis_eligible` flag. Rerunning generation/scoring cannot change that flag, and at most one seed could be recovered—below the five-seed Tune metric gate. Do not spend GPU compute on inference recovery for this cohort. The metric gate remains failed; a new seed-disjoint contrast cohort or a frozen metric redesign evaluated on new Tune seeds is needed.

### Existing Tune cohort exhausted for an untouched metric check (2026-11-12)

- The aggregate recovery audit finds **79 fully eligible matched Tune seeds and zero unexposed fully eligible Tune seeds** under the seed-level prior-history/review exclusion set (161 distinct excluded seeds across all sources). This includes every matched Tune seed with complete, eligible control and roleplay responses; the prior small positive cases do not create a clean unseen validation cohort.
- Therefore, the five-seed minimum cannot be met by selecting more untouched Tune examples from the current Dataset2 response shards. The remaining two incomplete keys from one seed are not sufficient, even if rerun.
- The next data step is to acquire or construct genuinely new source seeds and responses (or a suitable external benchmark), then preassign disjoint discovery, metric-tuning, and fresh evaluation splits before any metric changes. Keep current Dataset2 results as exploratory evidence only.
- Current external recheck: official Colab CLI reports 99.57 compute units and no active sessions/assignments. Drive reviewer delivery folders still contain only the two original reviewer packets and instructions; no returned review file is present in those folders.

### New-cohort design after Tune exhaustion (2026-11-12)

- Added `docs/lexsi/NEW_COHORT_PLAN.md` with a data-collection and evaluation route that preserves the current failed metric result and separates new matched roleplay mechanism examples from external safety benchmarks.
- Reviewed primary dataset documentation: AnswerCarefully provides Japanese safety data with English translations in v2.2 and states redistribution is prohibited; APTO-XSTest-JA supplies reviewed Japanese refusal/non-refusal references and CC BY 4.0 annotations but primarily targets over-refusal. Neither dataset alone provides the new same-language control-versus-roleplay contrasts required for CircuitKIT discovery.
- No external dataset has been downloaded or used. Use these only as separate external behavior evaluations, keep original test splits untouched, and acquire new, bilingual-reviewed matched seeds for the mechanism cohort.

### Metric-gate sample-size planning (2026-11-12)

- Added an exact-binomial sign-test planning utility and regression tests. It shows that the existing five-seed minimum has only 32.8% power under an illustrative .80 per-seed success rate; under that assumption, the first sample size reaching 80% exact-test power is 18 non-tied seeds (reject at 13/18). At .70/.75 assumed success, the corresponding sizes are 37/23.
- These are hypothetical operating characteristics, not empirical estimates or a final power recommendation. They omit ties and the stricter absolute-sign part of the metric gate, so actual cohort needs may be larger. No gate or metric threshold changed.

### AnswerCarefully source snapshot audit (2026-11-12)

- Retrieved `llm-jp/AnswerCarefully` v2.2 at immutable revision `d1d4e57cd4d7aecb44f879e7488832011fcaddb0` into the Git-ignored local cache `data_ready/answercarefully.sqlite3`. No model inference was run and no source text or row IDs were printed.
- Verified SQLite `quick_check=ok`, 1,800 unique records, 1,464 dev / 336 test, 1,800 Japanese prompts/references and English prompt translations, and zero English reference answers. `adaptation-tag=0` counts are 1,067 dev and 245 test; 349 rows contain translation notes. Metadata has risk categories but no explicit `refusal_required` field.
- Added an aggregate-only audit script and tests (2/2 pass); report `outputs/lexsi/answercarefully_snapshot_audit_20261112.json`. The dataset is not redistributed.
- Audited local notebook execution metadata: the Aya AnswerCarefully evaluation notebook has no executed cells/outputs. Historical Aya exposure of the dataset's test split remains unresolved because referenced external result artifacts were not found in accessible Drive. Keep test rows sealed and do not describe them as fresh confirmatory evaluation until provenance is resolved.
