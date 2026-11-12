# CircuitKIT Colab execution log

## 2026-11-12 — Capture unclipped signed causal outcomes

- The current CircuitKIT safety case study 23 source captures full pillar result dictionaries because the standard faithfulness summary clips scores to `[0,1]`; its source notebook has no execution outputs. Case study 24 is explicitly a benign GPT-2/custom-data plumbing demo, not a safety result.
- Audit found notebook 03's per-language bootstrap summaries used only the clipped report scores even though raw values were nested in the report metadata. Updated the validation loop to capture `Pillar1_CausalPatching.run` and `Pillar2_Ablation.run` results and bootstrap raw signed refusal-margin scores, clean/corrupt baselines, and unclipped ratios. It also preserves denominator/status flags per seed. This matches CircuitKIT's public pillar implementation and keeps clipped scores secondary.
- Source inspection showed CircuitKIT's degenerate-denominator raw ratio is a numeric `0.0` sentinel. Added `eval/track2_extension/intervention_summary.py`; ratio summaries now include only finite values with both denominator flags explicitly false. Per-seed sentinel and flag remain preserved for audit. Three focused tests cover degenerate, inverted, missing, nonfinite, and valid values.
- After the correction, the local Track-2 suite passed **50 tests with one expected skip**. Notebook JSON/code AST parsing and `git diff --check` pass. No Colab execution occurred; GPU balance remains 0.00 CUs.

## 2026-11-12 — Recoverable Pipeline3 seed evidence is incomplete

- Directly listed the user-shared `Pipeline1111/combined/` Drive folder after filename search missed nested files. Read the 22-row `pipeline3_safety_pairs.jsonl` and the 24-row `pipeline3_seed_prompts.json` in memory, emitted only structural metadata, and did not save or reproduce prompt/response contents locally.
- The accepted preference pairs are distributed EN/control=6, EN/roleplay=5, JA/control=6, JA/roleplay=5. The pair schema contains `instruction`, `chosen`, `rejected`, and opaque `source_sample_id`; the companion prompt records contain language/framing/culture metadata. Neither has Dataset2 `seed_id`, so it cannot establish seed-level exclusion.
- Wrote `outputs/lexsi/prior_seed_inventory_partial_20261112.json`, explicitly `complete=false`. Pipeline2 probe seed IDs and any additional historical tuning exposures remain missing; do not run fresh-evaluation causal analysis until this gap is reconciled from source records.
- Follow-up exact join: all 22 `source_sample_id` values were found in the `prompt_id_a`/`prompt_id_b` columns of `matched_observations.csv` (85,884 rows); each maps to one seed, yielding 21 distinct Pipeline3 training seeds, with one seed represented twice. This resolves Pipeline3 only. Updated the partial inventory and created `outputs/lexsi/circuitkit_prior_seed_registry.json` with `complete=false`; the notebook must continue to fail closed because Pipeline2 probe seeds and any other tuning exposures are still unknown.
- Searched Drive by `pipeline2_prompts_small`, `pipeline2_prompts.json`, and `pipeline2_results.json`, and listed the Pipeline1111 `notebooks/` folder. The ad-hoc Pipeline2 input sample is not in the accessible Drive artifacts. `pipeline2_results.json` is present under `combined/`, but it does not provide a durable prompt/seed manifest. Do not infer its seed IDs from the summary statistics.

## 2026-11-12 — Reviewer submission and seed provenance audit

- Read the current Drive reviewer folders without exporting their prompt/response contents. Each reviewer sheet has 87 rows and all 87 `review_status` cells are blank. No review, independent agreement, or adjudication has been completed; no CircuitKIT input export is authorized.
- Static audit of `03_circuit_validation.ipynb` found that it checked fresh-evaluation seeds against discovery and Phase 1 metric-validation seeds, but had no required inventory for historical SFT/training/tuning exposure. A row's `previously_exposed=false` flag alone is not evidence of independence.
- Added `eval/track2_extension/seed_audit.py` and wired notebook 03 to require a complete, sourced `circuitkit_prior_seed_registry.json`; it rejects missing/incomplete inventories, seed overlap, duplicate rows, undersized language strata, and cohorts above the predeclared cap. Added unit tests. No inputs were exported and no Colab execution occurred.
- First local test run exposed a test assertion targeting the wrong language for a deliberately 8/7 split. Corrected the expected message to the Japanese stratum; implementation behavior was as intended. Follow-up suite result is recorded in the current test log.

## Exact-match candidate manifest build (2026-11-12; CPU Colab)

- Implemented `eval/track2_extension/data_io.py`, `manifest.py`, `build_manifest.py` and tests. The first real-data run passed SQLite integrity and read all 21,184 rows, but produced an empty manifest. Diagnosis: the SQLite stores `mix_direction=NULL` for English/Japanese rows; the first builder incorrectly treated this inapplicable field as missing. Changed the rule only for `en`/`ja` (the selected population) and added a regression test. No label or selection criteria were relaxed.
- One `colab exec` attempt lost its connection during remote work. The CPU runtime was still present and the Google Drive mount still existed; no session was provisioned. Used the already-authorized SSH path with the existing temporary key to complete the corrected builder. One subsequent command using the default SSH identity failed with `Permission denied (publickey)`; the previously provisioned task-scoped key succeeded. These were connection/auth route failures, not analysis errors.
- Successful actual-data run: source SQLite SHA256 `ecffa936c76c1009717b1a11d6597e3803878cc6e8e8b3443c0cfc39474b4e11`, 439,144,448 bytes, `PRAGMA quick_check=ok`; output has 89 exact US-neutral pairs / 178 prompts / 35 seeds. All are heldout and flagged as previously exposed; all human fields remain null, with 0 confirmatory-eligible rows. Manifest SHA256 `6faec95ffe31771f36fbc7fde798d177c7093a066a35c30e21c4e13af836ef82`, 180,580 bytes. Private Drive copy and local ignored copy are hash-identical; local structural validation passed.
- This 89-pair candidate cohort is smaller than the broad 792-row eligible count because exact strata matching plus completed/eligible/duplicate/conflict gates remove unpaired or invalid rows. Exact framing/language counts are in private `manifest_breakdown_private.json`; do not publish the prompt-bearing manifest.

## Matched three-arm smoke and persistence blockers (2026-11-12)

- **Successful bounded run:** On a real Colab NVIDIA L4, executed the resumable HF evaluation for baseline, restricted-SFT and broad-SFT on the same 8 heldout prompts (4 US-neutral pairs, 4 seeds). All model arms loaded NF4; all prompts were under the 7,168-token input cap and token hashes matched across arms. The official Qwen3Guard model judged all 24 outputs. The actual captured execution output and private artifacts are in ignored `outputs/lexsi/track2_pilot_20261112_nf4_smoke/`.
- **Result:** each arm had 7 Safe / 1 Unsafe proxy labels; the paired risk difference was 0 for both tuned arms and the four-seed bootstrap CI was `[0,0]`. This degenerate 8-row smoke result is only an execution/pairing check. It is not evidence of equivalence or improvement. Generation used greedy decoding with `max_new_tokens=192` because available GPU compute was constrained; do not compare it as the planned full validation.
- **Warnings:** Transformers reports that `torch_dtype` is deprecated; the loaded saved-checkpoint generation config contains an invalid/ignored `cache_implementation` value. They did not prevent generation but should be cleaned in a future focused update. Generation resource use was baseline 50.35 s / peak 2.471 GiB allocated, restricted 62.81 s / 3.511 GiB, broad 62.90 s / 3.511 GiB. Guard peak allocation was approximately 1.21 GiB per arm.
- **Drive mount blocker:** `colab drivemount -s lexsi-track2-rerun` failed while propagating authorization with HTTP 400 for both default and alternate `authuser`. A local patch to the installed official CLI to honor `COLAB_AUTHUSER` did not fix it. The user had already authorized Google; repeated OAuth was not a valid remedy. A separate already-authorized CPU session had Drive mounted, so small smoke artifacts were transferred there and verified by hashes.
- **Large checkpoint transfer blocker:** the restricted and broad `model.safetensors` files are each about 3.6 GB and exist only in the L4 runtime scratch. The official CLI contents download attempts to materialize whole files through base64; a direct attempt hung. Cross-session SSH piping transferred only about 23 MB in roughly two minutes and was stopped. Do not claim weights are persisted to Drive; recover through a supported mounted Drive in the L4 runtime or another durable artifact route before stopping that runtime.
- **Compute blocker:** `colab usage` reports 0.00 compute units (L4 rate 2.10 units/hour). Do not provision/run further GPU jobs until the account has compute units again. No purchase or quota bypass was attempted.
- **Preservation:** all small pilot artifacts (including private prompts and responses) were downloaded locally under ignored `outputs/lexsi/track2_pilot_20261112_nf4_smoke/`; the authorized Drive copy remains in `Pipeline1111/codex_extension_20261111/pilot_eval_20261112/run_nf4_smoke/`. `run_manifest.json` lists hashes, versions, generation settings, runtime, and memory. Nine CPU tests pass.
- **New Drive checkpoint candidate inspection:** connected Drive search/listing found a new `checkpoints/pipeline4_restricted` folder with `config.json` (2,500 bytes) and `model.safetensors` (25,231,360 bytes), versus 3,599,063,241 bytes for the same-named file in L4 scratch. The `pipeline4_full` Drive folder is empty. Searches for adapter and shard index names found none. The Drive candidate is incomplete; it was not downloaded or loaded. Private IDs and metadata are in `outputs/lexsi/checkpoint_drive_candidate_inventory_20261112.json`.

## Track-2 Drive artifact inventory (2026-11-11)

- Reused the already-authorized official Colab CLI session `lexsi-track2-drive` (CPU, standard shape) with the user's Google Drive mounted at `/content/drive`. No GPU was provisioned for this inventory.
- Searched checkpoint/adapter-like names and model-weight/archive extensions recursively under mounted `/content/drive/MyDrive`, then separately inspected project-folder matches. Search evidence contains relative paths and byte sizes only, not prompt or response data: `outputs/lexsi/checkpoint_search_all_mydrive.json` and `outputs/lexsi/checkpoint_search_project_folders.json`.
- The all-MyDrive search returned six weight/archive candidates: one unrelated attachment archive, three llama-cuda cache archives, the historical `tiny-aya-water-q4_k_m.gguf` (2,143,977,056 bytes), and the Qwen3Guard GGUF (804,753,408 bytes). It found no `.safetensors`, PyTorch checkpoint, adapter, or files named for `pipeline4_restricted`/`pipeline4_full`.
- The Pipeline1111 folder contains the five pipeline notebooks and `combined/pipeline4_results.json`; its checkpoint values are `/content/pipeline4_restricted` and `/content/pipeline4_full`, not durable Drive paths. Thus the teammate's adapter training is reported but the resulting weights were not recoverable from mounted MyDrive in this search. A separate shared-drive or teammate-owned location remains unsearched.
- The inventory JSON was saved under the private ignored `outputs/lexsi/` directory and copied into the private Drive extension folder. The CPU session was stopped after the audit. No new GPU compute was used; Colab usage showed about 2.28 compute units remaining before shutdown.

## Dataset2 source and raw-label audit (2026-11-11)

- A second official Colab CLI CPU session was mounted to the already-authorized Drive after the fresh runtime required a new Google OAuth approval. The user approved in-browser; no auth bypass was attempted.
- A streaming, read-only audit verified `combined/observations.csv`: 21,184 rows, 320 unique seeds (160/100/60 by analysis/tune/heldout), file SHA256 `7a38bedf6d2d3730154c65a3c6e1b022c5216a82850de3017deefe373d57ac0b`; 17,795 generation-completed, 3,387 truncated and 2 error rows. Stored binary safety scores remain 13,730.
- The merged 439 MB SQLite export was readable and passed `PRAGMA quick_check`. It has an `observations.raw_guard_outputs` column and 21,184 observations. A read-only parser run recovered 17,373 binary response labels (15,749 Safe, 1,624 Unsafe); 3,643 additions came from rows whose prompt-level guard was Controversial. It verified 21,184 unique prompt IDs, zero malformed raw record shapes, and zero changes to the 13,730 previously published binary labels. A private corrected-label JSONL (IDs, split and guard verdicts only; no prompt/answer text) was written to the dedicated Drive extension folder and checksummed.
- The SQLite `comparisons` table contains seed-bootstrap summaries for US versus neutral on the original exported outcomes: analysis unsafe paired delta +6.76 pp (157 seeds; stored interval +3.31 to +10.11), tune +6.62 pp (100 seeds; +3.07 to +10.39), heldout +3.29 pp (60 seeds; -2.92 to +8.69). The table labels these `seed bootstrap descriptive unadjusted`; these summaries have not yet been independently recomputed and use the old 13,730-score export, not the recovered 17,373 labels. The historical heldout set is exposed.
- **Caught issue:** the first audit-summary script mistakenly left the per-split label subtotals at zero. Its global recovery checks and private label output were correct. The intermediate Drive `guard_recovery_audit.json` is not authoritative for per-split subtotals; the corrected, metadata-only canonical summary is `outputs/lexsi/raw_guard_recovery_and_seed_comparisons_20261111.json`. This failure and correction are recorded rather than hidden.
- The corrected summary is in the ignored local outputs and was uploaded/read back through the connected Drive app as `guard_recovery_audit_corrected.json` in the private extension folder. The first intermediate Drive summary was renamed `guard_recovery_audit_INTERMEDIATE_INVALID_SPLIT_TOTALS.json`; its global totals are correct but its split label subtotals are zero because of the bug. Do not use that file for analysis.
- The official CPU session was stopped after the read-only work. No GPU was provisioned and no model inference was repeated.
- The corrected real-data reanalysis is implemented in `eval/track2_extension/reconcile_dataset2.py` and passes synthetic SQLite tests. Running it against the private 439 MB Drive SQLite still requires a fresh Colab CPU runtime with Drive mounted; the runtime used for raw recovery was already stopped. No data reanalysis results are claimed from this implementation yet.
- Outcome: checkpoint recovery is unresolved, not successful. Do not attempt a before/after adapter evaluation until the exact weights/config/base revision are recovered and reload-tested. If no other authorized storage has them, plan a replacement experiment and label it as new work, not a reproduction of the old adapter.

## Phase 1 run on 2026-11-10

- Google OAuth completed. The official Colab CLI 0.8.0 provisioned `lexsi-phase1` on a real NVIDIA L4 with 23,034 MiB of VRAM and driver 580.82.07.
- Installation and imports succeeded for PyTorch 2.14.1+cu130, Transformers 5.15.0, CircuitKIT 0.2.2, TransformerLens 3.8.0 and Hugging Face Hub 1.33.0. The first run measured 194 MiB in use after imports. The retry executed a CUDA matrix multiplication and measured 256 MiB in use. These are setup measurements before model loading.
- The first run raised `google.colab.userdata.TimeoutException` in the model cell. The error says that Colab Secrets can only be fetched from the browser UI. The CLI continued to later cells, which failed because the model checks had stopped. The CLI returned exit status 0 despite the notebook cell errors. At that point the original checkpoint, tokenizer, and backend parity were still unverified; the authorized checkpoint run below supersedes that interim status.
- Updated the notebook to use an authorized runtime environment or cached Hugging Face credential and explicitly guard the downstream cells. The retry passed the CUDA setup cell and stopped at the missing runtime credential. An independent request from this Mac still got HTTP 403 with the existing local Hugging Face credential for the pinned model config.
- The approved export remains empty. The five provisional English control-versus-roleplay candidates include four analysis seeds and one tune seed. Four of these seeds also appear in the 50-pair bilingual shortlist. A two-reviewer sheet and candidate-specific rubric are available at `outputs/lexsi/human_adjudication_5_control_roleplay.csv` and `outputs/lexsi/HUMAN_ADJUDICATION_GUIDE.md`. Record independent judgments, adjudication, and verbatim response prefixes before generating the execution export.
- At this early point in the sequence, the original checkpoint, tokenizer and backend parity remained unverified. The authorized checkpoint run below later verified them. The behavioral metric and circuit discovery remain pending. Raw logs, executed notebooks with traceback cells, dependency versions and GPU measurements are saved under `outputs/lexsi/colab_phase1/authorized/`.

## Authorized checkpoint run

The new Hugging Face credential returned HTTP 200 for the pinned Tiny Aya Water configuration. The authorized Colab run used an isolated runtime token cache. The logs contain no credential value.

| Attempt | Observed failure | Correction |
| --- | --- | --- |
| 1 | Transformers exposed `chat_template` as a mapping. The fingerprint step attempted `.encode()` on the mapping. | Hash the selected `default` template as UTF-8. Preserve the tokenizer and the generation prompt. |
| 2 | Original weights loaded and the teacher-forcing check passed. The 4,196-token TransformerLens parity probe exhausted the L4 while trying to allocate 1.05 GiB, with 21.17 GiB already allocated by PyTorch. | Process the same long sequence in 256-token chunks with a persistent key/value cache. Compute vocabulary logits only for the final token. |
| 3 | Short English and Japanese parity probes passed. The long probe's cache expected four KV heads while the EAP-IG configuration expanded them to 16. | Use compact GQA for the long cached probe and restore the EAP-IG setting afterward. Short probes still exercise the EAP-IG configuration. |

Attempt 4 passed the architecture and chat-template checks, the dense-versus-chunked teacher-forcing check, and all three backend parity probes. English and Japanese used 372 and 371 tokens. The long probe used 4,196 tokens across a 4,096-token sliding window. All top tokens matched. The largest absolute log-probability difference was 7.44e-5. The near-zero negative KL values for two probes reflect floating-point roundoff. The final cell reported that the approved export was empty. Peak PyTorch allocation was 13.22 GiB during Hugging Face scoring and 20.72 GiB during CircuitKIT parity. The temporary runtime credential was removed, and the CLI confirmed the GPU session terminated.

The saved Attempt 4 notebook output records the exact final-cell exception: `RuntimeError: No human-approved contrasts available. Export adjudicated control/roleplay pairs.` This is a deliberate research-data gate, not a passing metric test. There is no `behavior_validation.json`; the likelihood-margin behavioral gate remains pending.

The checkpoint revision, input tokens, FP32 precision, and parity thresholds stayed fixed throughout these retries. Executed notebooks and raw tracebacks are in `outputs/lexsi/colab_phase1/authorized/`.

## Resume on Colab

Request access on the [Tiny Aya Water model page](https://huggingface.co/CohereLabs/tiny-aya-water) and sign in on this Mac with `hf auth login` using an authorized read token. For a CLI run, provision the credential in the private runtime environment or Hugging Face token cache before executing the notebook. Browser Colab Secrets require an attached notebook UI. Keep credentials out of the notebook and the logs.

Use a named session and run from the repository root. The FP32 parity check is sized for an L4 or larger GPU. If the requested accelerator is unavailable, record Colab's response before changing hardware or numeric settings.

```sh
colab_cli="$HOME/.local/bin/colab"
session=lexsi-phase1
remote=/content/lexsi
logdir=outputs/lexsi/colab_phase1
mkdir -p "$logdir"

"$colab_cli" sessions
"$colab_cli" new -s "$session" --gpu L4
"$colab_cli" status -s "$session" | tee "$logdir/session.txt"
"$colab_cli" install -s "$session" 'circuitkit==0.2.2' 'transformers==5.15.0'
printf '%s\n' 'from pathlib import Path; Path("/content/lexsi").mkdir(exist_ok=True)' |
  "$colab_cli" exec -s "$session"
"$colab_cli" upload -s "$session" eval/dataset2_likelihood.py "$remote/dataset2_likelihood.py"

# Add the adjudicated export once available. The setup cells can run before this step.
if test -s outputs/lexsi/circuitkit_pairs.jsonl; then
  "$colab_cli" upload -s "$session" outputs/lexsi/circuitkit_pairs.jsonl "$remote/circuitkit_pairs.jsonl"
fi

cp notebooks/01_prepare_circuitkit_data.ipynb "$logdir/01_prepare_circuitkit_data.executed.ipynb"
set -o pipefail
"$colab_cli" exec -s "$session" --timeout 0 -f "$logdir/01_prepare_circuitkit_data.executed.ipynb" \
  2>&1 | tee "$logdir/execution.log"
"$colab_cli" log -s "$session" -o "$logdir/session_history.jsonl"
"$colab_cli" ls -s "$session" "$remote/phase1"
# Download the files produced by the cells that finished.
# Example:
"$colab_cli" download -s "$session" "$remote/phase1/environment.json" "$logdir/environment.json"
"$colab_cli" stop -s "$session"
```

Download `pip-freeze.txt`, `checkpoint.json`, `huggingface_memory.json`, `backend_parity.json`, and `behavior_validation.json` when present. Preserve the executed notebook, command exit status, actual cell tracebacks, GPU peak allocation and reservation, and any hardware or dependency change. Inspect the executed notebook for cell errors because the CLI can exit 0 even when cells fail. A missing or failed behavioral validation report leaves discovery pending.

The parity probes use the checkpoint's native chat template and cover English, Japanese, and input beyond the sliding window. CircuitKIT documents its Cohere2 port as experimental and describes its real-weight parity checks in the [Tiny Aya guide](https://circuitkit.lexsi.ai/advanced/tiny-aya/). The official CLI documents provisioning, notebook execution, file transfer, and shutdown in the [Colab CLI repository](https://github.com/googlecolab/google-colab-cli).

## 2026-11-11 — Corrected Dataset2 analysis blocked at Drive mount

- Added and locally validated `eval/track2_extension/reconcile_dataset2.py`, a reproducible SQLite parser and seed-matched analysis command. Its unit tests cover exact-label parsing, unknown handling, preservation of previously published binary labels, seed/stratum matching, and output serialization. The focused test modules passed: 7 tests, 0 failures. CLI `--help` also executed successfully.
- Provisioned an official Colab CLI 0.8.0 CPU session named `lexsi-track2-corrected-analysis`; no GPU was requested because this is a CPU analysis task.
- The CLI reached the Drive authorization handoff, opened Google's authorization page, then reported HTTP 400 while propagating Drive credentials to the VM. The source SQLite was therefore not available in this runtime and the actual-data reanalysis did not run. No workaround or alternate credential path was used.
- Stopped the CPU session immediately after the failed mount. No Colab session remains active.
- Next exact step: rerun `colab drivemount -s lexsi-track2-corrected-analysis` from an interactive terminal, complete the Google Drive authorization page in the browser, and press Enter in the same terminal only after approval. If Google again returns HTTP 400, inspect/update the official Colab CLI OAuth setup or use a user-mounted runtime and provide access through the supported Drive mount. Then run the documented reconciliation command against `Pipeline1111/combined/results.sqlite3` and save aggregate outputs privately. This remains provisional automated-guard analysis, not human validation or causal evidence.

## 2026-11-12 — EAP continuation-margin adapter prototype (local CPU)

- Audited CircuitKIT upstream `main` at `e3501bc72b349963db76df324f5dbd41ced2a02f`. The generic EAP CSV path exposes one-token labels; the built-in WinoGrande TaskSpec carries variable-length suffix IDs in labels and scores mean suffix log-likelihood. This supports a custom TaskSpec approach, but WinoGrande's shared-suffix objective is not our two-target refusal/compliance margin.
- Added `eval/track2_extension/circuitkit_metric.py` and `circuitkit_data.py`. The metric scores variable-length, right-aligned continuation targets using gathered logits and logsumexp (avoids constructing a full [batch, sequence, vocabulary] log-softmax buffer), applies +1/-1 refusal/compliance signs, and requires balanced batches. The TaskSpec row builder rejects provisional/unreviewed pairs and tokenization mismatches; it creates refusal/compliance rows with identical continuation token IDs across clean/corrupt prompt sides.
- Added `tests/track2_extension/test_circuitkit_metric.py`. Focused and full `tests/track2_extension` runs passed: 18 tests total. New synthetic tests compare row construction and the metric against hand-computed expected values, require finite nonzero gradients, reject provisional input, and ensure invalid batches/positions fail loudly.
- No remote Colab command or Tiny Aya forward pass occurred in this step. CircuitKIT was not installed in the local `.venv` at this point in the timeline. This observation was later followed by an actual pinned 0.2.2 CPU install and TaskSpec/collator smoke (see subsequent entries); Tiny Aya, EAP-IG and intervention gates remained pending.
- Local exact-source import probe: adding the cloned 0.2.2 source at the audited commit to `PYTHONPATH` and importing `circuitkit.backends.eap.eap_utils.collate_EAP` fails with `ModuleNotFoundError: No module named 'transformer_lens'`. The repository `.venv` has Transformers 4.57.6, while this CircuitKIT release requires Transformers >=5.15,<6 and pins TransformerLens 3.8.0. I did not mutate the repository environment or claim CircuitKIT integration. The actual Colab runtime has the compatible stack from the earlier model parity run, but current quota prevents further GPU execution; install/source verification can resume when a runtime is available.

## 2026-11-12 — Real CircuitKIT TaskSpec/collator smoke on Colab CPU

- Reused existing CPU session `lexsi-browser-export` (ID `m-s-ass1a2-1mpkjw3i6iyic`); it remained mounted to Drive. Installed through the official CLI: `circuitkit==0.2.2`, `transformers==5.15.0`, `transformer-lens==3.8.0`. A follow-up metadata check recorded Python 3.13.18, PyTorch 2.14.1+cpu, and CUDA unavailable.
- Uploaded a small source zip under `/content/` and executed a synthetic smoke notebook through `colab exec`. CircuitKIT imported, the custom TaskSpec registered with `ck.register_task`, `build_dataloader` imported and used CircuitKIT's actual `collate_EAP`, and the batch had labels shape `(2,16)` and refusal/compliance signs `[1,-1]`. The integration smoke passed.
- Failure 1: a version-print cell read `transformer_lens.__version__`, causing `AttributeError`; fixed to use `importlib.metadata.version('transformer-lens')`.
- Failure 2: the smoke bundle imported as `eval.track2_extension`, but this kernel had already loaded a different `eval` package from `/content/lexsi_track2_manifest_code.zip`, causing `ModuleNotFoundError`. Fixed by using a unique `lexsi_track2_extension` top-level package in the smoke bundle; the TaskSpec then imported and passed.
- Two intermediate CLI calls showed `RuntimeError: Connection was lost`; `colab status` confirmed the same session remained live and IDLE, and subsequent execution on that session succeeded. No session was recreated and no GPU job started.
- Durable evidence: `outputs/lexsi/circuitkit_metric_cpu_smoke_20261112/adapter_api_smoke.executed.ipynb`, `environment_check.executed.ipynb`, `execution.log`, `failures_and_fixes.md`, and `run_manifest.json` (all ignored/private). This is only API/data-loader integration: no model load, forward metric, Tiny Aya execution, EAP/EAP-IG, behavioral validation, or causal test occurred. The 0.00 GPU-unit balance remains the blocker for required GPU execution.

## 2026-11-12 — HF teacher-forcing parity on a tiny CPU model

- Uploaded the repository's `eval/dataset2_likelihood.py` into the existing CPU Colab session and executed a second smoke notebook. It built a deterministic random `GPT2LMHeadModel` (vocab 32, context 16, hidden size 16, one layer, two heads; seed 20261112), then compared a two-token refusal target and one-token compliance target.
- The CircuitKIT metric and the project scorer both produced margin `0.0189359188079834`; absolute difference `0.0`. Backpropagation through the metric produced finite, nonzero gradients. Output notebook: `outputs/lexsi/circuitkit_metric_hf_cpu_20261112/metric_reference_parity.executed.ipynb`.
- The first CLI attempt returned `RuntimeError: Connection was lost` before the notebook appeared in session history. `colab status` confirmed the same session remained IDLE; retrying the same notebook on the same runtime succeeded. No runtime was recreated.
- This tests generic causal-LM token shifts/normalization and the custom metric's gradient only. It does not validate refusal-vs-compliance behavior, Tiny Aya likelihood, EAP-IG, or causal intervention.

## 2026-11-12 — Approval-provenance audit and corrected CPU smoke

- Audit finding: the first synthetic TaskSpec CPU notebook used `review_status='human_approved'` and placeholder reviewer IDs solely to pass the real-data validator. This was not human review, and the metadata was misleading even though the row was a toy and contained no project prompts. The earlier executed notebook is preserved as `adapter_api_smoke.superseded_fake_approval_schema.ipynb` in the private smoke folder and is explicitly superseded.
- Corrected `circuitkit_data.py`: toy records now use `data_mode='synthetic_smoke'` and are accepted only when the TaskSpec is explicitly constructed with `allow_synthetic_smoke=True`. Normal execution still demands `human_approved` plus two distinct reviewer IDs; provisional data remains rejected. No synthetic approval or reviewer provenance is fabricated.
- Reran on the same actual Colab CPU session using CircuitKIT 0.2.2 and its EAP collator. The corrected synthetic mode passed, with signs `[1,-1]`; no model/EAP-IG was run. See the corrected executed notebook and explicit correction note in the private smoke folder.
- Expanded local track2 tests: 21 passed, including explicit synthetic opt-in, rejection of provisional real-mode rows, and prohibition on approval metadata in synthetic rows.

- Persisted the corrected CPU smoke notebooks, source bundles, environment outputs, logs, correction note, and run manifests to a new private Drive folder: `MyDrive/ez-penta/Pipeline1111/codex_extension_20261111/circuitkit_cpu_smokes_20261112/`. The Drive-side copy cell validated SHA-256 for all 10 run artifacts and its receipt was downloaded locally to `outputs/lexsi/circuitkit_cpu_smokes_20261112/drive_copy_receipt.json`. This folder contains synthetic-only smoke evidence; the first false-approval notebook is clearly marked superseded.

## 2026-11-11 — Follow-up runtime and alignment audit

- Rechecked the project inventory: the five rows in `human_adjudication_5_control_roleplay.csv` have blank adjudication status and no completed pair of reviewers/adjudicator. They remain provisional; the approved export is still empty. No research inputs were exported.
- Read the pinned CircuitKIT EAP tokenizer and attribution implementation. Clean/corrupt inputs are padded from the end so continuation positions align, but CircuitKIT warns that unequal prompt lengths or mid-sequence edits can make positional activation attribution unreliable. The adapter currently verifies shared continuation token IDs and chat-boundary stability, but it does not yet certify prompt-side positional correspondence. This is a real Phase 2 gate; the synthetic collator smoke did not test it.
- Attempted a fresh CPU Colab CLI call against the existing `lexsi-browser-export` session. The CLI returned `RuntimeError: Connection was lost` while the session status still reported IDLE. Local reruns are also unavailable in the present Python environments: the repository venv has no `pytest`, and system Python 3.9 has no `torch`. No dependencies or labels were changed to force a pass. Earlier captured CPU Colab evidence remains intact and hash-valid; this failed retry adds no new test result.
- GPU quota remains 0.00 compute units. No GPU execution was attempted. Next work is to require auditable prompt/answer position alignment on actual human-approved tune examples, restore a usable Colab CPU connection for integration tests, and wait for GPU units before model-level EAP-IG or causal interventions.

## 2026-11-11 — CPU metric-test rerun: stale notebook-kernel modules

- Uploaded the current source bundle and observed 3 failures in 8 tests. The uploaded ZIP and extracted `circuitkit_data.py` both matched the current repository SHA-256; its function signature included the expected `allow_synthetic_smoke` argument.
- Root cause: the persistent Colab IPython kernel retained both the earlier `test_circuitkit_metric` module and a `lexsi_track2_extension` package loaded from `/content/lexsi_track2_adapter_v2/`. Re-extracting the new ZIP did not replace those Python module objects, so the first rerun executed stale code despite the files on disk being current.
- Corrected the test command to remove the cached test module and all `lexsi_track2_extension` entries from `sys.modules`, then re-import from `/content/lexsi_recheck`. All 8 focused metric/data tests passed on the existing Colab CPU runtime (PyTorch 2.14.1+cpu; CUDA unavailable). No research labels or metrics changed.
- This resolves the source-version mismatch for the focused unit suite only. It does not rerun all 21 Track-2 tests, Tiny Aya forward validation, the approved-data metric gate, EAP-IG, or causal interventions.
# 2026-11-12 — EAP-IG position-map core audit

- Re-read CircuitKIT source pinned at commit `e3501bc72b349963db76df324f5dbd41ced2a02f`. In `get_scores_eap_ig()`, batches are unpacked as `(clean, corrupted, label)` and no row-level alignment metadata reaches `make_hooks_and_matrices()`. That function accumulates corrupted-minus-clean activation differences at identical position indices; its gradient hook multiplies those differences by same-index gradients.
- This confirms the repository's `alignment.py` helper is only a mapping contract, not an attribution fix. A TaskSpec-only change cannot consume it at intervention sites.
- Additional failure mode identified before implementation: EAP-IG derives embedding interpolation endpoints from the difference buffer. If unmapped framing positions are represented as zero differences without explicitly preserving the clean embedding, interpolation can move those positions toward zero and create a spurious input intervention. The optional core path must preserve clean embeddings at invalid/unmapped coordinates and mask their attribution gradients. The separate-activation normalization branch must be mapped correctly or reject maps explicitly.
- No upstream clone changes or runtime tests were performed during this audit; the pinned clone remains disposable under `/tmp/circuitkit-inspect-20261112`. The only live Colab session is CPU-only; no GPU or real model experiment was attempted. No contrast or approval metadata was changed.
- Fix plan: prototype an opt-in `position_maps` argument in the pinned core, retain the no-map code path byte-for-byte if possible, add hook-level toy tests covering identity maps, shifted prefix positions, omitted divergent spans, continuation positions, invalid-gradient masking, and input interpolation endpoints, then run the tests on Colab CPU. Do not wire this into real runs until those tests pass and source/version hashes are recorded.

## 2026-11-12 — Attempted Colab CPU rerun after alignment changes

- Bundled the current `eval/track2_extension` and `tests/track2_extension` trees and uploaded them to the already-authorized `lexsi-browser-export` CPU session. The local full suite had just passed 31/31.
- The first CLI invocation used a remote path with `colab exec --file`; CLI help shows `--file` expects a local path, so it failed before execution with `FileNotFoundError`. Corrected the command to provide the local runner path.
- The corrected execution attempt lost the Jupyter connection (`RuntimeError: Connection was lost`) before returning any notebook/test output. A subsequent `colab sessions/status` still showed the same session ID `m-s-ass1a2-1mpkjw3i6iyic` as IDLE. This attempt produced no Colab test result and no traceback from the remote test process.
- The source bundle remains uploaded at `/content/lexsi_eap_audit_suite_20261112.zip`; the runner at `/content/run_lexsi_eap_audit_suite.py`. Do not interpret presence of these files as execution. The current Colab CPU suite evidence remains the earlier 21/21 run; the newer 31/31 result is local CPU only.
- No dependencies, labels, metrics, approvals, or research conclusions were changed. No GPU was provisioned or used. Next, recover a working supported Colab execution channel (or run the core-hook tests locally after installing pinned CPU test dependencies) before claiming refreshed remote validation.

### Recovery and successful rerun

- A short probe confirmed the session was still reachable and reported Python 3.13.18, PyTorch 2.14.1+cpu, `cuda_available=False`, and zero CUDA devices. Installed package versions were Transformers 5.15.0, TransformerLens 3.8.0, CircuitKIT 0.2.2, and einops 0.8.2.
- The first connected test run reported one import error and two metric-test failures. Printing the imported module path showed that the persistent IPython kernel had loaded `circuitkit_metric.py` from the older `/content/lexsi_full_recheck_run` extraction. The runner had not replaced the cached top-level `eval` package; this was an environment contamination issue, not evidence of a source regression.
- Restarted the Colab kernel, explicitly cleared cached `eval.*` and `test_*` modules, put the new source bundle root first on `sys.path`, and reran. Confirmed the imported file was `/content/lexsi_eap_audit_suite_20261112/eval/track2_extension/circuitkit_metric.py`. All 31 tests passed (0 failures, 0 errors; 0.203 seconds).
- Captured run manifest, package versions, limitations, and source hashes in `outputs/lexsi/circuitkit_eap_alignment_cpu_20261112/`. This verifies the current CPU unit suite only. It does not test a patched CircuitKIT core hook, CUDA, Tiny Aya behavior, EAP-IG, or causal effects.

## 2026-11-12 — Patched EAP position-map integration prototype

- Re-cloned/audited CircuitKIT at exact commit `e3501bc72b349963db76df324f5dbd41ced2a02f`; added an opt-in patch for map-aware EAP/EAP-IG hooks and a variable-length map collator. The project TaskSpec now emits exact common-prefix and continuation maps, and its collator translates them to the same left-padding offsets used by CircuitKIT. The differing roleplay framing span remains unmapped.
- Patch behavior: gather corrupted activations into clean coordinates; mask gradients outside mapped positions; retain full clean embeddings as the unmapped EAP-IG endpoint; preserve the legacy three-field path; reject map use with the unimplemented separate-activation normalization branch.
- First expanded run exposed a wrong test assertion: it expected target coordinates from the clean string length, even though CircuitKIT pads jointly to the longest clean/corrupt string. Corrected the assertion to compute joint maximum token length. The final rerun passed 41/41 tests.
- Final Colab environment: CPU, Python 3.13.18, PyTorch 2.14.1+cpu, Transformers 5.15.0, TransformerLens 3.8.0, CircuitKIT 0.2.2, einops 0.8.2; no CUDA and no GPU-memory reading. Eight patch tests plus all 33 current Track-2 tests passed, including TaskSpec-to-collator map construction and actual EAP-IG on a random one-layer TransformerLens model. Identity mapping matched legacy EAP-IG scores; a partial-map EAP-IG run returned finite scores.
- The exact patch and instructions are `docs/lexsi/patches/circuitkit_eap_position_maps_e3501bc.patch` and `docs/lexsi/patches/README.md`; regression tests are in `tests/circuitkit_patches/`. Executed source bundle, test bundle, manifest, logs, and hashes are under `outputs/lexsi/circuitkit_position_maps_cpu_20261112/`.
- This does not pass the real-model gates. No approved examples or Dataset2 prompt text were used; Tiny Aya was not loaded; no GPU discovery, metric discrimination on human-approved behavior, circuit attribution, or causal intervention occurred. A production Colab notebook must install this exact patched source and verify its import path before using the TaskSpec.
- After saving and hashing the run bundle, confirmed the CPU session was IDLE and stopped `lexsi-browser-export` with the official CLI. No Colab session remains active from this work.

## 2026-11-12 — Follow-up runtime and Drive artifact check

- Rechecked the official Colab CLI before continuing: balance `0.00` compute units, usage rate `0.00/hr`, and zero active assignments; `colab sessions` returned no active sessions. No GPU provisioning or repeated OAuth attempt was made because there is no current assignment/quota to use.
- Connected Drive search and exact folder listings reconfirmed that the restricted candidate is a 25,231,360-byte partial weight file plus config, while the broad checkpoint folder is empty. No complete Pipeline 4 arm is recoverable from the searched project Drive folders.
- The same Drive connection successfully accepted the private reviewer packet upload. Reviewer 1 and reviewer 2 have separate folders; the private adjudication key is stored outside both. Metadata readback shows only the project owner and connected project account, with no reviewers granted access. Receipt: `outputs/lexsi/blind_review_packets_20261112/drive_upload_receipt_20261112.json`.
- No GPU experiment was attempted, no auth bypass was attempted, and no checkpoint was loaded or modified.

# 2026-11-12 — CPU source-analysis runtime Drive mount retry

- **Attempt:** created official Colab CLI CPU session `lexsi-corrected-match-audit` (`m-s-usw1c3-29ksxjvjub504`) to run corrected-label seed/cell analysis without consuming GPU compute units.
- **Failure:** `colab drivemount --session lexsi-corrected-match-audit` reached Google's Drive authorization handoff but credential propagation returned HTTP 400. No Drive contents became available in that runtime; no source data or analysis was transferred or run.
- **Diagnosis / prior evidence:** the installed CLI already honors `COLAB_AUTHUSER` in both request paths. Earlier documented attempts with default and alternate authuser also failed with HTTP 400. The Drive connector can read small artifacts but rejects the 375,017,472-byte reduced SQLite source because its raw-download limit is 268,435,456 bytes.
- **Action:** stopped the failed mount process and terminated the CPU runtime. The CLI `sessions` command returned no named active sessions. However, account-level `colab usage` continued to report one active assignment at 0.08 units/hour, which the CLI did not map to a session name. I did not stop an unidentified assignment that could belong to another user/browser session. Did not repeat the same auth propagation attempt or use an unsupported credential workaround.
- **Result:** corrected-label seed/cell-matched analysis from raw SQLite remains blocked. The existing seed-only summary was independently checked from the saved 665-row per-seed CSV; that does not substitute for raw-source reanalysis.

## 2026-11-12 — Corrected-label matcher NULL-dimension bug found before rerun

- **Audit finding:** source inspection of `eval/track2_extension/reconcile_dataset2.py` found that the generic matched-strata helper excludes any row with a NULL matching field. The actual SQLite uses `mix_direction=NULL` for monolingual English/Japanese because the dimension is inapplicable there. The previous reported zero English/Japanese matched strata came from this implementation behavior and is **not evidence that the source has no eligible matched examples**.
- **Fix:** encode monolingual `mix_direction` as an explicit not-applicable sentinel before matching; continue excluding mixed-language rows whose direction is NULL or blank. Added regression cases for both conditions.
- **Verification:** synthetic SQLite tests cover source-style NULL direction for English/Japanese and missing direction for mixed-language rows. Focused reconciliation/bootstrap tests passed (5 tests), and the full suite passed (37 passed, 1 expected skip; CircuitKIT is not installed locally). The raw-source Colab run remains blocked by failed Drive credential propagation.
- **Interpretation:** retract the earlier no-English/Japanese-match conclusion. Do not report corrected-label matched estimates until fixed code runs against the real SQLite source and resulting counts are audited.


## 2026-11-12 — public CircuitKIT EAP-IG position-map integration correction

**Issue found during pinned-source inspection:** The experiment notebook calls `circuitkit.api.discover_circuit()`, which imports and dispatches through `circuitkit.backends.eap.attribute_node.attribute_node`. The existing position-map patch modified the older `attribute.py` path and `eap_utils.make_hooks_and_matrices`, but not the local `make_hooks_and_matrices()` or the `get_scores_eap[_ig]()` functions in `attribute_node.py`. Thus the previous random-model smoke exercised a real EAP-IG implementation, but not the implementation used by `discover_circuit`; it did not validate the intended public discovery path.

**First public-path CPU test:** Added a regression test against CircuitKIT 0.2.2 / TransformerLens 3.8.0 in a real official Colab CPU session. The first run failed in the EAP head-gradient hook: head gradients have an extra head axis, and the `[batch, position]` validity mask was only expanded by one dimension. It also exposed that the graph in the test needed `node_scores=True`, matching `discover_circuit()`'s graph construction. The failure was an actual EAP-IG score accumulation shape mismatch, not a data or model label issue.

**Fix:** Extended the pinned patch to the actual `attribute_node.py` EAP and EAP-IG loops. It now accepts the optional fourth position-map batch field, gathers corrupted activations into clean coordinates, captures full clean and mapped-corrupt embedding endpoints, keeps clean embeddings at unmapped framing positions during EAP-IG interpolation, masks unmapped gradients with a rank-aware broadcast, and explicitly rejects the unsupported separate-activation normalization branch. The public-path regression test initializes the graph with node scores as the production API does.

**Verification:** A new official Colab CPU session (`lexsi-patch-map-cpu`) installed CircuitKIT 0.2.2 and TransformerLens 3.8.0. After uploading the exact commit-pinned patched source and clearing the long-lived kernel, the complete position-map suite passed **9/9**. The captured output is `outputs/lexsi/circuitkit_position_map_api_cpu_20261112/execution.log`; patched source archive and test copy are alongside it. Runtime reported PyTorch `2.14.1+cpu`, `cuda_available=false`; it was stopped after the test. The patch applies cleanly to commit `e3501bc72b349963db76df324f5dbd41ced2a02f`.

**Limits:** This closes only the synthetic public-API integration gap. It does not validate Tiny Aya behavior, approved real examples, token maps on real prompts, EAP-IG on Tiny Aya, GPU memory/performance, or any causal claim. No GPU was available (account balance 0.00 units). The discovery notebook now verifies that the patched checkout is imported before CircuitKIT and calls the API path that the test covers.

## Public `discover_circuit()` EAP-IG synthetic integration (2026-11-12; official Colab CPU)

- Created official Colab CLI CPU session `lexsi-public-discovery-cpu`, installed CircuitKIT 0.2.2, TransformerLens 3.8.0, Transformers 5.15.0, and ran the public `circuitkit.discover_circuit()` API with patched CircuitKIT source at upstream commit `e3501bc72b349963db76df324f5dbd41ced2a02f`. PyTorch was 2.14.1+cpu, Python 3.13.18, CUDA unavailable.
- **Failure 1:** the synthetic `GPT2TokenizerFast` was assigned a chat template, but `HookedTransformer`'s wrapped tokenizer did not retain it. `build_dataloader()` raised `ValueError: Cannot use chat template functions because tokenizer.chat_template is not set`. **Fix:** assign the template explicitly to `model.tokenizer.chat_template` after model construction.
- **Failure 2:** public EAP-IG raised `IndexError: index 31 is out of bounds for dimension 1 with size 31`; CircuitKIT warned both contrast prompts had unequal token counts (24 vs. 31), invalidating same-position assumptions for this smoke. **Fix:** use a deliberately equal-token-length synthetic clean/corrupt prompt pair. Do not generalize this fix to real prompts: it does not solve the known length-changing roleplay framing alignment issue.
- **Pass:** public EAP-IG completed in 0.2 seconds, wrote circuit and score `.pt`/`.json` artifacts, and the wrapper preserved eight signed edge scores before public node aggregation. Two node score entries were returned. The tiny model had one head and one MLP node; 30% pruning resolved to zero pruned nodes. This is API integration evidence only, not a meaningful circuit, not Tiny Aya, and not evidence of safety behavior or causality.
- Exact execution events, outputs, exceptions, and tracebacks are preserved in `outputs/lexsi/circuitkit_public_api_smoke_20261112/colab_session_log.jsonl` and `execution.log`. Artifacts and reproduction inputs are in the same directory. No GPU memory figure applies because the runtime was CPU-only. The runtime was stopped after artifact download.
- A no-token authenticated-HF warning appeared while downloading the public GPT-2 tokenizer/config; the public assets downloaded successfully. No gated Tiny Aya model was accessed. The smoke had zero approved or real examples.
- Post-stop account check: `colab sessions` reported no active sessions, but `colab usage` reported 0.00 compute units and 1 active assignment. This is inconsistent with the session list; the runtime itself is terminated. Treat the account assignment count as unresolved/stale and do not launch another job at zero quota.

## Checkpoint candidate byte-level validation and smoke artifact backup (2026-11-12)

- Re-listed the connected Drive checkpoint folders. Restricted still contains only `config.json` and a 25,231,360-byte `model.safetensors`; broad remains empty. Downloaded the restricted candidate into ignored local output for a parser check; SHA256 `62eb698703051260588f1cc0c2e85a1bbbfd4117b740e64d2eca028241feadc1`. `.venv` `safetensors.safe_open(..., framework='pt', device='cpu')` fails with `SafetensorError: Error while deserializing header: incomplete metadata, file not fully covered`. This establishes truncation/incompleteness and confirms it is not loadable. No model load was attempted.
- Uploaded the synthetic public-API smoke outputs, source bundles, and full Colab CLI session history to a new folder under the project Drive extension root: `https://drive.google.com/drive/folders/1p_tYjBiSIPuZ_d--RIJLZzRh3TaoUJxG`. Read-back listed all ten uploaded filenames with matching byte sizes. Local SHA256 values and Drive file IDs are saved in `outputs/lexsi/circuitkit_public_api_smoke_20261112/drive_upload_receipt.json`; remote SHA256 was not independently computed.
- Colab status after stopping the CPU runtime: `sessions` lists none; `usage` reports 0.00 compute units and one active assignment. Treat the mismatch as unresolved/stale; no new GPU job was started.

## 2026-11-12 — Archived Pipeline 2 seed-provenance recovery attempt

- Retrieved the teammate's archived `pipeline2_circuit_discovery.ipynb` and `pipeline2_results.json` from the shared Drive folder. The notebook source says it consumed a separately uploaded `/content/pipeline2_prompts.json` created ad hoc from `observations.csv`; that prompt file is not in the recovered Drive artifacts.
- The archived notebook has no saved execution counts or cell outputs, so it cannot supply a run-time sample manifest. The results JSON stores per-language layer/neuron summaries and four aggregate projection tests, with no prompt IDs, seed IDs, or row-level observations.
- Therefore Pipeline 2 exposure cannot be joined to Dataset2 seeds from these artifacts. This is an unresolved data-provenance blocker, not a reason to infer or regenerate the sample. `outputs/lexsi/circuitkit_prior_seed_registry.json` remains `complete: false`; discovery and evaluation data builders must keep failing closed until all historical exposures are sourced.
- No prompt/response content was copied into the audit log. No labels or experimental conclusions were changed, and no Colab session was started.

## 2026-11-12 — Reviewer packet state rechecked

- Read back both owner-shared Drive reviewer CSVs and parsed quoted multiline CSV records correctly: 87 rows in each, `review_status` blank in all 87 rows. The local five-row adjudication worksheet also has blank adjudication status for every row.
- No human approval or adjudication is inferred from packet creation, Drive access, or this state check. The approved CircuitKIT export remains empty; no input was exported or executed.
- This is a data-state check only. Colab still reports zero active sessions/assignments and 0.00 compute units; no runtime was provisioned.

## 2026-11-12 — Checkpoint Drive revision and alternate-copy audit

- Queried Google Drive revision history for the restricted `model.safetensors` candidate. It has one 25,231,360-byte revision and no prior revision ID; the truncated file cannot be restored from version history.
- Repeated global Drive search for restricted/full checkpoint folder names and common PEFT/model weight filenames; no alternate full checkpoint or adapter copy was found. Existing incomplete safetensors validation result is unchanged.
- This closes another recovery avenue only; it does not prove the files never existed elsewhere. Inventory updated at `outputs/lexsi/checkpoint_drive_candidate_inventory_20261112.json`. No weight was loaded, modified, or exported.

## 2026-11-12 — Prior-review provenance audit for the “50 valid pairs”

- Read the saved annotation records without printing their prompt/response text. All 514 rows in `provisional_reviews.jsonl` are `reviewer_type=codex_assisted`, `annotator_id=codex`, and `review_status=provisional`; all 24 rows in `bilingual_provisional.jsonl` have the same provisional Codex provenance.
- The earlier 50/52 bilingual meaningful-difference shortlist therefore records automated review judgments, not a human sign-off. Those bilingual rows also do not satisfy the current primary design's same-language, same-intent control-versus-roleplay contrast requirement.
- They may support provisional dataset analysis, but they cannot populate the approved CircuitKIT export or validate the refusal/compliance metric under the repository's approval rule. No labels were promoted and no input was exported.

- A limited Drive search for reviewer return files and approved refusal/roleplay pairs returned no such artifacts. This keyword search is not exhaustive evidence that no copy exists elsewhere; the directly inspected reviewer CSVs remain the authoritative available review state and are all blank.

## 2026-11-12 — Resumable evaluation provenance hardening

- **Audit finding:** `colab_pilot_eval.py` previously keyed generation/judgment resume only by `prompt_id` (and, for generation, separately checked prompt-token hashes). A cached guard result could therefore be reused after a generated answer or guard model/config changed. Final and partial JSONL files were also not merged: resuming an incomplete final file could replace it with only newly generated rows. The script could begin baseline inference before discovering a required local adapter path was absent.
- **Fix:** added `resume_integrity.py` with content-addressed model-file fingerprints, generation/judgment key construction, duplicate/stale-record guards, and local-arm preflight. The runner now pins the resolved guard revision, validates exact answers and tokenized judge inputs before resuming, merges identical final/partial records, and rejects conflicting duplicates or mismatched prompt sets. No existing private outputs were rewritten.
- **Verification:** `tests/track2_extension/test_resume_integrity.py` covers content changes, configuration changes, missing adapter preflight, stale/missing/duplicate provenance, and journal merge/conflict behavior. Full local suite: 57 tests run, 56 passed, one expected CircuitKIT-not-installed skip. Python compilation passed. This does not verify a real runtime resume, GPU inference, or any safety effect.

- Follow-up preflight audit found the first adapter-path test fixture created only an empty directory. Once preflight began checking for real HF files, the test correctly failed, revealing the fixture—not the checker—was incomplete. Replaced it with a tiny valid config plus a Safetensors tensor. Focused resume/preflight tests then passed 9/9; full Track-2 suite ran 59 tests, 58 passed and one expected skip. Python compilation passed.

## 2026-11-12 — Recovering interrupted JSONL journal tails

- **Failure mode:** a VM shutdown can interrupt a JSONL append after a complete record but before the final newline/closing bytes. The previous loader treated this as fatal, and the previous journal merge left a torn fragment untouched when it was the only remaining row. That would prevent automatic resume or append after a disconnect.
- **Fix:** only partial journals may discard one malformed, unterminated final fragment. Malformed completed files, malformed newline-terminated rows, or corruption before the tail still fail closed. The merged partial journal is now rewritten through a same-directory temporary file, flushed/fsynced, and atomically replaced.
- **Verification:** tests cover preserving earlier rows, dropping the torn tail, clearing a tail-only partial, and rejecting interior/completed-file corruption. Full Track-2 suite ran 62 tests: 61 passed, 1 expected skip; Python compilation passed. No real Colab disconnect/resume was simulated and no private run artifacts were modified.

- Fresh external-state check after this code change: both Drive review packets still contain 87 blank statuses; official Colab CLI reports no sessions, 0 active assignments, and 0.00 compute units.

## 2026-11-12 — Shared generation settings and run-environment provenance

- **Audit finding:** despite identical explicit `do_sample=False`/token caps, each checkpoint could still contribute its own saved `GenerationConfig` defaults (including EOS and other generation controls). This could confound baseline-versus-adapter comparisons.
- **Fix:** the runner now resolves one pinned base-checkpoint `GenerationConfig`, overrides behavior-critical deterministic settings and tokenizer special-token IDs, and passes the same config to every arm. The complete effective config is included in each generation resume key and run summary. Run metadata now includes checkpoint fingerprint summaries, resolved guard revision, Python/CUDA/GPU properties, relevant dependency versions, and runner SHA256. Per-row records store only the model fingerprint hash to avoid repeating a full shard inventory.
- **Verification:** a local test confirms checkpoint defaults are overridden and EOS/BOS/PAD IDs come from the shared tokenizer. Full Track-2 suite ran 63 tests: 62 passed, 1 expected skip; Python compilation passed. This validates config construction, not actual Tiny Aya generation on Colab.

## 2026-11-12 — Matched dense model precision in the three-arm runner

- **Audit finding:** the baseline branch requested NF4 4-bit quantization, but both local tuned-checkpoint branches loaded BF16 without quantization. This would make the arm comparison sensitive to numerical precision and memory behavior.
- **Design revision:** an initial correction proposed applying NF4 to every arm. Inspection of the reported Pipeline4 save path showed the tuned files are merged full-model checkpoints, so quantizing them again could add error to the learned deltas. Removed that approach and its dependency on `bitsandbytes` for inference.
- **Correction:** the baseline and both merged checkpoint arms now load densely at the same dtype, selected as BF16 only when `torch.cuda.is_bf16_supported()` reports support and FP16 otherwise. Identical `torch_dtype` and `device_map` are bound into generation provenance; any unexpected 4-bit load fails. Each arm uses a fresh model load with the same scalar settings; only the base model gets the pinned HF revision.
- **Debug cycle:** the first unit test instantiated an actual 4-bit config on the local Mac, where `bitsandbytes` is absent, and failed at Transformers' package-version check. The final dense path removes this dependency and its loader tests mock only model construction, verifying that baseline and local checkpoint branches receive the same dtype/device-map settings.
- **Verification:** focused resume suite 15/15 passed; full Track-2 suite 65 total, 64 passed, 1 expected skip; Python compilation and `git diff --check` passed. No Colab runtime was started because the account has 0.00 compute units and the requisite two model checkpoints are unavailable. Dense loading and 8K-context memory fit remain unverified.
- **Fresh dependency readback:** official CLI reports no sessions, no assignments, and 0.00 compute units. Both Drive reviewer CSVs still have 87 blank statuses. Restricted checkpoint folder contains only the known incomplete 25,231,360-byte weight candidate and config; broad folder is empty.

## 2026-11-12 — Local reviewer reconciliation and export gate

- Added a fail-closed importer for the two blinded 87-row reviewer packets. It validates case/key/source joins, unmodified source prompts and responses, reviewer IDs, rubric values and exact evidence spans; creates a pending adjudication worksheet with source hashes; and never converts review completion into approval.
- Added an explicit export path requiring two distinct completed reviewers, a separate adjudicator, equivalent harmful requests, a safe clear-refusal control, an unsafe complied roleplay response, exact distinct continuation prefixes, and analysis/tune split eligibility. Export rechecks current source packet/key hashes and worksheet provenance before writing private JSONL and a manifest.
- Initial focused run exposed a test assertion mismatch: semantically equivalent CSV re-encoding is rejected first by canonical worksheet/hash comparison, as intended. The test now accepts either explicit changed-field or packet-provenance rejection. Final focused suite passed 5/5; full Track-2 suite passed 69 with one expected skip; `py_compile`, CLI help, and `git diff --check` passed. First full-suite count is 70 total.
- Read-only check of current private packets: reviewer 1 and reviewer 2 each have 87 rows and zero completed statuses. Reconciliation correctly failed closed with 176 validation issues; no adjudication worksheet or CircuitKIT export was created. No Colab runtime or model inference was used.

## 2026-11-12 — Local prompt-only coverage audit (no Colab transfer)

- Current offline inventory showed the only local Dataset2 SQLite is the 27 MB validated prompt-construction source, not the 375 MB raw outputs SQLite: `eval.dataset2_data.load_dataset()` validated 21,184 prompt rows, 320 seeds, source hash `296be6962f573eac1133f099eac098a16318b3d708e1cc9e541ea6fb53eea21f`, and 42 complete 84-variant source seeds. The file contains no generated responses or guard labels, so it cannot run the missing matched response analysis.
- Added a private aggregate structural coverage audit for exact-request harmful control/rp prompt structures. Result: 18 split/culture/language strata; analysis US had 134 structurally matched seeds per language (18 complete source seeds), tune US had 84 per language (11 complete source seeds). These are inventory counts, not validated behavior changes or approved CircuitKIT examples.
- Synthetic tests pass 2/2; full Track-2 suite 72 total (71 passed, one expected skip); script compilation and `git diff --check` pass. No model inference, data transfer, or Colab session occurred.
# 2026-11-12 — Local raw-shard reconciliation and test suite

- **Transfer constraint:** the connected Drive interface rejects an individual file larger than 268,435,456 bytes, so the historical merged 375 MB DB could not be fetched directly. Located the two original result shards (221,552,640 and 217,358,336 bytes) and downloaded them separately into a private Application Support directory outside the Desktop/iCloud-synchronized repository.
- **Failure:** the initial repository-local copy was marked by macOS as a `dataless` iCloud placeholder; opening SQLite blocked on file reads. Stopped the read-only process and fetched complete copies outside iCloud. Verified expected sizes and SHA256s before use; no data was mutated.
- **Failure/fix:** invoking the shard wrapper as a file (`python scripts/lexsi/reconcile_dataset2_shards.py`) did not put the project root on `sys.path` and raised `ModuleNotFoundError: eval`. Re-ran as a package module with the project virtualenv (`.venv/bin/python -m scripts.lexsi.reconcile_dataset2_shards`); the real-data reconciliation completed successfully. Repro instructions should retain module invocation.
- **Results:** two shards reconciled to 21,184 rows/320 seeds; exact corrected-label crosscheck 21,184/21,184 with zero mismatches; no changes to previously published binary labels; 9 strict seed/cell-matched split/language summaries. No model inference or GPU used.
- **Verification:** independent NumPy PCG64 bootstrap (20,000 draws per group) reproduced all seed-effect means; largest interval endpoint difference from the original 5,000-draw intervals was 0.52 percentage points. Full local Track-2 suite completed: 75 tests, 74 passed, one expected skip (CircuitKIT is not installed locally). This verifies local code, not the GPU notebooks.
- **Remaining:** Colab CLI reports no active sessions and 0.00 compute units. The original Pipeline 1 Mantel–Haenszel statistic still needs an independent reconstruction. Guard outcomes are proxies and this association is neither causal nor human-validated.

- **Pipeline 1 reconstruction:** a new read-only local shard script reproduced the 4,069-row cohort, exact language-specific binary tables, MH OR 1.9822, and continuity-corrected CMH p=2.21e-9 using the published `safety` column. After retrieving the archived Pipeline 1 notebook from Drive and inspecting its RBG formula, added the same variance calculation to the script; it reproduces CI [1.5831753, 2.4818013]. The script and aggregate result are recorded in the repo and private run manifest. Extended the script to recompute Japan-vs-neutral and roleplay-only comparisons; all four archived comparison ORs, CIs and p-values match their report values.
- **Synthetic test correction:** the first new MH unit-test fixture had the expected OR but only 12 rows per stratum, so its assertion that p<0.05 failed. Increased the synthetic counts while preserving cell proportions; this tests the p-value branch with adequate toy counts and does not change any real-data labels, method, or result. Final suite: 77 tests, 76 passed, 1 expected skip.


## Fresh blinded-review packet readback (2026-11-12)

- Connected Drive fetches of the two known reviewer CSV file IDs returned 87 cases apiece, with all 87 `review_status` values blank for both reviewers. Modified timestamps: 06:16:27Z and 06:16:19Z on 2026-11-12. The parser emitted only aggregate status counts; no prompt/response text or IDs were output. Approval gate remains closed.

## Expanded Pipeline 1 reproduction and regression fixture (2026-11-12)

- Extended the source-shard reproduction from US/all-framing only to all archived contrasts: US/Japan versus neutral, both all-framing and roleplay-only. The output now matches all four archived ORs, RBG CIs, and CMH p-values.
- First full test rerun failed because its synthetic table lacked the newly required `framing` field. After adding control and roleplay records for all three culture conditions, a second test exposed a fixture expectation error: the all-framing table intentionally includes both control and roleplay rows. Corrected the expected synthetic denominator from 1,200 to 2,400 without changing the real-data analysis.
- Final Track-2 local suite: 77 tests, 76 passed, one expected CircuitKIT-not-installed skip. No GPU or model inference occurred.


## Resource and checkpoint blocker recheck (2026-11-12 11:02 UTC)

- Official Colab CLI reports 0.00 compute units and no active sessions/assignments. No GPU runtime was started.
- Connected Drive search returned the same incomplete restricted Pipeline 4 weight file and empty broad checkpoint folder; no recovery source was found.
- Both reviewer packets remain blank. No real-input behavior validation, Tiny Aya EAP-IG, causal test, or model-arm evaluation can responsibly proceed without the required approvals and compute.


## 2026-11-12 — Current local verification and Colab PATH check

- Re-ran the complete Track-2 local unittest suite and captured output at `outputs/lexsi/latest_local_tests_20261112_current.log`: 77 tests, 76 passed, one expected skip because CircuitKIT is not installed in the local environment. This is a tooling/synthetic-data check only.
- The first shell invocation of `colab` returned command-not-found because `/Users/marcuschang/.local/bin` is not on the shell PATH. Located the installed official CLI at `/Users/marcuschang/.local/bin/colab` and invoked it directly. Authentication is active; current account readback is 0.00 compute units, zero assignments, no sessions. No GPU runtime was provisioned.
- Updated `outputs/lexsi/run_manifest.json`, `docs/lexsi/STATUS.md`, and `docs/lexsi/REPRODUCE.md` so their latest test count and current CLI state agree.
- Still blocked on user/reviewer actions or external resources: completed blind reviews/adjudication, recovery/retraining of complete durable checkpoints, recovery of the Pipeline 2 seed exposure manifest, a fresh untouched evaluation cohort, and restored Colab GPU compute. No results or labels were changed to bypass these gates.


## 2026-11-12 — Colab balance updated after user purchase

- Official CLI invoked directly at `/Users/marcuschang/.local/bin/colab`: `usage` reports 100.00 compute units, 0.00/hour, zero active assignments; `sessions` reports none.
- This supersedes earlier zero-quota statements as current state. No runtime was provisioned. No GPU usage occurred during this check.
- Manifest, status, experiment ledger, and decision log now reflect the new balance. Remaining research gates are data approval, historical exposure provenance, and valid checkpoint/evaluation inputs.

## 2026-11-12 — Phase 1 notebook gate and fresh-runtime credential recheck

- **Previous real-weight execution evidence:** the executed Phase 1 notebook loaded `CohereLabs/tiny-aya-water` at revision `41a99216d12b3ef4881df762934186e5e4d2bbc1` on an NVIDIA L4 (23,034 MiB). English, Japanese, and sliding-window HF↔CircuitKIT checks passed; maximum log-probability errors were `3.24e-05`, `3.34e-05`, and `7.44e-05`. Peak allocated GPU memory was 22,248,377,344 bytes.
- **Failure discovered by inspecting cell outputs:** the final cell raised `RuntimeError: No human-approved contrasts available. Export adjudicated control/roleplay pairs.` The CLI attempt file said exit code 0 even though the executed notebook contains this traceback. Therefore CLI exit status alone is not sufficient execution evidence.
- **Code correction:** `notebooks/01_prepare_circuitkit_data.ipynb` now writes `behavior_validation_status.json` with `BLOCKED_HUMAN_APPROVAL_REQUIRED` and exits that cell cleanly when the approved JSONL is absent or empty. If approved data exist but fail the preregistered behavioral threshold, it saves `VALIDATION_FAILED` and does not throw a misleading infrastructure exception. The metric and approval rules were not relaxed. A local execution of the no-approved-data branch produced the explicit blocked JSON; notebook code cells parse successfully.
- **Fresh Colab access check:** created an official Colab CLI L4 session, verified `NVIDIA L4, 23034 MiB`, and probed credential availability without printing secrets. Neither `HF_TOKEN` environment variable nor Hugging Face token cache was present in the new runtime, so no gated model download or notebook rerun was attempted. The notebook now retrieves `HF_TOKEN` through Colab `userdata` when configured and gives a precise setup error otherwise.
- **Cleanup:** stopped the L4 session after the access probe; `colab sessions` is empty, active assignments are zero, and usage returned to `0.00/hr`. Colab balance remains 100.00 CUs.
- **Current gate:** the historical real-weight parity evidence remains valid, but the corrected notebook has not yet been rerun on Colab. The owner must add a Hugging Face read token as a Colab Secret named `HF_TOKEN` and enable notebook access. Even after that, metric validation remains blocked until the approved contrast export is nonempty.

## 2026-11-12 — Notebook cell-error audit utility

- **Evidence:** the saved Phase 1 L4 execution returned CLI exit code 0 but had a `RuntimeError` output in cell 5. The first post-hoc review depended on manual inspection and could be repeated inconsistently.
- **Change:** added `scripts/lexsi/audit_notebook_execution.py`; it hashes the executed notebook, counts code cells, returns a nonzero exit for any `output_type=error`, and writes a private JSON report with exception name/message and traceback. Added synthetic tests for clean execution, cell-level exceptions, and malformed notebook input.
- **Check:** the auditor ran against the saved attempt-4 notebook and correctly returned `CELL_ERRORS` / exit code 1, identifying cell 5 and the missing human-approved contrasts error. Private report: `outputs/lexsi/colab_phase1/authorized/attempt4_cell_audit.json`.
- **Full regression:** 84 Track-2 tests, 83 passed, one expected skip. Log: `outputs/lexsi/latest_local_tests_20261112_execution_audit.log`.

## 2026-11-12 — Colab Secret UI requirement and Phase 1 staging

- **Credential probe:** `colab exec` on the active L4 called `google.colab.userdata.get("HF_TOKEN")`; Colab returned `TimeoutException: Requesting secret HF_TOKEN timed out. Secrets can only be fetched when running from the Colab UI.` The probe did not print or write the secret. A separate environment/cache probe was false, so the CLI cannot establish whether the owner already saved the secret.
- **Staging:** kept the same official CLI L4 session `lexsi-hf-secret-recheck` (`gpu-l4-s-ass1a1-gprt6hb6qmv8`) active and uploaded notebook 01, `eval/dataset2_likelihood.py`, and the empty approved-pairs file under `/content/lexsi/`. No model download, prompt data, or GPU inference has occurred.
- **Machine record:** private manifest `outputs/lexsi/colab_phase1/ui_secret_bootstrap_20261112/run_manifest.json` records the session, uploaded-file hashes, quota/rate, credential probe result, and pending state without a token value.
- **Next operation:** use the browser URL emitted by `colab url -s lexsi-hf-secret-recheck` to attach to this exact L4 runtime. In that UI, add/enable the `HF_TOKEN` Colab Secret if needed, run a small cell to set `os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")` without printing the value, then notify Codex. CLI `colab exec` can use the already-populated runtime environment afterward. Do not open a different runtime or paste the token into chat.


## 2026-11-12 — Fresh L4 dependency and credential probe

- Provisioned official Colab CLI session `lexsi-track2-gpu-audit-20261112` (`gpu-l4-s-ass1a0-1l7s0mxgkubfd`), confirmed NVIDIA L4 (23,034 MiB). Installed CircuitKIT 0.2.2, Transformers 5.15.0, and TransformerLens 3.8.0; PyTorch remained 2.14.1+cu130. Peak model allocation was zero because no model was loaded.
- `google.colab.userdata.get("HF_TOKEN")` through CLI execution timed out with `TimeoutException`; this only confirms that secret retrieval requires the attached Colab browser UI. It does not establish whether a secret is saved. No token was printed or transferred, and no model load/inference was attempted.
- Stopped the session after the probe. Official CLI now reports zero sessions/assignments, 0.00 CU/hour, and 99.90 CUs. Full sanitized report: `outputs/lexsi/colab_gpu_env_check_20261112.json`.
- This is dependency/runtime verification, not a rerun of the prior HF↔CircuitKIT parity experiment and not behavioral validation. The earlier real-weight parity evidence remains the separate L4 result documented above.

- Follow-up usage readback after settlement: 99.83 CUs, 0.00 CU/hour, zero active sessions/assignments. The immediate stop readback was 99.90 CUs; 0.07 CU settled afterward.


## 2026-11-12 — Phase 1 real-weight run with owner-authorized exploratory pairs

- **Credential/access resolution:** the project owner's local Hugging Face CLI identity was authenticated. To avoid Colab UI-only `userdata.get()` timeouts, a short-lived read-token file was uploaded directly to the requested Colab runtime, read into the runner environment, and deleted before notebook execution. The value was not printed or written into the notebook, execution output, or manifest. The runtime was stopped after downloads. This avoids requiring another browser-secret handoff for this run; any reproduction must provide its own authenticated HF access.
- **Execution:** official CLI created `lexsi-track2-phase1-diag-20261112` on NVIDIA L4. Uploaded the notebook, likelihood helper, private seven-pair input and runner; `nbclient` executed all notebook cells. Environment versions were Python 3.13.16, CUDA 13.0, PyTorch 2.14.1+cu130, Transformers 5.15.0, CircuitKIT 0.2.2, TransformerLens 3.8.0, and huggingface-hub 1.33.0.
- **Result:** real checkpoint/config/tokenizer/chat-template and backend parity checks passed. Max absolute log-probability differences were 3.2425e-5 (English), 3.3379e-5 (Japanese), and 7.4387e-5 (4,196-token sliding-window). Peak PyTorch memory was 22,248,377,344 bytes; parity `nvidia-smi` memory was 21,608/23,034 MiB. HF-only peak allocation was 14,197,043,200 bytes.
- **Scientific gate result:** seven owner-authorized LLM-provisional examples (six analysis seeds, one tune seed) were scored. Analysis had 6/6 positive relative control-minus-roleplay deltas but expected absolute signs in only 1/6. Tune had n=1, p=0.5 and 0/1 expected signs. Status is `INSUFFICIENT_VALIDATION_DATA`; this exposes insufficient metric discrimination and sample size. No EAP-IG or causal work followed.
- **Execution audit:** the notebook audit found five code cells and zero cell errors. The results and captured per-cell outputs are in `outputs/lexsi/circuitkit_owner_authorized_exploratory_20261112/`. The L4 was stopped; final usage 99.57 CUs, zero active assignments/sessions, 0.00 CU/hour.
- **Code/test correction:** the notebook accepts a separate owner-authorized exploratory schema while retaining LLM-provisional provenance and records a non-pass for fewer than five non-tied tune seeds. The formal human-adjudicated export remains unchanged. Full local regression: 84 tests, 83 passed and one expected skip.
