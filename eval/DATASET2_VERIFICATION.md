Dataset2 completion audit
=========================

Latest release: the default is now **GGUF / llama.cpp**, following the user's request to reuse the earlier working inference workflow. All **116 tests passed without skips** in 26.597 seconds with the opt-in real native CPU protocol test enabled. Actual Tiny Aya/Qwen GGUF GPU inference and the full Dataset2 evaluation have not been run locally. Earlier fp16 sections below are historical verification of retained optional helpers.

This file records implementation evidence, not a claim that the 21,184-prompt GPU evaluation was run. Team members execute the two full shards in Colab.

| Plan requirement | Current authoritative evidence |
|---|---|
| Inspect and reuse repository components | Repository audit in DATASET2_RESEARCH.md; existing guard parser, guard message builder, hashing and AuditKIT interfaces imported directly. Existing unrelated changes retained. |
| Two independent executable notebooks | eval_dataset2_part1.ipynb / part2.ipynb; identical code except SHARD_ID; default full_shard and no seed limit. Fixture execution tests run every pipeline cell in order for both notebooks. |
| Exact seed-preserving halves | Checked-in manifests/dataset2_shards.csv + JSON dataset/manifest fingerprints; make_manifest and verify_manifest; 160 seeds / 10,592 prompts per part, 21,184 union, no overlapping IDs or divided seeds. |
| Balance and reproducibility | Input-order perturbation test; exact 16 seeds per category and 80/50/30 original splits per shard; validate_dataset2 exports all balance/coverage tables. No independent random sampling in notebooks. |
| Dataset validation and original preservation | Read-only SQLite connection, integrity/schema/ID/metadata/count/condition checks across prompt and auxiliary seed relationships, source SHA256 regression test, quality_flags and missing_variants; validation CSV/JSON in each notebook. |
| Semantic uncertainty and incomplete seeds | All 320 seeds retained, generation_eligible always true after structural validation; identical twin contrasts excluded explicitly; all generated expectations unvalidated; review flags distinguish script heuristics and human review needs. |
| Colab environment, authentication and persistence | requirements-dataset2.txt; package_dataset2_colab embeds an identical integrity-checked source/data snapshot in both notebooks and can create an optional source ZIP; notebooks mount optional Drive and use HF Secrets/hidden login; no credential literals. |
| Tiny Aya settings and provenance | HFRunner uses pinned model/tokenizer commits, fp16, chat-template rendering with no duplicate special tokens, attention masks, specified greedy parameters and 1024-token cap; journal records original condition fields, exact token IDs/text/counts, EOS/truncation, config/software/run IDs and errors. |
| Independent safe checkpoints | init_run owns partN directories, Journal rejects foreign-shard records, flush/fsync per observation, damaged-tail backup/recovery, interior-corruption failure, deterministic duplicate resolution and explicit retry/permanent failures. |
| Valid resumption and cache invalidation | Exact prompt/config/implementation identities; answer identity included in score identity; tests cover partial resume, idempotence, changed inputs/models/parsers and rescoring without response regeneration. |
| AuditKIT/Qwen guard reuse | Real AuditKIT 1.1.3 fixture calls using precomputed answers and GuardJudge; full raw outputs saved separately from parsed metadata. Aya released before guard load. |
| Separate measurements and unknowns | Observation table preserves Safe, Unsafe, Controversial, Unparsed, missing/error and guard prompt safety; guard refusal and heuristic separate; unvalidated correctness/attack-success/over-refusal remain unknown. |
| Shard/global summaries | Headline and condition summaries include explicit assigned/eligible/evaluated/known/unknown/missing/error/seed counts; truncation separate; split-specific outputs. |
| Within-seed matched analysis | matched_observations joins every nontreatment condition within seed/split/kind; seed-level mean deltas and cluster bootstrap; no naive independent-prompt hypothesis tests. Missingness sensitivity exports and exploratory unadjusted CI policy explicit. |
| Merge compatibility and complete inventory | merge_dataset2_results validates source revision, manifest, model/tokenizer/template/generation/judge/parser/software identities, prompt inventory, exact original metadata and answer-score identities; duplicate/missing assignment rejection tests; real full-size union test with known and explicitly missing outcomes. |
| Combined exports and inspection | answers_current.jsonl / scores_current.jsonl, results.sqlite3, coverage, headline, denominators, matched observations, comparisons, sensitivity and full manual_inspection CSV; human fields blank. SQLite writes local, finalized snapshot copied to output. |
| Likelihood infrastructure | Optional teacher-forced module with boundary-prefix validation, exact token shift, bounded vocabulary-logit chunks, explicit EOS policy and caller-supplied candidates. Disabled by default; no full-dataset likelihood scoring. |
| CircuitKIT boundary | Research notes describe paired candidate validation, same-language alignment, differentiable metric engineering and causal validation; no CircuitKIT code executed or implemented. |
| Online technical investigation | Immutable Hub commits, official model cards and Transformers/CircuitKIT sources; installed AuditKIT source inspection; remaining gated/GPU uncertainty documented. |
| Tests and deployment limits | Latest final verification results are recorded below. No CUDA GPU is present; real Tiny Aya files require authorized Hugging Face access. No real Tiny Aya/Qwen GPU inference or full evaluation has been run. |

Final verification results
--------------------------

- `.venv/bin/python -m unittest discover -s tests -v`: **91 tests passed, zero skips**, in 23.375 seconds, using the final pinned Dataset2 runtime dependencies. This includes both notebook pipeline executions with fixtures, real AuditKIT precomputed scoring, the full 21,184-ID merge with known/missing outcomes, parser-only cache invalidation without Aya regeneration, refreshed validation flags on resume, auxiliary-table orphan checks, and wide-table count consistency.
- CPU model checks passed with PyTorch 2.8.0 / Transformers 4.57.6: EOS at the token cap is completed rather than truncated; attention masks and exact input/output IDs survive generation; teacher-forced log-probabilities from a tiny random Cohere2 model match a full-forward reference within 1e-5; tokenizer boundary changes fail explicitly.
- Both 31-cell notebooks pass nbformat 5.11.1 schema validation, compile as Python, contain all 14 required sections and differ only in shard configuration. Both pipeline cell sequences execute with CPU fixtures through scoring, analysis and SQLite export. Environment installation, login/Drive and actual GPU loading remain Colab-specific operations, not simulated GPU validation.
- `.venv/bin/python -m pip check`: **No broken requirements found.** The final requirements pin Datasets 4.4.1 to remain compatible with Transformers 4.57.6 / Hub 0.36.2; the repository's newer baseline Datasets 5.1.0 requires Hub >=1.31.0. Use the Dataset2 requirements file for this workflow.
- `.venv/bin/python -m eval.validate_dataset2 --output /tmp/dataset2_validation`: passed; 320 seeds, 21,184 prompts, 0 structural errors, 498 identical original/twin control pairs, 5,696 absent design variants. The source SHA256 still matches the immutable manifest metadata. Each shard has 160 seeds / 10,592 prompts; 16 seeds per category; 80 analysis / 50 tune / 30 heldout seeds; no overlapping IDs or missing assignments.
- Formatting, Python compilation and `git diff --check` passed. A ZIP roundtrip test checks all packaged source and dataset bytes. Each notebook contains its own identical embedded assets; `python -m eval.package_dataset2_colab --embed-notebooks` refreshes them and produces an optional source ZIP.
- Earlier dependency-download failures were resolved on continuation. They are no longer blockers, and none of the final tests are skipped because of missing libraries.
- **Not run:** actual Tiny Aya/Qwen3Guard GPU inference, the full evaluation, or CircuitKIT attribution. `torch.cuda.is_available()` is False. Tiny Aya model-file requests without credentials returned HTTP 401. The plan explicitly permits CPU/fixture verification when GPU resources or gated weights are unavailable; the run guide includes a separate one-seed Colab smoke mode before team execution of the full shards.

Delivered artifacts
-------------------

- Primary notebooks: `eval/eval_dataset2_part1.ipynb`, `eval/eval_dataset2_part2.ipynb`.
- Shared implementation: `dataset2_data.py`, `dataset2_runtime.py`, `dataset2_analysis.py`, `dataset2_likelihood.py` under `eval/`.
- Tools: `eval/validate_dataset2.py`, `eval/merge_dataset2_results.py`, `eval/package_dataset2_colab.py`.
- Runtime pins: `eval/requirements-dataset2.txt`.
- Authoritative manifest/provenance: `eval/manifests/dataset2_shards.csv`, `dataset2_shards.json`, `dataset2_validation_report.json`.
- Acceptance tests: `tests/test_dataset2_pipeline.py`.
- Execution/merge instructions: `eval/DATASET2_RUN.md`; research and scientific limitations: `eval/DATASET2_RESEARCH.md`.

All locally testable requirements are verified. Actual team GPU execution and subsequent human judgment validation remain explicitly separate from implementation completion.

Automatic Colab setup update: each notebook now embeds the dataset, manifest and shared modules. Its setup cell verifies and extracts them automatically, eliminating the separate ZIP upload and any dependence on published remote files. Use the updated notebook file rather than an older copy already open in Colab.

Automatic setup verification: **92 tests passed without skips** in 31.129 seconds. The added test executes each notebook's actual setup cell independently in fresh temporary folders, checks every extracted asset against the worktree, verifies identical payload hashes between parts, and verifies a repeat setup preserves an unchanged dataset file. Only pip installation is mocked for this extraction test; dependency installation was verified separately. The embedded setup cell is collapsed in Colab's normal form view.

Import recovery update: setup now explicitly imports Path and invalidates import caches. The data-loading cell restores its own repository path before any shared-module import and detects a missing extraction with an actionable setup message. A regression test uses a fresh isolated Python interpreter outside the repository for each notebook to verify recovery without relying on notebook globals or the test runner's import path.

Responsiveness correction: the compressed payload has moved out of cell source and into notebook metadata. All code cells stay under 15 KB. Setup reads metadata using the Colab bridge (or a local notebook-file path), verifies the payload hash, extracts assets, and installs dependencies. Tests cover both metadata sources and an absent frontend response. No live Colab UI performance claim is made.

Live-package consistency update: `dataset2_environment.py` detects preloaded/disc-version mismatches after dependency installation and the exact stale Hub utility/download import reported in Colab. Setup and the Tiny Aya loading cell stop with a clear Runtime > Restart session instruction. Tests cover changed root versions, a stale child module with matching root version, and a clean environment.

Environment regression verification: all **97 tests passed without skips** in 27.394 seconds, including fresh metadata extraction, import-path recovery, consistent Hub probes and restart detection. Setup now distinguishes successful asset extraction from completed dependency checks in its progress messages.

Restart-loop correction: dependency setup now compares exact pins with installed versions and skips pip on repeated matching runs, accepts CUDA local-version wheels for public Torch pins, and restarts only after loaded packages actually change or live imports fail their consistency check. It detects the extra unpinned upgrade command shown in the Colab traceback. Regression cases cover repeat setup, fresh installation, a real loaded-version change, and restoration of disk files to an already-loaded root version that may have stale child modules.

Idempotent setup verification: **101 tests passed without skips** in 25.619 seconds. Repeat setup skips pip, a fresh install without preloaded modules needs no restart, loaded-package changes require one, CUDA-tagged Torch wheels satisfy public pins, and an extra unpinned upgrade cell is rejected before installation. Both setup cells compile and remain below 4 KB of source.

GGUF backend update (current default): both notebooks use the earlier llama.cpp GGUF workflow, native greedy completion and exact token arrays, with public pinned model files and separate output paths. Shared launcher/build caching is reused. Tests now cover downloads/resumption/hash rejection, local/Drive model reuse, template/tokenization protocol, EOS/cap/context handling, process cleanup, real AuditKIT precomputed scoring with a native transport fixture, build-cache reuse and backend-mismatch merge rejection. CUDA compilation and inference with the actual Tiny Aya/Qwen GGUF weights remain unexecuted locally. Earlier fp16 verification remains historical evidence for the retained optional HF helpers.

Final GGUF release evidence
---------------------------

- `.venv/bin/python -m unittest discover -s tests -v` with DATASET2_LLAMA_CPU_BINARY and DATASET2_LLAMA_TEST_MODEL set: **116 passed, zero skips**. Without those opt-in paths, the real native integration test is explicitly skipped rather than claimed as executed.
- The pinned llama.cpp commit was compiled locally for CPU with Metal/BLAS GPU acceleration disabled. A public 1,185,376-byte TinyStories fixture GGUF was checksum-verified; its real HTTP server processed the current runner's `/props`, `/apply-template`, `/tokenize` and `/completion` requests, generating 8 tokens from 54 input tokens identically on repeat greedy calls. Process cleanup and offline health afterward passed. This is protocol evidence, not an Aya safety evaluation.
- An initial UI-asset build failure was resolved by disabling unused web UI/prebuilt UI provisioning in the headless build. The successful CPU build validates those flags against the pinned source. CUDA-specific compilation remains untested.
- Public actual Aya/guard GGUF endpoints returned HTTP 206 with GGUF magic bytes without credentials. Revisions, file sizes and LFS SHA256 values were obtained from publisher Hub metadata and pinned in MODEL_SPECS. Actual large model weights were not downloaded locally.
- Dataset validation still passes at 320 seeds / 21,184 retained prompts, with the same manifest/source hashes and exactly 10,592 prompts per shard.
- Both GGUF notebooks pass nbformat schema/Python compilation checks, contain identical shared assets and differ in shard configuration. Every code cell remains below 4 KB. Full pipeline fixture execution, actual AuditKIT precomputed scoring, cache/resume tests and cross-backend merge rejection pass.
- New default results: `My Drive/ez-penta/dataset2_eval_gguf/part1/` and `part2/`. Default dependencies omit Hub/Transformers/Torch and preserve compatible preinstalled NumPy/Pandas. Models and native builds are cached in Drive.

Automatic-port update: default GGUF launch now obtains a free OS loopback port and retries a detected pre-launch collision. Explicit requested ports retain strict ownership checks; unrelated processes are never stopped. Tests include a real non-HTTP occupied socket, automatic collision retry, explicit-busy rejection, port-independent inference configuration, and real native server execution on an automatically chosen port.

Automatic-port release verification: **120 tests passed without skips** in 26.410 seconds, with the real pinned native CPU server test enabled. The native test now uses the default automatic port and verifies that the server is offline after cleanup. Reexecuting a model-loading cell closes its prior notebook-owned runner. No actual Tiny Aya/Qwen GPU evaluation was run locally.
# Persistent resumption and startup recovery (2026-11-08)

Generation and judging interruption tests verify that saved observations are reused
without invoking their workers again. Model download tests simulate a lost runtime
and resume using the persistent Drive partial file in a new runtime directory.
Startup tests cover a bind collision after preflight and owned-process cleanup;
the automatic runner retries such collisions on another port. Setup now refreshes
loaded backend modules and closes previous owned runners. Compatible checkpoint
counts are printed before generation and scoring. The full CPU suite passed 123
tests, including the real pinned native-server protocol fixture. This does not
constitute an actual Aya/Qwen GPU Colab run.

