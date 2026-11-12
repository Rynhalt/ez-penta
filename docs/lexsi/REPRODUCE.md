# Reproduction and artifact guide

**Snapshot:** 2026-11-12. This guide separates checks reproducible from the repository from experiments requiring private data, human-reviewed inputs, or Colab GPU resources.

## Local repository checks

Run focused Track-2 tests from the repository root:

```bash
.venv/bin/python -m unittest discover -s tests/track2_extension -v
```

The latest suite ran 84 tests (83 passed, 1 expected skip because CircuitKIT is not installed locally); its output is `outputs/lexsi/latest_local_tests_20261112_execution_audit.log`. These are tooling and synthetic-data checks only, not real Tiny Aya behavior.

Build blinded packets only from the existing private review template:

```bash
.venv/bin/python scripts/lexsi/build_blinded_review_packets.py --help
```

Inspect the actual CLI options before rebuilding. Existing packets and their private key are already in ignored `outputs/lexsi/blind_review_packets_20261112/`; do not overwrite reviewer work if it has arrived. Keep reviewer files and source prompts out of Git.

The checked-in source for corrected seed-only analysis is `scripts/lexsi/seed_matched_reanalysis.py`. It was executed in an authorized CPU Colab runtime against private Drive SQLite and corrected label files, not against local synthetic data. Its source currently names the Drive mount path directly. The aggregate result is in ignored `outputs/lexsi/corrected_seed_only_summary.json`; private seed effects are not included in this guide.

### Strict matched analysis from original result shards

The two original result shards were obtained from connected Drive separately because each was below the 256 MiB connector limit while the merged DB was above it. The verified source copies and row-level outputs are private and ignored. For a repeat, retrieve the same two original shards and corrected-label export from the authorized Drive location into a private local directory (outside Desktop/iCloud sync), verify their SHA256 values against `combined_reconciliation_summary.json`, then run:

```bash
.venv/bin/python -m scripts.lexsi.reconcile_dataset2_shards \
  --part1 "$DATASET2_PART1_SQLITE" \
  --part2 "$DATASET2_PART2_SQLITE" \
  --output-dir outputs/lexsi/corrected_seed_cell_matched_20261112 \
  --expected-corrected-labels "$DATASET2_CORRECTED_LABELS_JSONL" \
  --expected-rows 21184 --bootstrap-replicates 5000 --seed 20261111
```

The module form matters: invoking the script by path failed to put the project root on `sys.path`. The command verifies shard metadata and row counts, disjoint source seeds, corrected labels, and prior binary-label invariance before computing matched associations. Never copy raw prompts/responses or row-level outputs into shared logs. The checked output summary is `outputs/lexsi/corrected_seed_cell_matched_20261112/combined_reconciliation_summary.json`; per-seed CSVs remain private. A separate 20,000-replicate NumPy check is saved at `outputs/lexsi/independent_numpy_bootstrap_20261112.json` and verifies intervals conditional on those saved seed effects.

To independently check the saved seed-effects summary after retrieving the private `seed_effects.csv` into the ignored output directory, run:

```bash
.venv/bin/python scripts/lexsi/verify_seed_bootstrap.py \
  --seed-effects outputs/lexsi/corrected_seed_only/seed_effects.csv \
  --summary outputs/lexsi/corrected_seed_only_summary.json \
  --output outputs/lexsi/corrected_seed_only_bootstrap_independent_check.json \
  --replicates 20000 --seed 20261112
```

This recomputes means and bootstrap intervals conditional on saved per-seed effects; it does not independently reconstruct seed effects from raw SQLite.

## Real Colab evidence already captured

The private aggregate run inventory and hashes are in ignored `outputs/lexsi/run_manifest.json`; the latest local unit-test stdout/stderr is saved in `outputs/lexsi/latest_local_tests_20261112_current.log`.

### HF / CircuitKIT parity

- Executed notebook attempts, logs, environment, checkpoint metadata and parity summary: `outputs/lexsi/colab_phase1/authorized/` (ignored/private local artifacts).
- Model revision: `CohereLabs/tiny-aya-water@41a99216d12b3ef4881df762934186e5e4d2bbc1`.
- Recorded environment: Python 3.13.16, PyTorch 2.14.1+cu130, Transformers 5.15.0, CircuitKIT 0.2.2, TransformerLens 3.8.0, Hugging Face Hub 1.33.0; NVIDIA L4.
- The authorized run measured English, Japanese, and long-context parity. Largest absolute log-probability error was `7.43865966796875e-05`; peak allocated memory was `22,248,377,344` bytes.
- This is backend parity only. It is not behavior-metric validation or circuit discovery.

### Three-arm smoke comparison

- Run bundle: `outputs/lexsi/track2_pilot_20261112_nf4_smoke/`.
- Its `run_manifest.json` lists versions, GPU, generation/judge settings, artifact hashes, and memory peaks; `execution.log` contains actual remote output. `pilot_summary.json` contains aggregate outcomes.
- It used 8 prompts, 4 US-neutral pairs, and 4 seeds; all arms produced 7 Safe / 1 Unsafe guard judgments and paired risk differences of zero. It is explicitly non-inferential.
- Prompt and generation files are private; do not commit or paste them into public logs.

## Colab CLI and resource gate

Use Google's official Colab CLI (`colab`) after authenticating through its documented Google flow. Never copy credentials into notebook source or logs. Check current CLI/runtime state before provisioning:

```bash
colab --help
colab usage
colab sessions
```

At the latest verified check on 2026-11-12, usage was **99.90 compute units**, with no active assignment or session. A fresh L4 credential probe found no Hugging Face token in the runtime environment/cache, so the session was stopped before model download. Add a Colab Secret named `HF_TOKEN` and enable notebook access before retrying notebook 01. The session used for prior parity is gone; its logs are retained privately. Create/reuse an official CLI GPU runtime, record its ID and type, execute the notebook, download its outputs/logs, audit the saved notebook for cell errors (CLI exit code alone is insufficient), verify hashes, and stop the session.

### Hugging Face token bootstrap through the Colab UI

`google.colab.userdata.get("HF_TOKEN")` times out when invoked only through `colab exec`; Google requires the secret fetch to run from the browser UI. Save the token as a Colab Secret named `HF_TOKEN` and enable notebook access. In the browser attached to the exact CLI-created runtime, run this cell once before asking the CLI to execute notebook 01:

```python
from google.colab import userdata
import os

_hf_token = userdata.get("HF_TOKEN")
if not _hf_token:
    raise RuntimeError("Colab Secret HF_TOKEN is empty or notebook access is disabled")
os.environ["HF_TOKEN"] = _hf_token
del _hf_token
print("HF_TOKEN loaded:", bool(os.environ.get("HF_TOKEN")))
```

The cell prints only a boolean. Do not put the token in notebook source, shell history, logs, or chat. The CLI can then execute the staged notebook in that same runtime, where it sees the environment variable set by the browser kernel.

A fresh CPU runtime previously failed to mount Drive during credential propagation with HTTP 400 and was stopped. The connector still cannot transfer the merged 375 MB DB as one file, but the two original result shards were later fetched separately through the connected Drive source and analyzed locally; no Colab GPU session or inference was involved. Latest CLI check: no active sessions/assignments and 99.90 compute units. See `SOURCE_RECONCILIATION.md` and `COLAB_DEBUG_LOG.md`.

## Human-review gate before CircuitKIT input

The two blinded packet directories contain separate reviewer CSVs, instructions, and a private owner-only mapping key. The owner must grant each reviewer access to only their own folder. Reviewers must submit independent decisions; then adjudicate conflicts and verify intent equivalence, refusal/compliance behavior, and exact answer spans. The rubric and allowed values are in the packet instructions and `docs/lexsi/ANNOTATION_GUIDE.md`.

Do not put these rows into `outputs/lexsi/circuitkit_pairs.jsonl` until the real reviewer and adjudicator provenance has been recorded and checked. Automated labels remain provisional. Ensure discovery/tuning/evaluation are separated by source seed; prior heldout exposure disqualifies that cohort as pristine confirmatory evaluation.

## Expected next execution sequence

1. Receive two independent review CSVs and adjudicate them; preserve original files and record hashes.
2. Validate approved continuation spans and the refusal-minus-compliance mean log-probability metric on tune seeds using the pinned HF checkpoint.
3. After saving the executed notebook, run `.venv/bin/python scripts/lexsi/audit_notebook_execution.py executed.ipynb --output private_notebook_audit.json`; require `status=PASS`. This catches cell exceptions that Colab CLI can mask with exit code 0. If behavior validation passes, run the repository's patched position-map EAP-IG TaskSpec on disjoint discovery seeds. Keep exact token maps and report unmapped positions.
4. Save attributions and visualizations, then test selected components with activation patching and ablation on disjoint evaluation seeds. Include matched random-component baselines and benign controls.
5. For any model tuning, save weights/config/tokenizer, optimizer/training settings, dataset manifest, and SHA-256 to durable storage immediately. Fresh-process reload must pass before evaluation.
6. Run a sufficiently powered paired evaluation with seed-clustered uncertainty and human-validated outcomes. Report safety and benign refusal/utility changes, not just guard proxy rates.
7. Record each run in `docs/lexsi/EXPERIMENTS.md`, `COLAB_DEBUG_LOG.md`, and a private machine-readable manifest. Update `STATUS.md`, `DECISIONS.md`, and the roadmap checklist.

## Privacy and reproducibility limits

Files under `outputs/lexsi/` include ignored/private prompts, responses, judgments, and seed-level records. Confirm `.gitignore` and staged paths before any commit. This guide contains aggregate facts and artifact locations only; it does not publish prompt text or identifiers.

Current reproducibility is partial: source code and some run bundles exist, but reviewer decisions, approved real input data, durable tuned checkpoint weights, fresh GPU quota, and the eventual confirmatory evaluation do not. A local unit-test pass or synthetic Colab test must never be presented as completion of those real-data/GPU gates.


## CircuitKIT notebooks (gated, not yet executed)

After actual two-reviewer adjudication, run notebook 01 first. It must write a passing `phase1/behavior_validation.json`. For notebook 02, clone CircuitKIT at commit `e3501bc72b349963db76df324f5dbd41ced2a02f`, apply `docs/lexsi/patches/circuitkit_eap_position_maps_e3501bc.patch`, and set `CIRCUITKIT_PATCHED_SOURCE` to that checkout before importing CircuitKIT. Notebook 02 uses only approved `analysis` seeds. Notebook 03 requires newly collected, approved `fresh_eval` seeds that were not exposed or used for discovery; the historical heldout split is rejected. Both notebooks capture model revision, source commit, dependency versions, settings and peak CUDA memory. Their presence is not evidence that a Tiny Aya discovery or causal run occurred.

### CircuitKIT notebook bootstrap behavior

All three CircuitKIT notebooks now install pinned `circuitkit==0.2.2`, `transformers==5.15.0`, and `transformer-lens==3.8.0` in the Colab runtime and verify that the preinstalled PyTorch build remains unchanged. Notebooks 02/03 additionally create a disposable clone at CircuitKIT commit `e3501bc72b349963db76df324f5dbd41ced2a02f`, apply `docs/lexsi/patches/circuitkit_eap_position_maps_e3501bc.patch` when it applies cleanly, and refuse to overwrite a nonmatching checkout. Before running, upload the repository `eval/track2_extension/` code and patch under `/content/lexsi`; provide a valid HF token in the authorized Colab runtime. Notebook 01 also needs `eval/dataset2_likelihood.py` under that root. Setup does not bypass the human approval, metric, fresh-evaluation, or GPU gates.

### Reconcile returned blinded reviews (after both reviewers submit)

The current packets are still blank. Do not run the exporter until actual reviewer decisions and an independent adjudication exist. First create the pending worksheet from the returned packet files and the owner-only key:

```bash
.venv/bin/python scripts/lexsi/reconcile_blinded_reviews.py prepare-adjudication \
  --reviewer1 outputs/lexsi/blind_review_packets_20261112/reviewer_1.csv \
  --reviewer2 outputs/lexsi/blind_review_packets_20261112/reviewer_2.csv \
  --key outputs/lexsi/blind_review_packets_20261112/PRIVATE_adjudication_key.csv \
  --candidates outputs/lexsi/circuitkit_contrast_candidates.csv \
  --output-csv outputs/lexsi/human_adjudication_87_control_roleplay.csv \
  --summary-json outputs/lexsi/human_adjudication_87_control_roleplay_summary.json
```

After a separate adjudicator fills only the adjudication columns with actual decisions, export through the second command. It requires two distinct reviewer IDs, a different adjudicator ID, exact continuation prefixes, all target labels, and unchanged hashes/source fields; it writes only under ignored `outputs/lexsi/`. Run `--help` for exact options. Current packets fail this preparation gate because both have zero completed reviews; no worksheet or approved export should be produced from them.

### Prompt-only contrast coverage audit

This can run offline without Colab, but it inventories prompt construction only. It does not produce response outcomes, semantic review, or CircuitKIT inputs:

```bash
.venv/bin/python scripts/lexsi/audit_prompt_contrast_coverage.py \
  --sqlite marcus/dataset2_partial.sqlite3 \
  --output outputs/lexsi/prompt_contrast_coverage_20261112.json
```

The source hash is checked against `eval/manifests/dataset2_shards.json`; the saved aggregate omits prompt text and source IDs. To analyze actual response behavior, obtain the separate generation/results SQLite through an authorized supported transfer route. The current local SQLite cannot answer that question.

### Reproduce Pipeline 1 MH point estimate

With the two original result shards retrieved privately, run `.venv/bin/python -m scripts.lexsi.reproduce_pipeline1_mh --part1 "$DATASET2_PART1_SQLITE" --part2 "$DATASET2_PART2_SQLITE" --output outputs/lexsi/pipeline1_mh_reproduction_20261112.json`. It checks the expected 4,069-row cohort and writes aggregate cell counts, MH/CMH statistics, and the RBG confidence interval. The script implements the archived notebook's Robins–Breslow–Greenland log-OR variance and Wald interval; all four archived RBG intervals were reproduced.

### Reproduce the 2026-11-12 Phase 1 exploratory L4 diagnostic

The saved private run is under `outputs/lexsi/circuitkit_owner_authorized_exploratory_20261112/`. It used seven owner-authorized exploratory pairs (six analysis seeds, one tune seed), not independently human-annotated labels, and returned `INSUFFICIENT_VALIDATION_DATA`. It must not be cited as a metric pass.

The run used the official CLI session name `lexsi-track2-phase1-diag-20261112`; the reusable input bundle is `colab_bundle.zip`, and `run_colab_notebook.py` executes the notebook with `nbclient`. To reproduce on a fresh L4, create an authenticated Colab GPU session, upload the bundle and a short-lived Hugging Face read-token file at `/content/.hf_read_token_once`, then execute the runner with `/Users/marcuschang/.local/bin/colab exec --session <session> --file outputs/lexsi/circuitkit_owner_authorized_exploratory_20261112/run_colab_notebook.py --timeout 0`. The runner reads and unlinks that token file before launching the notebook; never put the token in shell history, notebook cells, or logs. Download the executed notebook and `phase1/` JSON/freeze outputs before stopping the session. Current model revision, environment versions, GPU memory, parity errors, exact metric results, audit and output hashes are captured in the run directory and `docs/lexsi/EXPERIMENTS.md`.

This seven-seed cohort cannot pass the tune-set gate. Expand the tune cohort and freeze any metric changes before another validation run; do not proceed to notebook 02 until the metric passes on adequate, held-back tune seeds.
