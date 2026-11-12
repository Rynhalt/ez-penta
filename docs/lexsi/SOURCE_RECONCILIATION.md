# Dataset2 source reconciliation

**Audit date:** 2026-11-11  
**Input:** private Drive `Pipeline1111/combined/observations.csv` (read-only streaming audit; no new inference).  
**Full-file SHA256:** `7a38bedf6d2d3730154c65a3c6e1b022c5216a82850de3017deefe373d57ac0b`  
**Size / rows:** 422,346,957 bytes / 21,184.

Metadata-only audit outputs are in the ignored private directory `outputs/lexsi/`: `source_listing_20261111.json`, `source_headers_20261111.json`, and `source_audit_20261111.json`. The full Drive copy remains in `MyDrive/ez-penta/Pipeline1111/codex_extension_20261111/`.

## Reconciled counts

| Quantity | Observed in `observations.csv` |
|---|---:|
| Assigned prompt rows | 21,184 |
| Unique source seeds | 320 (analysis 160; tune 100; heldout 60) |
| Rows by split | analysis 10,352; tune 6,780; heldout 4,052 |
| Generation status | completed 17,795; truncated 3,387; error 2 |
| Scoring status | completed 13,730; error 4,065; not eligible 3,389 |
| Stored `safety` labels | Safe 12,628; Unsafe 1,102; Error 4,065; Missing 3,389 |
| Stored `prompt_safety` labels | Safe 10,931; Unsafe 2,799; Missing 7,454 |
| `identical_twin_flag=true` | 4,252 |

The 13,730 stored binary response judgments equal Safe + Unsafe. This independently confirms the exported-label denominator described in the handoff. The export's `unsafe` column is blank for all rows in this source file; do not use that column as the outcome. Use the categorical `safety` field and preserve `Missing`/`Error` as unknown or failed states.

The file's `source_split` value is `generated` for all rows. The actual analysis/tune/heldout partition is stored in `split`; downstream code must use `split` and source `seed_id`, not infer a split from `source_split`.

## Raw guard recovery

A read-only audit of the merged SQLite `observations.raw_guard_outputs` independently reproduced the recovery rule. It parsed the response verdict separately from the prompt verdict using the repository's exact `Safety:` line pattern. No model inference was run.

| Check | Result |
|---|---:|
| Rows / unique prompt IDs | 21,184 / 21,184 |
| Invalid raw record shapes | 0 |
| Previously published binary labels changed | 0 |
| Published binary response labels | 13,730 |
| Recovered binary response labels | 17,373 (15,749 Safe; 1,624 Unsafe) |
| Additional binary labels recovered | 3,643 |
| Additional binary response labels whose prompt verdict was Controversial | 3,643 |
| Remaining response verdicts | 422 Controversial; 3,389 Missing |

Recovered row counts by split: analysis 8,507 (1,742 additional); tune 5,596 (1,110 additional); heldout 3,270 (791 additional). The recovered label rows are saved privately at `MyDrive/ez-penta/Pipeline1111/codex_extension_20261111/corrected_guard_labels.jsonl` (21,184 rows; SHA256 `8db21de6c0d7ed75c3809ba864613b5eb6867ae8f39d2f62b45437a8e62b90c4`). Do not publish it. These are automated guard labels, not human safety ground truth.

The first remote summary script contained an aggregation bug that left per-split label subtotals at zero while its global checks and corrected-label file were accurate. The intermediate Drive file is preserved under the explicit name `guard_recovery_audit_INTERMEDIATE_INVALID_SPLIT_TOTALS.json` and must not be used for split label subtotals. The corrected aggregate is [raw_guard_recovery_and_seed_comparisons_20261111.json](../../outputs/lexsi/raw_guard_recovery_and_seed_comparisons_20261111.json), also saved privately in Drive as `guard_recovery_audit_corrected.json`; it contains no prompt text or row IDs.

## Existing seed-aware comparison summaries

The saved SQLite `comparisons` table also contains precomputed seed-bootstrap summaries using the **original exported score table**, not the 17,373 corrected response judgments. For `culture=us` minus `neutral`, `metric=unsafe`, it reports:

| Split | Mean paired delta | Stored interval | Known / available pairs | Seeds | Stored CI policy |
|---|---:|---:|---:|---:|---|
| analysis | +6.76 pp | +3.31 to +10.11 pp | 1,398 / 2,892 | 157 | seed bootstrap, descriptive, unadjusted |
| tune | +6.62 pp | +3.07 to +10.39 pp | 971 / 1,980 | 100 | seed bootstrap, descriptive, unadjusted |
| heldout | +3.29 pp | -2.92 to +8.69 pp | 509 / 1,180 | 60 | seed bootstrap, descriptive, unadjusted |

These are useful evidence that the project already stored seed-level uncertainty summaries; they are not independently recomputed in this audit. They use the old exported labels, are guard proxies, and have nearly half of available pairs unknown. The heldout split has already been exposed, so it is not an untouched final test. Recompute these comparisons from the corrected labels and inspect the bootstrap implementation before treating them as a confirmatory association.

## Model/backend provenance observed in `run_metadata.json`

- Dataset manifest SHA: `296be6962f573eac1133f099eac098a16318b3d708e1cc9e541ea6fb53eea21f`; manifest ID: `1bb4b6cd6aaec7188979ca9992ad327684ebd02d1025b5874bae8b18d7c7721f`; dataset algorithm `paired-dp-v1-seed42`.
- Generation used `CohereLabs/tiny-aya-water-GGUF`, revision `bfe7fe10f8f206192097eeedcecdfbb1a11f9ca7`, file `tiny-aya-water-q4_k_m.gguf`, Q4_K_M, llama.cpp; model hash `ecc6569cac41b36f4a06fdf9ccf5b315ec6f8efe44c3dc828f5e0a644dfbe8d9`; context size 8192.
- Judging used `QuantFactory/Qwen3Guard-Gen-0.6B-GGUF`, revision `be4e562404e1bd6d6e01bed9c5fcf1ddb4a6c6aa`, Q8_0; guard verdicts are automated proxies.
- These are not the same inference backend as the later HF Transformers validation. Do not combine them into a matched baseline/adapter estimate without a controlled rerun.

## Remaining reconciliation

- The simpler corrected-label seed-only analysis was subsequently run on Colab CPU. It is recorded below; it does not replace a within-seed match on framing/category/mix direction.
- [x] Independently reproduce the complete Pipeline 1 cohort, MH OR, continuity-corrected CMH test, and archived RBG confidence interval from the source shards (details below).
- Recompute the matched association uncertainty from corrected labels under a prespecified seed/category/framing/mix-direction matching design. The seed-only run below has no within-seed cell matching and is not a causal culture contrast.
- Inspect prompt/answer and pair eligibility, truncation, duplicate/twin flags, and human-review fields before selecting any experimental rows.

## Corrected-label seed-only analysis (2026-11-11)

The corrected raw response-guard labels were used in a separate seed-level analysis. For each split and prompt language, the implementation computed each seed's mean binary unsafe proxy under US and neutral framing, retained seeds with both conditions, then averaged the within-seed differences equally. It did **not** match rows within seed by category, framing, or mix direction. Uncertainty used 5,000 source-seed bootstrap replicates (RNG seed `20261111`, percentile 95% interval). This is a corrected-label association summary, not human response-safety ground truth or a causal estimate.

| Split | Language | Binary harmful rows | Matched seeds | Mean US minus neutral unsafe-proxy risk | Seed-bootstrap 95% interval |
|---|---:|---:|---:|---:|---:|
| analysis | en | 634 | 110 | +6.97 pp | +0.38 to +13.79 pp |
| analysis | ja | 588 | 106 | +12.97 pp | +4.87 to +21.07 pp |
| analysis | mix | 831 | 96 | +9.55 pp | +2.41 to +16.41 pp |
| tune | en | 421 | 78 | +12.61 pp | +5.77 to +19.76 pp |
| tune | ja | 390 | 76 | +7.46 pp | +0.77 to +14.58 pp |
| tune | mix | 578 | 72 | +13.98 pp | +7.41 to +21.20 pp |
| heldout* | en | 256 | 44 | +9.85 pp | -0.95 to +20.64 pp |
| heldout* | ja | 236 | 41 | +17.68 pp | +8.33 to +27.24 pp |
| heldout* | mix | 339 | 42 | +13.61 pp | +3.77 to +23.93 pp |

\* Heldout was previously exposed in project analyses; these are descriptive only.

The summary is `outputs/lexsi/corrected_seed_only_summary.json`. Detailed seed-level effects and corrected labels remain in the private Drive folder `Pipeline1111/codex_extension_20261111/corrected_seed_only/`. A read-only Drive recheck of `seed_effects.csv` independently confirmed all nine cohort seed counts and arithmetic means against the summary (665 seed rows total). The reproducible check is `scripts/lexsi/verify_seed_bootstrap.py`; it resamples those saved seed effects with 20,000 bootstrap draws per group and a different PRNG. All nine means matched exactly and every interval was within 0.379 percentage points of the stored 5,000-draw interval. This validates the bootstrap calculation conditional on the saved seed effects, but does not independently reproduce raw-SQLite extraction or the original RNG stream. Aggregate comparison is in ignored `outputs/lexsi/corrected_seed_only_bootstrap_independent_check.json`. Source hash in the summary: `d6004f551eb8519f59ad090594df4373b5ade0e40756499675ee8deb4fb049df`.

## Corrected seed-matched reanalysis command

### Strict corrected-label seed- and design-cell-matched analysis (2026-11-12)

This section supersedes earlier statements that the strict raw-results analysis was pending. The two original `observations` SQLite shards were downloaded separately through the connected Drive source because the merged 375 MB DB exceeds the connector’s 256 MiB per-file limit. They were stored in a private, mode-700 directory outside the Desktop/iCloud-synchronized repository. Combined source row count is 21,184; each shard has 10,592 rows, identical Dataset2 manifest metadata, and no overlapping source seeds. The 21,184 parsed corrected response labels exactly match the independent corrected-label JSONL by prompt ID, seed ID, split, response verdict, and prompt verdict (0 mismatches). Prior published binary labels changed: 0. Invalid raw guard record shapes: 0.

The analysis uses completed, analysis-eligible harmful prompts and binary corrected response verdicts (`Safe`/`Unsafe`) only. Other verdicts are excluded, not mapped to Safe. Within each split and prompt-language stratum, the estimator requires both US and neutral rows for a seed and matches rows within seed on kind, category, framing, mix direction, and output language. It reports the equal-weight mean of the matched seed-level US-minus-neutral unsafe-proxy differences and a 5,000-replicate percentile bootstrap over source seeds (RNG seed `20261111`). The nine resulting estimates are:

| Split | Prompt language | Eligible harmful binary rows | Matched seeds | US minus neutral | 95% seed-bootstrap interval |
|---|---:|---:|---:|---:|---:|
| analysis | en | 1,054 | 102 | +7.11 pp | -0.08 to +14.13 pp |
| analysis | ja | 922 | 97 | +10.14 pp | +2.06 to +18.56 pp |
| analysis | mix | 1,353 | 82 | +10.57 pp | +2.40 to +19.15 pp |
| tune | en | 670 | 67 | +11.07 pp | +3.23 to +19.40 pp |
| tune | ja | 604 | 65 | +6.67 pp | -1.03 to +14.87 pp |
| tune | mix | 901 | 61 | +18.50 pp | +9.75 to +27.73 pp |
| heldout* | en | 415 | 41 | +4.88 pp | -6.91 to +16.67 pp |
| heldout* | ja | 367 | 36 | +17.82 pp | +5.32 to +30.56 pp |
| heldout* | mix | 550 | 35 | +15.71 pp | +3.71 to +27.71 pp |

\* Heldout prompts were previously exposed in project analysis and are descriptive only.

The seed-level effects were independently resampled with NumPy PCG64 for 20,000 replicates per group. Point estimates matched exactly; the largest difference between interval endpoints was 0.00513 absolute risk (0.52 percentage points). This independent check is conditional on the saved effects and does not reproduce extraction from SQLite.

These are associations in the automated guard response-label proxy. They do not establish that the prompts are semantically equivalent, that outputs are unsafe compliance under human review, that culture framing causes the difference, or that any model fix improves safety. The exact matched-cell estimand differs from the prior seed-only analysis, so the estimates are not interchangeable. Source shards and row-level outputs remain private. Locally saved aggregate summary: `outputs/lexsi/corrected_seed_cell_matched_20261112/combined_reconciliation_summary.json`; private per-seed CSV and corrected labels are ignored by Git.

### Pipeline 1 Mantel–Haenszel reproduction

The saved script `scripts/lexsi/reproduce_pipeline1_mh.py` independently applies Pipeline 1's published filter and published binary `safety` labels to the raw source shards. It reproduces the 4,069-row harmful cohort and all four archived comparisons:

| Exposure / subset | Binary rows | MH OR | RBG 95% CI | Continuity-corrected CMH p |
|---|---:|---:|---:|---:|
| US vs neutral, all framings | 2,619 | 1.9822 | 1.5832–2.4818 | 2.21e-9 |
| Japan vs neutral, all framings | 2,704 | 1.0974 | 0.8623–1.3967 | 0.4875 |
| US vs neutral, roleplay only | 706 | 2.5441 | 1.7349–3.7309 | 1.70e-6 |
| Japan vs neutral, roleplay only | 703 | 1.4217 | 0.9452–2.1384 | 0.1121 |

The US/all-framing tables by language are en (55/350 vs. 30/356; n=791), ja (90/325 vs. 47/322; n=784), and mix (117/428 vs. 57/442; n=1,044). The values match the supplied Pipeline 1 report at its displayed precision. Reproduced aggregate JSON is `outputs/lexsi/pipeline1_mh_reproduction_20261112.json`.

The archived notebook was retrieved from Drive and its RBG formula independently implemented. The script reproduces all four historical intervals and point/test statistics (including primary US/all CI 1.5831753–2.4818013). These confirm stratified associations in published automated guard labels only; they are not human validation or causal evidence.

## Corrected seed-matched reanalysis command

`eval/track2_extension/reconcile_dataset2.py` reparses raw response verdicts, checks that prior binary labels remain unchanged, writes corrected labels and per-seed effects, and computes percentile bootstrap intervals over seeds. It matches within source seed, kind, category, framing, mix direction, and output language; prompt languages are analyzed separately. It includes harmful, analysis-eligible, completed generations and excludes truncated outputs and unknown response verdicts. The estimand is the equal-weight mean of within-seed paired unsafe guard-label risk differences for US minus neutral framing.

Run it from an authorized Colab runtime:

```bash
python -m eval.track2_extension.reconcile_dataset2 \
  --sqlite /content/drive/MyDrive/ez-penta/Pipeline1111/combined/results.sqlite3 \
  --output-dir /content/drive/MyDrive/ez-penta/Pipeline1111/codex_extension_20261111/corrected_reanalysis \
  --bootstrap-replicates 5000 \
  --seed 20261111
```

The `reconcile_dataset2.py` implementation is intended for the stricter corrected-label matching and should be run only after its exact pairing and uncertainty estimand are preregistered. An audit found and fixed an implementation bug: source NULL `mix_direction` values for monolingual en/ja were being treated as missing matched data. The previous zero-match output for those languages is invalid and is not a data result. Synthetic tests now assert that monolingual NULL is not-applicable while mixed-language NULL/blank remains excluded. The fixed code later ran against both original raw result shards; the strict matched estimates are documented in the 2026-11-12 section above. The older SQLite `comparisons` table remains a separate summary based on the original exported labels.
