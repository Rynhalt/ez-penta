# Hugging Face dataset release plan

**Status:** plan only, 2026-11-16. Nothing has been built, moved or uploaded.
**Goal:** every number in `reports/experiment_report.md` ("Safer, or Just More Refusals?") can be traced to a public
HF dataset row, and the before/after metrics (Unsafe rate, harmless refusal, fix rate) can be recomputed from it.

---

## 0. What exists today

### Already public on HF (`isaac369369`, post_train_2 pilot, CC BY-NC 4.0)

| Repo | Covers article section | Action |
|---|---|---|
| `datasets/isaac369369/tiny-aya-safety-dataset2` | §2 data origin (21,184 Dataset2 prompts + base answers) | Keep. Add a link to the new collection in the card. |
| `datasets/isaac369369/tiny-aya-safety-training-pairs` | §2 "restricted vs broad pilot" (195 kept / 37 rejected) | Keep. |
| `datasets/isaac369369/tiny-aya-safety-eval-results` | §2 pilot (684 answers, 12/145 → 1/147 / 1/148; 8 → 16 / 29 of 80) | Keep. Fix the "12/115" vs "12/145" denominator mismatch between `SUBMISSION_STATUS.md` and the article before linking. |
| models `…-lora-layer18`, `…-lora-broad`; Space `tiny-aya-safety-explorer` | pilot adapters + demo | Keep. |

Source: `~/App_Files/Code/tiny-aya-safety` (`hf/publish.py`, `hf/cards.py`, `hf/published.json`). **Reuse this publisher**
for the new repos instead of writing a new one.

### Not yet public: everything in the main study (post_train_3)

Full working copy (all experiments run on this Mac) is local at
`~/Library/CloudStorage/GoogleDrive-test-ezpenta/My Drive/ez-penta/post_train_3/` (below: `PT3/`, ~3.7 GB, 2.1 GB of it
XSTEER B activations). The Drive snapshot is
[runs_2026-11-15_en_bilingual_steering_followups](https://drive.google.com/drive/folders/1bH59zkx1BlKXcqvR3L-NAbh3fbby3ltQ)
(843 files, SHA-256-checked 2026-11-16). XSTEER has its own Drive folder (`PT3/x_steer/LINKS.md`).

**Held only by Marcus** (Drive, owner `marcuslime123`, read-only to us): `arm_ja/` (two folders) with the Japanese-written
side of the Japan-culture extension (`ja_s0` answers) and `INTERIM_REPORT.md`. Must be requested/copied.

### Gaps found while inventorying

1. **The article's own package is not in this checkout.** `reports/experiment_report.md`, `CLAIM_AUDIT.md`,
   `EXPERIMENT_ARTIFACT_MAP.md`, `reports/figures/`, `reports/tables/`, `post_train_3/reports/source_data/*.json` and
   `technical_appendix.md` are absent on `post-train-2` and `origin/main`, and not under `~`. Locate them (another machine /
   session) before building, because the figure aggregates must match the HF rows.
2. **B5 is recoverable.** The article says the B5 notebook was not found, but
   `post_train_3/bilingual/notebooks/B5_refusal_decomposition.ipynb` is committed (be57d43) — unexecuted, 0 outputs.
   It reuses `nbsrc/B4_analysis.py`. Running it on the BI result rows (CPU, `N_BOOT=0`) can produce the structured
   decomposition the article marks provisional. Update §6 afterwards.
3. **en_s0 has no single bilingual eval file.** The BI study reuses stored `arm_en/eval/b00{01..11}.jsonl` (US prompts)
   plus `bilingual/eval/reuse/` (3,748 re-judged rows). `base_us` has only 6,228 US rows; base JP rows come from the same
   reuse path. The release build must reassemble these exactly as `B4_analysis.py` does.
4. **Training items hold both languages per row** (`arm_en/train/items.jsonl`: `prompt`, `answer`, `pass` are
   `{en, ja}` dicts). Flattened to one row per (item, language).
6. **Seed-id reuse in the frozen pool.** 7 seed_ids appear in two or more batches with different text, so 84 pool
   prompt_ids repeat across batches. The evaluation uses each seed's first batch (`plan.json eval.first_batch_of_seed`);
   the release keys pool rows on (batch, prompt_id) and flags `in_bilingual_eval`.
7. **The article PDF's GitHub links are broken.** `Rynhalt/ez-penta` is public, but `reports/CLAIM_AUDIT.md`,
   `EXPERIMENT_ARTIFACT_MAP.md`, `REPRODUCE.md`, `technical_appendix.md` and
   `post_train_3/japan_culture/INTERIM_REPORT.md` return 404 on `main` (checked 2026-11-16). The PDF also links no HF
   datasets yet.
5. **XSTEER C** (`PT3/x_steer/C/selection` only) is unfinished — do not publish as a result.

---

## 1. What to publish (the "must" list)

Rule: publish **inputs** (prompts, training rows) and **per-row results** (answer + both judge labels) for every
experiment the article reports, plus the **aggregate JSON** each figure/table is computed from. Do **not** publish
checkpoints (`*_ckpt.jsonl`), `work/` dirs, raw activations or the AnswerCarefully SQLite files.

Three new dataset repos, each with HF `configs` so the viewer shows one table per experiment:

### A. `tiny-aya-safety-multilingual-prompts` — evaluation inputs

| Config | Source (`PT3/…`) | Rows | Article § |
|---|---|---|---|
| `pool_seeds` | `eval_pool/batches/b000{1,3,5,7,9}`,`b0011/seeds.jsonl` (+ `rejects.jsonl` as `pool_rejects`) | 519 seeds (6×~100; verify) | §2 frozen pool, 475 `paired_ok` |
| `bilingual_eval` | `bilingual/inputs/eval_prompts.jsonl` | 12,426 | §4–5 BI |
| `half_pool` | `followups/inputs/eval_half.jsonl` | 6,598 | §8 E1–E3 (276 seeds) |
| `e4_us_uk_matched` | `followups/e4/prompts/prompts_matched.jsonl` (+ `seed_checks`, `rejects`) | 5,776 | §7 E4 (150/276 matched) |
| `japan_culture` | `japan_culture/chunks/c00{01..11}/prompts/prompts_japan.jsonl` (+ checks/rejects) | ~6×1,156 | §7 Japan extension |

Schema (shared): `prompt_id, seed_id, us_prompt_id, culture{US,JP,UK}, lang{en,ja}, kind{harmful,twin}, framing{control,rp,rs,fc,…},
expected, category, subtopic, form, severity, prompt, batch`. Add `paired_ok` (from B4 logic) and `split`.

### B. `tiny-aya-safety-multilingual-training` — LoRA training rows

| Config | Source | Rows |
|---|---|---|
| `items` | `arm_en/train/items.jsonl` flattened to one row per (item, lang) | 823 items → 1,646 rows, with `survives`, `split`, `item_kind{refusal,helpful}` |
| `conditions` | `bilingual/plan.json` `rows` + `train_item_ids` | en 568, ja 568, bi 1,136, en2 1,136 — one row per (condition, step_index, item_id, lang) |
| `followup_conditions` | `followups/plan.json` | bi_h2 (1,704), ja2, en/ja halves A/B |
| `gate_stats` | `arm_en/train/gate_stats.json`, `gen.jsonl` (generator attempts) | as-is |

Card must state: LoRA r16/α32, target `model.layers.18.mlp.down_proj`, 208,896 params, AdamW 1e-4, 1 epoch, bs 1, max len 768,
plan hashes from `plan.json.hashes`.

### C. `tiny-aya-safety-multilingual-results` — answers + Qwen3Guard labels + aggregates

One long table per config, unified schema:
`study, model, adapter_cond{base,en,ja,bi,en2,bi_h2,ja2,…}, train_seed, intervention, layer, alpha, prompt_id, seed_id, culture,
lang, kind, framing, answer, tokens, finish_reason, answer_lang, g06_label, g06_unsafe, g06_refused, g4b_label, g4b_unsafe,
g4b_refused, hardware, serving`.
Keep `g06_raw`/`g4b_raw` (needed to show "unknown labels are preserved").

| Config | Source (`PT3/…`) | Rows (approx.) | Article § / figure |
|---|---|---|---|
| `bilingual` | `bilingual/eval/{base_us,ja_s*,bi_s*,en2_s*,en_s1_jp,en_s2_jp}/eval.jsonl` + `arm_en/eval/b*.jsonl` + `bilingual/eval/reuse/` | ~150k | §5, Fig 1–2, key-findings table |
| `steering_weight_edit` | `steering/final/{base,en_s*,A_base,A_en_s*}/{US,JP}/eval.jsonl`, `final_hf_base/answers.jsonl` | ~75k | §8, Fig 5 |
| `steering_selection` | `steering/s3_selectA`, `s4_selectBCD` answers+judge, `selection/selected.json` | ~30k | §8 "fallback setting" |
| `steering_quality` | `steering/quality/pairs.jsonl`, `judge_raw.jsonl` | small | 118 vs 104 pairwise |
| `followups_half` (E1–E3) | `followups/eval/half__*/eval.jsonl` (13 models), `followups/e1/final/` | ~95k | §8, Fig 4 |
| `e4_culture` | `followups/eval/e4__{base,en_s0,ja_s0,bi_s0}/eval.jsonl` | 23,104 | §7, Fig 3 |
| `japan_culture_interim` | `japan_culture/chunks/*/{base,en_s0}/eval.jsonl` + **Marcus's** `arm_ja` `ja_s0` chunks | ~30k | §7 (label `interim=True`, 4/6 chunks on JA side) |
| `xsteer_d_causal` | `x_steer/D/answers.jsonl` joined with `judge.jsonl` | 6,073 | §8, Fig 6 |
| `xsteer_ab_diagnostics` | `x_steer/A/{diagnostics,counts}.csv`, `x_steer/B/b_all_layers/{diagnostics,cross_model}.csv`, `hypothesis_H_by_layer.csv`, `qualifying_layers.json` | 36 layers × 4 models × 2 langs | §8 XSTEER A/B (AUC ≥ .80, |cos| ≤ .30) |
| `summary` | `{bilingual,steering,followups,japan_culture/en,x_steer/D}/results.json`, `x_steer/*/RESULTS.md`, B5 output, `reports/tables/*.csv` | small | every headline number |

Optional extra folder `artifacts/` (not a viewer config): `steering/s2_directions/directions.npz`,
`x_steer/A/directions_bal.npz` (5.7 MB), XSTEER B directions (20.7 MB). **Exclude** `x_steer/B/acts` (2.1 GB) and
`steering/s2_directions*/acts` unless a reviewer asks.

### Models (needed for "deployed publicly", out of dataset scope)

12 main-study adapters are not on HF yet: `arm_en/adapters/seed{0,1,2}`, `bilingual/adapters/{ja,bi,en2}_s{0,1,2}`
(each ~0.8 MB) plus follow-up adapters. Recommend one model repo with subfolders per condition/seed (reuse
`hf/publish.py`), so the Space can load `en/ja/bi` side by side.

---

## 2. Cleaning steps (one script, `scripts/build_hf_release.py`, run on the local working copy)

1. **Inventory + hash.** Walk `PT3/`, record path, size, SHA-256; check each `DONE` equals `sha256(manifest.json)`.
   Fail on mismatch. Write `release/MANIFEST.sha256`.
2. **Load only finalized outputs.** Use `eval.jsonl` / `answers.jsonl` inside folders with a valid `DONE`; ignore
   `*_ckpt.jsonl`, `work/`, `superseded/` (e.g. `japan_culture/superseded/c0005_base_duplicate_run`).
3. **Reassemble BI exactly like `B4_analysis.py`** (import its loader rather than re-implementing), so per-arm pooled rates
   reproduce the article: en 6.78 / ja 6.45 / bi 4.03 % Unsafe (EN prompts); 15.68 / 8.33 / 9.68 % (JA prompts);
   refusal 20.4 / 13.2 / 36.0 % and 6.4 / 17.1 / 15.4 %. **These assertions are the release gate.**
4. **Normalize.** Parse stringified dicts; flatten to one row per (item/prompt, lang); add `adapter_cond`, `train_seed`,
   `study`; map labels to `{Safe, Unsafe, Controversial, null}`; keep raw judge text.
5. **Join prompts to answers** by `prompt_id` so result rows are self-contained (prompt text included).
6. **Safety/privacy pass.** Secret scan (HF tokens, local paths like `/Users/isaac`, `/content/drive`, emails) over all text
   and manifests; strip absolute paths from manifests. Add `content_warning` column/card banner (harmful prompts and
   judged-Unsafe completions in EN/JA).
7. **Recompute aggregates from the cleaned parquet** and diff against `results.json` / `reports/tables/*.csv`
   (tolerance 0.01 pp). Run B5 here and save `b5_decomposition.json`.
8. **Write parquet** (`zstd`) per config, plus `README.md` card from `hf/cards.py` template: task, schema, row counts,
   source notebook per config, judge model + revision, licence CC BY-NC 4.0 (matches Tiny Aya Water), limitations
   (automated labels, no human review, interim Japan-culture, XSTEER D single adapter/fold/256-token cap).
9. **Dry-run locally** with `datasets.load_dataset("parquet", data_files=…)` for each config; then push as **private**,
   verify viewer + row counts, then flip to public.
10. **Link back:** add the three repos to the existing dataset collection; update `EXPERIMENT_ARTIFACT_MAP.md`,
    `REPRODUCE.md` and the article's "Reproducibility" section with HF URLs; tick the CircuitKIT roadmap only if any
    CircuitKIT artifact is published (none planned — it has no validated Tiny Aya result).

---

## Build status

| Repo | Status | Local output |
|---|---|---|
| A. `tiny-aya-safety-multilingual-prompts` | **Built locally**, 9 configs, 16 checks passed, loads with `datasets` | `outputs/hf_release/tiny-aya-safety-multilingual-prompts/` (6.2 MB) |
| B. `tiny-aya-safety-multilingual-training` | **Built locally**, 4 configs, 20 checks passed (incl. 0 train/eval seed or text overlap) | `outputs/hf_release/tiny-aya-safety-multilingual-training/` (0.9 MB) |
| C. `tiny-aya-safety-multilingual-results` | Not started | — |

Builder: `scripts/build_hf_release.py` (`.venv/bin/python scripts/build_hf_release.py [--only prompts|training]`).
Each output folder has `README.md` (card with HF `configs`), `data/*.parquet` and `manifest.json` (source and
output SHA-256, checks passed). Nothing uploaded.

## 3. Order of work

1. Find the article package (`reports/…`, `source_data/*.json`) — blocker for step 2.7.
2. Get Marcus's `arm_ja` Japan-culture `ja_s0` chunks + `INTERIM_REPORT.md`.
3. Build A (prompts) and B (training) — smallest, no joins.
4. Build C `bilingual` + `summary` with the B4 assertions; run B5.
5. Build remaining C configs (steering, followups, E4, Japan culture, XSTEER).
6. Push private → verify → public; publish main-study adapters; update Space to read from the results repo.
7. Update article links and the before/after table (base → en/ja/bi: Unsafe rate, harmless refusal, fix rate).

## 4. Decisions needed

- **Gating:** previous repos are public and un-gated. Raw harmful completions in two languages argue for gated
  (auto-approve) access; the submission requires "publicly deployed". Recommendation: un-gated prompts + training repos,
  results repo gated with auto-approval and a content warning.
- **Repo granularity:** three repos with configs (recommended) vs one repo per experiment.
- **Activations:** publish XSTEER/steering directions only (recommended), not the 2.1 GB activations.
