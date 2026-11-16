# post_train_3 follow-ups: four experiments (run unattended)

Drive: `MyDrive/ez-penta/post_train_3/followups/` (`FU/`). Notebooks: `post_train_3/followups/notebooks/`.
Logs: `FU/ERROR_LOG.md`, `FU/DEVIATIONS.md`. Stop file: `FU/STOP`.

## Common rules
- Same models, decoding, judges, LoRA recipe and metrics as `arm_en` / `bilingual` / `steering`; code is copied
  from those notebooks, not rewritten. Adapters are served as merged bf16 models with vLLM.
- **Evaluation set ("pool half")**: the frozen-pool seeds of batches 1, 3, 5 (276 seeds): their 3,312 US prompts
  and the Japanese-culture prompts of chunks 1, 3, 5. Paired analyses use `paired_ok` seeds. Existing adapters
  (base, en, ja, bi, en2) are compared on exactly these prompts (their stored answers).
  Half the pool keeps the four experiments inside the compute budget (~93 CU).
- Every stage checkpoints, publishes `manifest.json` then `DONE`, and is resumable. Errors are fixed and logged.
- Budget guard: no new GPU stage starts when fewer than 12 CU remain; analysis runs on what exists.

## E1 — Bilingual adapter + steering edit (VM A)
- **Question:** does the variant-A edit (remove f⊥ from the weights) cut the bilingual adapter's over-refusal
  while keeping its safety gain?
- **Selection data (fresh):** one new batch (b0013, 100 seeds, 1,200 US prompts) from the `03_eval_batch`
  generator; seeds near-duplicate (cos > 0.85) of any frozen-pool seed are dropped. Dataset2 `heldout` is used up.
- **Directions:** S1 labels (extraction twins only) and S2 directions for bi_s0, bi_s1, bi_s2 (each its own).
- **Grid (bi_s0):** Lsrc ∈ {12, 18, 24} × λ ∈ {0.5, 1.0}, plus unedited bi_s0 and base. Selection rule = steering
  4.3 (admissible = over-refusal ≤ base + 2 pt → lowest unsafe; else lowest over-refusal within 1 pt of
  unedited safety; else fallback L18 λ0.5).
- **Final:** A*(bi_s0..2), each with its own direction, on the pool half; compared with stored bi_s0..2.

## E2 — More helpful examples (VM B)
- Condition `bi_h2`: bilingual rows, helpful items shown twice (en, ja, en, ja), refusal items once (en, ja):
  1,704 steps; seeds 0–2. Compared with `bi` (1:1, 1,136 steps). Dose is not matched (stated).

## E3 — Japanese-only replication (VM B)
- `ja2`: Japanese row twice per item (1,136 steps), seeds 0–2: dose control for `ja` (mirrors `en2`).
- Disjoint item halves: items split by seed (`sha256(seed_id) % 2`) into A and B; `en_A`, `ja_A`, `en_B`, `ja_B`
  (one row per item, training seed 0): does "Japanese training ≈ English training on English prompts, and much
  better on Japanese prompts" replicate on two independent item sets?

## E4 — US vs UK culture with matched cues (VM B)
- For the pool-half seeds, Qwen3-4B rewrites each neutral request (harmful and twin together, one cue for both)
  into **US-matched** and **UK-matched** versions with the same rule (one benign cue; no crime-associated
  group; act, target, severity, legality and specificity unchanged); fc/rp framings with the same template and a
  neutral "keep the existing cue, add no other" culture line. 24 new prompts per seed.
- Models: base, en_s0, ja_s0, bi_s0. Compared with their stored US-original and Japan answers.
- Tests: UK-m vs US-m (culture), US-original vs US-m (cue content), each × prompt language; adapter transfer.

## Analysis (local)
`FA_analysis`: per experiment, metrics (safety, fix rate, over-refusal, over-caution, wrong language,
truncation) by prompt language × culture, paired tests (exact McNemar per training seed, seed-cluster or
two-level bootstrap), 4B-judge confirmation; writes `FU/report.md` and `FU/results.json`.
