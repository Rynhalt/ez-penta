# post_train_2 Report: Seed-Disjoint, Unquantized Circuit-Restricted Fix

Second run of the Track-2 pipeline on **Tiny Aya Water** (`CohereLabs/tiny-aya-water`, 3.35B
parameters). The first run (`post_train_1/`) had three problems:

1. Its training "refusals" were not refusals.
2. Its validation prompts shared seeds with its training prompts.
3. It trained a 4-bit quantized model.

This run fixes all three and adds training examples that teach the model when *not* to
refuse. Unless stated otherwise, numbers below come from the final run, which replaced an
earlier refusal-only iteration of post_train_2.

Run date: 2026-11-12. Pipelines 4 and 5 ran on a Colab G4 (RTX PRO 6000, 97 GB). Pipeline 2
ran on a T4, and pipeline 3 ran locally on CPU.

---

## How results are reported

Every answer gets two labels from Qwen3Guard-Gen-0.6B, judged on the model's **answer** (with the
prompt as context): **Safe / Unsafe / Controversial**, and **refusal yes / no**. The judge never
sees the prompt's category. The three metrics below combine those labels with the prompt type
from dataset2:

| Metric | Prompts it is measured on | Counted as a failure | Lower is better |
|---|---|---|---|
| **Safety** (unsafe rate) | All **harmful** prompts: clearly harmful + borderline, merged | The answer is labelled **Unsafe** | yes |
| **Over-refusal** | **Harmless twins**: the harmless version of a harmful request | The answer is a **refusal** | yes |
| **Over-caution** | **Borderline** ("answer carefully") harmful prompts | The answer is a **refusal** | yes |

- **Safety** is about the answer's content: did the model say something harmful? Refusing or
  not does not enter into it. Answers labelled `Controversial` (or unparseable) have no
  safe/unsafe verdict and are left out of the denominator.
- **Over-refusal** is refusing something plainly harmless.
- **Over-caution** is refusing a sensitive but legitimate request that a careful assistant
  should answer with care (safety information, context, a better alternative). It is a softer
  signal than over-refusal: the borderline labels were assigned when dataset2 was generated and
  have not been validated by a person, and some refusals of borderline prompts are defensible.
- Refusing a **clearly harmful** prompt is the correct behaviour, so it is not a metric here.
- A refusal flag alone does not prove a refusal was unnecessary; confirming that needs a human
  check of the prompt. Treat over-refusal and over-caution as upper estimates.

All numbers below are recomputed from the saved answers by
[`recompute_three_metrics.py`](notebooks/recompute_three_metrics.py)
(output: `notebooks/combined/three_metric_summary.json`).

---

## 1. Headline result

On 228 validation prompts (English and Japanese) whose seeds were never used to build the fix:

| | Prompts | Baseline | Restricted LoRA (layer 18 `down_proj`) | Broad LoRA (all layers) |
|---|---|---|---|---|
| **Safety**: unsafe answers | 148 harmful | 8.3% (12/145) | **0.7%** (1/147) | 0.7% (1/148) |
| **Over-refusal**: refused | 80 harmless twins | 10.0% (8/80) | **20.0%** (16/80) | 36.3% (29/80) |
| **Over-caution**: refused | 30 borderline | 30.0% (9/30) | 46.7% (14/30) | 43.3% (13/30) |
| Average harmless-answer length | 80 harmless twins | 1,072 chars | 700 | 292 |

All 12 unsafe baseline answers came from clearly harmful prompts; no model gave an unsafe answer
to a borderline prompt. Safety denominators leave out `Controversial` answers (baseline 3,
restricted 1, broad 0).

Paired tests on the same prompts (exact McNemar; "better" = unsafe → safe or refused → answered):

| Comparison vs baseline | Paired prompts | Better | Worse | p |
|---|---|---|---|---|
| Restricted, safety | 144 | 10 | 0 | **0.002** |
| Broad, safety | 145 | 11 | 0 | 0.001 |
| Restricted, over-refusal | 80 | 1 | 9 | 0.021 |
| Broad, over-refusal | 80 | 1 | 22 | 0.00001 |
| Restricted, over-caution | 30 | 0 | 5 | 0.062 |
| Broad, over-caution | 30 | 0 | 4 | 0.12 |

Restricted vs broad on the same prompts: equally safe (1 prompt each way), but the broad fix
refuses 16 harmless twins the restricted fix answers, against 3 the other way (p = 0.004).
On over-caution they barely differ (1 vs 0).

Seed-level robustness (each source request is one unit; 10,000-sample seed bootstrap, sign test
over seeds):

| | Prompts / seeds | Change | 95% CI | Seeds down / up | p |
|---|---|---|---|---|---|
| Restricted, safety | 144 / 54 | −6.9 pts | −11.7 to −2.9 | 9 / 0 | 0.004 |
| Broad, safety | 145 / 54 | −7.6 pts | −11.8 to −3.9 | 11 / 0 | 0.001 |
| Restricted, over-refusal | 80 / 33 | +10.0 pts | +2.5 to +19.1 | 1 / 8 | 0.039 |
| Broad, over-refusal | 80 / 33 | +26.3 pts | +17.1 to +36.2 | 0 / 16 | 0.00003 |
| Restricted, over-caution | 30 / 13 | +16.7 pts | 0.0 to +34.2 | 0 / 3 | 0.25 |
| Broad, over-caution | 30 / 13 | +13.3 pts | 0.0 to +26.7 | 0 / 3 | 0.25 |

**In one sentence:** the circuit-restricted fix removes almost all unsafe answers, at the cost
of doubling over-refusal (10% → 20%). The broad fix is no safer and over-refuses almost twice as
much again, with much terser answers. **The restricted fix is the better trade-off.** Both fixes
also increase over-caution, but with only 30 borderline prompts that change is not significant.

### By language, culture and jailbreak technique

| | Safety: baseline → restricted → broad | Over-refusal: baseline → restricted → broad | Over-caution: baseline → restricted → broad |
|---|---|---|---|
| English | 6.9% → 1.7% → 0.0% | 6.8% → 13.6% → 45.5% | 2/6 → 4/6 → 4/6 |
| Japanese | 9.2% → 0.0% → 1.1% | 13.9% → 27.8% → 25.0% | 29.2% → 41.7% → 37.5% |
| US culture | 13.9% → 1.4% → 1.4% | 10.0% → 22.5% → 45.0% | 26.7% → 46.7% → 40.0% |
| Neutral culture | 2.7% → 0.0% → 0.0% | 10.0% → 17.5% → 27.5% | 33.3% → 46.7% → 46.7% |
| Direct request | 13.5% → 1.9% → 1.9% | 8.3% → 19.4% → 30.6% | 1/10 → 4/10 → 3/10 |
| Role-play | 3.6% → 0.0% → 0.0% | 3/12 → 4/12 → 4/12 | 8/10 → 10/10 → 10/10 |
| Fiction | 3/15 → 0/15 → 0/16 | 0/18 → 0/18 → 8/18 | 0/2 → 0/2 → 0/2 |
| Refusal suppression | 0/22 → 0/22 → 0/22 | 2/14 → 5/14 → 6/14 | 0/8 → 0/8 → 0/8 |

Cells with fewer than 20 prompts are shown as counts.

### Did it close the US-vs-neutral culture gap?

Yes, by removing nearly all unsafe answers in both conditions (safety metric, harmful prompts).

| Bucket "US unsafe / neutral safe" | Baseline | Restricted | Broad |
|---|---|---|---|
| US-framed prompts unsafe | 23.8% (10/42) | 2.4% (1/42) | 2.3% (1/43) |
| Neutral-framed prompts unsafe | 2.3% (1/43) | 0% (0/43) | 0% (0/43) |

| Safety, US vs neutral | Baseline | Restricted | Broad |
|---|---|---|---|
| English | 10.7% vs 3.3% (gap 7.4 pts) | 3.4% vs 0% | 0% vs 0% |
| Japanese | 15.9% vs 2.3% (gap 13.6 pts) | 0% vs 0% | 2.3% vs 0% |

The stratified Mantel–Haenszel culture odds ratio was **5.7 (p = 0.035)** at baseline on this
set. After either fix it is undefined (p ≈ 1): almost no unsafe answers remain, so there is no
gap left to measure. For reference, pipeline 1 measured OR 1.98 (95% CI 1.58–2.48,
p = 2.2e-9) on the full dataset.

---

## 2. What changed from post_train_1

| | post_train_1 | post_train_2 |
|---|---|---|
| Precision | 4-bit QLoRA for training; fixed model merged from quantized base | **bf16 everywhere**, baseline and fixed loaded the same way |
| Training data | 22 pairs; `single_call` mode, so "chosen" answers were thorough *compliance*, not refusals | **195 Qwen3Guard-verified pairs**: 116 refusals + 79 helpful answers to harmless twins |
| Train/validation separation | Split by prompt; rewordings of the same seed landed in both | **Split by seed**: training uses `tune` seeds, validation uses `analysis` seeds, `heldout` untouched |
| Validation | 40 prompts; harmless twins mixed into the harmful buckets | **228 prompts**: harmful pairs in your 50/20/30 buckets, plus a separate 40-pair harmless "twin probe" for over-refusal |
| Pipeline 2 prompts | 8 per condition | 48–60 per condition, with bootstrap stability selection (30 resamples) |
| Target layers | 18 and 23 | **18** (picked by both languages, top-5 in 100% of resamples) |

---

## 3. Data: what was used where

Every dataset2 seed belongs to exactly one split (stratified by category):
analysis 50%, tune 30%, heldout 20%. A script asserts zero prompt overlap and zero seed
overlap between training and validation, and that no heldout prompt is used.

| Use | Split | Contents | Built by |
|---|---|---|---|
| Pipeline 2 (layer discovery) | `tune` | en/ja, control framing. Harmful-neutral / twin-neutral / harmful-US: **en 53/60/54, ja 48/52/56** | [`build_pipeline2_prompts.py`](notebooks/build_pipeline2_prompts.py) |
| Pipeline 3 refusal seeds | `tune` | 135 harmful, US-framed, `expected=decline` prompts (36 were real jailbreaks at baseline) | [`build_sampling_pools.py`](notebooks/build_sampling_pools.py) |
| Pipeline 3 helpful seeds | `tune` | 101 harmless twins matched to those seeds (same seed, culture, language, framing) | same |
| Pipeline 5 validation | `analysis` | 228 prompts in 114 US/neutral pairs, detailed below | same |
| Final evaluation | `heldout` | **Untouched**, reserved for you | — |

Validation pairs (each pair is the same request with US framing and with neutral framing):

| Bucket | Pairs | Meaning (from the original baseline eval) |
|---|---|---|
| `us_unsafe_neutral_safe` | 43 (all available) | The target pattern |
| `both_safe` | 17 | Sampled at 20:50 relative to the target bucket |
| `us_safe_neutral_unsafe` | 14 (all available) | The reverse pattern |
| `twin_probe` | 40 | Harmless twins, used only to measure over-refusal |

The bucket labels come from the original dataset2 run, which used a 4-bit quantized model
(GGUF Q4_K_M through llama.cpp, greedy decoding). This run uses the full-precision bf16 model
through transformers, so the baseline does not reproduce those labels exactly: the target bucket
is 23.8% unsafe here, not 100%. The paired before/after comparisons are unaffected, because all
three models answer the same prompts with the same settings.

---

## 4. Pipeline by pipeline

### Pipeline 1 — stratified diagnosis (not rerun)
[`pipeline1_stratified_diagnosis.ipynb`](notebooks/pipeline1_stratified_diagnosis.ipynb). Its
full-dataset Mantel–Haenszel result (OR 1.98, p = 2.2e-9) is reused as the reference
baseline. Nothing in it depends on the post_train_2 changes.

### Pipeline 2 — refusal direction and target layer
[`pipeline2_circuit_discovery.ipynb`](notebooks/pipeline2_circuit_discovery.ipynb) →
`combined/pipeline2_results.json`

- **Method.**
  - The refusal direction is the per-layer difference of mean last-token activations between
    harmful and harmless neutral prompts, extracted with SafeTune's `SteeringVectorExtractor`.
  - Safety neurons are ranked by weight (16 per layer).
  - A layer is selected if it appears in the top 5 across 30 bootstrap resamples.
- **Alignment check.** Our own hooks and SafeTune's extractor agree on every layer: mean cosine
  0.9997, minimum 0.997. Comparing each layer with its neighbor gives only 0.77, so the
  layers are not off by one.
- **Stable layer.**
  - **18 for both English and Japanese**, in the top 5 in 100% of resamples.
  - Layer 15 was in the top 5 in 97% of resamples for both languages.
  - The target set is therefore just `[18]`.
- **Projection test.** Do US-framed prompts sit lower on the refusal direction than held-out
  neutral prompts?

  | Direction from → applied to | Gap (neutral − US) | p |
  |---|---|---|
  | en → en | 2.38 | 0.18 |
  | en → ja | 2.33 | 0.15 |
  | ja → en | 2.86 | 0.12 |
  | ja → ja | 2.54 | 0.21 |

  All four gaps point the same way: US framing pushes activations *away* from refusal, in both
  languages and across languages. **None is significant on its own.** An earlier iteration of this run did
  get significant p-values (0.0007–0.047), but its prompts were not restricted to `tune`
  seeds and overlapped the validation seeds, so this weaker, cleaner result is the one to
  trust. The layer choice does not depend
  on these p-values; it rests on the bootstrap stability above.

### Pipeline 3 — training data (CuratorKIT + Qwen3Guard gate)
[`pipeline3_curatorkit_data.ipynb`](notebooks/pipeline3_curatorkit_data.ipynb) →
`combined/pipeline3_safety_pairs_v2.jsonl`

- **Generation.** CuratorKIT `PreferenceGenerationTask` in **`two_pass`** mode, with Tiny Aya
  itself (GGUF, local llama-server) as the writer.
  - *Refusal pass.* Chosen = a refusal, rejected = compliance. 133 pairs generated.
  - *Helpful pass.* Chosen = a helpful answer to the harmless twin, rejected = a needless
    refusal. 99 pairs generated.
- **Guard gate.** Qwen3Guard-Gen-0.6B judged every "chosen" answer.

  | Pass | Kept only if | Kept | Notes |
  |---|---|---|---|
  | Refusal | Safe *and* a refusal | **116 / 133** | 33 of the kept pairs come from real jailbreak seeds |
  | Helpful | Safe *and* not a refusal | **79 / 99** | 13 judged Unsafe, 7 Controversial |

  The 37 dropped pairs are in `combined/pipeline3_guard_dropped_v2.jsonl` for inspection.
- **Total: 195 verified pairs.**

### Pipeline 4 — circuit-restricted and broad LoRA
[`pipeline4_circuit_restricted_fix.ipynb`](notebooks/pipeline4_circuit_restricted_fix.ipynb) →
`combined/pipeline4_results.json`, `combined/pipeline4_adapters.zip`

- **Setup.**
  - bf16 base model: 6.70 GB on the GPU, and every parameter checked to be bf16.
  - Supervised fine-tuning on the 195 (instruction, chosen) pairs, with loss on the
    assistant turn only.
  - Manual PyTorch loop: 1 epoch, batch 1, AdamW, learning rate 1e-4, max length 768.
- **Results.**

  | Arm | LoRA placement | Trainable params | Loss, first → last quarter | Peak GPU memory |
  |---|---|---|---|---|
  | Restricted | `layers.18.mlp.down_proj` only | 208,896 (0.006%) | 1.28 → 0.63 | 9.6 GB |
  | Broad | all q/k/v/o/gate/up/down, every layer | 30.2M (0.9%) | 0.64 → 0.48 | 9.9 GB |

- **Adapter check.** Both adapters' `lora_B` weights moved away from zero, so training really
  updated them.
- **Saved outputs.**
  - The adapters (113 MB zipped) are the durable result.
  - The merged 6.7 GB checkpoints lived only on the Colab VM, for pipeline 5.
- Peak memory under 10 GB means this run would also fit on a 14.5 GB T4.

### Pipeline 5 — validation
[`pipeline5_validation.ipynb`](notebooks/pipeline5_validation.ipynb) →
`combined/pipeline5_results.json`, `combined/pipeline5_answers.jsonl`

- **Answer generation.** All three models answered all 228 prompts: greedy decoding, up to
  512 new tokens, batch 8.
- **Judging.** Qwen3Guard-Gen-0.6B (bf16, no truncation) judged every answer for safety and
  refusal.
  - Labels: 666 Safe, 14 Unsafe, 4 Controversial, 0 unparseable.
  - Controversial answers are excluded from unsafe-rate denominators, so paired n is 144–145
    rather than 148.
- **Results.** See section 1 (safety, over-refusal and over-caution).
- **Per-row data.** `pipeline5_answers.jsonl` holds every answer and verdict.

---

## 5. Interpretation and caveats

- **The safety gain is real but rests on small counts.** Ten harmful prompts flipped from
  unsafe to safe and none flipped back (p = 0.002; seed-level CI −11.7 to −2.9 points). Baseline
  Tiny Aya is already safe on most of these prompts, so only 12 unsafe answers were available to
  fix, all on clearly harmful prompts.
- **Over-refusal is the main cost.**
  - It doubled under the restricted fix (10% → 20%) and more than tripled under the broad one
    (→ 36.3%).
  - It is uneven by language: English rose 6.8% → 13.6% and Japanese **13.9% → 27.8%**.
- **Over-caution rose too, but is weakly measured.** Borderline prompts drew more refusals
  (30% → 46.7% restricted, 43.3% broad), but with 30 prompts neither change is significant
  (p = 0.062 and 0.12). For the restricted fix the extra refusals are on direct requests
  (1/10 → 4/10) and role-play (8/10 → 10/10); fiction and refusal-suppression borderline prompts
  were never refused by any model.
- **Some "harmless twins" are borderline.** Several newly refused twins wrap a harmless
  question in jailbreak framing, for example an "unrestricted mafia assistant that never
  refuses" persona. Refusing those is arguably acceptable, so 20% overstates genuine
  over-refusal somewhat.
- **Answers got shorter.** Harmless answers shrank by about a third under the restricted fix
  and by about three quarters under the broad fix. The guard does not measure helpfulness,
  so this is the clearest sign of capability cost.
- **Adding helpful pairs worked.** An earlier iteration of this run trained on refusal pairs
  only, and its outputs were overwritten by this run. Against that iteration:

  | Restricted fix | Earlier iteration (refusals only) | This run (refusals + helpful) |
  |---|---|---|
  | Over-refusal | 10% → 46% | 10% → 20% |
  | Over-caution | 26% → 72% | 30% → 46.7% |
  | Unsafe answers, clearly harmful prompts only | 13% → 0% | 10.4% → 0.9% |

  The comparison is indicative only: the earlier iteration's validation set shared 42 seeds
  with its training data and was a different prompt set. Its answers no longer exist, so its
  unsafe rate cannot be recomputed with the merged safety metric; the last row uses the old
  clearly-harmful-only definition for both runs.
- **The guard is the only judge.** Qwen3Guard-0.6B decides "unsafe" and "refused". Its errors
  apply equally to all three models, so the paired comparisons are fair, but absolute rates
  inherit its biases.
- **This is still validation, not a final evaluation.** The `heldout` seeds have not been
  touched.

---

## 6. Assumptions made

1. **Seed-level split.** `tune` seeds build the fix (pipelines 2–4), `analysis` seeds validate
   it, and `heldout` is reserved.
2. **Training size is 195, not the requested 300.** The seed-disjoint split leaves 135 refusal
   seeds and 101 matched twins in `tune`, and the guard gate dropped 37 pairs. I did not pad
   with lower-quality data.
3. **`answer_carefully` seeds are excluded from training**, per your choice. They still appear
   in validation (30 prompts).
4. **Helpful pairs** use each refusal seed's matched harmless twin, with a needless refusal as
   the rejected answer. Generation cap: 512 tokens for helpful answers, 256 for refusals.
5. **Guard criteria for training data.** Refusal examples were kept only if Safe and a
   refusal; helpful examples only if Safe and not a refusal. The guard does not check
   factual accuracy, so some helpful answers may contain hallucinations.
6. **Validation** uses all available target and reverse pairs, both-safe pairs at the 20:50
   ratio, and 40 twin-probe pairs. That gives 228 prompts and modest statistical power.
7. **Pipeline 2** used every available `tune` prompt (48–60 per cell, below the 64 target)
   and held out half the neutral prompts for the projection test.
8. **Pipeline 1 was not rerun**; its full-dataset statistics are reused.
9. **Hyperparameters** were kept from post_train_1 (1 epoch, lr 1e-4, batch 1) so the two
   runs stay comparable. Nothing was tuned on the validation set.
10. **Decoding and precision.** Validation uses greedy decoding with up to 512 new tokens on
    the full-precision bf16 model. The original dataset2 eval was also greedy, but on a 4-bit
    GGUF (Q4_K_M) model through llama.cpp, so baseline rates differ from dataset2's labels.
    All three arms here use identical settings, so they are directly comparable.
11. **GPU.** G4 when available, T4 as fallback.

---

## 7. Problems found and fixed during this run

| Problem | Fix |
|---|---|
| `colab exec` reports success even when a cell fails, so failures cascaded | A checker scans each output notebook and stops the chain on any cell error |
| `colab status` reports success for a session that doesn't exist, so the chain never started | Check for the session's `Hardware:` line instead |
| The HF token leaked into failed notebooks' tracebacks | Scrubbed automatically after every run; a grep confirms no token in `post_train_2/` |
| Pipeline 2 ran out of memory on a T4 (262k-vocabulary logits) | Hook the decoder layers directly (no logits); extractor batch size 4 |
| Pipeline 2's alignment assert was too strict | Require each layer to best-match itself, plus a mean-cosine threshold |
| Pipeline 2's projection test was off by one layer and scored in-sample | Use the layer-k output and score only the held-out neutral half |
| Pipeline 4 ran out of memory on a T4: all notebooks shared one Python kernel, so pipeline 2's model stayed resident (13.4 GB) | Restart the kernel before each notebook; a first-cell guard checks the GPU is empty; padding is trimmed per batch (loss unchanged, verified) |
| The Colab session kept running after the chain failed | The chain now stops the session on any exit |

---

## 8. Files

| File | What it is |
|---|---|
| `notebooks/pipeline{1..5}_*.ipynb` | The notebooks, with outputs from this run |
| `notebooks/build_sampling_pools.py`, `build_pipeline2_prompts.py` | Deterministic builders for every prompt set (seed 42) |
| `notebooks/pipeline2_prompts_v2.json`, `pipeline3_*seed_prompts_v2.json`, `pipeline5_validation_prompts_v2.json` | The built prompt sets |
| `notebooks/combined/pipeline2_results.json` | Layer selection, bootstrap frequencies, projection tests |
| `notebooks/combined/pipeline3_safety_pairs_v2.jsonl` | 195 training pairs (`type` = refusal or helpful) |
| `notebooks/combined/pipeline3_guard_dropped_v2.jsonl` | 37 pairs the guard rejected |
| `notebooks/combined/pipeline4_results.json` | Training summary |
| `notebooks/combined/pipeline4_adapters.zip` | Both LoRA adapters. **Local only:** 113 MB, over GitHub's 100 MB limit, so it is gitignored. Share it separately (Drive, Hugging Face) if teammates need it. |
| `notebooks/combined/pipeline5_results.json`, `pipeline5_answers.jsonl` | All statistics; every answer and guard verdict |
| `notebooks/recompute_three_metrics.py`, `notebooks/combined/three_metric_summary.json` | Recomputes every number in this report as safety / over-refusal / over-caution, from the saved answers |

The build scripts also read `notebooks/combined/observations.csv` at the repo root (412 MB).
That file is gitignored too, so anyone rerunning the prompt builders needs a copy of it.

To rebuild a fixed model: load `CohereLabs/tiny-aya-water` in bf16, apply the adapter from the
zip with `peft.PeftModel.from_pretrained`, then call `merge_and_unload()`.

---

## 9. Suggested next steps

1. **Reduce over-refusal, especially in Japanese.** Add more Japanese helpful pairs, include
   `answer_carefully` seeds as "careful answer" pairs, or lower the learning rate.
2. **Spot-check the newly refused twins by hand** to separate genuine over-refusal from
   reasonable refusals of jailbreak-framed prompts.
3. **Measure helpfulness, not just safety.** For example, use an LLM judge on answer quality
   for the twin probe, since answer length dropped noticeably.
4. **Run the final evaluation on the `heldout` seeds** once the configuration is settled.
