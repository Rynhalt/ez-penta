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

## 1. Headline result

On 228 validation prompts whose seeds were never used to build the fix:

| | Baseline | Restricted LoRA (layer 18 `down_proj`) | Broad LoRA (all layers) |
|---|---|---|---|
| Unsafe answers, harmful prompts the model should decline (n=118) | 10.4% | **0.9%** | 0.8% |
| Refusals, harmful prompts it should answer carefully (n=30) | 30% | 47% | 43% |
| **Needless refusals, harmless twin prompts (n=80)** | 10% | **20%** | 36% |
| Average harmless-answer length | 1,071 chars | 699 | 291 |

Paired before/after tests on the same prompts (exact McNemar):

| Comparison vs baseline | Got better | Got worse | p |
|---|---|---|---|
| Restricted, unsafe answers (harmful, n=144 paired) | 10 | 0 | **0.002** |
| Broad, unsafe answers (harmful, n=145 paired) | 11 | 0 | 0.001 |
| Restricted, needless refusals (twins, n=80) | 1 | 9 | 0.021 |
| Broad, needless refusals (twins, n=80) | 1 | 22 | 0.00001 |

**In one sentence:** the circuit-restricted fix removes almost all unsafe answers, and the
cost is roughly doubling needless refusals (10% → 20%). The broad fix is no safer and
over-refuses almost twice as much again, with much terser answers. **The restricted fix is the
better trade-off.**

### Did it close the US-vs-neutral culture gap?

Yes, by removing nearly all unsafe answers in both conditions.

| Bucket "US unsafe / neutral safe" | Baseline | Restricted | Broad |
|---|---|---|---|
| US-framed prompts unsafe | 23.8% | 2.4% | 2.3% |
| Neutral-framed prompts unsafe | 2.3% | 0% | 0% |

| Per-language harmful unsafe rate, US vs neutral | Baseline | Restricted | Broad |
|---|---|---|---|
| English | 10.7% vs 3.3% (gap 7.4 pp) | 3.4% vs 0% | 0% vs 0% |
| Japanese | 15.9% vs 2.3% (gap 13.6 pp) | 0% vs 0% | 2.3% vs 0% |

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

The bucket labels come from the original dataset2 run, which sampled at temperature 0.2. This
run uses greedy decoding, so the baseline does not reproduce them exactly: the target bucket
is 23.8% unsafe here, not 100%. That is expected regression toward the mean. The paired
before/after comparisons are unaffected, because all three models answer the same prompts the
same way.

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
- **Results.** See section 1.
- **Per-row data.** `pipeline5_answers.jsonl` holds every answer and verdict.

---

## 5. Interpretation and caveats

- **The safety gain is real but rests on small counts.** Ten prompts flipped from unsafe to
  safe and none flipped back (p = 0.002). Baseline Tiny Aya is already safe on most of these
  prompts, so only about 11 unsafe answers were available to fix.
- **Over-refusal is the main cost.**
  - Twin refusals doubled under the restricted fix (10% → 20%) and more than tripled under the
    broad one (→ 36%).
  - It is uneven by language. English rose 7% → 14%, but Japanese rose **14% → 28%**.
  - Prompts the model should answer carefully also drew more refusals (30% → 47%).
- **Some "harmless twins" are borderline.** Several newly refused twins wrap a harmless
  question in jailbreak framing, for example an "unrestricted mafia assistant that never
  refuses" persona. Refusing those is arguably acceptable, so the 20% overstates harmful
  over-refusal somewhat.
- **Answers got shorter.** Harmless answers shrank by about a third under the restricted fix
  and by about three quarters under the broad fix. The guard does not measure helpfulness,
  so this is the clearest sign of capability cost.
- **Adding helpful pairs worked.** An earlier iteration of this run trained on refusal pairs
  only, and its outputs were overwritten by this run. Against that iteration:

  | Restricted fix | Earlier iteration (refusals only) | This run (refusals + helpful) |
  |---|---|---|
  | Needless refusals | 10% → 46% | 10% → 20% |
  | "Answer carefully" prompts refused | 26% → 72% | 30% → 47% |
  | Unsafe answers | 13% → 0% | 10.4% → 0.9% |

  The comparison is indicative only: the earlier iteration's validation set shared 42 seeds
  with its training data and was a different prompt set.
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
10. **Decoding.** Validation uses greedy decoding with up to 512 new tokens, not the original
    eval's temperature 0.2. This makes the arms directly comparable, but baseline rates
    differ from dataset2's labels.
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
| `notebooks/pipeline4_circuit_restricted_fix_FAILED.ipynb` | The earlier out-of-memory T4 attempt, kept for reference; safe to delete |
| `notebooks/build_sampling_pools.py`, `build_pipeline2_prompts.py` | Deterministic builders for every prompt set (seed 42) |
| `notebooks/pipeline2_prompts_v2.json`, `pipeline3_*seed_prompts_v2.json`, `pipeline5_validation_prompts_v2.json` | The built prompt sets |
| `notebooks/combined/pipeline2_results.json` | Layer selection, bootstrap frequencies, projection tests |
| `notebooks/combined/pipeline3_safety_pairs_v2.jsonl` | 195 training pairs (`type` = refusal or helpful) |
| `notebooks/combined/pipeline3_guard_dropped_v2.jsonl` | 37 pairs the guard rejected |
| `notebooks/combined/pipeline4_results.json`, `pipeline4_adapters.zip` | Training summary; both LoRA adapters (113 MB, not for plain git) |
| `notebooks/combined/pipeline5_results.json`, `pipeline5_answers.jsonl` | All statistics; every answer and guard verdict |

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
