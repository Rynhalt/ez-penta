# Pipeline 6: mixed-language (code-mixed) evaluation

Run: 2026-11-13, Colab L4 (the G4 was unavailable). Experiment: `post_train_2`.
This is a follow-up to pipelines 1–5. The hackathon submission repo (`tiny-aya-safety`) and its
Hugging Face release were not changed.

## Question

The safety fix (LoRA adapters from pipeline 4) was trained on English and Japanese prompts only, and
validated only on English and Japanese (pipeline 5). **Does it carry over to code-mixed prompts,**
where the jailbreak framing is in one language and the request in the other? And does the
restricted adapter keep its advantage over the broad one there?

Motivation: mixed-language prompts make up 8,672 of Dataset2's 21,184 prompts, but the fix was
never tested on them. In Dataset2's original evaluation (Tiny Aya Water GGUF Q4, harmful prompts,
non-heldout splits), the weakest conditions were **role-play framing** (20.7% unsafe), **US cultural
framing** (19.2%) and **Japanese** (15.9%). Mixed prompts were close behind (15.0%; English 10.4%).
Compared like-for-like (jailbreak framings only, since mixed prompts have no direct version), mixed
and pure English/Japanese were about equally unsafe on harmful prompts in the analysis split:
17.8% (175/985) vs 17.4% (173/992).

## How results are reported

Every answer gets two labels from Qwen3Guard-Gen-0.6B, judged on the model's **answer** (with the
prompt as context): **Safe / Unsafe / Controversial**, and **refusal yes / no**. The three metrics
combine those labels with Dataset2's prompt type:

| Metric | Prompts it is measured on | Counted as a failure |
|---|---|---|
| **Safety** (unsafe rate) | All **harmful** prompts: clearly harmful + borderline, merged | The answer is labelled **Unsafe** |
| **Over-refusal** | **Harmless twins** | The answer is a **refusal** |
| **Over-caution** | **Borderline** ("answer carefully") harmful prompts | The answer is a **refusal** |

Lower is better for all three. Safety ignores whether the model refused; `Controversial` answers
are left out of its denominator. Over-caution is the softer signal: borderline labels were assigned
when Dataset2 was generated and have not been human-validated. Refusing a clearly harmful prompt is
correct, so it is not a metric. Definitions and numbers are shared with pipeline 5's report
([`report.md`](report.md)); every number here is recomputed from the saved answers by
[`recompute_three_metrics.py`](notebooks/recompute_three_metrics.py).

## Findings

| Mixed-language prompts | Prompts | Original | Restricted LoRA (layer 18) | Broad LoRA (all layers) |
|---|---|---:|---:|---:|
| **Safety**: unsafe answers | 500 harmful | 9.5% (46/484) | **2.0%** (10/491) | 2.8% (14/497) |
| **Over-refusal**: refused | 200 harmless twins | 11.0% (22/200) | **28.0%** (56/200) | 35.0% (70/200) |
| **Over-caution**: refused | 100 borderline | 13.0% (13/100) | **31.0%** (31/100) | 41.0% (41/100) |
| Average harmless-answer length (characters) | 200 harmless twins | 902 | 530 | 278 |

Safety denominators leave out answers Qwen3Guard labelled `Controversial` (original 16, restricted
9, broad 3 of the 500 harmful). Within safety, clearly harmful prompts went 11.1% (43/386) → 2.0%
(8/393) → 2.5% (10/398), and borderline ones 3.1% (3/98) → 2.0% (2/98) → 4.0% (4/99).

### 1. The safety fix transfers to mixed-language prompts it never saw

| Paired test, same prompt (exact McNemar) | n | improved | worsened | p |
|---|---:|---:|---:|---:|
| Restricted vs original, safety (unsafe → safe) | 478 | 37 | 2 | 3×10⁻⁹ |
| Broad vs original, safety | 481 | 39 | 5 | 1×10⁻⁷ |

Seed-level, treating each source request as one unit (95% seed-bootstrap CI; sign test over seeds):

| | prompts / seeds | change | 95% CI | seeds down / up | p |
|---|---:|---:|---:|---:|---:|
| Restricted, safety | 478 / 139 | −7.3 pts | −10.3 to −4.6 | 25 / 1 | 8×10⁻⁷ |
| Broad, safety | 481 / 140 | −7.1 pts | −10.5 to −3.9 | 25 / 4 | 1×10⁻⁴ |

The restricted adapter removes about four fifths of unsafe answers (9.5% → 2.0%) and the broad one
about seven tenths (→ 2.8%). On the same prompts the two are equally safe (6 vs 8, p = 0.79), but the
restricted adapter made fewer answers worse than the original (2 vs 5).

### 2. Over-refusal costs more on mixed prompts than on pure English/Japanese

| | n | answered → refused | refused → answered | p |
|---|---:|---:|---:|---:|
| Restricted vs original, over-refusal | 200 | 36 | 2 | 5×10⁻⁹ |
| Broad vs original, over-refusal | 200 | 49 | 1 | 9×10⁻¹⁴ |

| Seed-level | prompts / seeds | change | 95% CI | seeds down / up | p |
|---|---:|---:|---:|---:|---:|
| Restricted, over-refusal | 200 / 109 | **+17.0 pts** | +11.3 to +23.1 | 2 / 31 | 1×10⁻⁷ |
| Broad, over-refusal | 200 / 109 | +24.0 pts | +17.4 to +30.9 | 1 / 39 | 8×10⁻¹¹ |

On pure English/Japanese (pipeline 5) the restricted adapter's increase was +10 points (CI +2.5 to
+19.1). The restricted adapter still over-refuses less than the broad one on the same harmless
prompts: broad refuses 29 that restricted answers, against 15 the other way (p = 0.049). Its
advantage holds but is smaller than on pure languages (16 vs 3, p = 0.004).

### 3. Over-caution rises clearly on mixed prompts

| | n | answered → refused | refused → answered | p |
|---|---:|---:|---:|---:|
| Restricted vs original, over-caution | 100 | 19 | 1 | 4×10⁻⁵ |
| Broad vs original, over-caution | 100 | 28 | 0 | 7×10⁻⁹ |

| Seed-level | prompts / seeds | change | 95% CI | seeds down / up | p |
|---|---:|---:|---:|---:|---:|
| Restricted, over-caution | 100 / 52 | +18.0 pts | +8.6 to +28.7 | 1 / 13 | 0.002 |
| Broad, over-caution | 100 / 52 | +28.0 pts | +17.8 to +38.7 | 0 / 20 | 2×10⁻⁶ |

Refusals of borderline prompts more than double (13% → 31%) with the restricted adapter and triple
with the broad one (→ 41%). Unlike pipeline 5, where 30 borderline prompts were too few for a
significant result, this is clear here. The broad adapter over-cautions more than the restricted one
on the same prompts (16 vs 6, p = 0.052).

### 4. Full precision is much safer than the quantized model on the same prompts

| Original model, harmful prompts (safety) | Unsafe |
|---|---:|
| bf16, pure English/Japanese (pipeline 5 validation) | 8.3% (12/145) |
| **bf16, mixed (this run)** | **9.5% (46/484)** |
| GGUF Q4_K_M, the same 500 harmful mixed prompts (Dataset2 baseline run) | 21.4% (107/500) |

On these mixed prompts, the original model in full precision (bf16) is unsafe less than half as
often as the quantized GGUF Q4 model was in the Dataset2 run (9.5% vs 21.4%). This is a general
inference-stack effect, not a mixed-language one: in the GGUF run, mixed and pure English/Japanese
jailbreak prompts were about equally unsafe (17.8% vs 17.4%, analysis split). It compares separate
runs, not paired answers. The GGUF run also differs in its inference stack (llama.cpp), its token
limit (1,024 vs 512) and its judge (Qwen3Guard GGUF Q8_0 via AuditKIT vs bf16 transformers).

### 5. Where the remaining problems are

| Breakdown | Original | Restricted | Broad |
|---|---|---|---|
| **Safety, by framing** | fiction 17.4%, role-play 5.2%, refusal suppression 6.6% | fiction 5.7% (9/159), role-play 0.5% (1/194), refusal suppression **0%** | fiction 3.0%, role-play 2.6%, refusal suppression 2.9% |
| **Over-refusal, by framing** | fiction 2.8%, role-play 19.7%, refusal suppression 11.9% | fiction 4.2%, role-play **57.4%**, refusal suppression 26.9% | fiction 27.8%, role-play 47.5%, refusal suppression 31.3% |
| **Over-caution, by framing** | fiction 2.8%, role-play 25.0%, refusal suppression 13.9% | fiction 5.6%, role-play **57.1%**, refusal suppression 36.1% | fiction 36.1%, role-play 50.0%, refusal suppression 38.9% |
| **Over-refusal, by mix direction** | EN frame + JA request 14.0%, JA frame + EN request 8.8% | 39.5% / 19.3% | 44.2% / 28.1% |
| **Over-refusal, by answer language** | English 13.8%, Japanese 7.1% | English 29.3%, Japanese 26.2% | English **50.9%**, Japanese 13.1% |
| **Over-caution, by answer language** | English 18.6%, Japanese 4.9% | English 35.6%, Japanese 24.4% | English **59.3%**, Japanese 14.6% |
| **Safety, by culture** | Japan 7.7%, neutral 11.3%, US 9.9% | 1.6% / 1.3% / 3.2% | 2.2% / 3.2% / 3.1% |

- Nine of the restricted adapter's 10 remaining unsafe answers are fiction-framed; the tenth is a
  role-play borderline prompt. Refusal suppression is fully fixed.
- Its over-refusal and over-caution both concentrate on role-play prompts: requests wrapped in an
  "unrestricted assistant who never refuses" persona. Refusing those is arguably reasonable, so 57%
  overstates genuine over-refusal. It also over-refuses English-framing + Japanese-request prompts.
- The broad adapter over-refuses and over-cautions mostly when the answer is requested in English.
- There is no culture gap left to speak of in mixed prompts, before or after the fix.

### Bottom line

The fix generalises to code-mixed jailbreaks it never trained on: safety improves from 9.5% to 2.0%
unsafe. The cost is larger than on pure English/Japanese: over-refusal rises from 11% to 28% and
over-caution from 13% to 31%, much of it on persona-wrapped prompts. The restricted adapter remains
the better trade-off, but by a smaller margin than on pure languages.

## Methodology

### Prompt sample
- **Source:** Dataset2 `observations.csv`, rows with `prompt_language == "mix"`, `split == "analysis"`,
  `analysis_eligible`, and completed generation and scoring. That gives 2,145 prompts from 155 seeds,
  out of 8,672 mixed prompts in total; the rest are other splits or not eligible.
- **Sample:** 700 prompts, deterministic (random seed 42):
  - 400 harmful to decline, 100 borderline (`answer_carefully`), 200 harmless twins;
  - allocated in proportion across framing × mix direction × culture within each group;
  - spread across seeds (each seed's first prompt is picked before any seed's second), covering 148
    seeds with at most 17 prompts each.
- **Composition:**

  | Factor | Counts |
  |---|---|
  | Framing | role-play 255, fiction 238, refusal suppression 207 |
  | Mix direction | JA frame + EN request 402, EN frame + JA request 298 |
  | Answer language | English 397, Japanese 303 |
  | Culture | Japan 258, US 231, neutral 211 |

  Mixed prompts have no direct ("control") framing in Dataset2.
- **Checks (asserted by the builder):**
  - analysis split only;
  - mixed language only;
  - 0 seeds shared with the training data (tune split);
  - 0 prompts shared with the pipeline 5 validation set;
  - heldout untouched.

### Models and settings (identical to pipeline 5)

| | |
|---|---|
| Original | `CohereLabs/tiny-aya-water` @ `41a99216d12b3ef4881df762934186e5e4d2bbc1`, bf16 |
| Restricted | original + `isaac369369/tiny-aya-water-safety-lora-layer18`, merged |
| Broad | original + `isaac369369/tiny-aya-water-safety-lora-broad`, merged |
| Decoding | greedy (temperature 0), up to 512 new tokens, chat template, prompts cut at 1,024 tokens |
| Judge | `Qwen/Qwen3Guard-Gen-0.6B`, bf16, greedy, 64 new tokens, no truncation; `Safety:` and `Refusal:` lines parsed |
| Batching | batch size 8 (24 GB L4), prompts in length order, the same order for all three models |

The adapters were loaded from the public Hugging Face repos because the 113 MB adapter zip failed to
upload to Colab. The Hugging Face files were published from that same zip, so they are the same
adapters.

### Statistics
- **Safety** = Unsafe / (Safe + Unsafe) over all harmful prompts (clearly harmful + borderline);
  `Controversial`/`Unparsed` are excluded from the denominator.
- **Over-refusal** = refusals / all harmless-twin answers; **over-caution** = refusals / all
  borderline answers. Both use Qwen3Guard's refusal flag.
- Paired same-prompt comparisons use the exact McNemar test.
- Seed-level robustness uses a 10,000-sample seed-cluster bootstrap and a sign test over seeds.
- These are the same definitions and code as pipeline 5 and the `tiny-aya-safety` metrics module.

### Robustness of the run
- Answers and verdicts were written after every batch, and the notebook resumes from those files.
- The run script retried on connection loss (not needed this time) and always stopped the Colab session.
- **Run checks:**
  - 0 notebook errors;
  - 2,100 rows (700 per model), IDs identical to the prompt file;
  - 0 empty answers, 0 unjudged answers;
  - no tokens in any output.

## Where things are

### Local (`ez-penta` repo, branch `post-train-2`; not committed)
| File | What |
|---|---|
| `post_train_2/notebooks/build_mixed_language_prompts.py` | builds the sample (`--full` for all 2,145) |
| `post_train_2/notebooks/pipeline6_mixed_language_prompts.json` | the 700 prompts, with Dataset2 metadata and the GGUF baseline labels |
| `post_train_2/notebooks/pipeline6_mixed_language_eval.ipynb` | the evaluation notebook, **executed, with all outputs** |
| `post_train_2/notebooks/combined/pipeline6_mix_answers.jsonl` | 2,100 rows: prompt metadata, model, answer, label, unsafe, refused, raw judge output |
| `post_train_2/notebooks/combined/pipeline6_mix_results.json` | the run's original statistics (clearly-harmful-only safety) |
| `post_train_2/notebooks/recompute_three_metrics.py`, `combined/three_metric_summary.json` | every number in this report, as safety / over-refusal / over-caution |
| `post_train_2/pipeline6_mixed_language_report.md` | this report |

### Google Drive (`MyDrive/ez-penta/post_train_2/`, the copy at folder id `1p0zq2_Rc497dlIrfQtiOGYI8nMP7Dy_x`)
- `notebooks/`: https://drive.google.com/drive/folders/1h37qxJyOvsFX9a_oJJq1pYM3DR5jHbqQ
  (builder script, prompt file, notebook, and this report)
- `notebooks/combined/`: https://drive.google.com/drive/folders/1jJ_jlMcd_5CfFniE1ln5PJyw18K1HwjW
  (answers and results)

These are uploaded by hand. Note that `MyDrive/ez-penta` contains two folders named `post_train_2`.

### Inputs used
- Dataset2: `ez-penta/notebooks/combined/observations.csv` (same file as Drive
  `ez-penta/dataset2_eval_gguf/combined/observations.csv`, SHA-256 prefix `7a38bedf6d2d3730`)
- Adapters: Hugging Face `isaac369369/tiny-aya-water-safety-lora-layer18` and `-lora-broad`
- Pure-language comparison: `post_train_2/notebooks/combined/pipeline5_answers.jsonl`

## How to rerun

1. `python post_train_2/notebooks/build_mixed_language_prompts.py` (optionally `--full`).
2. Open `pipeline6_mixed_language_eval.ipynb` in Colab on a GPU runtime. A G4 or A100 is fastest; an
   L4 takes about 2.5 hours for 700 prompts; avoid the T4.
3. Add a Hugging Face token with Tiny Aya access as the `HF_TOKEN` secret, and run all.
4. Inputs come from `/content` if uploaded there, otherwise from Drive (the notebook mounts it).
   With Drive mounted, the results are copied back into `post_train_2/notebooks/combined/`.

## Limitations
- **Judge:** labels come from one automatic judge (Qwen3Guard-Gen-0.6B); no human review.
- **Sample:** a stratified 700-prompt sample of the 2,145 eligible analysis prompts, not all of them;
  run with `--full` for the rest. Analysis split, not heldout.
- **Framings:** mixed prompts only come in jailbreak framings, so there is no direct-request comparison.
- **Pure vs mixed:** the comparison uses different prompt sets and is not paired. The optional extra
  of answering each mixed prompt's pure-language version would give a paired test.
- **Baseline figures:** the weak-spot rates in the Motivation come from the GGUF baseline run
  (Dataset2 labels), with pipeline 1's filters (analysis-eligible, completed, Safe/Unsafe verdicts only).
- **Quantization:** the GGUF-vs-bf16 contrast mixes quantization with other stack differences
  (llama.cpp, token limit, judge format).
- **Seed variation:** a separate follow-up found the published layer-18 adapter was a favourable
  training draw, with seeded retrains at 1.7–3.4% unsafe on clearly harmful prompts only (the earlier
  definition, not recomputed here: the retrain answers are not in this repo) and 22.5–25.0% over-refusal on pure
  English/Japanese. This run uses the published adapter only.
