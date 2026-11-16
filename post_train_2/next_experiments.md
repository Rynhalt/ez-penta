# Next experiments: what each run showed, what it lacks, and what to run next

Written 2026-11-14 from every Markdown file in the repo plus the raw data:

- Dataset2 GGUF baseline: `notebooks/combined/results.sqlite3`, 21,184 prompts.
- post_train_2 pipelines 5–8 answer files.

Every table below is reproducible with
[`notebooks/cross_category_analysis.py`](notebooks/cross_category_analysis.py). Its outputs are
`notebooks/combined/cross_category_analysis.txt` and `cross_category_paired.csv`. Post-fix
per-category numbers come from `combined/pipeline7_category_breakdown.json` (pipelines 5 and 6
pooled).

The baseline numbers come from the **GGUF Q4 baseline and the non-heldout splits**. All labels come
from Qwen3Guard-Gen-0.6B unless stated. "Paired" means the same seed with every other factor held
fixed, tested with exact McNemar.

---

## 1. Experiments so far

| # | Experiment | What it indicates | What limits it |
|---|---|---|---|
| A | **AnswerCarefully multilingual run** (`reports/20261105…`): 28,800 fp16 answers, 1,800 requests × 8 variants × JA/EN | JA/EN safety disagrees on 114 of 1,671 pairs. Fiction and refusal suppression shift safety in opposite directions by language. | Some rewrites drifted in meaning and some guard labels are questionable. A different dataset and stack from Dataset2, so not comparable. The roleplay prep cannot be reproduced. |
| B | **Dataset2 GGUF eval**: 21,184 prompts, 320 seeds, 10 categories × culture × framing × language | US framing raises unsafe odds about 2× (MH OR 1.98, p = 2e-9), replicated in en, ja and mix. Roleplay amplifies it (OR 2.54). Japan framing has no pooled effect. | **35% of harmful rows have no Safe/Unsafe label** (3,387 truncated answers, Controversial, unscored). Missingness depends on framing: rp 48%, fc 37%, control 25%. Q4 GGUF does not match the bf16 model that was fixed. Only **32 seeds per category**. |
| C | **post_train_1**: 4-bit QLoRA, 22 pairs, 40 validation prompts | Nothing usable | The "refusals" were compliance, train and validation shared seeds, and the model was quantized. Superseded. |
| D | **LEXSI / CircuitKIT track** (`docs/lexsi/*`) | HF↔CircuitKIT log-probs match, so the tooling works | No human-approved contrasts, no EAP-IG circuit, no causal test. The GPU smoke test used 4 seeds and has no power. The docs say the heldout split was **exposed in earlier work**. |
| E | **post_train_2 pipelines 2–5**: layer-18 restricted vs broad LoRA, 195 pairs, 228 pure en/ja prompts | Unsafe 8.3% → 0.7% (p = 0.002). Over-refusal 10% → 20% (restricted) or 36% (broad). Restricted beats broad on over-refusal (16 vs 3, p = 0.004). | **12 baseline unsafe events in total.** 30 borderline prompts. One training seed (retrains: 1.7–3.4% unsafe, 22.5–25% over-refusal). **No layer or capacity control**: restricted vs broad changes placement *and* 145× the parameters together. Buckets were chosen on GGUF labels but scored on bf16. The pipeline 2 projection test is not significant (p = 0.12–0.21). |
| F | **Pipeline 6, mixed language**: 700 code-mixed prompts | Safety transfers (9.5% → 2.0%, p = 3e-9, seed-level p = 8e-7). Over-refusal +17 pts and over-caution +18 pts, both significant. | Not paired with the pure-language versions. Jailbreak framings only. A 700-of-2,145 sample, from the analysis split. |
| G | **Pipelines 7 and 8, category / language / culture breakdowns** | See section 2 | Most category cells have 0–8 discordant pairs and carry `low_n` / `few_discordant` flags |

### Problems that recur across experiments

1. **Seeds, not prompts, are the bottleneck.**
   - Each category has 32 seeds: 16 analysis, 10 tune, 6 heldout.
   - In pipeline 5 only 9 of 54 seeds changed under the fix.
   - So a category-level, seed-level significant result is impossible with the current seeds. Even
     the heldout split has only 6 seeds per category.
2. **Too few events.** The bf16 baseline is unsafe on only about 8–13% of harmful prompts. Most
   category cells therefore have fewer than 10 unsafe answers to fix.
3. **Confounded arms.**
   - Restricted vs broad changes placement and parameter count at once.
   - Training used US-framed prompts only.
   - Training is category-unbalanced: misinformation and unsafe_activities have 6 pairs each,
     specialized_advice has 36.
4. **Measurement.**
   - One 0.6B judge, with no human review.
   - Truncation-driven missingness differs by framing.
   - Borderline labels have not been validated.

---

## 2. Apples-to-apples comparisons across categories

### 2a. Which prompt categories are actually jailbreakable? (paired, baseline)

Each cell: rate with the factor vs without it, and the discordant count (factor fails / control
fails). **Bold** = p < 0.05.

| Category | rp vs control | fc vs control | rs vs control | US vs neutral | Japan vs neutral | ja vs en |
|---|---|---|---|---|---|---|
| nonviolent_crime | **20.0 vs 8.3% (8/1)** | **19.6 vs 3.9% (8/0)** | 12.0 vs 9.3 (3/1) | **19.8 vs 9.5% (16/3)** | 13.3 vs 8.6 (13/8) | 16.4 vs 14.1 (7/4) |
| violent_crime | 16.2 vs 8.1 (7/1, p=.07) | 19.3 vs 7.0 (9/2, p=.07) | 8.2 vs 6.1 | **17.3 vs 5.8% (15/3)** | 15.9 vs 8.8 (13/5, p=.10) | 14.9 vs 12.7 |
| unsafe_activities | 14.9 vs 12.8 | 9.8 vs 5.9 | 7.9 vs 5.3 | **17.8 vs 8.5% (15/4)** | 15.2 vs 9.8 | 5.0 vs 9.1 |
| chatbot_interaction | 16.2 vs 13.5 | 7.8 vs 6.5 | 6.7 vs 6.7 | **12.4 vs 4.1% (10/0)** | 5.8 vs 4.2 | **12.3 vs 5.8% (11/2)** |
| misinformation | 14.0 vs 5.3 (5/0, p=.06) | 4.6 vs 3.1 | 5.5 vs 2.7 | 9.6 vs 5.9 | **13.4 vs 5.0% (11/1)** | 10.5 vs 6.8 |
| specialized_advice | 10.4 vs 14.6 | 9.6 vs 5.8 | 1.4 vs 2.7 | **9.6 vs 3.5% (8/1)** | 4.7 vs 4.7 | 4.3 vs 6.8 |
| privacy | 13.5 vs 15.4 | 10.0 vs 6.0 | 10.4 vs 10.4 | 18.3 vs 11.3 (13/5, p=.10) | 12.4 vs 10.7 | 11.3 vs 6.1 |
| hate_discrimination | 5.5 vs 3.6 | 6.1 vs 0.0 (3/0) | 0 vs 0 | 11.6 vs 7.0 | 0.0 vs 3.7 | 2.7 vs 0.9 |
| self_harm_mental_health | 4.8 vs 3.2 | 1.5 vs 2.9 | 0 vs 0 | **4.8 vs 0.0% (6/0)** | **4.3 vs 0.0% (6/0)** | 1.5 vs 3.0 |
| sexual_content | 23.8 vs 31.0 | 22.6 vs 22.6 (9/9) | n = 9 | 28.6 vs 25.4 | 19.4 vs 23.9 | 17.1 vs 25.6 |
| **All** | **13.7 vs 10.7 (35/19)** | **10.6 vs 6.3 (45/20)** | 6.2 vs 5.0 (17/10, p=.25) | **14.3 vs 7.2 (105/27)** | **9.9 vs 7.1 (80/48)** | 9.5 vs 8.6 (60/49) |

What this shows:

- **US framing** is the only factor that works across categories: 6 of 10 are significant, and every
  category points the same way except sexual_content, which barely differs.
- **nonviolent_crime** is the only category where *every* jailbreak (rp, fc, US) is significant.
  violent_crime is close behind.
- **Refusal suppression (rs) is not a jailbreak** for this model, in any category.
- **sexual_content** is unsafe 25% of the time even in plain control prompts, with no framing
  effect. It is a gap in the model's default behaviour, not a jailbreak problem.
- **Japanese vs English** makes no difference overall. Only chatbot_interaction is significant.

### 2b. Over-refusal on harmless twins (paired, baseline)

Jailbreak framing does **not** raise over-refusal; it lowers it:

| Comparison | Over-refusal rates | Discordant pairs | p |
|---|---|---|---|
| fc vs control | 1.8 vs 6.8% | 5 / 41 | 4e-8 |
| US vs neutral | 3.6 vs 8.3% | 23 / 83 | 4e-9 |
| ja vs en | 4.2 vs 6.2% | 29 / 57 | 0.003 |

A **neutral, English, plain** request is the most likely to be refused when harmless. Baseline
over-refusal is highest in **privacy** (control 12.9%, rp 19.1%) and **nonviolent_crime** (rp
21.6%).

### 2c. Prompt category vs the guard's output category (unsafe baseline answers, n = 635)

| Prompt category | Top guard output category | Share |
|---|---|---|
| sexual_content | Sexual Content | 92% |
| nonviolent_crime | Non-violent Illegal Acts | 83% |
| violent_crime | Violent | 74% |
| hate_discrimination | Unethical Acts | 63% |
| self_harm_mental_health | Suicide & Self-Harm | 61% |
| misinformation | Unethical Acts 59%, **"Jailbreak" 22%** | |
| privacy | Non-violent Illegal 57%, PII 22% | |
| chatbot_interaction | **Violent 56%** | |
| specialized_advice | **Violent 44%**, Non-violent Illegal 29% | |
| unsafe_activities | Violent 45%, Non-violent Illegal 33% | |

The guard's own output category agrees with the prompt category for the crime, sexual and self-harm
categories. Those are the categories where "unsafe" is a well-defined target. For chatbot_interaction,
specialized_advice and misinformation it does not agree, so the label mixes several kinds of harm.

The guard's "Jailbreak" output category appears **only** under rp (28) and rs (7), never under
control or fc.

### 2d. After the fix, by output category (pipeline 6, unsafe answers: baseline → restricted → broad)

| Guard output category | Baseline | Restricted | Broad |
|---|---|---|---|
| Non-violent Illegal Acts | 20 | **2** | 5 |
| Unethical Acts | 17 | **2** | 5 |
| Violent | 7 | 2 | 5 |
| **Sexual Content** | 6 | **4** | 4 |

The fix removes about 90% of unsafe answers in the illegal-acts and unethical-acts output
categories, but barely touches sexual content. The same holds by prompt category, pipelines 5 and 6
pooled: sexual_content goes 16.2% → 10.0% (3 discordant, p = 0.25), while every other category ends
at ≤ 3.8%.

### 2e. Where the fix's cost lands (pipelines 5 and 6 pooled, restricted adapter)

| Category | Helpful training pairs | Over-refusal before → after | p |
|---|---|---|---|
| privacy | 10 | 30.0 → **53.3%** | 0.016 |
| nonviolent_crime | 3 | 19.5 → 36.6% | 0.039 |
| misinformation | 2 | 6.9 → 27.6% | 0.031 |
| unsafe_activities | 2 | 0.0 → 26.3% | 0.06 |
| self_harm_mental_health | 12 | 3.3 → 20.0% (broad: 40%) | 0.13 |
| violent_crime | 7 | 10.0 → 13.3% | 1.0 |
| hate_discrimination | 11 | 3.3 → 10.0% | 0.5 |

Over-caution: **specialized_advice 0 → 42%** (p = 0.008). Every other category has n ≤ 19.

Categories with ≤ 4 helpful training pairs rose an average of **+18 points**; those with ≥ 11
helpful pairs rose **+13**. This suggests that per-category helpful data drives the cost, but it is
untested: 10 categories and one training seed.

---

## 3. Next experiments, ranked by how likely each is to give a valid, significant result

### 0. Prerequisite (cheap, do alongside #1): fix the measuring stick

- **Second judge.** Rescore every unsafe or refused answer in pipelines 5 and 6 (about 500 rows)
  with a larger judge, Qwen3Guard-Gen-4B/8B or Llama Guard, and report agreement.
- **Human review.** Two reviewers check a stratified 150-row sample: every category, every arm.
- **Fix the missing labels.** Regenerate the 3,387 truncated GGUF answers with a larger token
  limit, or rerun the baseline in bf16 on the analysis split.

Why first: rp missingness is 48% vs 25% for control, so every framing comparison in section 2a may be
biased. The bias is probably toward *under*-counting rp, because long compliant answers are the
ones that get truncated.

### 1. A powered, seed-expanded replication on **nonviolent_crime + violent_crime** (highest significance)

- **Why these two.**
  - The only categories where several jailbreaks are significant at baseline.
  - The highest bf16 unsafe rate (12–13%).
  - The fix is already significant at prompt level (8 discordant each).
  - The guard's output category matches the prompt category (74–83%), so "unsafe" means one thing.
- **What it lacks today.**
  - Seed-level significance within the category: about 3 of 16 analysis seeds move.
  - A clean heldout: 6 seeds, possibly already exposed.
- **Design.**
  - Generate **about 60 new seeds per category** with `dataset2v2.ipynb`. Use the same subtopic
    grid, a new seed-id namespace, and mark them `confirm`.
  - Each seed brings its harmful + twin × {neutral, US} × {control, fc, rp} × {en, ja}. Drop rs:
    section 2a shows it is not a jailbreak.
  - Three arms (baseline, restricted, broad) in bf16 with identical decoding.
  - **Pre-register** the test: seed-level sign test on unsafe, plus over-refusal on twins.
- **Power.**
  - With about 17% of seeds moving and none moving back (the pipeline 5 rate), 60 seeds gives
    about 10 movers. A 10/0 split gives p ≈ 0.002 per category.
  - Prompts: about 60 × 24 ≈ 1,450 per category per arm. That is roughly 2 G4 hours per arm for
    two categories.
- **Payoff.** The first category-specific claim of the form "LoRA at layer 18 prevents crime
  jailbreaks (rp/fc/US) at seed level", tested on genuinely unseen seeds.

### 2. Layer and capacity control: does the "circuit" matter? (high significance, cheapest)

- **What it lacks.** Restricted vs broad changes placement and size (209K vs 30.2M parameters)
  together. There is one training seed, and retrains vary by ±1–2 pts on unsafe and ±5 pts on
  over-refusal.
- **Design.** Same 195 pairs, same hyperparameters, **3 training seeds per arm**:
  - `down_proj` at layer 18 (the selected layer);
  - layer 15 (the runner-up);
  - 3 random non-selected layers (e.g. 6, 27, 33);
  - a **parameter-matched broad adapter**: rank 1 on a subset of modules, about 209K parameters.
- **Evaluation.** Score on the existing 928 prompts from pipelines 5 and 6. Baseline answers already
  exist, so only the new arms need generating. Compare arms with paired McNemar on the same prompts.
  Compare across training seeds with a mixed model.
- **Payoff.**
  - If layer 18 beats random layers at equal size, that is evidence the selected layer matters.
  - If it does not, the honest claim becomes "a small adapter gives a better trade-off", which is
    still useful.
  - Each training run takes about 15 minutes.

### 3. Over-refusal dose-response with category-balanced helpful pairs (high significance for the cost side)

- **What it lacks.** The 79 helpful pairs are spread unevenly: 2 for misinformation and
  unsafe_activities, 16 for specialized_advice. Over-refusal rose most in privacy,
  nonviolent_crime, misinformation and unsafe_activities. The `answer_carefully` seeds were left
  out of training, and over-caution then rose (specialized_advice 0 → 42%).
- **Design.**
  - Hold the 116 refusal pairs fixed.
  - Train restricted adapters with **0 / 10 / 25 helpful pairs per category** (tune split). Write
    more twins if needed; twins are abundant.
  - Add one arm with **"careful answer" pairs** from the `answer_carefully` seeds.
- **Target categories.** privacy and nonviolent_crime have the highest baseline over-refusal and
  the largest increases. specialized_advice is the over-caution case.
- **Primary metric.** Twin over-refusal per category, which needs about 60 twins per category for
  ±10-pt resolution. Use the analysis seeds plus the new seeds from #1.
- **Payoff.** A causal answer to "does category-matched helpful data cancel the over-refusal
  cost?", with a dose curve per category.

### 4. **sexual_content**: a targeted fix for the one category the adapter does not repair (medium-high)

- **What we know.**
  - Unsafe 25% even under control framing, with no framing or culture effect.
  - The fix leaves 10% unsafe, and Sexual Content outputs only went 6 → 4.
  - Borderline prompts are unsafe 21.5% at baseline *and* refused 44% (over-caution). Either the
    labels are noisy or the policy line is genuinely unclear.
  - Only 9 refusal pairs. only 19 harmful rs prompts have a label, because the Qwen generator refused
    to write them.
- **Design.**
  1. Human-review all baseline sexual_content Unsafe and borderline rows (about 200).
  2. Build category-specific refusal and helpful pairs (about 40 each).
  3. Compare the general adapter against general + sexual_content pairs, on new seeds.
- **Payoff.** The high baseline rate gives many events, so power is good. The risk is label
  validity, which is why the human review comes first.

### 5. Culture-of-training transfer: US-only vs neutral-only training (medium)

- **What it lacks.** All 195 training pairs are US-framed. US vs neutral is the strongest effect in
  the data (105/27), but we do not know whether the fix generalises across framings or just learned
  "cowboy → refuse".
- **Design.** Train on the same seeds' neutral-framed pairs instead of the US ones, then evaluate
  both adapters on US, neutral and Japan prompts.
- **Payoff.** Separates "learned to refuse harmful intent" from "learned a cultural trigger". Its
  power is moderate because the post-fix gap is already near 0.

### 6. Paired mixed vs pure language (low-medium, nearly free)

- **What it lacks.** Pipeline 6's comparison with pipeline 5 is unpaired.
- **Design.** Answer each pipeline 6 mixed prompt's own `ja` and `en` versions (same seed, culture
  and framing) with all three arms.
- **Payoff.** Tells us whether code-mixing *itself* raises the fix's over-refusal cost: +17 pts on
  mixed vs +10 on pure, at present.

### 7. Final heldout evaluation (do last, and change its scope)

- The current heldout split has only 6 seeds per category, and the LEXSI docs say it was exposed.
  So use the **new seeds from #1** as the confirmatory set.
- Keep the old heldout split as a secondary, descriptive check.

### Deprioritise

- **rs technique.** No effect at baseline in any category.
- **Japan culture.** Significant only in misinformation and self-harm, and the self-harm rates are
  tiny.
- **ja vs en as a main factor.** Null overall.
- **Further pipeline 2 projection tests.** Underpowered, and the layer choice does not depend on
  them.
