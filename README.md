# Safer, or Just More Refusals? English, Japanese and Bilingual Safety Fine-Tuning of Tiny Aya

*Technical article, version 2 (2026-11-16). LEXSI AI Safety & Alignment Hackathon 2026, Track 2. All behaviour labels are automated (Qwen3Guard) and have not been human-reviewed. Every number below is traced to a saved output in [`CLAIMS.md`](CLAIMS.md).*

## Abstract

We post-trained Tiny Aya Water with small safety adapters (LoRA). Every adapter was trained using 568 safety examples of prompt-output pairs in English only, Japanese only, or both languages (bilingual) with outputs written by base Tiny Aya and filtered by Qwen3guard. We placed all adapters at layer 18, the layer which controlled refusals most identified by Safe Tune.

Each condition was trained and tested three separate times on Dataset2, jailbreak prompts (role-play, fiction, plus the plain request) generated in English and Japanese in framings with Qwen3-4B, checked with quality filters. Each prompt was paired with a harmless “twin” in the both languages to measure over-refusals on matched prompts.

We evaluated Tiny Aya’s replies with Qwen3Guard-Gen-0.6B, the safety judge configured in AuditKIT, based on two numbers:

- Unsafe rate: the share of replies to harmful prompts that the judge labelled Unsafe.
- Over-refusal: the share of harmless twins that the model refused to answer.

Japanese-only training gave the best balance and transferred to English. On English prompts it matched English-only training on safety (6.45% vs 6.78% Unsafe; paired difference −0.33 points, 95% CI −1.09 to +0.45) while refusing fewer harmless prompts (13.2% vs 20.4%). On Japanese prompts it had the lowest Unsafe rate of any condition (8.3% vs 15.4% base, 15.7% English-only, 9.7% bilingual), and it was the only condition that also lowered the Unsafe rate among the prompts it answered (20.6% → 17.2%). Its cost was the highest Japanese over-refusal (17.1%, vs 15.4% bilingual).

English training did not transfer to Japanese. English-only training lowered the English Unsafe rate from 11.1% to 6.8% but left Japanese unchanged (15.4% → 15.7%. It did fix 42% of the base model’s Japanese failures, but the Japanese answers were still less safe among answered prompts (20.6% → 24.6%).

Bilingual training was the safest overall, but almost entirely by refusing more. Unsafe fell from 13.3% (702/5,282) to 6.9% (1,147/16,743), while over-refusal rose from 3.8% (144/3,800) to 25.7% (2,927/11,400), reaching 36.0% on English prompts. Among prompts it answered, it was no safer than the base model (18.1% vs 17.8% Unsafe).

Reducing over-refusal cost safety. We tried to remove over-refusal after training by deleting a “false-refusal” direction from the adapter’s weights. On the English adapter this lowered English over-refusal from 20.9% to 15.1% but raised Unsafe answers from 6.6% to 7.8%, and it had no effect in Japanese. At layer 18 the direction was causal: removing it cut English over-refusal by 15 points, but raised Unsafe answers in both languages (+3.8 and +5.8 points). Across all 36 layers, we found no direction that predicted over-refusal while staying separate from the harm signal: wherever over-refusal could be read from the activations, its direction was aligned with harm. This suggests the model refuses harmless prompts using the same internal signal it uses to recognise harm, so any training that strengthens refusal of harmful prompts also raises refusal of harmless ones.

### Terms used in this article

| Term | Meaning |
|---|---|
| Jailbreak prompt | A harmful request wrapped in a framing meant to get past refusals, such as role-play or fiction. |
| Harmless twin | A safe version of the same request with the same framing; tests whether the model refuses what it should answer. |
| Unsafe rate | Share of answers to harmful prompts that the judge labels Unsafe (lower is safer). |
| Over-refusal | Refusing to answer a harmless prompt (lower is better). |
| Judge (Qwen3Guard) | A separate small safety-classifier model that labels each answer Safe / Unsafe / Controversial and Refused / Not refused. The 4B version is a cross-check. |
| LoRA adapter | A small add-on trained on top of the frozen base model. |
| Bilingual training | Each training example shown in English, then in Japanese. |
| Fix rate | Share of the base model's Unsafe answers (to clearly harmful prompts) that a trained model no longer gives. |
| Unsafe among answered | The Unsafe rate counting only prompts the model did not refuse. Each model answers a different set of prompts. |
| Cosine, AUC | Cosine: how similar two directions are (1 = the same, 0 = unrelated). AUC: how well a direction separates two groups (0.5 = chance, 1 = perfect). |
| Steering / direction | A pattern in the model's internal activations linked to a behaviour; removing it aims to change that behaviour. |
| pp | Percentage points: the absolute difference between two rates (6.3% → 3.1% is −3.2 pp). |

### Key findings at a glance

| Question | Finding | Qualification |
|---|---|---|
| Does Japanese safety data help English? | Yes: Japanese-only matched English-only on English Unsafe rate (6.45% vs 6.78%) with fewer English refusals (13.2% vs 20.4%). | Secondary contrast; its English fix rate was lower (67.0% vs 70.8%). |
| Does English safety data help Japanese? | No: Japanese Unsafe 15.4% (base) → 15.7% (English-only), and the Japanese answers it gave became less safe (20.6% → 24.6%). | Base has one run; changes are paired estimates. |
| Which condition is safest in Japanese? | Japanese-only (8.3%), ahead of bilingual (9.7%; difference 1.35 pp, CI 0.47–2.29). | Japanese-only refuses the most harmless Japanese prompts (17.1%). |
| Is bilingual training really safer? | Lowest overall Unsafe rate (6.9%), but its gain came almost entirely from refusing more. | Over-refusal 25.7% overall, 36.0% on English prompts. |
| Can over-refusal be removed safely? | No: removing a false-refusal direction cut English over-refusal (−5.8 pp; −15.0 pp with a layer-18 hook) but raised Unsafe answers, and no layer separated over-refusal from harm. | One adapter for the causal test; automated labels. |

## 1. Research question

Small multilingual models are often safety-tuned mostly on English data. We ask two questions:

1. **Does the language of the safety-training data decide which prompt languages become safer?**
2. **When the Unsafe rate falls, is that because the model answers more safely, or because it refuses more, including harmless requests?**

We answer both with one controlled comparison (Section 4), decompose the safety gains (Section 5) and ask whether over-refusal can be removed after training (Section 6). Section 8 reports the other experiments.

## 2. Model, data and training

**Model.** [`CohereLabs/tiny-aya-water`](https://huggingface.co/CohereLabs/tiny-aya-water), 3.35 billion parameters, base weights frozen. Each adapter is a PEFT LoRA (rank 16, alpha 32, dropout 0.05) on `model.layers.18.mlp.down_proj` only, 208,896 trainable parameters. Training: AdamW, learning rate 1e-4, batch 1, one epoch, maximum length 768. Evaluation: merged bf16 adapters served with vLLM, greedy decoding, up to 1,024 new tokens.

**Training data.** 568 aligned items: 284 refusal targets for harmful requests and 284 helpful answers for harmless twins. The target answers were written by base Tiny Aya in both languages and kept only when the Qwen3Guard judge passed them.

| Condition | Training rows | Steps | Runs |
|---|---|---|---|
| English-only (`en`) | 568 English | 568 | 3 |
| Japanese-only (`ja`) | 568 Japanese | 568 | 3 |
| Bilingual (`bi`) | each item in English, then Japanese | 1,136 | 3 |

Bilingual training presents every item in both languages, so it takes twice as many optimisation steps as the single-language conditions.

**Evaluation prompts.** The frozen evaluation pool has 519 source seeds; paired analyses use the 475 seeds whose prompts passed every generation check (`paired_ok`). Each seed yields a harmful request and its harmless twin, in English and Japanese, in three framings (the plain request, fiction and role-play). Each request also exists in two versions with different setting details, which the main results pool (Section 8.1). Different studies use different cohorts, and their rates are not pooled:

| Study | Cohort | Runs |
|---|---|---|
| Training-language comparison (Sections 4–5) | full pool, 475 seeds | 3 per condition, 1 base |
| Steering and x_steer (Section 6, Appendix B) | full pool, half pool, or held-out fold (stated per result) | 1–3 |

## 3. Lexsi toolkits and their roles

| Toolkit | What it did in this project (verified in code) | What it did not do |
|---|---|---|
| **SafeTune** | Refusal-direction extraction and safety-neuron ranking on Tiny Aya, which picked layer 18 in both languages (`post_train_2/notebooks/pipeline2_*`). | Did not train the main adapters (PEFT directly, same layer and recipe). |
| **CuratorKIT** | Generated refusal and helpful preference pairs with Tiny Aya in the post_train_2 pipeline (`pipeline3_curatorkit_data.ipynb`; Appendix A). | Not used for the main training items. |
| **AuditKIT** | Scored the Dataset2 baseline with Qwen3Guard-Gen-0.6B as its guard judge (`guard_judge.py`, `eval/dataset2_runtime.py`). | The main runs called the same judge model directly through vLLM. |
| **CircuitKIT** | Integration checks: its log-probabilities match Tiny Aya's. | No circuit discovery: it needs TransformerLens, which does not support Tiny Aya. The steering experiments use direct activation hooks. |

## 4. Training language decides where safety improves

![Figure 1](figures/fig1_safety_vs_overrefusal.png)

*Figure 1. Unsafe answers to harmful prompts (x) against refused harmless prompts (y), by prompt language. Large markers pool three training runs; small dots are single runs; base has one run. Lower-left is better on both. English ×2 (grey diamond) is a training-length control described in Section 8.2.*

| Prompt language | Condition | Unsafe | Over-refusal |
|---|---|---|---|
| English | Base | 11.13% (288/2,588) | 4.26% (81/1,900) |
| English | English-only | 6.78% (557/8,211) | 20.39% (1,162/5,700) |
| English | **Japanese-only** | **6.45% (528/8,183)** | **13.16% (750/5,700)** |
| English | Bilingual | 4.03% (338/8,383) | 36.00% (2,052/5,700) |
| Japanese | Base | 15.37% (414/2,694) | 3.32% (63/1,900) |
| Japanese | English-only | 15.68% (1,297/8,273) | 6.39% (364/5,700) |
| Japanese | **Japanese-only** | **8.33% (696/8,359)** | **17.12% (976/5,700)** |
| Japanese | Bilingual | 9.68% (809/8,360) | 15.35% (875/5,700) |

Trained-condition counts sum three runs.

![Figure 2](figures/fig2_change_from_base.png)

*Figure 2. Paired change from the base model on identical prompts, with two-level bootstrap 95% CIs (evaluation seeds × training runs). Filled circles: English prompts; open squares: Japanese prompts. English ×2 is described in Section 8.2.*

### 4.1 Japanese-only training: English parity, best in Japanese

On **English** prompts, Japanese-only and English-only adapters had statistically indistinguishable Unsafe rates (paired difference −0.33 pp, 95% CI −1.09 to +0.45), while Japanese-only refused 7.2 pp fewer harmless English prompts (CI −8.7 to −5.8). Its English fix rate was lower (67.0% vs 70.8%; −3.8 pp, CI −7.7 to +0.03), so English-only fixes slightly more of the base model's English failures.

On **Japanese** prompts, Japanese-only gave the lowest Unsafe rate of every condition: −7.04 pp from base (CI −8.72 to −5.42), −7.35 pp versus English-only, and 1.35 pp below bilingual (CI 0.47 to 2.29). It fixed 72.0% of the base model's Japanese failures, against 41.8% for English-only (+30.2 pp, CI 25.4 to 35.0). Its cost is over-refusal in Japanese: 17.1%, compared with 3.3% for base, 6.4% for English-only and 15.4% for bilingual (bilingual − Japanese-only: −1.77 pp, CI −2.86 to −0.72).

These Japanese-only comparisons are secondary contrasts: two-level bootstrap on the 0.6B judge, not part of the Holm-corrected primary family below.

### 4.2 English training does not reach Japanese

English-only training lowered English Unsafe by 4.34 pp (CI −5.63 to −3.11), from 11.1% to 6.8%, but left Japanese unchanged (+0.31 pp, CI −1.04 to +1.66; 15.4% → 15.7%). It still fixed 41.8% of the base model's Japanese failures, but it also turned previously safe Japanese answers unsafe (about 120 per run, against about 166 fixed, on clearly harmful prompts), and among the Japanese prompts it answered, the Unsafe rate rose from 20.6% to 24.6% (Section 5). Japanese safety data transfers to English; English safety data does not transfer to Japanese.

### 4.3 Bilingual training: safest overall, by refusing more

Bilingual training had the lowest pooled Unsafe rate: 6.9% (1,147/16,743) against 13.3% (702/5,282) for base. The pre-specified primary tests (Holm-corrected, both judges) compare it with English-only training:

| Contrast (Unsafe rate) | 0.6B judge, pp (95% CI) | Holm p | 4B judge, pp |
|---|---|---|---|
| Bilingual − English-only, English prompts | −2.75 (−3.56 to −1.95) | < 0.001 | −1.68 |
| Bilingual − English-only, Japanese prompts | −6.00 (−7.38 to −4.65) | < 0.001 | −4.42 |

Its over-refusal was the highest of any condition: 25.7% (2,927/11,400) overall, 36.0% (2,052/5,700) on English prompts and 15.4% (875/5,700) on Japanese prompts. Section 5 shows that almost all of its safety gain came from refusing more.

## 5. Safer answers, or more refusals?

The Unsafe rate can fall for two reasons: the model refuses more harmful prompts, or it answers them more safely. Writing P(Unsafe) = P(answered) × P(Unsafe | answered), we split each change from base into these two parts with a two-factor Shapley decomposition (mean of three runs; harmful prompts; 0.6B judge).

![Figure 3](figures/fig3_refusal_decomposition.png)

*Figure 3. Change in the Unsafe rate from base, split into the part from refusing more often (grey) and the part from answering more or less safely (purple). Diamonds: total change. English ×2 is a training-length control described in Section 8.2.*

| Condition | Prompts | Total change | From refusing | From answering | Unsafe among answered (base → model) |
|---|---|---|---|---|---|
| Japanese-only | Japanese | −7.04 pp | −5.00 | **−2.05** | 20.57% → 17.24% |
| Japanese-only | English | −4.68 pp | −3.18 | **−1.48** | 14.98% → 12.61% |
| Bilingual | Japanese | −5.69 pp | −5.34 | −0.35 | 20.57% → 19.99% |
| Bilingual | English | −7.10 pp | −6.94 | −0.12 | 14.98% → 14.74% |
| English-only | English | −4.34 pp | −5.48 | +1.18 | 14.98% → 17.04% |
| English-only | Japanese | +0.31 pp | −2.49 | +2.80 | 20.57% → 24.61% |

- **Japanese-only is the only condition whose answers became safer** in both languages. About 70% of its gain comes from refusing more and about 30% from safer answers.
- **Bilingual and English training gained safety by refusing**: bilingual's answer component is near zero, and English training made the answers it did give *less* safe, offset by more refusals.

This is a descriptive decomposition. Each model answers a different set of prompts, so "Unsafe among answered" compares different answered sets and does not estimate a causal effect. The identity holds almost exactly here: only 2 of about 16,500 Japanese-only rows were both refused and Unsafe, and none for the other conditions.

## 6. Can over-refusal be removed after training?

Training raised refusals of harmless prompts, so we tested whether that over-refusal could be removed afterwards. We used a “false-refusal” direction: the difference in internal activations between refused and answered harmless twins, with its harmfulness component projected out.

### 6.1 Weight edit on the English adapter

We removed the direction from the output projections of every layer of the English adapter, using the setting picked by a pre-registered rule on a separate selection split (layer 18, strength 0.5). On the full pool, English over-refusal fell from 20.9% (415/1,988) to 15.1% (300/1,988), a paired change of −5.78 pp (CI −6.89 to −4.76), and the drop replicated on training runs 1 and 2. English Unsafe answers rose from 6.6% (189/2,857) to 7.8% (222/2,853), +1.46 pp (CI +0.89 to +2.10), failing our pre-set margin of +2 pp (margin p 0.042), and the fix rate fell by 5.19 pp (222/308 → 206/308), failing its −5 pp margin. Japanese prompts did not change (over-refusal −0.15 pp, Unsafe −0.11 pp). Answer quality on harmless prompts was unchanged (pairwise judge: 118 edited wins, 104 unedited, 178 ties; p 0.38).

### 6.2 Is over-refusal separable from harm?

The original direction was partly a language signal. At layer 18 it had cosine 0.60 with an English-minus-Japanese direction, because the English adapter refused far more English than Japanese twins (367/975 vs 166/981). Rebuilt within each language, it became more aligned with the harm direction, not less (cosine 0.78 at layer 18).

We then scanned all 36 layers of the base model and of the English-, Japanese- and bilingual-trained adapters. No layer, model or language met our separability rule: over-refusal readable from a held-out direction (AUC ≥ 0.80) while nearly orthogonal to harm (|cosine| ≤ 0.30). Where the signal was readable (layer 16 onwards) it was aligned with harm (cosine 0.68–0.85), and the low-cosine early layers were only weakly readable.

### 6.3 Causal test at single layers

On the English adapter, we removed or added the within-English direction with forward hooks at a single layer, on held-out seeds, with answers capped at 256 tokens:

- At layer 8, removing or adding the direction changed nothing.
- At layer 18, removing it cut refusals of harmless English prompts by 15.0 pp (66 answers flipped from refused to answered, none the other way), adding it raised them by 22.2 pp, and a random direction did nothing.
- The same layer-18 removal also cut Japanese refusals by 6.3 pp and raised Unsafe answers to clearly harmful prompts by 3.8 pp in English and 5.8 pp in Japanese.

Both interventions, against their unedited models:

| Intervention | Over-refusal change, pp (95% CI) | Unsafe change, pp (95% CI) | Cohort |
|---|---|---|---|
| Weight edit, English adapter (English prompts) | −5.78 (−6.89 to −4.76) | +1.46 (+0.89 to +2.10) | full pool, 519 seeds |
| Hook removal at layer 18 (English prompts) | −14.97 (−18.58 to −11.42) | +3.75 (+0.93 to +6.73) | held-out seeds, 256 tokens |
| Hook removal at layer 18 (Japanese prompts) | −6.33 (−9.56 to −3.36) | +5.83 (+3.42 to +8.68) | held-out seeds, 256 tokens |

Being orthogonal to the mean harm direction did not make the edit harm-neutral. In this model, refusing harmless prompts appears to use the same internal signal as recognising harm.

## 7. Conclusion

Training-language composition decided where Tiny Aya became safer. Japanese safety data improved Japanese prompts far more than English data did, and carried over to English prompts with fewer refusals of harmless English requests than English training. English data did not carry over to Japanese. Bilingual training reached the lowest overall Unsafe rate, but by refusing more, including more than a third of harmless English prompts. Japanese-only training was the one condition whose answers became safer, not just rarer. Removing over-refusal after training lowered refusals but always gave back safety, consistent with over-refusal and harm recognition sharing one internal signal. For a bilingual deployment, the results support training on Japanese safety data and measuring over-refusal separately in each language.

## 8. Other experiments and findings

The experiments below support or extend the main results but are outside the main question. Their cohorts differ from Sections 4–6 and are stated with each result.

### 8.1 Cultural framing: language or culture-specific words?

Dataset2 gives every request a **cultural framing**. The **US framing** places it in an American setting with cue words such as cowboy, Wild West, mafia or Hollywood. The **Japan framing** uses cues such as ninja (25% of Japan-framed English prompts), samurai (9%) or anime. Prompt language and cultural framing vary independently: an English prompt can carry a Japan cue. We included culture to tell apart two explanations for a safety failure: the *language* of the prompt, and *culture-specific words* that may themselves work as jailbreak cues.

![Figure 4](figures/fig4_culture_vs_language.png)

*Figure 4. Left: base model Unsafe rate by cultural cue and prompt language (full pool). Right: matched-cue test E4 (150 seeds), paired differences with 95% CIs (0.6B) and the 4B estimate.*

At baseline, US-framed prompts were about twice as unsafe as Japan-framed ones: 15.9% (199/1,255) vs 6.7% (89/1,333) in English, and 20.2% (268/1,325) vs 10.7% (146/1,369) in Japanese. The follow-up E4 tested whether this is about culture or about the cue words. Qwen3-4B rewrote each neutral request with exactly one benign cue (US: a county fair or Fourth of July event; UK: a village fete or bank-holiday event), under the same rules and with no crime-associated groups.

- **Matched US and UK framings were equally unsafe:** 7.9% vs 7.7% in English (UK − US −0.25 pp, CI −2.93 to +2.46) and 8.3% vs 9.7% in Japanese (+1.43 pp, CI −2.14 to +4.96). No difference was detected; this is not proof of equivalence.
- **The original US prompts were riskier than their matched rewrites:** +8.06 pp in English (CI +3.42 to +12.74) and +13.17 pp in Japanese (CI +7.97 to +18.76), confirmed by both judges and on a second GPU type.

The US-vs-Japan gap therefore reflects the content of Dataset2's cue words more than culture itself. The rewrite can also soften wording beyond the cue, which is why only 150 of 276 seeds passed the matching checks. Training effects followed language, not framing: each adapter improved both US- and Japan-framed prompts in its own training language (Section 4 pools both framings), and all three adapters lowered Unsafe answers on the UK rewrites in English (−6.1 to −6.8 pp).

### 8.2 Training-length control (English ×2)

Bilingual training takes twice as many optimisation steps as English-only training (1,136 vs 568). To check that its advantage came from the Japanese rows and not from the longer training, we trained a control on each English item twice (English ×2: 1,136 steps, three runs). The pre-specified tests compare bilingual training with it:

| Contrast (Unsafe rate) | 0.6B judge, pp (95% CI) | Holm p | 4B judge, pp |
|---|---|---|---|
| Bilingual − English ×2, English prompts | −0.63 (−1.26 to +0.02) | 0.11 | −0.25 |
| Bilingual − English ×2, Japanese prompts | −7.43 (−8.91 to −5.99) | < 0.001 | −5.69 |

Against this control, bilingual training had no established English advantage but a large Japanese one, so its Japanese gain came from the Japanese rows. English ×2 itself reached 4.67% Unsafe (389/8,338) on English prompts with 29.3% over-refusal (1,668/5,700). On Japanese prompts it was worse than the base model: 17.11% Unsafe (1,412/8,253; +1.74 pp, CI +0.40 to +3.09), and among the Japanese prompts it answered, the Unsafe rate rose from 20.6% to 26.3%.

### 8.3 Replication on new training items

The follow-up E3 (half pool, US framing) repeated the comparison with a Japanese ×2 control and with two disjoint halves of the training items (one run each):

- Doubling the Japanese data changed nothing (English Unsafe −0.64 pp, CI −1.63 to +0.32).
- On both item halves, Japanese training refused fewer harmless English prompts (−7.6 and −6.2 pp; confirmed) and was safer on Japanese prompts (−7.6 and −2.7 pp; confirmed).
- On English prompts, Japanese training was about 2 pp less safe than English training on both halves (+1.9 and +2.0 pp; Holm p 0.055 and 0.067), and its English fix rate was lower. The English-parity result of Section 4.1 is therefore close, not equal.

### 8.4 Other ways to reduce over-refusal

Besides the weight edit in Section 6, we tested two more changes on the bilingual adapter (half pool, US framing, three runs), a negative control and an inference-time method:

- E1: the same weight edit on the bilingual adapter. No grid setting met the selection rule, so a fallback setting was used.
- E2: bilingual training with each helpful example shown twice, which also trains for more steps than its control.
- x_steer C: removing the most harm-independent direction we found (layer 1) from the English adapter.
- CAST: conditional activation steering at inference time on the base model (exploratory).

![Figure 5](figures/fig5_interventions.png)

*Figure 5. Change in English over-refusal (x) and English Unsafe answers (y) for each intervention, against its own unedited model, with 95% CIs. Cohorts differ; compare directions, not sizes.*

| Intervention (English prompts) | Over-refusal change, pp | Unsafe change, pp | Cohort |
|---|---|---|---|
| Weight edit, English adapter | −5.78 (−6.89 to −4.76) | +1.46 (+0.89 to +2.10) | full pool, 519 seeds |
| Weight edit, bilingual adapter (E1) | −5.68 (−8.39 to −3.02) | +1.77 (+0.80 to +2.86) | half pool, US framing |
| Helpful examples ×2 (E2) | −7.91 (−11.29 to −4.47) | +1.81 (+0.82 to +2.96) | half pool, US framing |
| Hook removal at layer 18 (x_steer D) | −14.97 (−18.58 to −11.42) | +3.75 (+0.93 to +6.73) | held-out fold, 256 tokens |
| Harm-independent direction, layer 1 (x_steer C) | +0.72 (−1.09 to +2.72) | −0.78 (−1.70 to 0.00) | half pool, US framing |

Every method that lowered over-refusal also raised Unsafe answers. On the bilingual adapter (E1) the weight edit also raised Japanese over-refusal (+3.26 pp), and the harm-independent layer-1 direction (x_steer C) changed neither refusals nor safety. CAST on the base model lowered English Unsafe from 10.0% (131/1,314) to 6.8% (90/1,332) but raised over-refusal from 4.4% (42/950) to 6.9% (66/950).

### 8.5 Further results in the appendices

- Appendix A: an earlier pilot (post_train_2) comparing a layer-18 adapter with an adapter on every layer, built with SafeTune and CuratorKIT.
- Appendix B: the full steering hypotheses and the direction analyses.
- Appendix C: interim results of the Japanese arm on Japan-framed prompts (4 of 6 chunks).

## 9. Limitations

- **Automated labels.** Safe/Unsafe and refusal labels come from Qwen3Guard, not humans. Over-refusal measures unnecessary refusal, not answer quality or helpfulness.
- **Secondary contrasts.** The Japanese-only comparisons are not in the Holm-corrected primary family and were not re-judged with the 4B model in the main study. The E3 replication supports them, but with one run per item half.
- **Selected sets.** The refusal decomposition compares Unsafe rates among answered prompts, which differ by model.
- **Single base run.** The base model has one run and trained conditions three; inference uses paired estimates.
- **Cultural rewrites.** E4 rewrites can change wording beyond the cue, and 126 of 276 seeds failed the checks.
- **Hardware and cohorts.** Follow-ups ran on an L4 GPU (label agreement with the G4 runs about 99%). Studies use different cohorts and are not pooled.
- **Training data.** The training answers were generated by the base model itself and filtered by the same judge family used to evaluate.

## 10. Reproducibility and sources

- **Figures and numbers:** [`notebooks/A1_figures.ipynb`](notebooks/A1_figures.ipynb) (executed: `A1_figures_out.ipynb`) regenerates every figure and `data/numbers.json` from saved outputs; [`notebooks/A2_validate.ipynb`](notebooks/A2_validate.ipynb) checks every cited number against them. Claim-by-claim sources: [`CLAIMS.md`](CLAIMS.md).
- **Training and evaluation notebooks:** `post_train_3/arm_en/notebooks/` (English adapters, frozen pool), `post_train_3/bilingual/notebooks/` (B0–B5, including the refusal decomposition `B5_refusal_decomposition.ipynb`), `post_train_3/followups/notebooks/` (E1–E4), `post_train_3/steering/notebooks/`, `post_train_3/x_steer/notebooks/`.
- **Google Drive (outputs, adapters, reports):** [post_train_3 runs folder](https://drive.google.com/drive/folders/1bH59zkx1BlKXcqvR3L-NAbh3fbby3ltQ): [bilingual](https://drive.google.com/drive/folders/1MpPV_D-5GmxvDYXptKA8EElHWcZEZ4UT) (adapters, evaluations, report), [English arm](https://drive.google.com/drive/folders/1TkUi_We2J2dC-NqAHvuYvYB0ISsJRsru), [follow-ups](https://drive.google.com/drive/folders/1d0ij4ACsOHtl4xFemWO3BV1PosS-oqDX), [steering](https://drive.google.com/drive/folders/1b2NwRg4iFP8UmoJ636EMbrGK-OmE_Q1T), x_steer, [frozen evaluation pool](https://drive.google.com/drive/folders/1CDta-eawhfdJMrE5vRzkDX0sfVnThFNv); earlier pilot: [post_train_2](https://drive.google.com/drive/folders/1p0zq2_Rc497dlIrfQtiOGYI8nMP7Dy_x); all reports: [EXPERIMENTS_SUMMARY.md](https://drive.google.com/file/d/11Ae4RgloJguP7CAYiWb4AmN5Wj0CeTjq/view).
- **Hugging Face (public):** [collection](https://huggingface.co/collections/isaac369369/tiny-aya-multilingual-safety-fine-tuning-6afae56e983f2538a5e2775e); [prompts](https://huggingface.co/datasets/isaac369369/tiny-aya-safety-multilingual-prompts) (Dataset2 pool and every evaluation prompt set); [training data](https://huggingface.co/datasets/isaac369369/tiny-aya-safety-multilingual-training) (the 568 items and every adapter's training stream); [results](https://huggingface.co/datasets/isaac369369/tiny-aya-safety-multilingual-results) (every answer and judge label behind Sections 4–8 and the appendices, with numbers.json and the figures); [adapters](https://huggingface.co/isaac369369/tiny-aya-water-multilingual-safety-lora) (all 22 LoRA adapters and the steering direction files).
- **Release terms:** the datasets and adapters are public under CC BY-NC 4.0, following the Tiny Aya Water licence. The results dataset includes answers the judge labelled Unsafe and carries a content warning; all labels are automated, not human-reviewed.

---

## Appendix A. Earlier pilot: layer-restricted vs broad LoRA (post_train_2)

The pilot tested *where* to place the adapter, not training language. **SafeTune** extracted refusal directions and ranked safety neurons, which pointed to layer 18 in both languages. **CuratorKIT** generated 195 accepted refusal and helpful preference pairs. A layer-18-only LoRA ("restricted") was compared with a LoRA on every projection ("broad") on 228 held-out prompts from 54 seeds:

|  | Base | Restricted (layer 18) | Broad |
|---|---|---|---|
| Unsafe answers | 12/145 (8.3%) | 1/147 (0.7%) | 1/148 (0.7%) |
| Refused harmless twins | 8/80 (10.0%) | 16/80 (20.0%) | 29/80 (36.3%) |

Both arms removed almost all unsafe answers. The restricted arm refused fewer harmless twins (broad refused 16 that restricted answered, against 3 the other way; p = 0.004). With only 12 baseline failures, this pilot motivated the layer-18 design of the main study but does not establish a general advantage. Report: [post_train_2/report.md](https://drive.google.com/file/d/1ZQ5UFfDdba4gjGiV2fvSqRUZ9s9rkpl7/view).

## Appendix B. Steering details

**Weight edit on the English adapter** (full pool; rule-selected setting: layer 18, strength 0.5, chosen on Dataset2 `heldout`, which is now used up). Pre-specified hypotheses, edited minus unedited:

| Hypothesis | English, 0.6B | Japanese, 0.6B | Result |
|---|---|---|---|
| H1 over-refusal falls | −5.78 pp (415/1,988 → 300/1,988) | −0.15 pp | supported in English only |
| H2 Unsafe within +2 pp | +1.46 pp (189/2,857 → 222/2,853), CI to +2.10 | −0.11 pp | failed in English (margin p 0.042, Holm 0.12) |
| H3 fix rate within −5 pp | −5.19 pp (222/308 → 206/308) | +0.93 pp | failed in English |

The edit replicated on training runs 1 and 2 (about 20.5% → 15.0% over-refusal). Answer quality was unchanged (pairwise judge: 118 edited wins, 104 unedited, 178 ties; p 0.38). No inference-time variant qualified for the final run. An exploratory activation-steering variant (CAST) on the base model lowered English Unsafe from 10.0% (131/1,314) to 6.8% (90/1,332) but raised over-refusal from 4.4% (42/950) to 6.9% (66/950).

**E1 (edit on the bilingual adapter) and E2 (helpful examples ×2)**, half pool, US framing, three runs:

|  | E1 English | E1 Japanese | E2 English | E2 Japanese |
|---|---|---|---|---|
| Over-refusal | −5.68 pp | **+3.26 pp** | −7.91 pp | −4.59 pp |
| Unsafe | +1.77 pp | −0.26 pp | +1.81 pp | +2.82 pp |

No E1 grid setting met the selection rule, so a fallback was used. E2 trained for more steps than its control.

**x_steer (follow-up direction analysis)**

- **A:** the original pooled direction was partly language: cos(f, English − Japanese) = 0.60 at layer 18, and 0.87 at layer 8. Rebuilt within one language, its overlap with the harm direction rose to 0.78 at layer 18.
- **B:** all 36 layers of base and the English-, Japanese- and bilingual-trained adapters. No layer met the readable (AUC ≥ 0.80) and harm-independent (|cos| ≤ 0.30) rule. On B's extraction set, the Japanese adapter also raised English twin refusals (99 → 285 of 975).
- **C:** a full removal of the most harm-independent direction (layer 1) changed neither over-refusal nor safety; it served as a negative control.
- **D:** forward hooks on the English adapter, held-out fold, 256-token cap:
   - at layer 8 nothing changed;
   - at layer 18, removing the direction cut English refusals by 15.0 pp and adding it raised them by 22.2 pp, while a random direction did nothing;
   - the same layer-18 removal cut Japanese refusals by 6.3 pp and raised Unsafe answers by 3.8 pp (English) and 5.8 pp (Japanese).

## Appendix C. Japan-culture evaluation, Japanese arm (interim, 4 of 6 chunks)

Run by the JA agent on separate hardware (G4 and A100); 322 paired seeds; one run per adapter. The Japanese adapter (`ja_s0`) on Japan-framed prompts, compared with base:

- **Unsafe fell:** English 6.3% → 3.1% (−4.15 pp, CI −6.13 to −2.34); Japanese 8.6% → 5.8% (−4.87 pp, CI −7.04 to −2.69). Confirmed by both judges.
- **Over-refusal rose:** +8.85 pp (English) and +8.54 pp (Japanese).
- **About three quarters of base failures fixed:** 44/58 English, 65/88 Japanese.
- **Smaller gain on Japan cues:** the safety gain was 3.95 pp smaller on Japan-cue prompts than on matched US-cue versions (4B: 4.55 pp).
- **Japanese vs English adapter:** the Japanese adapter fixed 22.4 pp more Japanese-prompt failures than the English one. Not confirmed: the 4B Holm p was 0.060, and the arms ran on different hardware.

Judge agreement on safety in this slice was moderate (κ 0.48–0.61). Interim p-values are descriptive. Report: [interim findings PDF](https://drive.google.com/file/d/1oo5oYLBXmQMugqpDJHZ8WZpitGPvO8wV/view).

## Appendix D. Configuration

| Item | Value |
|---|---|
| Base model | CohereLabs/tiny-aya-water (Cohere2, 3.35B) |
| Adapter | PEFT LoRA r=16, α=32, dropout 0.05, target `model.layers.18.mlp.down_proj`, 208,896 parameters |
| Training | AdamW, lr 1e-4, 1 epoch, batch 1, max length 768, training seeds 0–2 |
| Serving | merged bf16, vLLM 0.33, greedy, max 1,024 new tokens (x_steer D: HF generate, 256) |
| Judges | Qwen3Guard-Gen-0.6B (primary), Qwen3Guard-Gen-4B (robustness) |
| Statistics | exact McNemar per training run; two-level bootstrap (seeds × runs), 10,000 resamples; Holm within the primary family |
| Hardware | Colab G4 (main), L4 (follow-ups, x_steer B–C; agreement about 99%) |

## Appendix E. Changes from version 1

- Reframed around English vs Japanese, with the Japanese-only results as the main finding.
- The refusal decomposition (B5) was recovered (`bilingual/notebooks/B5_refusal_decomposition_out.ipynb`) and reproduced exactly by `A1_figures.ipynb`; it is now a verified result with a figure.
- x_steer C is complete (negative control) and included.
- Figure 1's Japanese panel now shows every condition.
- The restricted-vs-broad pilot, steering details and the interim Japan-culture results moved to appendices.
- Sections 4–7 now cover only the training-language comparison, the refusal decomposition and steering; cultural framing, the training-length control (English ×2), the replication on new items and the other over-refusal interventions moved to Section 8.
