# ez-penta: every experiment, with expected and unexpected findings

The model is Tiny Aya Water throughout. Unless stated, unsafe and refusal labels come from Qwen3Guard-Gen-0.6B. Results are
automated analyses and still need human review. Each experiment links to its report on Google Drive.

---

## post_train_1: first circuit-restricted fix (4-bit, small)
Report: [post_train_1/report.md](https://drive.google.com/file/d/1uXmH_h5ePAr2aDutwacR0ERD1kg-hCPk/view)

**Expected**
- US cultural framing roughly doubles the odds of an unsafe answer, even after controlling for language (Mantel–Haenszel OR 1.98, p < 1e-6). The effect is stronger under role-play.
- Within each language, US framing pushes activations away from the refusal direction.

**Unexpected**
- Japan framing has no effect (OR 1.10, p 0.49).
- English and Japanese both pick layer 18, but the refusal *direction* doesn't transfer between languages.
- Validation was inconclusive: 39 of 40 baseline answers were already Safe.
- The training "refusals" turned out to be compliance, training and validation shared seeds, and the model was 4-bit. All three were fixed in post_train_2.

## post_train_2, pipelines 2–5: layer-18 restricted LoRA vs broad LoRA (bf16, seed-disjoint)
Reports: [report.md](https://drive.google.com/file/d/1ZQ5UFfDdba4gjGiV2fvSqRUZ9s9rkpl7/view) · [pipeline flow](https://drive.google.com/file/d/1Q48sRG5dy8CgiQT9aTKxzTjDjRQatCEl/view)

**Expected**
- Both fixes remove almost all unsafe answers (8.3% → 0.7%, p = 0.002) and close the US-vs-neutral gap.
- The restricted fix over-refuses much less than the broad one (20% vs 36%; 16 vs 3 discordant, p = 0.004).
- Adding helpful training pairs cut the cost: over-refusal went from 10% → 46% in the refusal-only iteration to 10% → 20%.

**Unexpected**
- Restricting the fix to one layer bought no extra safety; it only reduced the cost.
- Over-refusal doubled even for the restricted fix, and grew more in Japanese (13.9% → 27.8%) than in English (6.8% → 13.6%).
- Harmless answers got about a third shorter (restricted) or three quarters shorter (broad).
- Only 12 baseline unsafe events existed to fix, so the result rests on small counts.

## post_train_2, pipeline 6: code-mixed prompts (framing in one language, request in the other)
Report: [pipeline6_mixed_language_report.md](https://drive.google.com/file/d/1iK1g1KLTJrEAnpqPaeWdPXrGXytlrn09/view)

**Expected**
- The safety fix carries over to mixed prompts it never saw: 9.5% → 2.0% unsafe (p = 3e-9).
- The restricted adapter is still the better trade-off.

**Unexpected**
- The cost is larger than on pure languages: over-refusal 11% → 28%, over-caution 13% → 31%. Most of it falls on persona-wrapped role-play prompts.
- The bf16 model is less than half as unsafe as the quantized GGUF model on the same prompts (9.5% vs 21.4%).
- 9 of the 10 remaining unsafe answers are fiction-framed.

## post_train_2, pipelines 7–9 and cross-category analysis (re-analysis, no new generation)
Report: [next_experiments.md](https://drive.google.com/file/d/1UQrubMQkT1Jc4kT0o2_xPgP98WFuJ4H4/view)

**Expected**
- US framing is the one jailbreak factor that works across categories: 6 of 10 categories are significant, and pooled it raises unsafe from 7.2% to 14.3%.
- The fix lowers unsafe in nonviolent and violent crime (pooled p < 0.05).

**Unexpected**
- Refusal suppression is not a jailbreak for this model in any category.
- sexual_content is unsafe about 25% of the time even in plain prompts, and the fix barely touches it (16% → 10%).
- Jailbreak framing *lowers* over-refusal on harmless twins: plain, neutral, English requests are refused most.
- Over-refusal cost concentrates in privacy (30% → 53%) and nonviolent crime.
- The broad adapter's over-refusal is far worse in English than in Japanese (difference about 28–31 points, p < 0.01). The restricted adapter shows no language difference.

## post_train_3, English arm: English-trained adapter on the frozen pool (519 seeds, 3 training seeds)
Report: [arm_en/report.md](https://drive.google.com/file/d/1OlO0XXUFe5N219d0GDQW34qT-6HogCVH/view) · [detail breakdown](https://drive.google.com/file/d/1oT-TI3HottAm6fYCwYdpLjosQfh45tbC/view)

**Expected**
- English unsafe 15.7% → 9.1%, and about 70% of English base failures fixed, consistently across 3 training seeds.

**Unexpected**
- No safety gain on Japanese prompts (20.0% → 20.0%), even though about 39% of Japanese base failures are fixed. Many previously safe answers break (about 120 broken vs 166 fixed per run), so it's mostly churn, not improvement.
- English over-refusal rises about 18 points (about 38 under role-play).

## post_train_3, Japanese-culture evaluation (EN arm): same seeds re-framed with Japanese cultural cues
Report: [japan_culture/en/report.md](https://drive.google.com/file/d/1YxZsNT8nM8SHuycZ3rBfAsHmmwFdouX0/view)

**Expected**
- The English adapter also improves Japan-framed English prompts (unsafe 6.7% → 4.0%; fixes 79% of base failures).

**Unexpected**
- Base is much *safer* on Japan-framed prompts than on US-framed ones (English 6.7% vs 16.5%; Japanese 10.7% vs 20.2%).
- On Japanese-language prompts the English adapter still doesn't help (10.7% → 10.8%).

## post_train_3, Japanese arm on Japan-culture prompts (run by the JA agent on another machine; interim, 4 of 6 chunks: 1, 3, 5, 7; 322 paired seeds)
Reports: [interim findings (PDF, 4 of 6 chunks)](https://drive.google.com/file/d/1oo5oYLBXmQMugqpDJHZ8WZpitGPvO8wV/view) · [japan_culture/ja/report.md](https://drive.google.com/file/d/1YbvR3KrA67XGk-5vjt5Sr5aotwe_G2x3/view) · [japan_culture/combined/report.md](https://drive.google.com/file/d/1nLlMeo1vACff4zB1W1wSagj9o50lXoNE/view)

**Expected**
- **J2:** the Japanese-trained adapter (`ja_s0`) lowers unsafe on Japan-culture prompts: English 6.3% → 3.1% (−4.2 points), Japanese 8.6% → 5.8% (−4.9 points). Confirmed by both judges.
- **J2:** it fixes about three quarters of base failures (76% English, 74% Japanese).
- **J4, harmless-twin refusals:** each adapter refuses more in its own training language. `ja_s0` refuses more on Japanese twins, `en_s0` more on English twins.

**Unexpected**
- **J2:** the cost is over-refusal of about +9 points in both languages (English 3.8% → 12.4%, Japanese 3.4% → 11.6%; confirmed).
- **J3:** the safety gain is about 4 points *smaller* on Japan-cue prompts than on the matched US-cue versions (+3.95 points with the 0.6B judge, +4.55 with the 4B; confirmed). This is about the adapter's effect, not proof that Japanese culture is riskier.
- **J4, fix rate on Japanese prompts:** `ja_s0` fixes 22 points more Japanese-prompt base failures than `en_s0`. This is suggestive but not confirmed: the 4B judge's Holm-adjusted p is 0.060, and the runs used different hardware (G4 vs A100).
- **J5:** no special "Japanese culture × Japanese language" effect. A secondary 4B odds-ratio interval disagrees with the prespecified test, so J5 stays inconclusive.
- Judge agreement on safety is only moderate here (κ 0.48–0.61). p-values stay descriptive until chunks 9 and 11 are in.

## post_train_3, steering: remove the over-refusal direction from the English adapter
Report: [steering/report.md](https://drive.google.com/file/d/1BRjUfQNt5jwAohKtxNherr3qM4UMxi6S/view)

**Expected**
- English over-refusal 20.9% → 15.1% (confirmed by both judges), replicated on training seeds 1 and 2.
- Answer quality unchanged (pairwise 118 vs 104 wins, p 0.38).

**Unexpected**
- It costs safety: English unsafe 6.6% → 7.8%, fix rate 72% → 67%. Both miss the pre-set margins.
- The "refuse harmless" and "harmful" directions overlap heavily (cosine about 0.6 at layer 18).
- No effect at all on Japanese prompts.
- No inference-time steering variant qualified. On base, the activation-steering variant (CAST) *raised* over-refusal.

## post_train_3, bilingual training: en / ja / both / English twice, same 568 items, 3 training seeds each
Report: [bilingual/report.md](https://drive.google.com/file/d/1i7pmZDYK6Nav93ho_BchRwzKwfU_7Rm6/view)

**Expected**
- Training on both languages is much safer on Japanese prompts (9.7% vs 15.7% unsafe). This isn't the extra dose: English twice gives 17.1%.
- Each training language mainly improves that language.

**Unexpected**
- `bi` is also safer on English (4.0% vs 6.8%) but over-refuses far more (39.8% vs 23.0%).
- Japanese-only training nearly matches English training on English prompts (net fixes +139 vs +149) and over-refuses less there.
- Doubling the English data made Japanese *worse* (net fixes +17 vs +46).
- Over-refusal rises in whichever language you train on.

## post_train_3, follow-ups (one unattended run: E1–E4 plus a hardware check; new models on L4)
Report: [followups/report.md](https://drive.google.com/file/d/1BJO1o68QKZ652Ke3_oYzHzWgTYwdeQsZ/view) · [overview](https://drive.google.com/file/d/1XItP7AHRnyOsz7kHX2AkSC2kLiYAbfdS/view)

**Hardware check (L4 vs G4)**
- *Expected:* L4 answers match the stored G4 labels about 99% of the time, with no significant drift.
- *Unexpected:* nothing.

**E1: steering edit on the bilingual adapter**
- *Expected:* English over-refusal 39.9% → 34.2% (confirmed).
- *Unexpected:* no grid setting met the selection rule (fallback used); unsafe rose 3.7% → 5.4%; Japanese over-refusal went *up* (17.6% → 20.9%).

**E2: helpful examples shown twice**
- *Expected:* over-refusal fell in every cell (English 39.9% → 31.9%).
- *Unexpected:* unsafe rose about as much as with steering (English 4.3% → 6.1%, Japanese 8.8% → 11.6%).

**E3: Japanese-only replication (dose control and two independent item halves)**
- *Expected:* doubling the Japanese data changes nothing, and the en-vs-ja pattern replicates on both item halves.
- *Unexpected:* on English prompts, Japanese training is less safe than dose-matched English training (7.3% vs 3.9%).

**E4: US vs UK culture with matched cues**
- *Expected:* adapters trained on US prompts also cut unsafe on UK prompts in English (8.9% → 2.1%).
- *Unexpected:* with matched benign cues, US ≈ UK (7.7% vs 7.9%). The *original* US prompts are about twice as risky as their matched rewrites (16.1% vs 8.1% in English; 20.5% vs 7.3% in Japanese). So the "US culture is more dangerous" finding comes from the content of the cues, not the culture itself.

---

## All markdown files on Drive

| Experiment | Main report | Other notes |
|---|---|---|
| post_train_1 | [report.md](https://drive.google.com/file/d/1uXmH_h5ePAr2aDutwacR0ERD1kg-hCPk/view) | – |
| post_train_2 | [report.md](https://drive.google.com/file/d/1ZQ5UFfDdba4gjGiV2fvSqRUZ9s9rkpl7/view) | [post_train_2_pipeline.md](https://drive.google.com/file/d/1Q48sRG5dy8CgiQT9aTKxzTjDjRQatCEl/view), [pipeline6_mixed_language_report.md](https://drive.google.com/file/d/1iK1g1KLTJrEAnpqPaeWdPXrGXytlrn09/view), [next_experiments.md](https://drive.google.com/file/d/1UQrubMQkT1Jc4kT0o2_xPgP98WFuJ4H4/view) |
| post_train_3 English arm | [report.md](https://drive.google.com/file/d/1OlO0XXUFe5N219d0GDQW34qT-6HogCVH/view) | [detail_breakdown.md](https://drive.google.com/file/d/1oT-TI3HottAm6fYCwYdpLjosQfh45tbC/view) |
| Japanese culture (EN arm) | [report.md](https://drive.google.com/file/d/1YxZsNT8nM8SHuycZ3rBfAsHmmwFdouX0/view) | [ERROR_LOG.md](https://drive.google.com/file/d/12CFh3VMP9bJ7uaHSMbo-OdvvoSbPdMlB/view), [DEVIATIONS.md](https://drive.google.com/file/d/1g0umk7dV5nkK5W_tSMJP_zGsdU-v-kvP/view) |
| Japanese culture (JA arm, combined; interim) | [interim findings PDF (4/6 chunks)](https://drive.google.com/file/d/1oo5oYLBXmQMugqpDJHZ8WZpitGPvO8wV/view), [ja/report.md](https://drive.google.com/file/d/1YbvR3KrA67XGk-5vjt5Sr5aotwe_G2x3/view), [combined/report.md](https://drive.google.com/file/d/1nLlMeo1vACff4zB1W1wSagj9o50lXoNE/view) | [ERROR_LOG.md](https://drive.google.com/file/d/1kwALYEY2bNpFOVfGfXBqbmfsrbQ8q3Xa/view), [DEVIATIONS.md](https://drive.google.com/file/d/1DH3IlsrSW4MV3ml2sA7M6EOU2qGnFttn/view) |
| Steering | [report.md](https://drive.google.com/file/d/1BRjUfQNt5jwAohKtxNherr3qM4UMxi6S/view) | [ERROR_LOG.md](https://drive.google.com/file/d/1fRjJr6H7CyCdUbJqTovgCubufJud9epZ/view), [ASSUMPTIONS.md](https://drive.google.com/file/d/12iu1usfz7MmAiWWc_bOuA_wf51Vzj-Fs/view) |
| Bilingual | [report.md](https://drive.google.com/file/d/1i7pmZDYK6Nav93ho_BchRwzKwfU_7Rm6/view) | [ERROR_LOG.md](https://drive.google.com/file/d/1dm0N1xAzYiEESTr6bn0_1CZdzl8Ue-s3/view), [ASSUMPTIONS.md](https://drive.google.com/file/d/1sZMGZCjvxfmWj_zDfuk6F0Qhiv89lTVe/view) |
| Follow-ups (E1–E4) | [report.md](https://drive.google.com/file/d/1BJO1o68QKZ652Ke3_oYzHzWgTYwdeQsZ/view) | [PLAN.md](https://drive.google.com/file/d/1Lz9Oo0JfphoxzFdLZ0I0KUhP2URhF1ha/view), [ERROR_LOG.md](https://drive.google.com/file/d/13kYrZYdWQOASXsJecYGOgEGxl7QekD6O/view), [DEVIATIONS.md](https://drive.google.com/file/d/1sdirOUScp4cHDjUOe-2PyNZyjUiB4k3e/view), [FILES.md](https://drive.google.com/file/d/1ykFrK_cnShMnicqcXNnul2mwTsbe_IeB/view) |
| post_train_3 overview and plan | [EXPERIMENTS_OVERVIEW.md](https://drive.google.com/file/d/1XItP7AHRnyOsz7kHX2AkSC2kLiYAbfdS/view) | [plan_post_train_3.md](https://drive.google.com/file/d/1vClSfK_Z2n5hMYwxv3T0V1hptCtNOsVv/view) |

Data folders: [post_train_1](https://drive.google.com/drive/folders/1XhYdi1aH-1LY517k0bkk-cJ3zwME7_Ik) ·
[post_train_2](https://drive.google.com/drive/folders/1p0zq2_Rc497dlIrfQtiOGYI8nMP7Dy_x) ·
[post_train_3 runs (English arm, steering, bilingual, follow-ups, Japanese culture EN)](https://drive.google.com/drive/folders/1bH59zkx1BlKXcqvR3L-NAbh3fbby3ltQ) ·
[post_train_3_shared/japan_culture (JA arm)](https://drive.google.com/drive/folders/17K2ru3s-xiceeE8NBcurc48HAhZxTFM2)
