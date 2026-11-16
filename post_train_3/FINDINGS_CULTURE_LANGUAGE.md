# Making Tiny Aya safer in English and Japanese: what worked

## Abstract

We fine-tuned the small multilingual model Tiny Aya Water with SafeTune LoRA adapters to make it refuse harmful requests.
We tested it on prompts in English and Japanese, with US or Japanese cultural framing.

**There is always a trade-off between two errors:**
- **Unsafe answers:** the model helps with a harmful request.
- **Over-refusals:** the model refuses a harmless request written to *look* harmful.

Every method that reduced unsafe answers increased over-refusals.

**Results by training language:**
- **English training** makes the model safe in English only. Japanese prompts don't improve.
- **Japanese training** makes the model safe in Japanese, and almost as safe in English as English training does, while refusing fewer harmless English prompts. It is also the only adapter whose *answers* became safer; the others mostly got safer by refusing more.
- **Training on both languages** gives the safest model overall, but it refuses by far the most harmless prompts.

**Culture matters less than language.** US-framed prompts look about twice as risky as Japan-framed ones, but that comes from what the US prompts contained, not from US culture. Training also carries over from US-framed to Japan- and UK-framed prompts.

**Removing a "refusal direction" from the model's weights** (reverse abliteration) cut English over-refusals but cost some safety. It didn't help in Japanese.

---

## Key terms
- **Unsafe rate:** the share of harmful prompts that got a harmful answer. Lower is better.
- **Over-refusal rate:** the share of harmless "look-alike" prompts the model refused. Lower is better.
- **Adapter:** a small add-on (LoRA) trained on top of the model. All adapters used the same 568 training examples, written with US cultural framing, in English, Japanese, or both.
- All answers were scored by an automatic safety judge (Qwen3Guard), not by humans.

---

## Main results

### 1. Safety depends on the training language

Unsafe rate, averaged over US- and Japan-culture prompts:

| Model | English prompts | Japanese prompts |
|---|---|---|
| No training (base) | 11% (288/2588) | 15% (414/2694) |
| English-trained | 7% (557/8211) | 16% (1297/8273), **no improvement** |
| Japanese-trained | 7% (528/8183) | **8% (696/8359)** |
| Trained on both | **4% (338/8383)** | 10% (809/8360) |

- **English training doesn't carry over to Japanese.** Doubling the English data made Japanese slightly worse.
- **Japanese training carries over to English** almost as well as English training does. It isn't *better* than English training on English prompts: with matched amounts of data, English training wins (6% vs 9%).

### 2. The price: refusing harmless prompts

Over-refusal rate:

| Model | English prompts | Japanese prompts |
|---|---|---|
| No training (base) | 4% (81/1900) | 3% (63/1900) |
| English-trained | 20% (1162/5700) | 6% (364/5700) |
| Japanese-trained | 13% (750/5700) | 17% (976/5700) |
| Trained on both | **36% (2052/5700)** | 15% (875/5700) |

- Each adapter refuses most in the language it was trained in.
- The bilingual adapter's extra safety costs a lot of extra refusals.

### 3. Most of the safety comes from refusing more

We split each model's safety gain into two parts:
- **refusing more often;**
- **answering more safely** when it does answer.

| Model | Share of the gain from refusing more | Unsafe rate when it does answer (base: 15% English, 21% Japanese) |
|---|---|---|
| English-trained | about 100% or more | English 17%, Japanese 25%: answers got *worse* |
| Trained on both | about 95–100% | English 15%, Japanese 20%: unchanged |
| **Japanese-trained** | **about 70%** | **English 13%, Japanese 17%: answers got safer** |

Only Japanese training made the model's answers themselves safer, in both languages and both cultures.

### 4. Culture: US vs Japan

- **At baseline, US-framed prompts look twice as unsafe as Japan-framed ones:** 16% (199/1255) vs 7% (89/1333) in English.
- **But that's the prompts, not the culture.** When we rewrote the same requests with a simple, comparable cultural detail, US and UK versions were equally unsafe: 8% (50/670) vs 8% (55/684). The original US prompts were about twice as risky as their rewrites: 15% (111/720) vs 8%.
- **Training carries over across cultures.** Every adapter was trained only on US-framed examples, yet each became safer on Japan- and UK-framed prompts in its training language. For example, English training on English prompts:
  - US culture: 16% → 10% (393/4042);
  - Japan culture: 7% → 4% (164/4169);
  - UK culture: 8% → 2% (16/700).

### 5. Two fixes for over-refusal didn't escape the trade-off

| Fix | Effect on over-refusal | Effect on safety |
|---|---|---|
| **Reverse abliteration** (remove the over-refusal direction from the weights), English adapter | English 21% → 15% (415/1988 → 300/1988) | English unsafe 6.6% → 7.8%; no effect in Japanese |
| Same, on the bilingual adapter | English 40% → 34%, but Japanese *rose* 18% → 21% | English unsafe 5.0% → 6.4% |
| **Show helpful examples twice** (bilingual) | English 40% → 32%, Japanese 18% → 13% | English unsafe 5.0% → 7.0%, Japanese 10% → 14% |

Both fixes slide along the same trade-off: fewer refusals means less safety.

### 6. Which model is "best"?

That depends on how much worse an unsafe answer is than an unnecessary refusal:
- **Equally bad:** the untrained model scores best, because training adds more refusals than it removes harm.
- **One unsafe answer ≈ 2 or more refusals:** Japanese-trained is best.
- **Safety is all that matters:** the bilingual model is best.

---

## Caveats
- All scores come from an automatic judge, with no human review.
- The Japanese-arm results are interim (4 of 6 batches).
- Some experiments used half of the test prompts; numbers from different sections aren't from identical prompt sets.

## Where the numbers come from

| Topic | Experiment and report |
|---|---|
| Sections 1–3 and 6 | Bilingual training ([report](https://drive.google.com/file/d/1i7pmZDYK6Nav93ho_BchRwzKwfU_7Rm6/view)); refusal split in `bilingual/notebooks/B5_refusal_decomposition.ipynb` |
| Section 4, cultural rewrites and UK | Follow-ups, experiment E4 ([report](https://drive.google.com/file/d/1BJO1o68QKZ652Ke3_oYzHzWgTYwdeQsZ/view)) |
| Section 4, transfer to Japan culture | Bilingual training; JA arm ([interim PDF](https://drive.google.com/file/d/1oo5oYLBXmQMugqpDJHZ8WZpitGPvO8wV/view)) |
| Section 5, reverse abliteration | Steering ([report](https://drive.google.com/file/d/1BRjUfQNt5jwAohKtxNherr3qM4UMxi6S/view)); follow-ups E1 |
| Section 5, helpful examples | Follow-ups E2 |
| Matched-data comparison (section 1) | Follow-ups E3 |

All numbers, with raw counts for every cell: [FINDINGS_CULTURE_LANGUAGE_DETAILED.md](FINDINGS_CULTURE_LANGUAGE_DETAILED.md).
