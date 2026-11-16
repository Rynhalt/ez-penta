# Culture vs language: what training, data mix and abliteration fixed

Model: Tiny Aya Water. Judge: Qwen3Guard-Gen-0.6B. These are automated labels and still need human review.

**Rates and counts.** Every rate is followed by its raw count as `k/n`. Where an adapter has 3 training runs, the `k/n` is
**summed over the runs** (e.g. `393/4042` = 3 runs × ~1,350 prompts). Some report tables average the per-run rates instead,
so a few percentages differ there by ±0.1 pt.

**Metrics.**
- **Unsafe:** Unsafe / (Safe + Unsafe) on harmful prompts (all severities).
- **Over-refusal:** refused harmless twins (control + fiction framings).
- **Fix rate:** the share of base failures (clear harmful prompts where base was Unsafe) that the model no longer fails.

**Where each number comes from.** Each number is tagged with its experiment; links are in [Sources](#sources).

| Tag | Experiment | Prompt set |
|---|---|---|
| **[BI]** | Bilingual training: en / ja / bi / en2 adapters trained on the same 568 US-culture items, 3 training runs each | Full frozen pool, 475 paired seeds |
| **[B5]** | Refusal split, computed from the [BI] answers | Same as [BI] |
| **[ST]** | Steering: "abliteration" in reverse on the English adapter | Full pool |
| **[E1]** | Follow-up E1: the same edit on the bilingual adapter | Pool half, 245 paired seeds |
| **[E2]** | Follow-up E2: helpful examples shown twice | Pool half |
| **[E3]** | Follow-up E3: Japanese-only replication | Pool half |
| **[E4]** | Follow-up E4: US vs UK with matched cues | Pool half |
| **[JA]** | JA arm's Japan-culture evaluation (interim) | Chunks 1, 3, 5, 7; 322 seeds |
| **[PT1]**, **[PT2]** | post_train_1 and post_train_2 | Earlier, smaller sets |

---

## 1. Headline answers

| Question | Answer | Main evidence |
|---|---|---|
| Is US culture riskier than Japan culture? | At baseline, yes: about 2× more unsafe in both languages. But this comes mostly from the content of the US cues, not from culture. | [BI] base; [E4] matched cues; [PT1] |
| Does training transfer across culture? | Yes. Every adapter (trained only on US-culture items) improves Japan-culture and UK prompts in its training language. | [BI], [E4], [JA] |
| Does training transfer across language? | English → Japanese: **no**. Japanese → English: **mostly yes**. | [BI], [E3] |
| How does each adapter get safer? | English and bilingual training: almost entirely by **refusing more**. Japanese training: about 70% refusing, about 30% **safer answers**. | [B5] |
| Does abliteration (removing the over-refusal direction) help? | It lowers English over-refusal in both cultures, costs some safety, does nothing in Japanese on the English adapter, and backfires in Japanese on the bilingual adapter. | [ST], [E1] |

---

## 2. US vs Japan culture

### 2.1 Base model: US-culture prompts are about twice as unsafe [BI]

| Prompt language | US culture | Japan culture |
|---|---|---|
| English | 15.9% (199/1255) | 6.7% (89/1333) |
| Japanese | 20.2% (268/1325) | 10.7% (146/1369) |

**Earlier work agrees.**
- **[PT1]:** across the full Dataset2 GGUF evaluation, US framing about doubles the odds of an unsafe answer (Mantel–Haenszel OR 1.98, 95% CI 1.58–2.48). Japan framing doesn't (OR 1.10, p = 0.49).
- **[PT2] cross-category re-analysis:** US 14.3% vs neutral 7.2% unsafe.

### 2.2 The cause is cue content, not culture [E4]
Qwen3-4B rewrote each neutral request into a US version and a UK version, using the same rule: one benign cue each.

| Base model, unsafe | Original US prompt | US, matched cue | UK, matched cue |
|---|---|---|---|
| English | 15.4% (111/720) | 7.5% (50/670) | 8.0% (55/684) |
| Japanese | 20.0% (153/765) | 7.2% (49/685) | 7.9% (54/684) |

- **US vs UK with matched cues:** no difference (English test: 7.7% vs 7.9%, p = 1).
- **Original US prompts vs matched rewrites:** about twice as unsafe. On the 150 seeds that passed the checks in both cultures: English 16.1% vs 8.1%, Japanese 20.5% vs 7.3%. Both confirmed by both judges, and the same on L4 hardware.

### 2.3 Training transfers across culture [BI]
Every adapter was trained on US-culture items only. Unsafe on English prompts:

| Model | US culture | Japan culture | Relative cut, US / Japan |
|---|---|---|---|
| Base | 15.9% (199/1255) | 6.7% (89/1333) | – |
| English-trained | 9.7% (393/4042) | 3.9% (164/4169) | −39% / −41% |
| Japanese-trained | 9.2% (369/4016) | 3.8% (159/4167) | −42% / −43% |
| Bilingual | 5.3% (220/4149) | 2.8% (118/4234) | −67% / −58% |

**Fix rate on base failures is just as high, or higher, for Japan culture:**

| Model | US culture, English | Japan culture, English | US culture, Japanese | Japan culture, Japanese |
|---|---|---|---|---|
| English-trained | 68.1% (385/565) | 76.9% (193/251) | 38.9% (301/774) | 47.1% (198/420) |
| Japanese-trained | 63.6% (342/538) | 74.5% (184/247) | 69.8% (548/785) | 76.1% (322/423) |
| Bilingual | 82.5% (466/565) | 86.5% (218/252) | 68.8% (535/778) | 71.9% (304/423) |

**Other evidence of transfer:**
- **To UK prompts [E4]** (unsafe, English, UK-matched): base 8.0% (55/684) → English-trained 2.3% (16/700), Japanese-trained 3.2% (22/698), bilingual 1.7% (12/708).
- **JA arm [JA]** (the JA arm's own Japanese adapter, Japan-culture prompts):
  - English prompts: 6.3% (63/998) → 3.1% (32/1031).
  - Japanese prompts: 8.6% (88/1023) → 5.8% (61/1043).
  - Fix rate: 76% English (44/58), 74% Japanese (65/88).

**Why the gain is smaller on Japan-culture prompts.** In absolute points it's smaller. [JA] J3 puts it at +3.95 pt (0.6B judge) and +4.55 pt (4B judge). The likeliest reason is that Japan-culture prompts start safer, leaving less to fix: the *relative* cut and the fix rates above are the same or higher.

---

## 3. English vs Japanese

### 3.1 Unsafe rate by prompt language [BI]

| Model | English prompts | Japanese prompts | All prompts |
|---|---|---|---|
| Base | 11.1% (288/2588) | 15.4% (414/2694) | 13.3% (702/5282) |
| English-trained | 6.8% (557/8211) | 15.7% (1297/8273) | 11.2% (1854/16484) |
| Japanese-trained | 6.5% (528/8183) | **8.3% (696/8359)** | 7.4% (1224/16542) |
| Bilingual | **4.0% (338/8383)** | 9.7% (809/8360) | **6.9% (1147/16743)** |
| English ×2 | 4.7% (389/8338) | 17.1% (1412/8253) | 10.9% (1801/16591) |

### 3.2 Over-refusal by prompt language [BI]

| Model | English prompts | Japanese prompts | All prompts |
|---|---|---|---|
| Base | **4.3% (81/1900)** | **3.3% (63/1900)** | **3.8% (144/3800)** |
| English-trained | 20.4% (1162/5700) | 6.4% (364/5700) | 13.4% (1526/11400) |
| Japanese-trained | 13.2% (750/5700) | 17.1% (976/5700) | 15.1% (1726/11400) |
| Bilingual | 36.0% (2052/5700) | 15.4% (875/5700) | 25.7% (2927/11400) |
| English ×2 | 29.3% (1668/5700) | 5.4% (307/5700) | 17.3% (1975/11400) |

### 3.3 The same tables by culture × language [BI]

| Unsafe | Base | English-trained | Japanese-trained | Bilingual | English ×2 |
|---|---|---|---|---|---|
| US culture, English | 15.9% (199/1255) | 9.7% (393/4042) | 9.2% (369/4016) | 5.3% (220/4149) | 6.2% (256/4112) |
| US culture, Japanese | 20.2% (268/1325) | 20.2% (831/4115) | 9.8% (407/4160) | 11.3% (471/4162) | 21.5% (881/4094) |
| Japan culture, English | 6.7% (89/1333) | 3.9% (164/4169) | 3.8% (159/4167) | 2.8% (118/4234) | 3.1% (133/4226) |
| Japan culture, Japanese | 10.7% (146/1369) | 11.2% (466/4158) | 6.9% (289/4199) | 8.1% (338/4198) | 12.8% (531/4159) |

| Over-refusal | Base | English-trained | Japanese-trained | Bilingual | English ×2 |
|---|---|---|---|---|---|
| US culture, English | 5.5% (52/950) | 23.0% (656/2850) | 15.7% (448/2850) | 39.8% (1135/2850) | 32.9% (937/2850) |
| US culture, Japanese | 4.1% (39/950) | 7.5% (214/2850) | 21.0% (599/2850) | 19.2% (547/2850) | 6.1% (174/2850) |
| Japan culture, English | 3.1% (29/950) | 17.8% (506/2850) | 10.6% (302/2850) | 32.2% (917/2850) | 25.6% (731/2850) |
| Japan culture, Japanese | 2.5% (24/950) | 5.3% (150/2850) | 13.2% (377/2850) | 11.5% (328/2850) | 4.7% (133/2850) |

### 3.4 True or false: Japanese-only vs bilingual vs English-only [BI]

| Statement | Verdict | Numbers |
|---|---|---|
| Japanese-only is best for Japanese unsafe | **True** | 8.3% vs English-trained 15.7%, bilingual 9.7% |
| Japanese-only is best for Japanese over-refusal | **False**: it's the worst | 17.1% vs English-trained 6.4%, bilingual 15.4% |
| Japanese-only ≈ English-only on English unsafe | **True** (a tie, not better) | 6.5% vs 6.8% |
| Japanese-only beats English-only on English over-refusal | **True** | 13.2% vs 20.4% |
| Bilingual is safest overall | **True** | 6.9% vs Japanese-only 7.4% |
| Bilingual has the worst English over-refusal | **True** | 36.0% |

**Is Japanese training *better* than English on English prompts? No.**
- The 9.2% vs 9.7% gap (US culture) sits within run-to-run spread. Per run:
  - English-trained: 128/1338, 133/1355, 132/1349.
  - Japanese-trained: 118/1333, 130/1343, 121/1340.
- Fix rates favour English training (68.1% vs 63.6%).
- **With matched training length [E3]** (twice the data each), English training wins on English prompts:
  - US culture: en2 5.8% (140/2400) vs ja2 9.1% (213/2348).
  - Japan culture: en2 3.0% (65/2177) vs ja2 3.9% (83/2154).
- **On the two independent item halves [E3]** (US culture, English): en_A 6.8% (53/783) vs ja_A 8.7% (68/779); en_B 8.0% (63/784) vs ja_B 8.8% (68/775).

**English training doesn't transfer to Japanese; Japanese training does to English.**
- **[E3], US culture, Japanese prompts:** en2 22.2% (529/2381) vs ja2 7.8% (188/2412).
- **[E3] item halves:** en_A 17.6% (139/788) vs ja_A 10.1% (80/795); en_B 12.8% (101/787) vs ja_B 10.3% (81/790).

**Does Japanese training help English because it learned US cues? No.** All adapters saw the same US-culture items. Japanese training's cut on English prompts is the same in relative terms in US and Japan culture (−42% / −43%, see 2.3), and it carries over to UK prompts it never saw [E4].

---

## 4. Is the safety gain bought with refusals? [B5]

**The split:** P(unsafe) = P(not refused) × P(unsafe | answered).

The change from base splits (Shapley, two factors) into:
- **from refusing:** the change in P(not refused);
- **from answers:** the change in P(unsafe | answered).

**Refusal share** = the part of the unsafe drop that comes from refusing.
- **100%:** the model got safer only by refusing.
- **Below 100%:** some of the gain is safer answers.
- **Above 100%:** its answers actually got *less* safe.

A refusal is almost never labelled Unsafe (≤0.1%), so the split is exact.

| Prompts | Model | Unsafe | Refuses harmful prompts | Unsafe when it answers | From refusing | From answers | **Refusal share** |
|---|---|---|---|---|---|---|---|
| English | Base | 11.1% (288/2588) | 26.0% (672/2588) | 15.0% (287/1916) | – | – | – |
| English | English-trained | 6.8% (557/8211) | 60.2% (4943/8211) | 17.0% (557/3268) ↑ | −5.5 pt | +1.2 pt | **127%** |
| English | Japanese-trained | 6.5% (528/8183) | 49.0% (4011/8183) | **12.6% (526/4172) ↓** | −3.2 pt | −1.5 pt | **68%** |
| English | Bilingual | 4.0% (338/8383) | 72.7% (6092/8383) | 14.7% (338/2291) | −6.9 pt | −0.1 pt | **98%** |
| English | English ×2 | 4.7% (389/8338) | 69.9% (5829/8338) | 15.5% (389/2509) | −6.7 pt | +0.3 pt | **104%** |
| Japanese | Base | 15.4% (414/2694) | 25.3% (681/2694) | 20.6% (414/2013) | – | – | – |
| Japanese | English-trained | 15.7% (1297/8273) | 36.3% (3002/8273) | 24.6% (1297/5271) ↑ | −2.5 pt | +2.8 pt | net worse |
| Japanese | Japanese-trained | 8.3% (696/8359) | 51.7% (4322/8359) | **17.2% (696/4037) ↓** | −5.0 pt | −2.1 pt | **71%** |
| Japanese | Bilingual | 9.7% (809/8360) | 51.6% (4316/8360) | 20.0% (809/4044) | −5.3 pt | −0.4 pt | **94%** |
| Japanese | English ×2 | 17.1% (1412/8253) | 34.9% (2880/8253) | 26.3% (1412/5373) ↑ | −2.3 pt | +4.0 pt | net worse |
| All | Japanese-trained | 7.4% (1224/16542) | 50.4% (8333/16542) | 14.9% (1222/8209) | −4.1 pt | −1.8 pt | **69%** |
| All | Bilingual | 6.9% (1147/16743) | 62.2% (10408/16743) | 18.1% (1147/6335) | −6.6 pt | +0.1 pt | **102%** |

**Same split by culture** (refusal share; unsafe when it answers, base → model):

| Cell | English-trained | Japanese-trained | Bilingual |
|---|---|---|---|
| US culture, English | 126% (22.0% → 24.8%) | 71% (22.0% → 18.8%) | 93% (22.0% → 20.5%) |
| US culture, Japanese | net 0 (27.6% → 33.0%) | 74% (27.6% → 22.9%) | 93% (27.6% → 26.5%) |
| Japan culture, English | 121% (8.8% → 9.8%) | 63% (8.8% → 7.1%) | 112% (8.8% → 9.7%) |
| Japan culture, Japanese | net worse (14.0% → 16.9%) | 79% (14.0% → 12.8%) | 122% (14.0% → 14.9%) |

**Reading it:**
- **English training and bilingual training** get safer by refusing. Their answers are no safer, and English training's answers are actually *less* safe.
- **On Japanese prompts, English training's extra refusals are cancelled out by less safe answers.**
- **Japanese training is the only one that answers more safely,** in every cell.
- **Culture doesn't change the pattern.**

The share is meaningless when the net change is about zero; read the point columns instead. Computed in `bilingual/notebooks/B5_refusal_decomposition.ipynb` from the same answers as [BI].

---

## 5. Abliteration: removing the over-refusal direction [ST], [E1]

**What the edit is.** It is the reverse of classic abliteration:
1. Find the over-refusal direction: the direction that separates refused from answered harmless twins.
2. Remove its overlap with the harmfulness direction.
3. Subtract the result from the weights (variant A; layer 18; strength 0.5).

### 5.1 On the English adapter (training seed 0) [ST]

| Cell | Over-refusal: adapter → edited | Unsafe: adapter → edited |
|---|---|---|
| US culture, English | 23.5% (244/1038) → **17.0% (176/1038)** | 9.1% (133/1467) → 10.8% (159/1469) |
| Japan culture, English | 18.0% (171/950) → **13.1% (124/950)** | 4.0% (56/1390) → 4.6% (63/1384) |
| US culture, Japanese | 7.6% (79/1038) → 7.2% (75/1038) | 20.0% (300/1502) → 20.0% (298/1489) |
| Japan culture, Japanese | 5.3% (50/950) → 5.4% (51/950) | 10.8% (149/1383) → 11.1% (154/1385) |

**Results:**
- **English over-refusal (cultures pooled):** 20.9% (415/1988) → 15.1% (300/1988). Confirmed by both judges.
- **Replication:** seed 1 went 408/1988 → 298/1988, seed 2 went 403/1988 → 294/1988.
- **Cost:**
  - English unsafe 6.6% (189/2857) → 7.8% (222/2853).
  - Fix rate 72.1% (222/308) → 66.9% (206/308).
  - Both miss the pre-set margins.
- **On the base model** (layer 12): almost no change.
  - English unsafe: 11.7% (320/2736) → 11.3% (307/2707).
  - Japanese unsafe: 15.5% (438/2828) → 15.8% (446/2827).
  - English over-refusal: 4.2% (83/1988) → 4.0% (80/1988).

### 5.2 On the bilingual adapter [E1] (3 training runs, pool half)

| Cell | Over-refusal: adapter → edited | Unsafe: adapter → edited |
|---|---|---|
| US culture, English | 39.9% (660/1656) → **34.2% (566/1656)** | 5.0% (121/2403) → 6.4% (154/2401) |
| Japan culture, English | 28.5% (419/1470) → **22.9% (336/1470)** | 2.7% (59/2187) → 3.4% (74/2172) |
| US culture, Japanese | 17.6% (292/1656) → **20.9% (346/1656) ↑** | 10.4% (250/2402) → 10.8% (261/2414) |
| Japan culture, Japanese | 10.0% (147/1470) → **12.3% (181/1470) ↑** | 8.5% (185/2169) → 8.2% (178/2172) |

No grid setting met the selection rule, so the fallback (layer 18, strength 0.5) was used.

### 5.3 What abliteration shows

| | Same language as the over-refusal | Other language | Across culture |
|---|---|---|---|
| English adapter [ST] | English over-refusal −5 to −6.5 pt, unsafe +0.6 to +1.7 pt | No effect | Same effect in US and Japan culture (−28% and −27% relative) |
| Bilingual adapter [E1] | English over-refusal −5.6 to −5.7 pt, unsafe +0.7 to +1.4 pt | **Backfires:** Japanese over-refusal +2.3 to +3.3 pt | Same pattern in both cultures |

- **The over-refusal direction overlaps the harm direction** (cosine 0.61 at layer 18), so cutting refusals costs safety.
- **The English and Japanese over-refusal directions are similar** (cosine about 0.74). The English adapter barely over-refuses in Japanese (6.5%), so there is little to remove there.
- **On the bilingual adapter, one shared direction pushes Japanese the wrong way.**

### 5.4 The other knob: more helpful examples [E2] (bilingual, helpful items shown twice)

| Cell | Over-refusal: bilingual → bi_h2 | Unsafe: bilingual → bi_h2 |
|---|---|---|
| US culture, English | 39.9% (660/1656) → 31.9% (529/1656) | 5.0% (121/2403) → 7.0% (169/2409) |
| US culture, Japanese | 17.6% (292/1656) → 13.0% (216/1656) | 10.4% (250/2402) → 14.0% (338/2408) |
| Japan culture, English | 28.5% (419/1470) → 21.1% (310/1470) | 2.7% (59/2187) → 4.2% (91/2173) |
| Japan culture, Japanese | 10.0% (147/1470) → 6.9% (102/1470) | 8.5% (185/2169) → 9.9% (214/2157) |

It works in both languages and both cultures, but it costs about as much safety as the edit.

---

## 6. One-number view: weighted error [BI]

Treat unsafe as the miss (a false negative) and over-refusal as the false alarm (a false positive).

**Score** = (w × unsafe + over-refusal) / (w + 1), where w = how many over-refusals one unsafe answer is worth. All prompts:

| Model | Unsafe | Over-refusal | w = 1 (balanced error) | w = 3 |
|---|---|---|---|---|
| Base | 13.3% (702/5282) | 3.8% (144/3800) | **8.6%** | 10.9% |
| English-trained | 11.2% (1854/16484) | 13.4% (1526/11400) | 12.3% | 11.8% |
| Japanese-trained | 7.4% (1224/16542) | 15.1% (1726/11400) | 11.3% | **9.3%** |
| Bilingual | 6.9% (1147/16743) | 25.7% (2927/11400) | 16.3% | 11.6% |
| English ×2 | 10.9% (1801/16591) | 17.3% (1975/11400) | 14.1% | 12.5% |

- **Weighting the two errors equally, the untrained base model wins.**
- **Japanese-only becomes best once one unsafe answer is worth about 2 or more over-refusals.**
- **Bilingual wins only when w ≈ 9 or more.**

F1 isn't used: precision depends on the harmful-to-harmless ratio of the test set, which is set by its design (twins are one-to-one).

---

## 7. Caveats
- **Labels:** automated. Judge agreement on safety is moderate (κ about 0.5–0.65); on refusals it's high (κ about 0.9–0.97).
- **[JA]:** interim (4 of 6 chunks), with one training run per arm and hardware-mismatched runs. Its J4 English-vs-Japanese adapter comparison is not confirmed (4B Holm p = 0.060).
- **[E4]:** only 150 of 276 seeds passed the checks in both cultures; the tables above use all seeds that have both versions.
- **[E1]–[E4] hardware:** these ran on an L4. A re-run of base on the L4 agreed with the stored G4 labels about 99% of the time.
- **Prompt sets differ:** [BI]/[ST] use the full pool, [E1]–[E4] the pool half, [JA] chunks 1/3/5/7. Compare across sections loosely; within a table the prompts are identical.
- **Tables vs tests:** a few rates quoted earlier in chat come from the tests, which use only prompts every compared run judged Safe/Unsafe. They differ slightly from these table rates, e.g. E3 en2 vs ja2 on English, US culture: 3.9% vs 7.3% in the test, 5.8% vs 9.1% in the table.

---

## Sources

| Tag | Report on Drive | Local copy |
|---|---|---|
| [BI] | [bilingual/report.md](https://drive.google.com/file/d/1i7pmZDYK6Nav93ho_BchRwzKwfU_7Rm6/view) | `post_train_3/bilingual/report.md` |
| [B5] | (new, not on Drive yet) | `post_train_3/bilingual/notebooks/B5_refusal_decomposition_out.ipynb` |
| [ST] | [steering/report.md](https://drive.google.com/file/d/1BRjUfQNt5jwAohKtxNherr3qM4UMxi6S/view) | `post_train_3/steering/report.md` |
| [E1]–[E4], hardware check | [followups/report.md](https://drive.google.com/file/d/1BJO1o68QKZ652Ke3_oYzHzWgTYwdeQsZ/view) | `post_train_3/followups/report.md` |
| [JA] | [interim findings PDF (4/6 chunks)](https://drive.google.com/file/d/1oo5oYLBXmQMugqpDJHZ8WZpitGPvO8wV/view) · [ja/report.md](https://drive.google.com/file/d/1YbvR3KrA67XGk-5vjt5Sr5aotwe_G2x3/view) | `post_train_3/japan_culture/ja/JA_interim_findings_4of6.pdf` |
| EN-arm Japan culture | [japan_culture/en/report.md](https://drive.google.com/file/d/1YxZsNT8nM8SHuycZ3rBfAsHmmwFdouX0/view) | `post_train_3/japan_culture/en/report.md` |
| [PT1] | [post_train_1/report.md](https://drive.google.com/file/d/1uXmH_h5ePAr2aDutwacR0ERD1kg-hCPk/view) | `post_train_1/report.md` |
| [PT2] | [next_experiments.md](https://drive.google.com/file/d/1UQrubMQkT1Jc4kT0o2_xPgP98WFuJ4H4/view) | `post_train_2/next_experiments.md` |
| All experiments | [EXPERIMENTS_SUMMARY.md](https://drive.google.com/file/d/11Ae4RgloJguP7CAYiWb4AmN5Wj0CeTjq/view) | `EXPERIMENTS_SUMMARY.md` |
