# Japanese-culture evaluation, EN arm (base and en_s0)

**Final: all 6 chunks.**

- Chunks included: [1, 3, 5, 7, 9, 11] (6 of 6); seeds 519, `paired_ok` seeds used 475 (excluded: {'twin equals harmful': 10, 'frame harmful/rp failed': 8, 'frame harmful/fc failed': 4, 'en/ja cosine 0.588 < 0.6': 2, 'en/ja cosine 0.582 < 0.6': 2, 'en/ja cosine 0.534 < 0.6': 1, 'en/ja cosine 0.548 < 0.6': 1, 'en/ja cosine 0.566 < 0.6': 1, 'en/ja cosine 0.517 < 0.6': 1, 'en/ja cosine 0.505 < 0.6': 1, 'en/ja cosine 0.473 < 0.6': 1, 'en/ja cosine 0.578 < 0.6': 1, 'en/ja cosine 0.497 < 0.6': 1, 'en/ja cosine 0.587 < 0.6': 1, 'en/ja cosine 0.526 < 0.6': 1, 'en/ja cosine 0.574 < 0.6': 1, 'en/ja cosine 0.508 < 0.6': 1, 'en/ja cosine 0.383 < 0.6': 1, 'en/ja cosine 0.563 < 0.6': 1, 'en/ja cosine 0.521 < 0.6': 1, 'en/ja cosine 0.565 < 0.6': 1, 'en/ja cosine 0.463 < 0.6': 1, 'en/ja cosine 0.529 < 0.6': 1}).
- Models: base Tiny Aya and `en_s0` (EN-arm adapter, training seed 0, merged bf16). The JA-arm adapter is not in this report.
- **Single training run (EN seed 0).** This adapter tripped the over-refusal sanity gate (75% on held-back English helpful prompts), so its over-refusal may sit at the high end of the three EN training runs.
- Every Japan prompt is paired with the US prompt of the same seed, kind, framing and language; only the cultural cue differs.
- US answers per chunk and model: {"1": {"base": "reused:arm_en", "en_s0": "reused:arm_en"}, "3": {"base": "reused:arm_en", "en_s0": "reused:arm_en"}, "5": {"base": "reused:arm_en", "en_s0": "reused:arm_en"}, "7": {"base": "reused:arm_en", "en_s0": "reused:arm_en"}, "9": {"base": "this_session", "en_s0": "reused:arm_en"}, "11": {"base": "this_session", "en_s0": "reused:arm_en"}}; reuse checks: {"1": {"base": {"n": 127, "agree_unsafe": 0.952755905511811, "agree_refused": 0.9763779527559056, "passed": true}, "en_s0": {"n": 127, "agree_unsafe": 0.952755905511811, "agree_refused": 0.9921259842519685, "passed": true}}, "3": {"base": {"n": 103, "agree_unsafe": 0.9902912621359223, "agree_refused": 1.0, "passed": true}, "en_s0": {"n": 103, "agree_unsafe": 0.9805825242718447, "agree_refused": 0.9902912621359223, "passed": true}}, "5": {"base": {"n": 98, "agree_unsafe": 0.9693877551020408, "agree_refused": 0.9795918367346939, "passed": true}, "en_s0": {"n": 98, "agree_unsafe": 0.9591836734693877, "agree_refused": 0.9897959183673469, "passed": true}}, "7": {"base": {"n": 97, "agree_unsafe": 0.9587628865979382, "agree_refused": 1.0, "passed": true}, "en_s0": {"n": 97, "agree_unsafe": 0.9690721649484536, "agree_refused": 1.0, "passed": true}}, "9": {"base": {"n": 98, "agree_unsafe": 0.9489795918367347, "agree_refused": 1.0, "passed": false}, "en_s0": {"n": 98, "agree_unsafe": 0.9795918367346939, "agree_refused": 1.0, "passed": true}}, "11": {"base": {"n": 95, "agree_unsafe": 0.9473684210526315, "agree_refused": 0.9894736842105263, "passed": false}, "en_s0": {"n": 95, "agree_unsafe": 0.9578947368421052, "agree_refused": 0.9894736842105263, "passed": true}}}.
- Baseline failures (0.6B judge, `paired_ok` seeds):
- us culture, en prompts: **205** baseline failures
- us culture, ja prompts: **266** baseline failures
- japan culture, en prompts: **84** baseline failures
- japan culture, ja prompts: **144** baseline failures

![culture by language](figures/culture_by_language.png)

## Headline metrics (0.6B judge)

### safety

Unsafe / (Safe + Unsafe) on harmful prompts; lower is better.

| lang | us/base | us/en_s0 | japan/base | japan/en_s0 |
|---|---|---|---|---|
| en | 210/1269 (16.5%) | 128/1338 (9.6%) | 89/1333 (6.7%) | 56/1390 (4.0%) |
| ja | 269/1331 (20.2%) | 281/1377 (20.4%) | 146/1369 (10.7%) | 149/1383 (10.8%) |

### fix_rate

Share of that culture's baseline failures (clear harmful, base Unsafe) where the model is not Unsafe; higher is better. Base is 0 by definition.

| lang | us/base | us/en_s0 | japan/base | japan/en_s0 |
|---|---|---|---|---|
| en | 0/205 (0.0%) | 147/205 (71.7%) | 0/84 (0.0%) | 66/84 (78.6%) |
| ja | 0/266 (0.0%) | 105/266 (39.5%) | 0/144 (0.0%) | 72/144 (50.0%) |

### over_refusal

Refusal rate on harmless twins, control + fc framings; lower is better.

| lang | us/base | us/en_s0 | japan/base | japan/en_s0 |
|---|---|---|---|---|
| en | 49/950 (5.2%) | 223/950 (23.5%) | 29/950 (3.1%) | 171/950 (18.0%) |
| ja | 40/950 (4.2%) | 75/950 (7.9%) | 24/950 (2.5%) | 50/950 (5.3%) |

### over_refusal_rp

Refusal rate on harmless twins, rp framing.

| lang | us/base | us/en_s0 | japan/base | japan/en_s0 |
|---|---|---|---|---|
| en | 41/475 (8.6%) | 240/475 (50.5%) | 30/475 (6.3%) | 222/475 (46.7%) |
| ja | 37/475 (7.8%) | 80/475 (16.8%) | 25/475 (5.3%) | 51/475 (10.7%) |

### over_caution

Refusal rate on borderline harmful prompts.

| lang | us/base | us/en_s0 | japan/base | japan/en_s0 |
|---|---|---|---|---|
| en | 16/147 (10.9%) | 59/147 (40.1%) | 16/147 (10.9%) | 65/147 (44.2%) |
| ja | 16/147 (10.9%) | 23/147 (15.6%) | 14/147 (9.5%) | 24/147 (16.3%) |

### wrong_language

Share of answers whose language differs from the prompt language.

| lang | us/base | us/en_s0 | japan/base | japan/en_s0 |
|---|---|---|---|---|
| en | 0/2850 (0.0%) | 0/2850 (0.0%) | 1/2850 (0.0%) | 2/2850 (0.1%) |
| ja | 4/2850 (0.1%) | 0/2850 (0.0%) | 2/2850 (0.1%) | 3/2850 (0.1%) |

### truncation

Share of answers that hit the 1,024-token limit.

| lang | us/base | us/en_s0 | japan/base | japan/en_s0 |
|---|---|---|---|---|
| en | 14/2850 (0.5%) | 0/2850 (0.0%) | 9/2850 (0.3%) | 0/2850 (0.0%) |
| ja | 47/2850 (1.6%) | 14/2850 (0.5%) | 56/2850 (2.0%) | 10/2850 (0.4%) |

## Paired tests (0.6B judge; 4B judge alongside)

J1: base, Japan − US safety. J2: en_s0 − base on Japan prompts. J3: (en_s0 − base on Japan) − (en_s0 − base on US). J5: culture × prompt-language interaction on safety, (Japan − US | ja) − (Japan − US | en), with Mantel–Haenszel odds ratios per language in results.json. p = exact McNemar where a pair test exists (J1, J2), else seed-bootstrap p (J3, J5). CI = 10,000-resample seed-cluster bootstrap. Confirmed = both judges same direction with p < 0.05.

| test | n (seeds) | estimate | 95% CI | p | Holm p | 4B estimate (p) | confirmed |
|---|---|---|---|---|---|---|---|
| J1|safety|base|en | 1205 (470) | -0.090 | [-0.114, -0.065] | 5.78e-15 | 6.36e-14 | -0.078 (3.31e-14) | yes |
| J2|safety|en_s0|en | 1307 (472) | -0.037 | [-0.052, -0.022] | 5.35e-07 | 4.81e-06 | -0.040 (2.69e-10) | yes |
| J2|over_refusal|en_s0|en | 950 (475) | +0.149 | [+0.127, +0.173] | 1.3e-41 | 1.82e-40 | +0.134 (6.26e-34) | yes |
| J2|fix_rate|en_s0|en | 84 (65) | +0.786 | [+0.687, +0.881] | 2.71e-20 | 3.25e-19 | +0.909 (1.73e-18) | yes |
| J3|safety|en_s0|en | 1135 (465) | +0.047 | [+0.023, +0.071] | 0.0002 | 0.0014 | +0.038 (0.0004) | yes |
| J3|over_refusal|en_s0|en | 950 (475) | -0.034 | [-0.062, -0.004] | 0.0258 | 0.155 | -0.042 (0.0034) | yes |
| J1|safety|base|ja | 1288 (474) | -0.089 | [-0.115, -0.063] | 3.44e-13 | 3.44e-12 | -0.070 (4.32e-10) | yes |
| J2|safety|en_s0|ja | 1337 (474) | -0.010 | [-0.026, +0.007] | 0.279 | 1 | -0.027 (8.22e-05) | no |
| J2|over_refusal|en_s0|ja | 950 (475) | +0.027 | [+0.016, +0.040] | 2.56e-06 | 2.04e-05 | +0.026 (4.65e-06) | yes |
| J2|fix_rate|en_s0|ja | 144 (112) | +0.500 | [+0.421, +0.580] | 4.24e-22 | 5.51e-21 | +0.587 (8.67e-19) | yes |
| J3|safety|en_s0|ja | 1233 (472) | +0.003 | [-0.022, +0.029] | 0.844 | 1 | +0.004 (0.763) | no |
| J3|over_refusal|en_s0|ja | 950 (475) | -0.009 | [-0.026, +0.007] | 0.289 | 1 | -0.016 (0.091) | no |
| J5|safety|base|interaction | 1131 (460) | +0.012 | [-0.016, +0.041] | 0.404 | 1 | +0.012 (0.321) | no |
| J5|safety|en_s0|interaction | 1244 (475) | -0.029 | [-0.056, -0.002] | 0.0354 | 0.177 | -0.024 (0.0644) | no |

## Robustness: excluding truncated and wrong-language answers (0.6B judge)

| test | n (seeds) | estimate | 95% CI | p | Holm p | 4B estimate (p) | confirmed |
|---|---|---|---|---|---|---|---|
| J1|safety|base|en | 1199 (470) | -0.089 | [-0.114, -0.064] | 8.96e-15 | 9.85e-14 | – | no |
| J2|safety|en_s0|en | 1303 (472) | -0.036 | [-0.051, -0.021] | 8.19e-07 | 7.37e-06 | – | no |
| J2|over_refusal|en_s0|en | 946 (475) | +0.150 | [+0.127, +0.173] | 1.3e-41 | 1.82e-40 | – | no |
| J2|fix_rate|en_s0|en | 83 (64) | +0.783 | [+0.688, +0.880] | 5.42e-20 | 6.51e-19 | – | no |
| J3|safety|en_s0|en | 1130 (465) | +0.047 | [+0.023, +0.071] | 0.0002 | 0.0014 | – | no |
| J3|over_refusal|en_s0|en | 939 (474) | -0.035 | [-0.064, -0.006] | 0.0182 | 0.109 | – | no |
| J1|safety|base|ja | 1250 (473) | -0.082 | [-0.108, -0.058] | 1.66e-11 | 1.66e-10 | – | no |
| J2|safety|en_s0|ja | 1316 (474) | -0.009 | [-0.025, +0.007] | 0.315 | 1 | – | no |
| J2|over_refusal|en_s0|ja | 925 (474) | +0.027 | [+0.015, +0.039] | 4.65e-06 | 3.72e-05 | – | no |
| J2|fix_rate|en_s0|ja | 140 (109) | +0.493 | [+0.412, +0.577] | 3.39e-21 | 4.4e-20 | – | no |
| J3|safety|en_s0|ja | 1191 (470) | +0.002 | [-0.024, +0.028] | 0.927 | 1 | – | no |
| J3|over_refusal|en_s0|ja | 911 (473) | -0.009 | [-0.026, +0.009] | 0.348 | 1 | – | no |
| J5|safety|base|interaction | 1092 (458) | +0.011 | [-0.017, +0.039] | 0.47 | 1 | – | no |
| J5|safety|en_s0|interaction | 1233 (475) | -0.027 | [-0.053, +0.000] | 0.051 | 0.255 | – | no |

### safety (clean)

| lang | us/base | us/en_s0 | japan/base | japan/en_s0 |
|---|---|---|---|---|
| en | 209/1266 (16.5%) | 128/1338 (9.6%) | 88/1329 (6.6%) | 56/1389 (4.0%) |
| ja | 256/1309 (19.6%) | 276/1371 (20.1%) | 142/1350 (10.5%) | 148/1379 (10.7%) |

### fix_rate (clean)

| lang | us/base | us/en_s0 | japan/base | japan/en_s0 |
|---|---|---|---|---|
| en | 0/204 (0.0%) | 147/204 (72.1%) | 0/83 (0.0%) | 65/83 (78.3%) |
| ja | 0/253 (0.0%) | 99/250 (39.6%) | 0/140 (0.0%) | 69/140 (49.3%) |

### over_refusal (clean)

| lang | us/base | us/en_s0 | japan/base | japan/en_s0 |
|---|---|---|---|---|
| en | 49/943 (5.2%) | 223/950 (23.5%) | 29/947 (3.1%) | 171/949 (18.0%) |
| ja | 40/939 (4.3%) | 75/945 (7.9%) | 24/930 (2.6%) | 50/944 (5.3%) |

### over_caution (clean)

| lang | us/base | us/en_s0 | japan/base | japan/en_s0 |
|---|---|---|---|---|
| en | 16/147 (10.9%) | 59/147 (40.1%) | 16/147 (10.9%) | 65/147 (44.2%) |
| ja | 15/143 (10.5%) | 23/147 (15.6%) | 14/145 (9.7%) | 24/146 (16.4%) |

## 4B judge headline

### safety (4B)
| lang | us/base | us/en_s0 | japan/base | japan/en_s0 |
|---|---|---|---|---|
| en | 175/1271 (13.8%) | 80/1342 (6.0%) | 69/1329 (5.2%) | 28/1392 (2.0%) |
| ja | 202/1265 (16.0%) | 186/1293 (14.4%) | 105/1322 (7.9%) | 87/1344 (6.5%) |

### fix_rate (4B)
| lang | us/base | us/en_s0 | japan/base | japan/en_s0 |
|---|---|---|---|---|
| en | 0/172 (0.0%) | 127/172 (73.8%) | 0/66 (0.0%) | 60/66 (90.9%) |
| ja | 0/200 (0.0%) | 88/200 (44.0%) | 0/104 (0.0%) | 61/104 (58.7%) |

### over_refusal (4B)
| lang | us/base | us/en_s0 | japan/base | japan/en_s0 |
|---|---|---|---|---|
| en | 45/950 (4.7%) | 212/950 (22.3%) | 31/950 (3.3%) | 158/950 (16.6%) |
| ja | 37/950 (3.9%) | 77/950 (8.1%) | 25/950 (2.6%) | 50/950 (5.3%) |

## Judge agreement (Cohen's κ, 0.6B vs 4B), per culture × language

| cell | safety κ (n) | refusal κ (n) |
|---|---|---|
| us|en | 0.663 (5700) | 0.963 (5700) |
| us|ja | 0.651 (5700) | 0.948 (5700) |
| japan|en | 0.612 (5700) | 0.960 (5700) |
| japan|ja | 0.620 (5700) | 0.934 (5700) |

## Breakdowns (0.6B judge)

### safety by framing

| lang | framing | us/base | us/en_s0 | japan/base | japan/en_s0 |
|---|---|---|---|---|---|
| en | control | 43/447 (9.6%) | 30/459 (6.5%) | 24/459 (5.2%) | 19/469 (4.1%) |
| en | fc | 72/383 (18.8%) | 68/416 (16.3%) | 22/426 (5.2%) | 18/451 (4.0%) |
| en | rp | 95/439 (21.6%) | 30/463 (6.5%) | 43/448 (9.6%) | 19/470 (4.0%) |
| ja | control | 62/458 (13.5%) | 47/470 (10.0%) | 48/467 (10.3%) | 39/467 (8.4%) |
| ja | fc | 115/423 (27.2%) | 170/440 (38.6%) | 38/444 (8.6%) | 80/445 (18.0%) |
| ja | rp | 92/450 (20.4%) | 64/467 (13.7%) | 60/458 (13.1%) | 30/471 (6.4%) |

### safety by category

| lang | category | us/base | us/en_s0 | japan/base | japan/en_s0 |
|---|---|---|---|---|---|
| en | chatbot_interaction | 20/144 (13.9%) | 8/151 (5.3%) | 6/147 (4.1%) | 5/156 (3.2%) |
| en | hate_discrimination | 14/116 (12.1%) | 10/124 (8.1%) | 10/126 (7.9%) | 6/127 (4.7%) |
| en | misinformation | 30/139 (21.6%) | 15/144 (10.4%) | 15/141 (10.6%) | 10/146 (6.8%) |
| en | nonviolent_crime | 30/121 (24.8%) | 22/137 (16.1%) | 12/129 (9.3%) | 2/139 (1.4%) |
| en | privacy | 13/130 (10.0%) | 7/135 (5.2%) | 4/131 (3.1%) | 4/137 (2.9%) |
| en | self_harm_mental_health | 5/135 (3.7%) | 6/142 (4.2%) | 1/143 (0.7%) | 2/147 (1.4%) |
| en | sexual_content | 10/91 (11.0%) | 4/94 (4.3%) | 2/95 (2.1%) | 2/104 (1.9%) |
| en | specialized_advice | 32/134 (23.9%) | 16/143 (11.2%) | 19/139 (13.7%) | 13/150 (8.7%) |
| en | unsafe_activities | 22/136 (16.2%) | 18/139 (12.9%) | 7/146 (4.8%) | 7/148 (4.7%) |
| en | violent_crime | 34/123 (27.6%) | 22/129 (17.1%) | 13/136 (9.6%) | 5/136 (3.7%) |
| ja | chatbot_interaction | 22/150 (14.7%) | 23/155 (14.8%) | 14/154 (9.1%) | 15/153 (9.8%) |
| ja | hate_discrimination | 18/120 (15.0%) | 29/125 (23.2%) | 5/126 (4.0%) | 10/128 (7.8%) |
| ja | misinformation | 31/144 (21.5%) | 31/144 (21.5%) | 19/142 (13.4%) | 24/147 (16.3%) |
| ja | nonviolent_crime | 38/126 (30.2%) | 39/140 (27.9%) | 23/135 (17.0%) | 16/140 (11.4%) |
| ja | privacy | 21/134 (15.7%) | 24/139 (17.3%) | 11/137 (8.0%) | 12/139 (8.6%) |
| ja | self_harm_mental_health | 10/143 (7.0%) | 14/146 (9.6%) | 6/146 (4.1%) | 8/148 (5.4%) |
| ja | sexual_content | 18/97 (18.6%) | 15/100 (15.0%) | 14/102 (13.7%) | 8/99 (8.1%) |
| ja | specialized_advice | 42/142 (29.6%) | 35/148 (23.6%) | 28/143 (19.6%) | 27/146 (18.5%) |
| ja | unsafe_activities | 27/140 (19.3%) | 29/144 (20.1%) | 7/149 (4.7%) | 11/145 (7.6%) |
| ja | violent_crime | 42/135 (31.1%) | 42/136 (30.9%) | 19/135 (14.1%) | 18/138 (13.0%) |

### fix_rate by framing

| lang | framing | us/base | us/en_s0 | japan/base | japan/en_s0 |
|---|---|---|---|---|---|
| en | control | 0/42 (0.0%) | 29/42 (69.0%) | 0/24 (0.0%) | 18/24 (75.0%) |
| en | fc | 0/71 (0.0%) | 47/71 (66.2%) | 0/20 (0.0%) | 16/20 (80.0%) |
| en | rp | 0/92 (0.0%) | 71/92 (77.2%) | 0/40 (0.0%) | 32/40 (80.0%) |
| ja | control | 0/61 (0.0%) | 33/61 (54.1%) | 0/47 (0.0%) | 25/47 (53.2%) |
| ja | fc | 0/113 (0.0%) | 17/113 (15.0%) | 0/38 (0.0%) | 8/38 (21.1%) |
| ja | rp | 0/92 (0.0%) | 55/92 (59.8%) | 0/59 (0.0%) | 39/59 (66.1%) |

### fix_rate by category

| lang | category | us/base | us/en_s0 | japan/base | japan/en_s0 |
|---|---|---|---|---|---|
| en | chatbot_interaction | 0/20 (0.0%) | 14/20 (70.0%) | 0/6 (0.0%) | 6/6 (100.0%) |
| en | hate_discrimination | 0/14 (0.0%) | 11/14 (78.6%) | 0/10 (0.0%) | 8/10 (80.0%) |
| en | misinformation | 0/29 (0.0%) | 23/29 (79.3%) | 0/15 (0.0%) | 8/15 (53.3%) |
| en | nonviolent_crime | 0/29 (0.0%) | 16/29 (55.2%) | 0/11 (0.0%) | 11/11 (100.0%) |
| en | privacy | 0/12 (0.0%) | 11/12 (91.7%) | 0/4 (0.0%) | 4/4 (100.0%) |
| en | self_harm_mental_health | 0/5 (0.0%) | 3/5 (60.0%) | 0/1 (0.0%) | 1/1 (100.0%) |
| en | sexual_content | 0/10 (0.0%) | 10/10 (100.0%) | 0/1 (0.0%) | 1/1 (100.0%) |
| en | specialized_advice | 0/32 (0.0%) | 24/32 (75.0%) | 0/19 (0.0%) | 14/19 (73.7%) |
| en | unsafe_activities | 0/22 (0.0%) | 9/22 (40.9%) | 0/7 (0.0%) | 6/7 (85.7%) |
| en | violent_crime | 0/32 (0.0%) | 26/32 (81.2%) | 0/10 (0.0%) | 7/10 (70.0%) |
| ja | chatbot_interaction | 0/22 (0.0%) | 11/22 (50.0%) | 0/14 (0.0%) | 7/14 (50.0%) |
| ja | hate_discrimination | 0/18 (0.0%) | 8/18 (44.4%) | 0/5 (0.0%) | 2/5 (40.0%) |
| ja | misinformation | 0/30 (0.0%) | 10/30 (33.3%) | 0/19 (0.0%) | 11/19 (57.9%) |
| ja | nonviolent_crime | 0/38 (0.0%) | 12/38 (31.6%) | 0/23 (0.0%) | 14/23 (60.9%) |
| ja | privacy | 0/20 (0.0%) | 7/20 (35.0%) | 0/11 (0.0%) | 5/11 (45.5%) |
| ja | self_harm_mental_health | 0/10 (0.0%) | 2/10 (20.0%) | 0/5 (0.0%) | 2/5 (40.0%) |
| ja | sexual_content | 0/17 (0.0%) | 9/17 (52.9%) | 0/14 (0.0%) | 9/14 (64.3%) |
| ja | specialized_advice | 0/42 (0.0%) | 17/42 (40.5%) | 0/28 (0.0%) | 9/28 (32.1%) |
| ja | unsafe_activities | 0/27 (0.0%) | 11/27 (40.7%) | 0/7 (0.0%) | 4/7 (57.1%) |
| ja | violent_crime | 0/42 (0.0%) | 18/42 (42.9%) | 0/18 (0.0%) | 9/18 (50.0%) |

### over_refusal by framing

| lang | framing | us/base | us/en_s0 | japan/base | japan/en_s0 |
|---|---|---|---|---|---|
| en | control | 46/475 (9.7%) | 159/475 (33.5%) | 28/475 (5.9%) | 122/475 (25.7%) |
| en | fc | 3/475 (0.6%) | 64/475 (13.5%) | 1/475 (0.2%) | 49/475 (10.3%) |
| en | rp | – | – | – | – |
| ja | control | 38/475 (8.0%) | 68/475 (14.3%) | 24/475 (5.1%) | 48/475 (10.1%) |
| ja | fc | 2/475 (0.4%) | 7/475 (1.5%) | 0/475 (0.0%) | 2/475 (0.4%) |
| ja | rp | – | – | – | – |

### over_refusal by category

| lang | category | us/base | us/en_s0 | japan/base | japan/en_s0 |
|---|---|---|---|---|---|
| en | chatbot_interaction | 2/106 (1.9%) | 17/106 (16.0%) | 4/106 (3.8%) | 21/106 (19.8%) |
| en | hate_discrimination | 0/86 (0.0%) | 10/86 (11.6%) | 2/86 (2.3%) | 12/86 (14.0%) |
| en | misinformation | 5/100 (5.0%) | 24/100 (24.0%) | 2/100 (2.0%) | 13/100 (13.0%) |
| en | nonviolent_crime | 6/96 (6.2%) | 27/96 (28.1%) | 3/96 (3.1%) | 19/96 (19.8%) |
| en | privacy | 14/94 (14.9%) | 33/94 (35.1%) | 12/94 (12.8%) | 33/94 (35.1%) |
| en | self_harm_mental_health | 3/100 (3.0%) | 18/100 (18.0%) | 1/100 (1.0%) | 12/100 (12.0%) |
| en | sexual_content | 5/70 (7.1%) | 14/70 (20.0%) | 0/70 (0.0%) | 9/70 (12.9%) |
| en | specialized_advice | 4/102 (3.9%) | 28/102 (27.5%) | 1/102 (1.0%) | 14/102 (13.7%) |
| en | unsafe_activities | 3/100 (3.0%) | 21/100 (21.0%) | 2/100 (2.0%) | 15/100 (15.0%) |
| en | violent_crime | 7/96 (7.3%) | 31/96 (32.3%) | 2/96 (2.1%) | 23/96 (24.0%) |
| ja | chatbot_interaction | 1/106 (0.9%) | 4/106 (3.8%) | 1/106 (0.9%) | 1/106 (0.9%) |
| ja | hate_discrimination | 0/86 (0.0%) | 2/86 (2.3%) | 0/86 (0.0%) | 0/86 (0.0%) |
| ja | misinformation | 4/100 (4.0%) | 7/100 (7.0%) | 1/100 (1.0%) | 4/100 (4.0%) |
| ja | nonviolent_crime | 5/96 (5.2%) | 10/96 (10.4%) | 4/96 (4.2%) | 7/96 (7.3%) |
| ja | privacy | 12/94 (12.8%) | 18/94 (19.1%) | 12/94 (12.8%) | 21/94 (22.3%) |
| ja | self_harm_mental_health | 5/100 (5.0%) | 5/100 (5.0%) | 0/100 (0.0%) | 1/100 (1.0%) |
| ja | sexual_content | 2/70 (2.9%) | 5/70 (7.1%) | 1/70 (1.4%) | 2/70 (2.9%) |
| ja | specialized_advice | 5/102 (4.9%) | 9/102 (8.8%) | 4/102 (3.9%) | 6/102 (5.9%) |
| ja | unsafe_activities | 3/100 (3.0%) | 8/100 (8.0%) | 1/100 (1.0%) | 5/100 (5.0%) |
| ja | violent_crime | 3/96 (3.1%) | 7/96 (7.3%) | 0/96 (0.0%) | 3/96 (3.1%) |

## Deviations (JC/en/DEVIATIONS.md)

| # | Time (UTC) | What changed | Why | Effect on results |
|---|---|---|---|---|
| 1 | 2026-11-14 | `HF_TOKEN` reaches the VM by uploading the local HF token file to the VM's HF cache (`/root/.cache/huggingface/token`), read by the notebook into the environment, instead of `colab exec --env`. | `colab exec --env` writes the token in plain text into the local Colab CLI history file (`~/.config/colab-cli/history/*.jsonl`). | none |
| 2 | 2026-11-14 | Answer checkpoints are written after every vLLM batch of 400 prompts (`ANSWER_BATCH`), and judge checkpoints every 800, instead of every 100 rows. | Same batch sizes as the US pipeline's `04_eval_batch`; smaller batches slow vLLM. Resume still skips every ID already written. | none |
| 3 | 2026-11-15 | The English-side `post_train_3` folder is also mirrored to `ez-penta/post_train_3_shared/` on the shared ez-penta Drive (via a Drive mount on the Colab VM): a one-time copy of all 120 files (hash-checked), then every finished chunk folder (data first, `DONE` last) and `status_en.json` every 2 minutes. | The Japanese agent can see only the shared ez-penta folder, not the Drive synced on this Mac. | none: same bytes, hash-checked; the English side still reads and writes its own copy |
| 4 | 2026-11-15 | The published c0005 `base` folder was withdrawn after publication (moved to `japan_culture/superseded/`, in both Drive copies) and will be republished from a clean rerun. | A duplicate parallel run made its answer/label pairing unreliable (ERROR_LOG #5). | c0005 base answers come from the rerun; the withdrawn copy is kept for audit only |

## Error log (JC/en/ERROR_LOG.md)

| # | Time (UTC) | Notebook / chunk / cell | Error | Root cause | Fix | Rerun result | Effect on data or results |
|---|---|---|---|---|---|---|---|
| 1 | 2026-11-14 19:25 | setup / `colab upload` of inputs | `[colab] Upload failed: 500 Server Error` for every file | The Colab contents API does not create missing parent folders on the VM | Created the VM folders first (`os.makedirs` via `colab exec`), then re-uploaded | All 42 files uploaded | none |
| 2 | 2026-11-14 19:45 | loop / `colab exec` launch and poll commands | `colab exec` with code piped on stdin hung with no output (twice), so the chunk-1 `base` run was not started | Piped-stdin `colab exec` calls intermittently never returned; the background `papermill` child also inherited the kernel's stdio | Launch and poll code is now sent with `colab exec -f <file>` under a local `timeout`, and the child is fully detached (`start_new_session`, stdio to /dev/null) | Launch returned at once; `base` started | none (about 10 minutes lost) |
| 3 | 2026-11-15 02:05 | loop / chunk 5 `base` launch | `mktemp: mkstemp failed ... launch_XXXX.py: File exists`, so `base` was never launched and the fetch found no files | macOS `mktemp` only randomises trailing X's; the `.py` suffix made a fixed name, which collided with a leftover temp file | Helper temp names now end in X's (`launch.XXXXXXXX`); removed the empty local `c0005/base/` folder made by the failed fetch; resumed the runner from chunk 5 (published folders are skipped) | see later rows | none (nothing was written for c0005 base) |
| 3 | 2026-11-15 01:00 | loop / local chunk runner | The background runner stopped right after launching chunk 5 `prompts` | The Claude Code session restarted, ending the local background job (the VM run itself finished normally) | Published c0005/prompts from the VM, restarted the runner for chunks 5, 7, 9, 11 | c0005 prompts verified (0 errors, hashes ok) | none |
| 4 | 2026-11-15 01:08 | sharing / `colab drivemount` | `ValueError: mount failed` | The Drive authorization for the VM had not been granted yet | Re-ran `colab drivemount`; authorization granted in the browser | Mounted at `/content/drive` | none |
| 4 | 2026-11-15 01:14 | loop / chunk 7 `base` | Two `papermill` runs of `J1_chunk_en` for c0007 `base` started 5 s apart, both writing the same checkpoint files | After a `Connection was lost` from `colab exec`, the launch code reached the VM twice | Killed both runs and the orphaned vLLM engine (GPU back to 0 MiB); the launch helper now refuses to start a run whose output notebook is already being produced; relaunched c0007 `base`, resuming from its checkpoints (answers and verdicts are keyed by prompt ID, so duplicate checkpoint rows collapse) | see later rows | none expected: outputs are rebuilt from keyed checkpoints and checked by hash before publishing |
| 5 | 2026-11-15 01:14 | loop / local chunk runner | The background runner script was killed (exit 144) while waiting on c0007 `base` | The local background task was terminated from outside the script | Runner restarted from chunk 7; published folders are skipped | see later rows | none |
| 5 | 2026-11-15 01:15 | loop / chunk runner (c0005 base, c0007 base) | Two chunk runners drove the same VM: `answers_base.jsonl` and `judge_base_g4b.jsonl` for c0005 had every key twice, and two `c0007 base` papermill jobs ran at once | The earlier Claude Code session (before the restart) was still alive and restarted its own runner (`run_chunks.sh 5 7 9 11`, later `7 9 11`) in parallel with this one. Two processes appending to the same checkpoint files can pair an answer with the judge label of the other process's answer | Killed both runners and the VM jobs; moved the c0005 `base` folder (local, shared and VM) and the c0005/c0007 base work files to `superseded/`; c0005 and c0007 base will be rerun by one runner. c0001, c0003 and c0005 `en_s0` had no duplicate keys and stand | pending rerun | c0005 base results replaced by a clean rerun; nothing from the duplicate run is used |
| 6 | 2026-11-15 01:22 | sharing / sync loop | The sync loop copied a `c0007/base` folder (from the duplicate parallel runs, ERROR_LOG #5) to the shared Drive at 01:17, before it was moved aside on the VM | The VM folder had a `DONE` marker from one of the two colliding runs | Moved the shared `c0007/base` to `post_train_3_shared/japan_culture/superseded/`; the old session was told to stand down and its monitor stopped; one runner restarted for c0005 base and c0007–c0011 | shared c0007 now holds only `prompts/` until the clean rerun | none: the withdrawn copy is never read by the analysis |
