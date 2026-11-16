# post_train_3: experiments overview

Three runs on Tiny Aya Water, all on the same frozen evaluation pool: **(1) steering**, **(2) bilingual training**, and
**(3) the follow-ups**, one unattended run that held four experiments (E1–E4) plus a hardware check. All numbers use the
primary judge (Qwen3Guard-Gen-0.6B). "Confirmed" means the 4B judge agrees with Holm-adjusted p < 0.05. The results are
automated analyses and still need human review.

**Drive folder (everything):** [ez-penta / post_train_3 / runs_2026-11-15_en_bilingual_steering_followups](https://drive.google.com/drive/folders/1bH59zkx1BlKXcqvR3L-NAbh3fbby3ltQ)

---

## 1. Steering: remove the over-refusal direction from the English adapter

**Experiment.** We found the direction in the model's activations that separates refused from answered *harmless* prompts,
made it orthogonal to the direction that separates harmful from harmless, and removed it from the weights of the
English-trained adapter (`en_s0`). The settings (layer 18, strength 0.5) were picked on Dataset2 `heldout`. We also tried
inference-time variants.

| | |
|---|---|
| **Expected** | Over-refusal on harmless English prompts fell: 20.9% → 15.1% (confirmed). It replicated on training seeds 1 and 2 (about 20.5% → 15%). Answer quality didn't change: a pairwise judge preferred the edited model 118 times and the original 104 times (p 0.38). |
| **Unexpected** | It cost safety. English unsafe rose 6.6% → 7.8%, and fixed base failures fell 72% → 67%; both missed the pre-set margins. The "refuse harmless" and "harmful" directions largely overlap (cosine 0.61 at layer 18, up to 0.8 in base). Japanese prompts didn't change at all (6.5% → 6.3%). No inference-time variant qualified, and on base the activation-steering variant (CAST) *raised* over-refusal while lowering unsafe. |
| **What it tells us** | In this model, over-refusal isn't a separate feature you can cut out: it shares most of its direction with harm detection, so removing it gives back some safety. The adapter's over-refusal is also English-specific, and a direction found on English prompts doesn't touch Japanese behaviour. |

## 2. Bilingual training: does adding Japanese training rows fix Japanese prompts?

**Experiment.** We took the same 568 Dataset2 items and trained LoRA adapters four ways, 3 training seeds each: English only
(`en`), Japanese only (`ja`), both (`bi`: each English row followed by its Japanese row), and English twice (`en2`). `en2` has
the same number of steps as `bi`, so it controls for training dose.

| | |
|---|---|
| **Expected** | `bi` is much safer than `en` on Japanese prompts (9.7% vs 15.7% unsafe, confirmed), and this isn't the extra dose: `en2` scores 17.1%. Each training language mainly improves that language. |
| **Unexpected** | `bi` is also safer on English (4.0% vs 6.8%), but it refuses many more harmless prompts (39.8% vs 23.0% on US/English). Japanese-only training nearly matches English training *on English prompts* (net fixes +139 vs +149) and refuses less there (15.7% vs 23.0%). The transfer is lopsided: doubling the English data made Japanese *worse* (net fixes +17 vs +46). Training in a language raises refusals of harmless prompts in that language (`ja` on Japanese prompts: 21.0% vs 7.5% for `en`). |
| **What it tells us** | Japanese safety data is the biggest single lever for Japanese prompts and costs little on English. Japanese-to-English transfer is strong, while English-to-Japanese is weak. The cost is over-refusal, which follows the training language. |

## 3. Follow-ups: one unattended run, four experiments plus a hardware check

All of these use the pool half (frozen batches 1, 3, 5: 276 seeds, 3,312 US prompts and the matching Japan prompts).
No G4 GPU was available, so the new models ran on **L4**.

**Hardware check.** We re-ran base and `bi_s0` on the L4. About 99% of labels match the stored G4 answers, and no drift
test is significant, so the L4 results are comparable.

### E1: the steering edit on the bilingual adapter
| | |
|---|---|
| **Expected** | English over-refusal fell: 39.9% → 34.2% (confirmed). |
| **Unexpected** | No grid setting met the selection rule, so the fallback (layer 18, strength 0.5) was used. English unsafe rose 3.7% → 5.4%, and Japanese over-refusal went *up* (17.6% → 20.9%). |
| **What it tells us** | The bilingual adapter shows the same entanglement as the English one. A single weight edit can't remove its extra refusals without giving back safety, and in Japanese it backfires. |

### E2: showing helpful examples twice (`bi_h2`)
| | |
|---|---|
| **Expected** | Over-refusal fell in every cell: English 39.9% → 31.9%, Japanese 17.6% → 13.0% (confirmed). |
| **Unexpected** | Unsafe rose about as much as with steering: English 4.3% → 6.1%, Japanese 8.8% → 11.6%. Training length wasn't matched (1,704 vs 1,136 steps). |
| **What it tells us** | The mix of helpful and refusal examples is a dial along the same safety vs over-refusal trade-off, not a way around it. |

### E3: Japanese-only replication (dose control `ja2`, plus two disjoint item halves)
| | |
|---|---|
| **Expected** | `ja2` ≈ `ja`, so dose doesn't matter. The en-vs-ja pattern replicates on two independent item sets: Japanese training refuses much less on English prompts, and is much safer with more fixes on Japanese prompts (for example, Japanese-prompt unsafe: `ja2` 6.3% vs `en2` 19.9%; fix rate 79.6% vs 40.2%). |
| **Unexpected** | On English prompts, Japanese training is *less* safe than the dose-matched English adapter (7.3% vs 3.9% unsafe; fix rate 69% vs 82%). The two item halves show the same gap at about 2 points. |
| **What it tells us** | The bilingual findings hold beyond one item list. Japanese data transfers to English only partly: English data still matters for English safety. |

### E4: US vs UK culture with matched cues
| | |
|---|---|
| **Expected** | All three adapters, trained on US-culture prompts, also cut unsafe on UK prompts in English (base 8.9% → `en_s0` 2.1%). |
| **Unexpected** | With matched, benign cues, US and UK behave the same (7.7% vs 7.9% unsafe in English). The *original* US prompts are about twice as risky as their matched rewrites (English 16.1% vs 8.1%, Japanese 20.5% vs 7.3%; confirmed, also on L4). The English adapter didn't help on Japanese-language UK prompts (12.1% vs 9.5%). Only 150 of 276 seeds passed the checks in both cultures. |
| **What it tells us** | The earlier finding that "US culture is more dangerous" mostly reflects the content of the original cultural cues, not the culture itself. Claims about culture need cue-matched prompts. |

---

## Methodology (brief)

- **Model.** Tiny Aya Water (`CohereLabs/tiny-aya-water`). Adapters are LoRA r=16, α=32 on layer-18 `mlp.down_proj`: AdamW lr 1e-4, batch 1, one epoch, max_len 768, training seeds 0–2. They are served as merged bf16 models with vLLM.
- **Prompts.** Seeds come from Dataset2 (519 frozen seeds in 6 batches). Each seed has a harmful request and a harmless *twin*, in three framings (control, fiction/creative, role-play), two languages (English, Japanese) and two cultures (US, Japan); E4 adds matched US and UK cultures. Paired analyses use only seeds that pass every generation check (`paired_ok`).
- **Generation.** Greedy decoding, up to 1,024 new tokens.
- **Judging.** Qwen3Guard-Gen-0.6B is the primary judge, with Qwen3Guard-Gen-4B as a second judge.
- **Metrics.** Unsafe rate on harmful prompts; fix rate (base failures the model no longer fails); over-refusal (refused harmless twins); over-caution (refused borderline prompts); wrong-language answers; truncation.
- **Statistics.** Every model answers identical prompts. Tests are exact McNemar per training seed, a seed-cluster or two-level bootstrap (evaluation seeds × training runs), Holm correction within each experiment, and confirmation by the 4B judge.
- **Runs.** Every stage is a papermill notebook on Colab (G4; L4 for the follow-ups), checkpointed and resumable. Each output folder carries `manifest.json` and a `DONE` hash. Errors and deviations are logged as they happen.

---

## Where things are on Drive

| Folder | Contents |
|---|---|
| [steering/](https://drive.google.com/drive/folders/1b2NwRg4iFP8UmoJ636EMbrGK-OmE_Q1T) | Labels, directions, selection grid, final evaluations ([final/](https://drive.google.com/drive/folders/1h_GOX1asZAW1aGH-xtZ9vdG8mSbdpe8x)), quality check, [notebooks/](https://drive.google.com/drive/folders/1vgXQlRT1jqO75efaJnz6oa_90KhCQ3kJ) |
| [bilingual/](https://drive.google.com/drive/folders/1MpPV_D-5GmxvDYXptKA8EElHWcZEZ4UT) | [adapters/](https://drive.google.com/drive/folders/1gZg9ThfFukm3zgEKKvnDV65JsIIO3jRe) (ja, bi, en2 × 3), [eval/](https://drive.google.com/drive/folders/1gzrGycq-9U7FaoHvqmSULTWW2yMDLASg), [logs/](https://drive.google.com/drive/folders/1P1smpQsDKl229489kjMiqhwARy_fTb8H), [notebooks/](https://drive.google.com/drive/folders/1vMX28_SSW1Rr9mzz-Ca65Thfvq9zVjh5) |
| [followups/](https://drive.google.com/drive/folders/1d0ij4ACsOHtl4xFemWO3BV1PosS-oqDX) | [eval/](https://drive.google.com/drive/folders/1Mheaj-AleQGQhGt9GtgUFbDEy-M5Ubf_) (half and E4 answers), [e1/](https://drive.google.com/drive/folders/1yUFy5WtyrgGRl-mee1qGOur6Av6HFrEq), [e4/](https://drive.google.com/drive/folders/1ZfQIBMYarRqaIw-DzBjpFSB6ZaP7Yb0I), [adapters/](https://drive.google.com/drive/folders/1Mn29FyvE1jD2gRDsTztCajE0pMnU8kl-), [notebooks/](https://drive.google.com/drive/folders/1DBh87JLPVuQ18i4nZ_KE8MA5LhUvr4ca), [code/](https://drive.google.com/drive/folders/1kH2-09AdxMxejtQU1zt1dPAi82nEshVi), [logs/](https://drive.google.com/drive/folders/1KAVMxjbEpXGpKmMXKGvgTmvXTAVeAEVN), [figures/](https://drive.google.com/drive/folders/1FfJ1GA-gdPuQrdeN1r8C9f1WiCVkqLzq) |
| [japan_culture/](https://drive.google.com/drive/folders/1R0ekOnMRWcHKRZK9SMgCNHOd45350IWG) | Japan-culture prompts and answers per chunk ([chunks/](https://drive.google.com/drive/folders/1aEfa4IkQLFCswWhGeQwghl3Jci4t2uwZ)); EN analysis ([en/](https://drive.google.com/drive/folders/1ICFbZueLuz9ALB38uUC3vgirAnzH43yU)) |
| [arm_en/](https://drive.google.com/drive/folders/1TkUi_We2J2dC-NqAHvuYvYB0ISsJRsru) | English adapters (seeds 0–2), training items, pool evaluations used as the stored baseline |
| [eval_pool/](https://drive.google.com/drive/folders/1CDta-eawhfdJMrE5vRzkDX0sfVnThFNv) | Frozen evaluation pool (seed batches, decisions) |
| [prompts/](https://drive.google.com/drive/folders/1wnMfWAz0sMrGqlpaFwYpnkiZC2oWtBzh) | Agent prompts and the post_train_3 plan |

## Markdown files on Drive

| Experiment | Report | Logs and notes |
|---|---|---|
| Steering | [report.md](https://drive.google.com/file/d/1BRjUfQNt5jwAohKtxNherr3qM4UMxi6S/view) | [ERROR_LOG.md](https://drive.google.com/file/d/1fRjJr6H7CyCdUbJqTovgCubufJud9epZ/view), [ASSUMPTIONS.md](https://drive.google.com/file/d/12iu1usfz7MmAiWWc_bOuA_wf51Vzj-Fs/view) |
| Bilingual | [report.md](https://drive.google.com/file/d/1i7pmZDYK6Nav93ho_BchRwzKwfU_7Rm6/view) | [ERROR_LOG.md](https://drive.google.com/file/d/1dm0N1xAzYiEESTr6bn0_1CZdzl8Ue-s3/view), [ASSUMPTIONS.md](https://drive.google.com/file/d/1sZMGZCjvxfmWj_zDfuk6F0Qhiv89lTVe/view) |
| Follow-ups (E1–E4) | [report.md](https://drive.google.com/file/d/1BJO1o68QKZ652Ke3_oYzHzWgTYwdeQsZ/view) | [PLAN.md](https://drive.google.com/file/d/1Lz9Oo0JfphoxzFdLZ0I0KUhP2URhF1ha/view), [ERROR_LOG.md](https://drive.google.com/file/d/13kYrZYdWQOASXsJecYGOgEGxl7QekD6O/view), [DEVIATIONS.md](https://drive.google.com/file/d/1sdirOUScp4cHDjUOe-2PyNZyjUiB4k3e/view), [FILES.md](https://drive.google.com/file/d/1ykFrK_cnShMnicqcXNnul2mwTsbe_IeB/view) |
| Japan culture (EN) | [report.md](https://drive.google.com/file/d/1YxZsNT8nM8SHuycZ3rBfAsHmmwFdouX0/view) | [ERROR_LOG.md](https://drive.google.com/file/d/12CFh3VMP9bJ7uaHSMbo-OdvvoSbPdMlB/view), [DEVIATIONS.md](https://drive.google.com/file/d/1g0umk7dV5nkK5W_tSMJP_zGsdU-v-kvP/view) |
| Plan and agent prompts | [plan_post_train_3.md](https://drive.google.com/file/d/1vClSfK_Z2n5hMYwxv3T0V1hptCtNOsVv/view) | [EN](https://drive.google.com/file/d/1FcwMBFoAuzGwTD8HAAPXSFdrMDQfZSn1/view), [JA](https://drive.google.com/file/d/16feklwu4Wf3hC_mO8Vbqjff3PLfi4WGC/view), [STEERING](https://drive.google.com/file/d/1qtkbIQ_jlM5HU50Uw9fizJEqvxKIBqzQ/view), [BILINGUAL](https://drive.google.com/file/d/1gh6K9keOc9r-tA53hF4rRTYZ0GZ2Uvqz/view), [JAPAN_CULTURE_EN](https://drive.google.com/file/d/1Rx4IkKEE22Zv5QetThfX5tPMQzsXgwst/view), [JAPAN_CULTURE_JA](https://drive.google.com/file/d/1Qnx6-EaeLWh2e_7MDjJdc_GjwP4o2ZQL/view) |

Repo copies: `post_train_3/{steering,bilingual,followups,japan_culture/en}/report.md`, `post_train_3/followups/{PLAN,FILES}.md`, `post_train_3/*.md`.
