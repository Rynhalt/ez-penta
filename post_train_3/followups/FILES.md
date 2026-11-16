## Files and links (Google Drive)

All notebooks (source and executed `_out`), outputs, results, code and logs from the post_train_3 runs on this Mac
(English arm, eval pool, bilingual, steering, Japanese culture, follow-ups) are on Google Drive in one folder:

**Drive root:** [ez-penta / post_train_3 / runs_2026-11-15_en_bilingual_steering_followups](https://drive.google.com/drive/folders/1bH59zkx1BlKXcqvR3L-NAbh3fbby3ltQ)
— 843 files, each checked by SHA-256 against the local copy (uploaded 2026-11-16).

Note: the local folder `~/Library/CloudStorage/GoogleDrive-test-ezpenta/My Drive/ez-penta/post_train_3/` used by every
notebook is **not** synced (Google Drive for desktop is not installed). It is the working copy; the Drive folder above is a
snapshot of it. Paths below are relative to both.

| What | Path | Drive link | Repo copy |
|---|---|---|---|
| Follow-ups (E1–E4) folder | `followups/` | [folder](https://drive.google.com/drive/folders/1d0ij4ACsOHtl4xFemWO3BV1PosS-oqDX) | `post_train_3/followups/` |
| Follow-ups report | `followups/report.md` | [report.md](https://drive.google.com/file/d/1BJO1o68QKZ652Ke3_oYzHzWgTYwdeQsZ/view) | `post_train_3/followups/report.md` |
| Follow-ups results (all tests) | `followups/results.json` | [results.json](https://drive.google.com/file/d/12AsIx5AWWA4ED5FDr0Wy7QXC1bwSZ4pr/view) | – |
| Follow-ups plan | `followups/PLAN.md` | [PLAN.md](https://drive.google.com/file/d/1Lz9Oo0JfphoxzFdLZ0I0KUhP2URhF1ha/view) | `post_train_3/followups/PLAN.md` |
| Error log / deviations | `followups/ERROR_LOG.md`, `followups/DEVIATIONS.md` | [ERROR_LOG](https://drive.google.com/file/d/13kYrZYdWQOASXsJecYGOgEGxl7QekD6O/view), [DEVIATIONS](https://drive.google.com/file/d/1sdirOUScp4cHDjUOe-2PyNZyjUiB4k3e/view) | – |
| Notebooks (F0–F4, ST0–ST6, FA; executed `_out`) | `followups/notebooks/` | [folder](https://drive.google.com/drive/folders/1DBh87JLPVuQ18i4nZ_KE8MA5LhUvr4ca) | `post_train_3/followups/notebooks/` |
| Notebook builder and runner scripts | `followups/code/` (`build_fu.py`, `runA/B/C.sh`, helpers) | [folder](https://drive.google.com/drive/folders/1kH2-09AdxMxejtQU1zt1dPAi82nEshVi) | – |
| Runner logs | `followups/logs/` | [folder](https://drive.google.com/drive/folders/1KAVMxjbEpXGpKmMXKGvgTmvXTAVeAEVN) | – |
| Answers + judge labels (half and E4 prompt sets) | `followups/eval/{half,e4}__<model>/eval.jsonl` | [folder](https://drive.google.com/drive/folders/1Mheaj-AleQGQhGt9GtgUFbDEy-M5Ubf_) | – |
| E1 (selection batch b0013, labels, directions, grid, selection, final A*bi) | `followups/e1/` | [folder](https://drive.google.com/drive/folders/1yUFy5WtyrgGRl-mee1qGOur6Av6HFrEq) | – |
| E4 matched US/UK prompts and checks | `followups/e4/prompts/` | [folder](https://drive.google.com/drive/folders/1ZfQIBMYarRqaIw-DzBjpFSB6ZaP7Yb0I) | – |
| New adapters (bi_h2, ja2, en/ja item halves) | `followups/adapters/` | [folder](https://drive.google.com/drive/folders/1Mn29FyvE1jD2gRDsTztCajE0pMnU8kl-) | – |
| Figures | `followups/figures/` | [folder](https://drive.google.com/drive/folders/1FfJ1GA-gdPuQrdeN1r8C9f1WiCVkqLzq) | – |
| Steering folder | `steering/` | [folder](https://drive.google.com/drive/folders/1b2NwRg4iFP8UmoJ636EMbrGK-OmE_Q1T) | `post_train_3/steering/` |
| Steering report / results / assumptions / errors | `steering/{report.md,results.json,ASSUMPTIONS.md,ERROR_LOG.md}` | [report](https://drive.google.com/file/d/1BRjUfQNt5jwAohKtxNherr3qM4UMxi6S/view), [results](https://drive.google.com/file/d/1zaoWyMt835Aci4imYzz4DEHIGIT9u4yJ/view), [assumptions](https://drive.google.com/file/d/12iu1usfz7MmAiWWc_bOuA_wf51Vzj-Fs/view), [errors](https://drive.google.com/file/d/1fRjJr6H7CyCdUbJqTovgCubufJud9epZ/view) | `post_train_3/steering/report.md` |
| Steering notebooks (S0–S7; executed `_out`; builders) | `steering/notebooks/` | [folder](https://drive.google.com/drive/folders/1vgXQlRT1jqO75efaJnz6oa_90KhCQ3kJ) | `post_train_3/steering/notebooks/` |
| Steering final evaluations | `steering/final/` | [folder](https://drive.google.com/drive/folders/1h_GOX1asZAW1aGH-xtZ9vdG8mSbdpe8x) | – |
| Bilingual folder | `bilingual/` | [folder](https://drive.google.com/drive/folders/1MpPV_D-5GmxvDYXptKA8EElHWcZEZ4UT) | `post_train_3/bilingual/` |
| Bilingual report / results / errors | `bilingual/{report.md,results.json,ERROR_LOG.md}` | [report](https://drive.google.com/file/d/1i7pmZDYK6Nav93ho_BchRwzKwfU_7Rm6/view), [results](https://drive.google.com/file/d/1j5sBofw6gMVBrK0Qd-PdWfVSNgH7xjVd/view), [errors](https://drive.google.com/file/d/1dm0N1xAzYiEESTr6bn0_1CZdzl8Ue-s3/view) | `post_train_3/bilingual/report.md` |
| Bilingual notebooks (B0–B4; executed `_out`; `nbsrc/`) | `bilingual/notebooks/` | [folder](https://drive.google.com/drive/folders/1vMX28_SSW1Rr9mzz-Ca65Thfvq9zVjh5) | `post_train_3/bilingual/notebooks/` |
| Bilingual adapters (ja, bi, en2 × seeds 0–2) | `bilingual/adapters/` | [folder](https://drive.google.com/drive/folders/1gZg9ThfFukm3zgEKKvnDV65JsIIO3jRe) | – |
| Bilingual evaluations | `bilingual/eval/` | [folder](https://drive.google.com/drive/folders/1gzrGycq-9U7FaoHvqmSULTWW2yMDLASg) | – |
| Bilingual logs | `bilingual/logs/` | [folder](https://drive.google.com/drive/folders/1P1smpQsDKl229489kjMiqhwARy_fTb8H) | – |
| Japanese culture folder | `japan_culture/` | [folder](https://drive.google.com/drive/folders/1R0ekOnMRWcHKRZK9SMgCNHOd45350IWG) | `post_train_3/japan_culture/` |
| Japanese culture EN analysis (report, notebooks, code, logs) | `japan_culture/en/` | [folder](https://drive.google.com/drive/folders/1ICFbZueLuz9ALB38uUC3vgirAnzH43yU) | `post_train_3/japan_culture/en/` |
| Japanese-culture prompts and answers per chunk | `japan_culture/chunks/` | [folder](https://drive.google.com/drive/folders/1aEfa4IkQLFCswWhGeQwghl3Jci4t2uwZ) | – |
| English arm (adapters seed 0–2, training items, pool evals) | `arm_en/` | [folder](https://drive.google.com/drive/folders/1TkUi_We2J2dC-NqAHvuYvYB0ISsJRsru) | `post_train_3/arm_en/` |
| Frozen evaluation pool (batches, decisions) | `eval_pool/` | [folder](https://drive.google.com/drive/folders/1CDta-eawhfdJMrE5vRzkDX0sfVnThFNv) | – |
| Agent prompts and the post_train_3 plan | `prompts/` | [folder](https://drive.google.com/drive/folders/1wnMfWAz0sMrGqlpaFwYpnkiZC2oWtBzh) | `post_train_3/*.md` |

The shared `post_train_3/eval_pool` and `post_train_3/arm_ja` folders next to this one belong to the Japanese-arm run on
another machine and were not changed.
