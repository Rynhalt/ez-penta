# Agent 1: x_steer experiments A then D (run unattended)

You are Agent 1. Read `post_train_3/XSTEER_SHARED.md` first and follow it exactly. It holds the paths, definitions, run
rules, the C decision rule, and how to report. Agent 2 runs B and C in parallel on this same Mac. Both agents share the
local folder `PT3/x_steer/`. Never touch `PT3/x_steer/B`, `PT3/x_steer/C`, or VM `pt3-x2`.

Order: **A → report A → D → report D → final message.** Start A immediately.

---

## A: language-balanced directions on the saved layers (local CPU, no GPU)

**Question:** was the steering direction biased by language (and framing)? Within one language, is there a saved layer
where over-refusal is readable (held-out AUC ≥ 0.80) and independent of harm (|cos| ≤ 0.30)?

**Notebook:** `x_steer/notebooks/XA_balanced_directions.ipynb`, run locally with papermill.

1. Load and hash-check the inputs:
   - `steering/s2_directions/manifest.json` + `DONE`;
   - `acts/index.jsonl`;
   - `acts_{base,en_s0}_L{8,12,16,18,20,24,28}.npy`;
   - `steering/s1_labels/answers.jsonl`.

   Join labels to activation rows by `prompt_id`. Assert the row counts match (3,864 prompts).
2. Report refused/answered twin counts per model × language × framing × culture.
3. For each model ∈ {base, en_s0} and each saved layer, compute every direction in XSTEER_SHARED §2: h, h_en, h_ja, g,
   f_en, f_ja, f_bal, f_pooled_old, plus the ⊥h and ⊥hg versions.
4. Diagnostics per (model, layer, direction): the cosines of §2; the held-out and in-sample AUCs (within English, within
   Japanese, pooled); the language-only baseline AUC.
5. **Bias check:**
   - cos(f_pooled_old, g) and cos(f_pooled_old, f_bal);
   - re-check the steering result's "cos(f, h)" with f_bal in place of f_pooled_old.
6. Apply the §3 threshold to each saved layer, per model and language: `qualifies` yes/no. This is informational for A;
   C's actual choice uses B.
7. **Outputs** in `PT3/x_steer/A/`:
   - `directions_bal.npz`, with keys `{dir}_{model}_L{L}` for every direction above;
   - `diagnostics.csv` and `diagnostics.json`;
   - `report.md`;
   - `figures/`: cos(f_lang, h_lang) and held-out AUC vs layer, one line per model × language;
   - `manifest.json` and `DONE`.
8. **Report A now** (XSTEER_SHARED §5). In RESULTS.md answer directly:
   - How biased was the old direction? Give cos(f_pooled_old, g) and the language-only AUC.
   - Does the picture change with language-balanced directions?
   - Which (model, language, layer) cells qualify?

## D: causal test (GPU, VM `pt3-x1`; starts after A is DONE)

**Question:** does the language-specific over-refusal direction *cause* refusals of harmless prompts, and can it be
removed at one layer without making harmful prompts unsafe?

**Direction and layer:**
- **Direction:** d = `f_en⊥hg` of en_s0 from A, at **L_A** = the saved layer chosen by the §3 rule applied to A's numbers
  (English, en_s0). If no saved layer qualifies, use the layer maximising held-out AUC − |cos(f_en, h_en)|.
- **Comparison layer:** also run L = 18 (the steering layer), with d18 = `f_en⊥hg` at 18.
- **Fold rule:** use the direction fitted on fold 0 and test only on fold-1 seeds (the §2 fold), so the test prompts were
  not used to find d.

**Notebook:** `x_steer/notebooks/XD_causal.ipynb` (Colab GPU, HF `generate` with forward hooks; copy the hook code from
`steering/notebooks/S4_hfsteer.ipynb`). Greedy decoding, max_new_tokens 256; record that every arm uses 256. Model: merged
en_s0, bf16.

**Arms** (fold-1 seeds of the extraction set, each arm against the no-hook run on the same prompts):

| Arm | Prompts | Hook at layer L only, all positions |
|---|---|---|
| D0 baseline | all below | none |
| D1 remove | English twins | a ← a − (a·d) d |
| D2 remove (ja) | Japanese twins | same as D1 |
| D3 add | English twins that D0 answered | a ← a + κ d, where κ = mean projection of refused − answered English twins (fold 0) |
| D4 safety | English and Japanese clear harmful | same as D1 |
| D5 control | English twins + English clear harmful | remove a random unit direction orthogonal to h, h_en, g (fixed seed 0) |

Run each arm at L_A and at 18. Judge everything with the 0.6B judge. Report:
- refusal rate (twins) and unsafe rate (harmful) per arm;
- the flip counts vs D0, with an exact McNemar test;
- a seed-cluster bootstrap 95% CI.

**What would support hypothesis H:**
- D1 lowers English refusals clearly more than D5;
- D3 raises them;
- D2 changes Japanese little;
- D4 does not raise unsafe beyond D5.

State in RESULTS.md which of these held, at which layer.

**Outputs:** `PT3/x_steer/D/` (`answers.jsonl`, `judge.jsonl`, `results.json`, `report.md`, `RESULTS.md`, `manifest.json`,
`DONE`). Stop `pt3-x1`. **Report D now** (§5).

## Final message

Status of A and D; Drive links; all errors with fixes; deviations; and, in one paragraph, whether the evidence so far
supports H.
