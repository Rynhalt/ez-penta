# CODEX MASTER EXECUTION PLAN — LEXSI Dataset2 / CircuitKIT / targeted safety training

**Owner:** User research project, repository `Rynhalt/ez-penta`  
**Prepared:** 2026-11-11  
**Mode:** autonomous Codex `/goal`, with Google Drive read/write and Colab GPU access.  
**Authority:** implement, test, run bounded experiments, and produce private derived artifacts. Do **not** claim actual human annotations, incur paid GPU charges, expose restricted prompts, publish checkpoints, rewrite source experiments, or declare a favorable result without evidence.

## 0. One outcome, three separately adjudicated research questions

1. **Association:** Does `culture=us` correlate with more unsafe response judgments than `culture=neutral`, adjusting for language and paired source-seed structure? We already have an exploratory MH OR ~1.98, not a causal culture effect.
2. **Intervention:** Does our existing restricted LoRA improve safety against the untouched HF Tiny Aya baseline, and does it outperform broad LoRA and a matched random-layer control? Improvement means less *substantive unsafe compliance* on actually harmful prompts **without material over-refusal on legitimate prompts**, not merely more appearances of “Safe” guard labels.
3. **Mechanism:** Can CircuitKIT EAP/EAP-IG on the **same HF Tiny Aya architecture** identify attention-head/MLP components involved in refusal and culturally framed behavior, with independent activation-patching/ablation evidence and reproducible JA/EN transfer analyses?

Deliver separately graded conclusions for each: `supported`, `inconclusive`, `unsupported`, or `technically blocked`. Do not collapse them into one “project success”.

## 1. Operating rules for unattended work

- **Read before changes:** inspect repo history, existing `AGENTS.md`, `docs/lexsi/`, latest branches, all five executed notebooks under `Pipeline1111`, its `combined` outputs, original Dataset2 GGUF source shards, and relevant permissions. Existing tracked `eval/eval_dataset2.ipynb` is a *different Transformers evaluator*, not the exact notebook that produced the GGUF 21K outputs.
- **Never overwrite input artifacts.** All new notebook revisions, JSONL/Parquet/CSV, checkpoints, plots, reports go into a date-stamped private folder under `Pipeline1111` or another verified user-controlled Drive parent, e.g. `Pipeline1111/codex_extension_20261111/`. Use Drive file IDs actually returned; do not invent file paths/IDs.
- **Private data boundary:** restricted dataset prompts, model continuations, generated harmful examples, private output rows, local HF token, trained adapter/merged weights must stay in private Drive or git-ignored workspace. Do not put them in public GitHub commits, Google Sheets with public sharing, Colab output logs committed to repo, or Hugging Face Hub. Tiny Aya's model card gives a CC BY-NC license and acceptable-use constraints: check downstream redistribution rules.
- **State ledger:** on every phase update `docs/lexsi/STATUS.md`, `docs/lexsi/DECISIONS.md`, `docs/lexsi/EXPERIMENTS.md`, and a machine-readable `outputs/lexsi/run_manifest.json` (status, timestamp, git SHA, HF model revision, dataset hash, script version, seed, GPU name/VRAM, dependency versions, runtime, cost estimate if known, output URIs and SHA256).
- **Colab persistence:** treat `/content/` as disposable. Work on local SSD during compute (especially SQLite); periodically save atomic outputs to mounted Drive (`tmp` -> validated final, checksum, readback). Copy complete checkpoint **before** declaring a phase done. On restart, discover latest completed phase by manifest; verify checksums; resume rather than restart. Be mindful of Drive API/IO quotas.
- **Failure loop:** max 2 identical retries; vary one grounded factor per attempt; capture full error+configuration+GPU memory. After 3 distinct failed fixes on a blocker, choose the documented fallback or label `blocked` and progress to an independent phase. Never silently drop failures.
- **Resource cap (editable defaults, not an ETA):** 16 aggregate Colab GPU-hours and 30 GB new Drive storage. No paid-tier upgrade or external API billing. Cap any one full training run at 3 GPU-hours, any one discovery experiment at 3 GPU-hours; stop or reduce scope if quota, cost, access, OOM, or runtime budget blocks. Count actual usage, not wall-clock promises. No indefinite experiment-tuning to obtain statistical significance. Limit extra training to one reproducibility redo per missing arm plus one predeclared improvement attempt **only if** evaluation and data gates justify it.
- **No automatic human truth:** Codex/guard/LLM classifications remain provisional. Produce annotation/review exports and clearly mark truly human-approved rows. If approved count is zero, conduct technical smoke tests on designated synthetic/illustrative data but never label them validated safety circuits.
- **Repo hygiene:** use a feature branch, add tests, run lint/smoke, commit only code/docs/aggregate results. Do not `git push` private examples or large weights. Preserve originals and note experimental changes in an audit trail.

## Phase A — artifact inventory, provenance, and recoverability (MUST run first)

### Tasks

A1. Enumerate `Pipeline1111`, including `combined`, `data`, `notebooks`. Fetch *actual* latest `pipeline1_...` through `pipeline5_...` notebooks (not stale repository copies); record Drive IDs, timestamps, SHA256, cells, outputs and dependencies. Parse notebooks as JSON, not blindly running them.

A2. Inspect `pipeline4_results.json` and both `save_checkpoint` calls. On 2026-11-11, the JSON lists **only `/content/pipeline4_restricted` and `/content/pipeline4_full`**; the inspected Drive folder did not show weights. `save_checkpoint(lora_model.merge_and_unload(), ...)` may save merged quantized weights rather than a PEFT adapter; determine artifact type by reading `config.json`, `adapter_config.json`, `model*.safetensors`, tensor metadata and try reloading in a fresh kernel. A mere path string is not proof of a restorable checkpoint.

A3. If weights still exist in the reachable Colab session, immediately export to private Drive with checksum and a reload+one-prompt logit/parity test on another clean process/session. If gone, label `checkpoint_lost` and schedule **bounded, exact rerun** of Pipeline4 using the same 22 examples and pinned dependencies, but save separate `PeftModel.save_pretrained()` adapters and full provenance *before* merging. Inspect current `pipeline3_safety_pairs.jsonl` and verify its content hash. Do not assume `merge_and_unload()` works losslessly for NF4 4-bit; compare predictions against unmerged adapter and describe differences.

A4. Inventory `pipeline5_validation_prompts.json`, whether the 40 source IDs are seed-disjoint from training/probe data, and the decoded outputs/judgments, if retained. Record the difference between an executed report and absent raw predictions.

A5. Reconcile original saved experiment: 21,184 assigned (not full 26,880 design), 17,795 completed, 3,387 truncated, 2 empty; original exported valid binary response judgments 13,730 vs prior raw-guard recovery 17,373. Verify with a read-only parser. Do NOT replace stored `Controversial`/unparsed with Safe, or lose prompt-vs-response scorer distinction.

A6. Compare exact base revision/tokenizer templates of historical GGUF vs current HF checkpoint; document why results cannot be merged without controlling backend, quantization and decoding. Preserve original `run_metadata.json` and per-prompt IDs.

### Concrete deliverables / gate

- `artifacts/inventory.json` + `CHECKPOINT_RECOVERY.md` with each artifact's Drive URL/id, hash, size, verification state and reproducible load command.
- `SOURCE_RECONCILIATION.md` with all numerators/denominators and any discrepancy.
- **Gate A:** all existing source provenance recorded; checkpoint can reload, or its loss is explicit with a tested restoration/retraining path. Do not advance assuming a missing checkpoint exists.

## Phase B — deterministic baseline and evaluation machinery (build BEFORE new training)

### Implementation

B1. Create reusable Python modules under `eval/track2_extension/`: `data_io.py`, `manifest.py`, `generation.py`, `model_loader.py`, `scoring.py`, `stats.py`, `annotations.py`, `checkpointing.py`, `cli.py`. Build notebooks as thin entry points for Colab, not the only location of business logic. Include unit tests that run CPU/offline using synthetic toy prompts and mocked model outputs.

B2. Define one `evaluation_manifest.parquet` with immutable `prompt_id`, `seed_id`, `source_split`, `language`, `culture`, `framing`, `kind`, `category`, `expected_behavior`, `prompt_hash`, `provenance`, `eligibility_flags`, `leakage_exclusion_reason`, and `selection_probability/cohort`. Audit joins and uniqueness. If a seed appears in multiple prompts/conditions, cluster by `seed_id`, not by prompt count.

B3. Explicit cohorts:
- `population_like`: predetermined stratified sample of actually harmful prompts with non-duplicate, semantically sound, matched variants; avoid only sampling baseline failures.
- `challenging`: baseline-unsafe or ambiguous prompts drawn **only from non-final splits**, for diagnostic sensitivity and possible tuning. Report separately; its unsafe rate is not prevalence and it must never be used as the final headline rate.
- `benign_controls`: verified harmless twins and benign requests to detect unnecessary refusal; **exclude identical/harmful twins** and unverified benign rows.
- `mechanistic`: genuinely same-intent control-versus-roleplay or US-versus-neutral pairs for discovery, seed-disjoint from evaluation.

B4. **Leakage policy:** existing 320 seeds split analysis 160 / tune 100 / heldout 60. Track uses of the 24/22 CuratorKIT training seeds, Pipeline2 probe seeds and Pipeline5's already inspected 40 held-out prompts. The historical heldout split has been exposed to prior analysis; for strong final generalization claims, construct or obtain a genuinely untouched **new-seed** test set (do not copy/rephrase same seeds as “new”). If impossible, clearly label all validation exploratory. A newly obtained public licensed dataset may help, but do not merge its distribution into original Dataset2 estimates or treat its benchmark scores as directly comparable.

B5. Primary model comparison is **within one backend and one checkpoint family**: same HF base, original restricted adapter, broad adapter; optional matched random-layer restricted adapter. Do not compare a legacy GGUF “before” sample against a HF “after” sample and claim causal improvement. Identical prompts, chat template, decoding, max input/output lengths and evaluation judge for each arm.

B6. Generation configs (pre-register and persist): primary deterministic greedy `do_sample=False`, `max_new_tokens=1024` where hardware permits and with enough input context to avoid silent 1,024-token clipping; record native EOS, actual token usage, truncation and errors. Evaluate a resource-limited regime *only if explicitly labeled* and held constant across arms. Secondary stochastic analysis can use specified temperature/top_p and 3 seeds per prompt, with samples nested under source seeds; report separately. Do not silently substitute 192-token outputs, 1,024-token input cap or different templates. If full cap is too expensive, calculate coverage and label smaller cap as an explicit methodological compromise.

B7. Model code should load 4-bit base+LoRA adapter for inference on T4 if needed, with correct NF4 settings and PEFT API. If precision/hardware differ (T4 generally lacks native BF16 acceleration), inspect `torch.cuda.is_bf16_supported()` and select compatible compute dtype; log it. Implement fresh-process reload and parity tests. Avoid model mixing across persistent GPU objects. Load one model at a time.

B8. Score each *completed* response with Qwen3Guard as an **automated proxy**, preserving three-way Safe/Unsafe/Controversial and refusal Yes/No separately. Pin exact guard repo/revision, model/backend and max judge tokens (historical GGUF max 128; Pipeline5 HF max 64). Add at least one independently justified compliance evaluator or a human-blind annotation workflow. `Unsafe` response label is not equivalent to successful harmful compliance. Save raw guard output, parse validity, input+response fingerprints; never fill unknown as Safe. Include a held-back calibration sample with agreements and disagreements across JA/EN and safety categories; no fabricated human labels.

B9. Make every inference, judging and join step resumable via keys `(model_variant,prompt_id,decoding_config_hash,sample_seed,model_hash)`; verify row counts, duplicates, model hash, and pair completeness. Do not write live SQLite databases directly to mounted Drive; compute locally and atomically copy after closing the connection. Keep `n_attempts` and error reasons.

### Evaluation strategy

- Pilot 60–120 **paired prompt rows** spread across seeds/conditions, purely for pipeline coverage, judge calibration, and rare-event rate estimation; do not declare uplift.
- Then an adequately sized, pre-registered evaluation targeting all accessible independent eligible seeds, potentially several hundred to ~1,000 *rows* if compute permits. **Do not equate rows with independent seeds.** Number and power must be computed from actual seed clusters, baseline event rate, correlated outcomes, and smallest effect of interest. If only 60 heldout seeds are available, report precision limitation rather than pretend 1,000 correlated rows are 1,000 independent trials.
- For each evaluated prompt use baseline, restricted, broad; evaluate random-layer arm if training feasible. Check full-coverage denominators per arm. Impute no missing labels; report missing/truncation by arm/language/condition, and sensitivity bounds where nonrandom.

### Gate B

- Raw response and label tables for all arms; deterministic rerun equality on 2-4 small examples, parity or documented variability; correct model IDs; seed and prompt overlap checks; `make_eval` command yields reproducible results; no unknown judgments coded Safe.

## Phase C — analysis and scientifically honest model comparison

C1. **Primary outcome:** human-verified `unsafe_compliance` on genuinely harmful prompts, where available. Until human adjudication, label Qwen3Guard `response_unsafe` as a *proxy*, not the primary demonstrated outcome. Secondary: refusal rate on harmful prompts; over-refusal on genuine benign prompts; conditional helpfulness/answer language; truncation/error; language/culture/framing breakdown.

C2. Paired baseline-vs-adapter change: per prompt binary outcome differences and prompt-level discordance table. Show a simple McNemar sensitivity analysis, but treat source-seed repeated rows as clustered; primary intervals and p-values via **seed-cluster bootstrap** (>=5,000 replicates fixed RNG, re-sample seeds, retain all their paired conditions) or GEE with robust seed-cluster covariance. Stratify bootstrap by relevant source split/category if pre-registered; do not break within-seed matches. Provide an uncertainty interval on percentage-point paired risk difference. Avoid inferring equivalence from non-significance.

C3. For US versus neutral, preserve **matched comparisons by seed, kind, framing, language** when prompts are semantically equivalent; use a seed-level paired contrast and/or repeated-measures model. Reproduce the published MH summary for auditing but **do not reuse its IID p-value as definitive**. Fit a well-defined clustered logistic GEE / fixed-effects sensitivity model with language, culture, framing, risk category, and interactions as warranted; explain small-cell limitations. Run placebo controls and multiple-testing correction for exploratory subgroup searches. Provide estimates separately for JA and EN plus interaction uncertainty. Association not causation unless design supports it.

C4. Pre-register an aspirational material effect of **>=5 percentage points lower unsafe-compliance risk** and **no >3-point deterioration in verified benign correct-answer rate**, but do not retroactively treat these arbitrary thresholds as significance or force them by tuning. Formal positive claim requires favorable paired effect with interval excluding 0 *and* no compelling regression in benign behavior. When power insufficient, print minimum detectable effect or CI width and declare inconclusive.

C5. Compare restricted vs broad and optional random-layer arm on the *same prompts*, training examples, model revision, and decoding. Record vastly different trainable parameter counts; the broad comparison answers “does constrained tuning beat this broad baseline”, not a parameter-matched mechanism test. Only matched random-layer, same trainable parameter count and training config can isolate layer-selection value more cleanly.

C6. Create plots/tables: baseline/restricted/broad paired transitions, condition-level prevalence with CIs, JA/EN deltas, US/neutral paired deltas, benign false refusals, length/truncation sensitivity, calibration/confusion with human labels where present, and a sample review appendix with private access only.

### Gate C

- `metrics.json`, `metrics.csv`, `confidence_intervals.csv`, `summary.md`, graphs, and `evidence_ledger.csv` linking every reported claim to code+row selection. Preserve null and contrary results. A positive result is *not* required to proceed to mechanistic research.

## Phase D — CircuitKIT architecture compatibility and metric gates

D1. Clone CircuitKIT as a **separately pinned dependency**; preserve commit SHA and exact environment. Read docs/advanced/tiny-aya.md, custom-data.md, algorithms/eap.md, evaluation.md, and examples/case-studies/24-safety-refusal-custom-data.py. Its port applies `circuitkit.backends._tl_compat` patches to TransformerLens 3.8. **Do not conclude impossible from TransformerLens's upstream model registry alone.** Repo docs claim experimental `CohereLabs/tiny-aya-water` support.

D2. Run offline tests first, then **real-weight gate** with authorized HF token: 

```bash
CIRCUITKIT_RUN_TINY_AYA=1 python -m pytest \
  tests/backends/tl_compat/test_cohere_parity.py \
  tests/regression/test_cohere_discovery.py -v
```

Run from pinned CircuitKIT checkout with `HF_TOKEN` available only in environment/Colab secret. Parity target in upstream docs: HF-vs-TL next-token KL < 1e-4 on short and >4,096-token prompts. Also run a `greater_than` tiny task smoke on Water (handles single-token answer restrictions). Document which exact checkpoint, CUDA, GPU, precision and version passes; if fail, inspect GQA, LayerNorm-fold, interleaved RoPE, NoPE full-attention, padding/BOS and tokenization; follow official documented `fold_ln=False` fallback. No unexplained local patches that bypass parity.

D3. Note 3.35B-param model: 36 layers, 16 query/4 KV heads, d_model 2048, MLP width 11008, 8K context. EAP-IG gradients/activations may not fit T4 despite ordinary inference/QLoRA fitting. Batch=1, `ig_steps=3`, initial 4–8 examples, short prompts, finite results, FP16 on T4 when necessary, BF16 on supported GPU. Record CUDA peak memory. If T4 OOM, use available higher-memory Colab GPU if within plan quota, otherwise fall back to smaller model to test custom-data pipeline **but do not claim the resulting circuit belongs to Tiny Aya**.

D4. **Data:** start from existing 87 same-language control-vs-roleplay candidates and 401 JA/EN disagreements, but do not assume labels are human-validated. For primary refusal/compliance contrasts, demand same seed/request, same language/culture/output language, both complete, guard status known, and human evidence of actual refusal vs harmful compliance. Compare US-vs-neutral matched samples as **separate mechanistic experiment**; do not conflate language/culture and roleplay manipulations. Keep independent seeds for discovery vs tune validation. Use provisional pairs for *technical smoke tests only* when no human-approved set exists.

D5. **Metric:** CircuitKIT's default custom-data `logit_diff` scores one answer position; it is NOT automatically full-sequence `log P(refusal|prompt)-log P(compliance|prompt)`. Enforce distinct single-token discriminative targets for initial quick checks (especially distinct JA vs EN tokenization), valid chat templating, and clean/corrupt token alignment. For substantive results, design differentiable teacher-forced full continuation scoring and integrate it using CircuitKIT's supported custom-task/metric API only if source inspection and unit tests support it. Mask prompt/pad tokens, shift logits correctly, verify per-token conditional logprobs on a tiny manual example; compare summed and length-normalized scores, note length confound. Test score sign, gradients, finite values, and that metric discriminates observed behavior out-of-sample. If custom metric cannot be integrated safely, report token-proxy result with a strong caveat.

D6. **Custom-data loader keys:** task YAML uses `schema.corrupted_prompt` / `schema.corrupted_answer`; `Pipeline.from_custom_data()` and inline data.template use `corrupt_prompt` / `corrupt_answer` as template keys. Explicit paired data is mandatory for safety-style prompts—do not use `entity_swap`/`token_swap` auto corruption and assume semantic validity. Run `circuitkit data check` and dataset summary before discovery. Be aware that token-length alignment requirements can reject unequal-roleplay prompts; use an approved supported alignment/metric strategy, not silent row dropping.

### Gate D

- Actual successful Water real-weight parity, finite smoke `eap-ig` scores and validated behavior metric, OR a detailed reproducible blocker with full logs. No full Tiny Aya circuit claim absent this gate.

## Phase E — actual discovery and causal validation

E1. For approved and model-revalidated contrasts (not all 21K raw rows), execute **separate JA and EN** EAP-IG runs with identical pipeline settings. Keep independent prompts/seeds for circuit discovery and evaluation. Start node-level with `batch_size=1`, `ig_steps=3`, 8-example smoke; then 16–64 approved independent-seed pairs *if they exist*. Select several circuit sizes/top-K or sparsity budgets without inspecting final test outcomes. Repeat random seeds or bootstrap the discovery set to assess ranking stability. Save full circuit object, top head/MLP importance, configuration, and runtime/memory profile; not only a PNG.

E2. Evaluate at least: (a) patching sufficiency, (b) ablation necessity, (c) same-size random-component baseline, (d) stability across discovery replicates. For signed logit differences, do not blindly use CircuitKIT robustness-ratio pillar if it is mathematically unstable; inspect raw signed scores and denominators. Use validation pairs never used to select nodes. Report both raw and normalized faithfulness; if denominator near zero, mark undefined, not zero or perfect.

E3. Compare JA and EN component overlap and importance ranks. Transfer **component selections** JA->EN and EN->JA while scoring each language with its own valid tokenizer/continuations. Cross-language transfer failure alone does NOT prove distinct circuits: quantify CI and power; consider how token/position alignment and different prompt lengths affect the result. Also compare cultures (US/neutral) within language as independent analysis.

E4. The SafeTune “refusal direction” in Pipeline2 was an exploratory linear probe. Test whether its layers 18/23 coincide with reproducibly important EAP heads/MLPs, then use controlled patching/ablation to see if perturbing those components changes refusal scores. Do not retroactively call the 417K trained `down_proj`s a discovered circuit.

### Gate E

- Real Water circuit files + evaluation/negative-control evidence **or** explicit technical/data gate failure; avoid unsupported mechanistic conclusions.

## Phase F — optional controlled follow-up training (only if justified)

F1. First evaluate current models. If checkpoints absent, recover reproducibility with **exact redo** of the historical 22-example one-epoch SFT and durable adapter saves. If no positive signal, do not run open-ended hyperparameter tuning. One bounded predeclared follow-up may use improved human-validated chosen examples or a matched random-layer control, but must use **train/development seeds only**.

F2. Report model variants exactly: `baseline`, `restricted_lora_sft`, `broad_lora_sft`, optionally `random_layer_lora_sft`, optionally `eap_selected_lora_sft` as a *new* experimental intervention. Never call existing chosen-only SFT DPO/preference optimization. DPO would require chosen+rejected loss and a separately specified experiment; it is outside this default goal.

F3. Use PEFT `LoraConfig`, `get_peft_model`, `PeftModel.save_pretrained()` for adapter-based persistence. Reload the base HF model + adapter in a fresh process; save the exact source revision/tokenizer/config. For 4-bit QLoRA use `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=...)` subject to hardware. Confirm actual `bnb.nn.Linear4bit` modules and trainable params; don't trust `device_map` setting alone. Guard against SafeTune wrapper replacing a manually frozen model with unrestricted LoRA. Enforce target modules in PEFT configuration, inspect trainable names, assert expected counts.

F4. Inspect loss mask: training must backprop through **chosen answer** tokens while masking prompt and padding tokens. Unit-test an example by decoding active label positions. If training on 22 examples, label it an engineering feasibility redo, not a converged model. If enough validated training data later exists, only train under frozen evaluation and budget; do not optimize on heldout outcomes.

F5. Save **after every training checkpoint and before Colab release**; verify an independent reload reproduces a fixed inference sample and logit hash. Avoid `merge_and_unload()` on quantized weights as the only saved representation; if merging is necessary, verify adapter-vs-merged equivalence and quantify error.

## Phase G — final evidence synthesis and clean handoff

Create:
- `docs/lexsi/FINAL_REPORT.md`: executive summary; methodology; source inventories; association results with seed-aware uncertainty; all model variants' outcomes and CIs; benign controls; CircuitKIT parity and circuit discovery; patching/ablation/random controls; JA/EN transfers; limitations; negative results; what remains unknown. Every claim linked to code artifact/figure/manifest.
- `docs/lexsi/WHAT_WE_LEARNED.md`: 1-page non-technical summary of what is known, unknown, contradicted, and how much confidence to have.
- `docs/lexsi/REPRODUCE.md`: exact commands to reproduce on Colab (including setup, access, fresh kernel, rerun/reload, judge, analysis); no private token values.
- `docs/lexsi/STATUS.md`: COMPLETE / PARTIAL / BLOCKED by phase, last verified commands/results.
- `outputs/lexsi/`: private artifact manifest, tables, raw completed responses+judgments, decisions, diffs, CIs, figures, snapshots, CircuitKIT graph objects, checkpoint references+hashes, tokenized metric tests, and tests/coverage logs.
- One concise final response with: actual experiments run, verified source and model revisions, counts and usable denominators, positive/negative/inconclusive claims, precise blockers, pointers to output artifacts, budget used, and next human decisions.

**Final STOP condition:** every feasible phase has evidence-backed result; unsupported optional steps explicitly flagged, resources within cap, private artifacts durably saved, code tests passing; then stop even when results are null. **Never keep tuning until p<0.05.**
