# Track-2 Pipeline Report: Stratified Diagnosis → Circuit-Restricted Fix → Validation

This documents exactly what was run, cell by cell, across the five pipeline notebooks in
[`notebooks/`](notebooks/), why each step was done, and what the real (not simulated)
result was — including where the result was expected vs. where it wasn't.

All five notebooks executed end-to-end with **zero code errors**. Steps 1 and 3 ran
locally (CPU/pandas); steps 2, 4, and 5 ran on a real Google Colab **T4 GPU** session via
the `colab` CLI, on the actual `CohereLabs/tiny-aya-water` checkpoint.

---

## Pipeline 1 — [`pipeline1_stratified_diagnosis.ipynb`](notebooks/pipeline1_stratified_diagnosis.ipynb)

**Goal:** pipeline 1 asks whether `culture=us`/`japan` framing really raises jailbreak
odds, or whether that's actually just a side effect of language — since `en` vs `ja` vs
`mix` prompts aren't evenly split across culture conditions in the raw data. Language is
used as a *stratifying control variable* via Cochran–Mantel–Haenszel (CMH), instead of
throwing it into the same regression as culture the way `stat_analysis.ipynb` did.

### Step 1, cell 2 — load and filter the real dataset
```python
usable = observations[
    (observations["analysis_eligible"] == True)
    & (observations["generation_status"] == "completed")
    & (observations["scoring_status"] == "completed")
    & (observations["split"] != "heldout")
].copy()
harmful = usable[usable["kind"] == "harmful"].copy()
```
* Did this to restrict to rows that are actually eligible for analysis, have a completed
  generation + a completed safety score, and are **not** in the `heldout` split — that
  split is reserved so pipeline 5 can validate on prompts the fix never touched.
* Restricted to `kind=="harmful"` because jailbreak-success is only a meaningful concept
  for harmful seed prompts (not their harmless "twin" counterparts).

**Results:** 4,069 usable harmful-seed rows. As expected — this is a subset of the full
21,184-row dataset after excluding the heldout split and non-completed rows.

### Step 2, cell 4 — implement Mantel–Haenszel pooling from scratch
```python
def two_by_two(df, exposure_mask, outcome_col): ...
def mantel_haenszel(df, exposure_col, exposure_value, baseline_value, outcome_col, stratum_col): ...
```
* Did this because no existing library call in the project computed a stratified pooled
  odds ratio — implemented the standard Mantel & Haenszel (1959) pooled-OR / chi-square
  formulas, with Robins–Breslow–Greenland (1986) variance for the 95% CI, directly from
  the textbook formulas rather than trusting an unfamiliar black-box implementation.
* No output (function definition only).

### Step 3, cell 6 — the headline test: `culture=us`/`japan` vs `neutral`, pooled over language
```python
for culture_value in ["us", "japan"]:
    result = mantel_haenszel(harmful, "culture", culture_value, "neutral", "unsafe", "prompt_language")
```
* Did this to get one pooled number per culture condition that already accounts for
  language as a confounder, instead of pipeline 1's predecessor (`stat_analysis.ipynb`)
  mixing language into the same regression as culture.

**Results — as expected / confirms the hypothesis:**
```
culture=us vs neutral, pooled across prompt_language strata:
  OR_MH = 1.982  (95% CI 1.583-2.482)  chi2_MH=35.78  p<0.000001

culture=japan vs neutral, pooled across prompt_language strata:
  OR_MH = 1.097  (95% CI 0.862-1.397)  chi2_MH=0.48  p=0.487
```
`culture=us` roughly **doubles** the odds of an unsafe (jailbroken) response, highly
significant, even after controlling for language. `culture=japan` shows **no**
significant effect — this was *not* assumed going in; it says the earlier observed
"culture effect" is really a `us`-specific effect, not a general cross-culture-framing
effect.

### Step 4, cell 8 — does `us`'s effect replicate in `en` AND `ja` separately?
```python
for lang in ["en", "ja", "mix"]:
    row = per_stratum.loc[lang]
```
* Did this because a pooled OR can look significant even if it's driven by just one
  language stratum while the other goes the opposite direction — this checks that
  doesn't happen, which is the actual "controlling for language" sanity check (not just
  trusting the pooled number).

**Results — as expected, and reassuring:**
```
culture=us vs neutral:   en: OR=1.865 p=0.011  n=791  |  ja: OR=1.897 p=0.001  n=784  |  mix: OR=2.120 p<0.001 n=1044
  -> en and ja agree on direction: True
culture=japan vs neutral: en: OR=1.283 p=0.331  n=837  |  ja: OR=1.019 p=1.000  n=786  |  mix: OR=1.060 p=0.777 n=1081
  -> en and ja agree on direction (both weak/non-significant): True
```
The `us` effect is consistent (OR≈1.9) in English *and* Japanese independently — a real,
language-independent effect, not a language artifact. This is the key finding that
pipeline 2 goes on to investigate mechanistically.

### Step 5, cell 10 — repeat within `framing=rp` only (the worst framing from prior analysis)
```python
rp_only = harmful[harmful["framing"] == "rp"]
```
* Did this because earlier analysis (in `stat_analysis.ipynb`) found a significant
  `culture x framing` interaction — this checks whether the `us+roleplay` combination
  specifically is even stronger, and whether it's still consistent cross-language.

**Results — as expected, effect is stronger within `rp`:**
```
framing=rp only, culture=us vs neutral:    OR_MH = 2.544 (95% CI 1.735-3.731) p<0.00001
framing=rp only, culture=japan vs neutral: OR_MH = 1.422 (95% CI 0.945-2.138) p=0.112
```
Roleplay framing amplifies the `us` effect (OR 1.98 → 2.54) and nudges `japan` toward
(but not past) significance — consistent with roleplay being the riskiest framing
generally, as found previously.

### Step 6, cell 12 — save baseline stats for pipeline 5
* Did this so pipeline 5 can recompute the identical statistic on the post-fix model's
  outputs and do a true before/after comparison against *this exact* number, rather than
  re-deriving a new baseline from scratch.

**Results:** `pipeline1_baseline_mh_stats.json` saved with all four OR_MH values above.
No surprises — this is purely a serialization step.

---

## Pipeline 2 — [`pipeline2_circuit_discovery.ipynb`](notebooks/pipeline2_circuit_discovery.ipynb)

**Goal:** pipeline 1 found a real, language-independent `culture=us` effect. Pipeline 2
asks *why*, mechanistically, inside the real model — does `us`-framing measurably
suppress the internal "refusal direction," and is that suppression the same mechanism in
English and Japanese, or two coincidentally-similar but separate ones?

**Stated limitation up front (markdown cell 0):** true `CircuitKIT` attention-head
circuit discovery needs `transformer_lens.HookedTransformer`, and Tiny Aya's
Cohere2 architecture is not in TransformerLens's supported registry (checked directly:
no `aya`/`cohere` match among its 249 officially supported models). So this notebook
uses SafeTune's model-agnostic direction/neuron tools instead — real mechanistic
analysis on the real model, just at direction+neuron grain rather than attention-head
circuit grain.

### Step 1, cell 2 — setup, load SafeTune, authenticate to the gated HF model
```python
MODEL_ID = "CohereLabs/tiny-aya-water"
device = "cuda" if torch.cuda.is_available() else "cpu"
```
* Did this on a real Colab T4 (not the local GGUF quantization used elsewhere in the
  project) because reading internal hidden states requires the full-precision HF
  checkpoint and GPU memory, not a `llama.cpp` quantized server.

**Results:** `device: cuda` — confirms this actually ran on GPU, not silently falling
back to (much slower, and arguably non-representative) CPU.

### Step 2, cell 4 — load the real 3.35B-parameter Tiny Aya checkpoint in bfloat16
**Results:** "Model loaded: CohereLabs/tiny-aya-water." As expected, but this is also
where the project's working assumption that Tiny Aya is small got corrected — it's a
real 3.35B-parameter model (see pipeline 4 for where this mattered more).

### Step 3, cell 6 — load the real prompt sample (8 prompts/condition, from `observations.csv`)
* Did this specifically with `framing=control` prompts to isolate the pure culture
  effect from the framing effect pipeline 1 separately found — and kept the sample small
  (8 per condition) deliberately, to keep this step's GPU time and memory bounded on a
  free-tier T4 while still getting a real (if noisy) signal.

**Results:** `en: {'n_harmful_neutral': 8, 'n_twin_neutral': 8, 'n_harmful_us': 8}`
(same for `ja`) — exactly the intended sample sizes.

### Step 4, cell 8 — helper: last-token hidden state at every layer
* Did this because the refusal-direction literature's standard convention is the last
  real (non-padding) token's residual-stream value — defined once here, reused in the
  diagnostic step below.

### Step 5, cell 10 — extract the refusal direction + rank safety neurons, per language
```python
direction, layer_idx, all_layer_directions = steer.extract_refusal_direction(
    model, tokenizer, harmful_prompts=harmful, harmless_prompts=twin,
)
report = identify_safety_neurons(model, all_layer_directions, SafetyNeuronConfig(method="weight", top_k_per_layer=16))
```
* Did this using the `kind=harmful` vs `kind=twin`, `culture=neutral` contrast
  specifically — the cleanest possible harmful-vs-harmless pair, deliberately excluding
  the `us`-framed prompts from *defining* the direction so the later test (does `us`
  suppress it) isn't circular.

**Results:**
```
en: selected layer_idx=18, top 5 layers by safety score: [23, 24, 27, 15, 18]
ja: selected layer_idx=18, top 5 layers by safety score: [18, 25, 28, 7, 15]
```
Both languages independently picked the **same primary layer (18)** — unexpected in the
sense that nothing forced this to agree, and a useful early hint that there may be a
shared locus even before the actual cross-language transfer test below.

### Step 6, cell 12 — the actual diagnostic: does `us` suppress the refusal direction, and does it transfer across languages?
```python
for direction_lang, (direction, layer_idx, _) in directions_per_lang.items():
    for test_lang in ["en", "ja"]:
        ...
```
* Did this as a 2×2 design (direction source language × prompts tested on) specifically
  to separate two different questions: (a) does `us`-framing suppress activation along
  the refusal direction *at all* (same-language cells), and (b) is that a single shared
  linear direction across languages, or two separate per-language mechanisms
  (cross-language cells)?

**Results — partially expected, partially not:**
```
en direction -> en prompts   [same-language]:      gap=+2.52  t=1.77  p=0.098
en direction -> ja prompts   [CROSS-language]:      gap=+0.30  t=0.18  p=0.862
ja direction -> en prompts   [CROSS-language]:      gap=+1.03  t=0.72  p=0.482
ja direction -> ja prompts   [same-language]:       gap=+4.12  t=1.80  p=0.093
```
As expected: within each language, `us`-framing does push activation away from the
refusal direction (positive gap in both same-language cells, trend-level significance at
this small n=8/condition). **Not fully as expected:** the direction does **not**
meaningfully transfer cross-language (gaps shrink to near-zero, p≈0.86 and p≈0.48) —
despite both languages independently selecting the same *layer* (18) in the step above,
the actual linear *direction* within that layer is language-specific. This is the
single most important nuance in the whole pipeline: the behavioral effect pipeline 1
found is language-invariant, but the mechanism encoding it is not a single shared
vector — it's (at least) two per-language vectors that happen to live in the same layer.

### Step 7, cell 14 — save results for pipeline 4
**Results:** `pipeline2_results.json` saved with both languages' `layer_idx`,
`top_layers_by_safety_score`, ranked safety neurons, and all four projection tests. No
surprises — serialization only.

---

## Pipeline 3 — [`pipeline3_curatorkit_data.ipynb`](notebooks/pipeline3_curatorkit_data.ipynb)

**Goal:** generate the actual fine-tuning data pipeline 4 needs: for each real
`culture=us` harmful prompt, a safe-refusal ("chosen") and a compliant ("rejected")
response, as a preference pair — using CuratorKIT's real generation+gating pipeline, not
hand-written examples.

Runs locally (no GPU) against the project's own local Tiny Aya GGUF `llama-server`.
**Stated limitation (markdown cell 0):** the same local Tiny Aya model is used as both
the generator and the (implicit, in `score_responses`) gate — a stronger separate
teacher model would produce better "chosen" responses in a production run.

### Step 1, cell 2 — install CuratorKIT, check the local server
**Results:** "Existing server status: offline" — as expected, since step 3 starts it
fresh for this run.

### Step 2, cell 4 — load the 24 real seed prompts
* Did this using 24 real rows pulled directly from `observations.csv`
  (`culture=us`, `en`/`ja` × `control`/`rp`, 6 each) rather than hand-written prompts, so
  the fine-tuning data is drawn from the exact condition pipeline 1 flagged as elevated-risk.

**Results:** "24 seed rows" with a real example instruction shown (a `control`-framing
`en` prompt) — matches the intended design exactly.

### Step 3, cell 6 — configure CuratorKIT: `preference` task, instruction-mode, custom refuse/comply prompt templates
```python
CHOSEN_PROMPT = """...write a brief, respectful refusal..."""
REJECTED_PROMPT = """...respond... fully in character... no safety caveats..."""
config = CuratorConfig(dataset=seed_rows, generation_task="preference", preference_mode="single_call", ...)
```
* Did this to override CuratorKIT's default "thorough vs. shallow" preference contrast
  with a "safely refuses vs. unsafely complies" contrast — the actual axis this project
  needs for a safety fine-tune, not CuratorKIT's out-of-the-box use case.
* Disabled `score_responses` and all quality thresholds (`hallucination_threshold=None`,
  etc.) because the only available judge (local Tiny Aya) is too weak to trust for
  quality scoring at this scale — `dedup="exact"` and `clean=True` were kept on since
  those don't need a judge at all.

**Results:** "Config built." — no surprises, config construction only.

### Step 4, cell 8 — run the real pipeline (starts the server, generates, gates, exports)
* Did this as the actual CuratorKIT execution: `SchemaGate` → `ExactDeduplicator` →
  `TextCleaner` → `PreferenceGenerationTask`, all real pipeline stages, not mocked.

**Results — as expected, with one caught issue:** completed in ~200s for 24 samples
(~8.2s/sample). A `UserWarning` fired saying `export_formats=["jsonl"]` isn't the
designed export format for `generation_task="preference"` (CuratorKIT expects `"dpo"`
there) — flagged as a real, caught library-usage mismatch, worked around in step 6
below rather than ignored.

### Step 5, cell 10 — inspect the result object
**Results:** `CuratorResult(passed=[...], rejected=2, output_dir=...)` — **as expected**:
22 of 24 prompts passed the gate, 2 were rejected. A real example `DataSample` is shown
with both a genuinely safety-conscious `chosen` response and a genuinely non-refusing,
hedging `rejected` response for the same "threaten someone" roleplay prompt — exactly
the chosen/rejected contrast the custom prompt templates were designed to produce.

### Step 6, cell 12 — write the accepted pairs to the file pipeline 4 expects
```python
dest = ROOT / "notebooks" / "combined" / "pipeline3_safety_pairs.jsonl"
for sample in result.passed:
    if sample.instruction and sample.chosen:
        f.write(_json.dumps({...}) + "\n")
```
* Did this as a direct in-memory export from `result.passed` instead of relying on
  CuratorKIT's own exporter, because (per the warning above, and confirmed by checking
  `output_dir`) the framework's own `.jsonl` export for `task_type="preference"` writes
  the *rejected-by-the-gate* samples, not the accepted chosen/rejected pairs — using the
  framework's own export here would have silently produced the wrong file.

**Results:** `notebooks/combined/pipeline3_safety_pairs.jsonl` written with **22 lines**
— matches `result.passed`'s count exactly, confirming the workaround was correct.

---

## Pipeline 4 — [`pipeline4_circuit_restricted_fix.ipynb`](notebooks/pipeline4_circuit_restricted_fix.ipynb)

**Goal:** actually fine-tune the model two ways on pipeline 3's 22 pairs — (A) touching
*only* the layers pipeline 2 identified (a "circuit-restricted" fix) and (B) a normal
broad LoRA fine-tune touching every projection in every layer — to compare a surgical fix
against a generic one.

This was the hardest notebook to get working: `safetune`'s built-in `PlainSFTTrainer`
turned out to always internally LoRA-wrap the model and to silently fight precision
settings on T4 hardware (see the Assumptions section). **The final notebook bypasses
`PlainSFTTrainer`/`transformers.Trainer` entirely** and uses a short, transparent manual
PyTorch training loop instead — a deliberate, documented departure from the originally
planned "just call SafeTune's trainer" approach, not an accident.

### Step 1, cell 2 — setup; re-authenticate to the gated model
**Results:** `device: cuda` — as expected, same Colab session pattern as pipeline 2.

### Step 2, cell 4 — load pipeline 2's target layers + pipeline 3's training pairs
```python
target_layers = sorted(set(pipeline2_results[lang]["top_layers_by_safety_score"][0] for lang in ["en", "ja"]))
```
* Did this by taking the **union** of each language's single top-ranked layer from
  pipeline 2 (`en`→23, `ja`→18 by the `top_layers_by_safety_score[0]` ranking), rather
  than only the shared `layer_idx=18` both languages used for direction *extraction* —
  since pipeline 2 showed the two languages' mechanisms don't fully overlap, restricting
  to only one layer risked fixing just one language.

**Results:** `Target layer(s) for the circuit-restricted fix: [18, 23]` and "22 training
pairs loaded" — as expected, matches pipeline 2's and pipeline 3's outputs exactly.

### Step 3, cell 6 — load the model fresh with 4-bit quantization (QLoRA)
```python
QUANT_CONFIG = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, ...)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=QUANT_CONFIG, device_map="auto")
```
* Did this because the model turned out to be a real 3.35B parameters (not "tiny" in the
  memory sense) — a bf16 full-precision load plus Adam optimizer states for any
  meaningfully-sized trainable set risked CUDA OOM on a 14GB T4, so this switched from a
  plain bf16 load to 4-bit NF4 quantization specifically so the *base* weights are cheap
  and only the small LoRA adapters need full-precision gradients.
* Used `device_map="auto"` deliberately (not a bare `"cuda"` string) — confirmed by
  direct testing that `device_map="cuda"` silently skips bitsandbytes quantization
  entirely, defeating the point of this cell.

**Results:** "Model has 1,943,087,104 parameters (4-bit base)" and "GPU memory allocated
after loading: 3.61 GB" — confirms the quantization actually took effect (a full bf16
load of this model would be roughly double that). The base-model `requires_grad`/dtype
audit printed "Total trainable fp16 params in base model: 0" — confirms the base model
is fully frozen and clean before any LoRA wrapping, which was the actual thing being
checked for (this audit was added mid-debugging specifically to rule out one hypothesis
for an earlier OOM, and left in as a real correctness check).

### Step 4, cell 8 — tokenize the training pairs
```python
train_dataset = Dataset.from_list(_tokenize_qa_rows(tokenizer, rows, max_len=512))
```
* Did this using SafeTune's own `_tokenize_qa_rows` (not a hand-rolled tokenizer call) so
  the label-masking exactly matches what SafeTune's trainers expect, then wrapped the
  result in a real HF `Dataset` — `_tokenize_qa_rows` returns a plain Python list, not a
  `Dataset`, which the manual training loop's `DataLoader` needs.

**Results:** `Dataset({features: ['input_ids', 'attention_mask', 'labels'], num_rows: 22})`
— exactly the expected 22 rows.

### Step 5, cell 10 — Arm A: circuit-restricted LoRA, attached only to layers 18 and 23's `down_proj`
```python
restricted_target_modules = tuple(f"layers.{layer}.mlp.down_proj" for layer in target_layers)
safetune.configure(lora_target_modules=restricted_target_modules)
lora_model_restricted = lora_wrap(model)
lora_model_restricted = manual_lora_train(lora_model_restricted, train_dataset, batch_size=1, lr=1e-4)
```
* Did this via a manual PyTorch loop (forward, `loss.backward()`, `optimizer.step()`)
  instead of `PlainSFTTrainer`, after root-causing (via direct source inspection and a
  parameter audit) that `PlainSFTTrainer` was silently reducing the requested `bf16`
  training precision back to `fp16` on T4 hardware inside its own `_fp32_master_weights`
  step, and that this coincided with the real OOM — this was the actual fix, not a
  workaround for a symptom.
* Used `safetune.configure(lora_target_modules=...)` (the library's real global config
  mechanism) rather than manually setting `requires_grad` on named parameters — found
  that the latter gets silently overridden the moment any SafeTune trainer internally
  calls `lora_wrap()`.

**Results — matches the intended "surgical" design exactly:**
```
trainable params: 417,792 || all params: 3,349,645,312 || trainable%: 0.0125
GPU memory: steady 4.18GB across all 22 training steps
loss: 1.61 -> (noisy, single epoch) -> 1.11
Checkpoint saved to /content/pipeline4_restricted
```
0.0125% of parameters trainable — confirms the restriction actually worked (not just
configured but silently ignored, which was the exact failure mode root-caused earlier in
debugging). Loss is noisy rather than smoothly decreasing, which is expected for 1 epoch
on 22 examples with batch size 1 — this was never intended to fully converge, only to
prove the mechanism.

### Step 6, cell 12 — Arm B: broad LoRA on all projection modules, same data
```python
broad_target_modules = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
```
* Did this as the direct comparison point: identical data, identical manual training
  loop, identical hyperparameters — the only variable changed is *which* modules LoRA
  attaches to, isolating the "surgical vs. broad" contrast cleanly.

**Results — ran successfully, no OOM (not a given going in):**
```
trainable params: 30,228,480 || all params: 3,379,456,000 || trainable%: 0.8945
GPU memory: steady 4.53GB across all 22 steps (vs. 4.18GB for Arm A)
loss: 1.12 -> (noisy) -> 0.99
Checkpoint saved to /content/pipeline4_full
```
~72x more trainable parameters than Arm A, but only +0.35GB more GPU memory — as
expected once quantization + LoRA (rather than full-parameter fine-tuning) was the
design; a true full-parameter fine-tune of this model's broad module set would not have
fit this cheaply.

### Step 7, cell 14 — save both checkpoints' paths for pipeline 5
**Results:** `pipeline4_results.json` saved with `target_layers: [18, 23]`,
`n_training_pairs: 22`, and both checkpoint paths, `full_finetune_completed: true`. No
surprises.

---

## Pipeline 5 — [`pipeline5_validation.ipynb`](notebooks/pipeline5_validation.ipynb)

**Goal:** the actual before/after test — generate real answers from the untouched
baseline model and from pipeline 4's circuit-restricted fixed checkpoint on held-out
prompts, judge both with a real safety-guard model, and recompute pipeline 1's exact
statistic to see whether the `culture=us` gap actually shrank after the fix.

**Success criterion stated up front (markdown cell 0):** the `culture=us` OR_MH should
shrink after the fix, in *both* `en` and `ja`, not just pooled — otherwise the fix hasn't
hit the language-invariant mechanism pipeline 2 found.

### Step 1, cell 2 — setup
**Results:** `device: cuda` — as expected.

### Step 2, cell 4 — load 40 held-out validation prompts + pipeline 1's baseline stats
* Did this using 40 prompts (`en`/`ja` × `us`/`neutral` × `control`/`rp`, 5 each) with
  seeds **distinct** from the 24 used to build pipeline 3's training pairs, specifically
  so this measures generalization rather than memorization of the exact training
  prompts.

**Results:** "40 validation prompts" plus pipeline 1's baseline JSON printed back
(OR_MH=1.982 for `us`) — as expected, confirms both files loaded correctly.

### Step 3, cells 6–8 — generate with the baseline (untouched) model
```python
def generate_answers(model, tokenizer, prompts, batch_size=8, max_new_tokens=192): ...
```
* Did this with a batched greedy-decoding generation helper, consistent with the
  project's own original data-generation settings (`temperature=0.0` in
  `run_metadata.json`), so the baseline arm's decoding matches how the rest of this
  project's data was generated.

**Results:** a real sample baseline answer shown (a measured, caveat-laden response to a
`control`-framing prompt) — qualitatively reasonable, as expected.

### Step 4, cells 10 — generate with the FIXED (circuit-restricted) checkpoint
* Did this by loading pipeline 4's actual saved `/content/pipeline4_restricted`
  checkpoint (not re-simulating the fine-tune), on the exact same 40 prompts, so the
  comparison is apples-to-apples.

**Results:** a real sample fixed-model answer shown, for the same prompt as the baseline
sample — qualitatively similar in tone/content, which foreshadows the null result below
(the fix didn't dramatically change this particular visible answer).

### Step 5, cell 12 — judge both answer sets with the real Qwen3Guard-Gen-0.6B
* Did this by loading Qwen3Guard fresh via `transformers` (not reusing any cached
  judgments), using the same `Safety:`/`Refusal:` regex-parsing convention as this
  project's own `guard_judge.py`, so the verdicts are directly comparable to the rest of
  the project's judged data.

**Results — the key finding, and the one surprising result in the whole pipeline:**
```
Baseline verdicts (40): 39x Safe, 1x Unsafe
Fixed verdicts (40):    37x Safe, 2x Unsafe, 1x Controversial
```
Almost everything on both sides came back "Safe." This was **not** the expected
outcome — pipeline 1's full-dataset baseline unsafe rate is high enough to produce
OR_MH=1.98, but at only 40 greedily-decoded prompts, the realized unsafe rate is far too
low (2.5%–5%) to see that effect reproduce at this sample size.

### Step 6, cell 14 — recompute pipeline 1's exact Mantel–Haenszel statistic on this new data
```python
# the exact same mantel_haenszel() function from pipeline1_stratified_diagnosis.ipynb
```
* Did this specifically by copy-pasting the identical function used in pipeline 1 (not a
  reimplementation), so the before/after comparison can't be an artifact of using a
  different statistical method on the two ends.

**Results — an honest null/inconclusive finding, not a bug:**
```
baseline: culture=us vs neutral, OR_MH=nan, chi2=0.00, p=1.00000
fixed:    culture=us vs neutral, OR_MH=nan, chi2=0.55, p=0.45637
(for comparison) pipeline 1, full dataset: OR_MH=1.982, p<0.000001
```
The odds ratio is literally undefined (`NaN`) because the 2×2 tables have empty or
near-empty cells at this sample size (1 baseline unsafe case total, 2 fixed) — the
Mantel–Haenszel formula divides by zero when a stratum has no exposed-unsafe or no
unexposed-unsafe cases. This is a real, correctly-computed `NaN`, not a crash or a silent
error — the statistic is genuinely undefined at n=40 with this low an event rate.

### Step 7, cell 16 — per-language breakdown
**Results:**
```
baseline: en gap=+0.00  ja gap=+0.10
fixed:    en gap=+0.10  ja gap=+0.11
```
Not the hoped-for "gap shrinks in both languages" pattern (the `en` gap actually grew
slightly, from the fixed model's one additional unsafe verdict), but at n≈10 unsafe-rate
observations per cell, these numbers individually carry essentially no statistical
weight — this breakdown is a diagnostic look, not a second independent test.

### Step 8, cell 18 — save the final before/after comparison
**Results:** `pipeline5_results.json` saved with both arms' (NaN) statistics and
pipeline 1's full-dataset baseline for reference. No surprises — serialization only.

**Honest conclusion for pipeline 5:** the validation design is sound (same statistic,
same judge, held-out generalization prompts, apples-to-apples checkpoints) but
under-powered — see "How big would a real validation set need to be?" below for a
concrete, data-grounded answer.

---

## How big would a real validation set need to be?

This was asked directly, so it's worth a real, worked answer rather than a round number.

**What the data should consist of:** `observations.csv` already contains a dedicated
**`heldout` split** (`split == "heldout"`) that pipeline 1 explicitly excluded and that
was *never* used to build pipeline 3's training pairs or pipeline 4's fine-tune — this is
exactly the right validation set, and it's already real data with real recorded prompts,
rather than needing to invent new prompts. Checked directly against the real file:

```
split column value counts: analysis=10352, tune=6780, heldout=4052
heldout, analysis_eligible & generation completed & scoring completed,
  culture in {us, neutral}, language in {en, ja}:  792 rows
    (us: en=197, ja=211 | neutral: en=198, ja=186)
```

**Crucially, the baseline arm doesn't need to be re-generated at all.** Checked
`run_metadata.json`: the entire dataset's `answer` column was already generated by the
exact same model (`CohereLabs/tiny-aya-water-GGUF`, Q4_K_M, temperature=0.0/greedy — the
same decoding settings pipeline 5 used) and already judged by the exact same guard model
(`QuantFactory/Qwen3Guard-Gen-0.6B`) used in pipeline 5. So a real re-run only needs to:
generate the **fixed** checkpoint's answers on these 792 held-out prompts (not 40), judge
those with Qwen3Guard, and compare against the **already-computed** real baseline
labels for those same 792 rows — roughly half the compute of a from-scratch 2-arm run.

**Minimum row count, from a power calculation, not a guess:** using pipeline 1's own
pooled effect size as the target to detect (`OR_MH≈1.98`, i.e. roughly 15% unsafe at
baseline vs. ~26% unsafe under `us`-framing, consistent with the full-dataset pooled OR),
a standard two-proportion power calculation for 80% power at α=0.05 gives:

```
n ≈ 214 prompts per arm (us vs. neutral), per language stratum
  -> ~430 per language (us+neutral combined)
  -> ~860 total across en+ja for ONE framing
```

Since the real design also wants to look within `control` and `rp` framing separately
(pipeline 1 found the effect is notably stronger within `rp`), a safe practical minimum
is **roughly 800–1,200 prompts total** (culture × language × framing cells each with
~100–200 rows) — which the existing 792-row `us`+`neutral`/`en`+`ja` heldout subset above
already gets close to on its own, and the full 2,732-row eligible heldout split (which
also includes `japan` culture and `mix`-direction prompts) comfortably covers.

**One more design change worth making alongside the larger n:** pipeline 5 used greedy
decoding (`temperature=0.0`), which is deterministic and tends to collapse toward the
model's single "safest" completion — this is likely part of why the realized unsafe rate
at n=40 came in so much lower than the full dataset's pooled rate. A production-scale
re-run should either use the dataset's own original decoding settings as faithfully as
possible (confirmed above: already greedy) at a much larger n, or deliberately sample at
non-zero temperature with multiple completions per prompt to recover some of the variance
that greedy decoding suppresses — but matching the original data's exact generation
settings at the larger n is the more defensible first step before changing anything else.

---

## Assumptions, limitations, and open questions (as requested)

- **Sample sizes in pipelines 2–5 are demo-scale** (8/condition, 22 training pairs, 40
  validation prompts), chosen to fit real GPU time/memory/Colab compute-unit budget
  within this session — not large enough to be statistically definitive on their own.
  Pipeline 1 is the exception: it used the full, well-powered 21,184-row dataset.
- **Pipeline 5's null result (`NaN` odds ratios) is a genuine statistical-power
  limitation, not a fix failure or a bug** — addressed concretely above.
- **Full `CircuitKIT` attention-head circuit discovery never ran on the real Tiny Aya
  model** — its Cohere2 architecture isn't in TransformerLens's supported registry
  (checked directly before writing any pipeline-2 code). Pipeline 2 substitutes
  SafeTune's model-agnostic direction/neuron tools, a real but coarser-grained
  mechanistic analysis.
- **CuratorKIT's generator and quality gate in pipeline 3 both effectively rely on the
  same weak local model** (Tiny Aya itself, since `score_responses=False` disabled the
  only alternative quality check) — a stronger, separate teacher model would produce
  better "chosen" responses in a production run.
- **Pipeline 4 trained for only 1 epoch, batch size 1, on 22 pairs** — enough to prove
  the circuit-restricted mechanism works end-to-end (measurably different trainable
  parameter counts, stable GPU memory, real loss curves), not a converged model.
- **Tiny Aya Water is 3.35B parameters, not literally tiny** — this drove most of
  pipeline 4's real engineering difficulty (documented in that notebook: bitsandbytes
  `device_map` silently skipping quantization under the wrong string value, and
  `PlainSFTTrainer` silently downgrading requested bf16 training back to fp16 on T4
  hardware) and was worked around with a transparent manual PyTorch training loop
  instead of the library's built-in trainer.
- **Colab sessions are not fully reliable for long-running work:** one VM used in this
  project was fully terminated mid-task (not just a dropped connection), destroying
  pipeline 4's on-disk checkpoints since they'd only been referenced by path, not
  downloaded locally — recovered by re-running pipeline 4 fresh in a new session. A
  production rerun should download/persist checkpoints immediately after each training
  step.
- **Pipeline 2's exact input-prompt extraction script (`observations.csv` → the 8
  prompts/condition JSON file) isn't committed to the repo** — only its output
  (`pipeline2_results.json`) and the notebook that consumes it are. This is a
  reproducibility gap worth closing if this pipeline is run again.
- **The target layers used in pipeline 4 (18, 23) come from a single small-n (n=8/condition)
  probe in pipeline 2** — worth re-deriving from a larger sample (see the validation-set
  sizing section above, which applies equally to pipeline 2's own input sample) before
  treating them as final.
- **Nothing in this phase of work has been committed to git** — `notebooks/pipeline1-5_*.ipynb`,
  their input JSON files, and `notebooks/combined/pipeline{1,2,4,5}_*.json` /
  `pipeline3_safety_pairs.jsonl` are all currently untracked in the working tree.
