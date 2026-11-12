# What you should expect when Codex finishes

**This is the acceptance checklist for the human project owner.** Technical completion is not a guarantee of a favorable statistical result. Judge the final package by the evidence it contains.

## 1. Concrete files to receive

### In the code repository (private-data-free)

| File / directory | Expected contents | How to validate |
|---|---|---|
| `docs/lexsi/STATUS.md` | Phase gate statuses A–G, exact evidence pointers, blockers | No undocumented `DONE` statuses |
| `docs/lexsi/DECISIONS.md` | What changed, why, alternatives, experiment IDs | Each consequential parameter choice recorded before final results |
| `docs/lexsi/EXPERIMENTS.md` | Executed commands, runs, GPU/runtime and seeds | Distinguishes planned from executed |
| `docs/lexsi/CHECKPOINT_RECOVERY.md` | Both adapter/merged checkpoint locations, verification | Independent clean-session reload passes or loss explicitly documented |
| `docs/lexsi/FINAL_REPORT.md` | Scientific methods, counts, effects, CIs, limitations, all model arms, circuits | Every important number traceable to code and artifact |
| `docs/lexsi/WHAT_WE_LEARNED.md` | One page: claims supported / unsupported / inconclusive | Understandable without reading notebooks |
| `docs/lexsi/REPRODUCE.md` | Exact end-to-end Colab commands, dependency pins, private-data setup | At least a sample rerun demonstrably succeeds |
| `eval/track2_extension/` | Modular scripts and Colab notebook entrypoints | `pytest` passes, no data leakage or hardcoded Colab paths |
| `tests/track2_extension/` | Unit/regression tests for selection, labels, metrics, model reload, resumption | Tests run without GPU/data for core logic |

### In PRIVATE Google Drive (not in public GitHub)

| Artifact | Purpose |
|---|---|
| `manifest.json` / `artifact_inventory.csv` | Unique run IDs, complete provenance, URI and SHA256 for each file |
| `checkpoints/restricted_adapter/` and `checkpoints/broad_adapter/` (or equivalent) | Durable model parameters; if absent, a precise recorded reason |
| `evaluation_manifest.parquet` | Matched prompt IDs and seed grouping with leakage flags |
| `responses.parquet`, `judgments.parquet` | Per-model raw generations and verdicts, errors, token counts, completion states |
| `metrics.csv`, `metrics.json`, `bootstrap_replicates.parquet` | Raw denominators, paired differences, uncertainty and stratified results |
| `figures/` | Risk differences, model transition counts, JA/EN and culture analyses, negative controls |
| `circuits/` | `.pt`/other actual CircuitKIT circuit objects + graph metadata, or explicit failed gate logs |
| `circuit_evaluation/` | Patching, ablation, size-matched random scores, transfer comparisons, or justified failure |
| `reviews/` | Human annotation guide and human-approval CSV or review interface; never fictitious approvals |
| `logs/` | Errors, environment freeze, CUDA peak, execution notebook snapshots, runtime restart records |

Every claim of success needs at least one durable file and a verification command; screenshots alone are insufficient.

## 2. Four plausible end states

### Outcome A: Model improvement + mechanism found (best case, not guaranteed)

Codex produces seed-aware results showing a meaningful drop in unsafe compliance in restricted/attribution-guided model vs untouched baseline, no material benign degradation, a fair broad/random control comparison, a valid CircuitKIT Cohere2 parity run, and patching/ablation evidence of an explanatory circuit. The article can cautiously claim a mechanism-informed targeted intervention, subject to small-sample, guard, license, annotation and external-validation caveats.

### Outcome B: Model improvement but no valid circuit (still publishable as an engineering study)

The intervention is empirically better on held-back data, but EAP-IG port/metric fails or causal tests do not pass. Conclude *targeted LoRA fine-tuning helped on this evaluation*, **not** that CircuitKIT identified a refusal circuit. Report SafeTune as an exploratory probe only.

### Outcome C: Circuit evidence but no improvement (a legitimate research outcome)

CircuitKIT reveals components that pass intervention tests, but LoRA training on 22 examples is ineffective or hurts safety. Conclude *mechanistic localization was feasible; this intervention did not establish improved safety*. Consider enhanced data quality and larger training as future work, not hidden iterative tuning of final evaluation.

### Outcome D: No measurable improvement or no validated circuit (still a completed, honest package)

If model behavior is overwhelmingly Safe, sample too small, model checkpoints cannot be restored, quality-gated pairs are too few, GPU resources are insufficient, or metric invalid: provide the strongest feasible technical result and a precise negative/inconclusive statement. Useful artifacts include a reliable evaluation harness, calibrated guard/human comparison, corrected cluster-aware association, notebook reproducibility, and fault logs. No manufactured success.

## 3. Questions Codex must answer in plain English

1. Were the two previously trained weights actually saved durably, and can we reload them independently? If not, did Codex redo exactly the experiment and save adapters correctly?
2. How many *unique seeds* and prompt rows were used in each phase, and what was excluded and why?
3. Did the new model reduce **unsafe compliance** (not just guard Unsafe), by how many percentage points, with which 95% seed-aware uncertainty interval and what reference population?
4. Did it make harmless requests more likely to be refused? Was that verified or only inferred from automated labels?
5. How does restricted LoRA compare with the **broad LoRA trained on the same data**, and (if run) a parameter-matched random-layer control?
6. Was CircuitKIT's Water `cohere2` compatibility actually tested on real weights? Report observed KL and exact pass/fail logs.
7. Did EAP-IG actually run on Tiny Aya? Which head/MLP nodes scored highly? Were results stable across rediscovery seeds?
8. Does patching those components change the chosen refusal metric? Does ablating them change it? Are effects stronger than random components?
9. Is there credible evidence of different Japanese vs English circuitry? What transfers, what does not, and what confounds remain?
10. What assumptions, limitations, failed experiments, budget constraints, and possible data leakage make each conclusion weaker?

## 4. Scientific claim ladder

| Evidence provided | Claim permitted |
|---|---|
| Historical 21K rows, MH OR only | “US-cultural framing was **associated with** more Unsafe guard labels in Dataset2” |
| SafeTune activation direction from 8 examples/condition | “A small probe identified a **candidate** refusal-related activation direction” |
| Successful two-arm LoRA training | “We successfully **trained** restricted/broad adapters” |
| 40-prompt baseline vs restricted near-all-Safe | “The original pilot was **inconclusive**” |
| Large, seed-separated paired model evaluation with reliable labels and uncertainty | “On this evaluation distribution, intervention **improved / did not improve / was inconclusive**” |
| Actual Tiny Aya EAP-IG + held-back patching/ablation beating controls | “Selected internal components **causally contribute to the measured refusal metric** under these interventions” |
| Seed-disjoint replicated JA/EN comparison | “Shared/different components under a specified metric and tested settings” |
| Verified all of the above plus controlled intervention + fair training controls | “Evidence for **mechanism-informed improvement** on the defined evaluation” |

“Statistically insignificant” does **not** mean there is no effect; a null result with a wide CI is inconclusive. Likewise a very small p-value from 21K *correlated* prompt rows does not establish causal cultural framing.

## 5. Red flags: stop trusting the final report if any appear

- A file called `pipeline4_results.json` is presented as proof of saved model weights, but the only locations are `/content/...`.
- The “before” model uses historical GGUF outputs while the “after” uses HF generation with different caps/templates.
- `1/40 vs 2/40 Unsafe` is described as proven safety improvement (or definitive failure) with no paired outcome table or CI.
- All 792 held-out *rows* are called 792 independent examples despite only ~60 original heldout source seeds.
- Qwen3Guard Safe is equated with an appropriate refusal, or Controversial is treated as Safe.
- CuratorKIT's 22 auto-generated pairs are presented as human-approved ground truth.
- Chosen-only SFT is called DPO or preference optimization.
- Restricted LoRA layers 18/23 are automatically called a “CircuitKIT-discovered refusal circuit”.
- CircuitKIT is described as impossible solely because vanilla TransformerLens lacks Cohere2, without testing CircuitKIT's own compatibility port.
- Discovered attention-head ranking is called causal without patching/ablation and a same-size random baseline.
- A favorable result appears only after repeated undocumented tuning on heldout examples.
- No distinct benign controls, no prompt/seed manifest, no real raw outputs, or no reproducible commands.

## 6. Minimum acceptable completion

Regardless of scientific outcome, Codex should leave: (i) a passing tested codepath to inspect data and run a small reproducible inference/scoring experiment, (ii) valid multi-arm paired dataset analysis or a precise blocker, (iii) CircuitKIT port parity/discovery test logs, (iv) a full evidence-based report and known limitations, (v) all recoverable weights/checkpoints and private outputs backed up to Drive, and (vi) an actionable next-step decision no longer requiring guessing what was actually run.
