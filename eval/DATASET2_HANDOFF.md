Dataset2 handoff: evaluation outputs to future CircuitKIT work
============================================================

Start here when handing the experiment to another LLM or researcher. The implementation and technical investigation are documented in `eval/DATASET2_VERIFICATION.md` and `eval/DATASET2_RESEARCH.md`; execution, resumption and merging are in `eval/DATASET2_RUN.md`. This document is a project handoff file. The current notebooks export evaluation artifacts to Drive, rather than automatically exporting this newly added document.

Dataset2 has 320 seeds and 21,184 retained prompts. Part 1 and Part 2 each evaluate 160 complete seed groups / 10,592 retained prompts, with no overlap. Both now use local Tiny Aya Water GGUF Q4_K_M followed by AuditKIT/Qwen3Guard GGUF Q8_0 on saved answers. The notebooks collect observations and screening measurements. They do not run CircuitKIT or turn each guard label into a circuit-discovery objective. Local tests are not evidence that a full GPU run or human judgment validation has occurred.

Default shard result locations are:

```
My Drive/ez-penta/dataset2_eval_gguf/part1/
My Drive/ez-penta/dataset2_eval_gguf/part2/
```

Copy both complete shard directories into one environment. From the repository root (or `/content/ez-penta` after notebook setup), merge them with:

```bash
python -m eval.merge_dataset2_results \
  --part1 /path/to/part1 \
  --part2 /path/to/part2 \
  --output /path/to/combined
```

The combined output location is whatever `--output` specifies. Merging does not rerun either model. It checks dataset/manifest/configuration compatibility, accounts for all expected IDs and recomputes metrics from observations. Explicit failures and missing outcomes can remain in a merged export; inspect coverage before making scientific claims.

Every output has a distinct role:

| Artifact | What it contains | Connection to future CircuitKIT work |
|---|---|---|
| `answers.jsonl` | Incremental answer attempts, configurations, errors and successful completions; historical retries/configurations can coexist. | Audit/resumption evidence. Do not treat raw journal lines as independent examples. |
| `scores.jsonl` | Incremental guard-scoring attempts and raw guard evidence. | Audit/rescoring evidence. Preserve it, but use validated current exports for candidate selection. |
| `answers_current.jsonl` | One current exported record per assigned prompt ID, including explicit missing records; completed answers retain source metadata, exact prompt/rendered input, Aya input/output token IDs, token counts and generation identity. | Reconstruct exact model inputs and selected response sequences for validated candidate pairs and likelihood comparisons. |
| `scores_current.jsonl` | Current prompt/response guard judgments, raw outputs and score/answer identities, or explicit missing records. | Screen response-safety and refusal contrasts for human review. Guard classifications are not differentiable Tiny Aya metrics. |
| `observations.csv` / `results.sqlite3` | Observation-level prompt, seed, condition, answer, status, judgment and validation fields; SQLite also holds summary and metadata tables. | Main relational input for a future adapter. Join by `prompt_id`; pair by `seed_id` and the required matching conditions. |
| `manual_inspection.csv` | All observations with blank human request-safety, response-safety, refusal, appropriateness, notes and review-status fields. | Produce a separately reviewed subset and resolve ambiguous guard judgments before defining the discovery task. Blank fields are not human labels. |
| `validation.csv`, `validation_report.json` (per shard) | Structural checks, identical-twin and other review flags, source coverage counts. | Define comparison-specific eligibility. Retain flagged examples in the audit trail and state exclusions. |
| `missing_variants.csv` (per shard) | Absent conditions from the original full design; covers source-dataset missingness. | Determine which paired comparisons are possible. Absence here is different from a failed model response. |
| `coverage.json` | Assigned, completed, truncated, failed/unknown coverage and completion status. | Establish the usable sample count and failure bias before discovery. Full assignment coverage does not imply successful inference on every row. |
| `headline.csv`, `denominators.csv` | Automated rates and explicit known/unknown/eligible counts by split and condition. | Choose research questions and assess measurement quality. Aggregate rates are not direct circuit objectives. |
| `matched_observations.csv` | Matched prompt-ID pairs, seed/split/condition information and paired measured outcomes. | A starting list of potential contrasts, requiring further human and token-alignment checks. |
| `comparisons.csv`, `sensitivity.csv` | Seed-level contrast summaries, descriptive bootstrap intervals and alternative coverage/quality cohorts. | Check robustness and design a discovery/validation split. These statistics do not identify causal model components. |
| `run_metadata.json` and SQLite `run_metadata` | Dataset/manifest identity, model/tokenizer revisions, generation/judge settings, implementations and assigned IDs. | Recreate the experiment and reject incompatible checkpoint, template, precision or software combinations. |
| `likelihood_diagnostic.json` (optional) | A small non-sensitive teacher-forced scoring demonstration. | Checks scoring mechanics only. It does not contain validated refusal/compliance pairs or discovered circuits. |

The intended progression is:

```
Both shard exports
        ↓ compatibility/coverage-checked merge
Observation-level answers + judgments + quality flags
        ↓ human review and within-seed matching
Validated prompt contrasts + validated response candidates
        ↓ likelihood sanity checks and model-backend parity
Differentiable CircuitKIT task adapter
        ↓ EAP / EAP-IG discovery
Activation patching and causal validation on heldout examples
```

1. **Choose a valid behavioral contrast.** Begin with same-language examples. A possible contrast is control versus roleplay for the same seed, request kind, culture, input-language condition, requested output language and metadata split. Verify that the underlying request is preserved, and that one response actually refuses while another provides the behavior of interest. An Unsafe label does not establish compliance, and a Safe label does not establish refusal. Borderline originals may correctly expect cautious assistance rather than refusal. Do not automatically use identical original/twin inputs as an expected behavioral contrast.

2. **Create a reviewed pair manifest.** This is future work, not an existing export. A useful schema would record `pair_id`, `seed_id`, both `prompt_id` values, the contrast definition, original split, fixed conditions, reviewed request/response/refusal labels, review provenance, selected continuation text/token IDs and exclusion reasons. Specify the orientation of the clean/corrupted contrast rather than assuming clean means safe. Preserve the seed-level split and keep heldout examples out of task tuning. A seed needs the variants for the specific comparison; it does not need all 84 variants.

3. **Validate fixed candidate continuations.** Each generated prompt currently supplies one actual answer, not both a refusal and a compliance candidate. Select or construct candidates only with review. For a prompt x, a particular refusal continuation r and a particular compliance continuation c, a proposed objective is:

   `m(x) = log P(r | x) - log P(c | x)`

   Hold the candidate definitions fixed when comparing the paired inputs. Higher values favor the specified r relative to the specified c. This is not the probability of all possible refusals versus all possible compliance. Candidate wording, length, EOS inclusion, language and precision affect the measurement. State those choices before interpreting a margin.

4. **Use the current likelihood module for diagnostics.** `eval/dataset2_likelihood.py` implements teacher-forced token/sequence log-probabilities and caller-supplied candidate margins. Its token shift, prefix boundary checks and chunking have CPU tests. It uses `torch.inference_mode()` and converts results into Python numbers, so it is not a differentiable CircuitKIT objective. Exact saved token IDs allow the original generated sequences to be rescored without treating decoded/stripped text as a perfect reconstruction.

5. **Implement and test the actual adapter.** CircuitKIT tasks package data, metrics and paired loaders; EAP-family attribution requires model forwards and gradients of a scalar objective. See the official [task documentation](https://circuitkit.lexsi.ai/user-guide/tasks/) and [EAP documentation](https://circuitkit.lexsi.ai/algorithms/eap/). Reuse the installed version's single-token or differentiable span-scoring patterns where appropriate, but implement the required continuation passes, masks, gradients and token/position alignment for this task. Registering a TaskSpec alone does not supply that computation. Before discovery, pin CircuitKIT/backend dependencies and prove forward/logit and candidate-score parity for the exact Tiny Aya Water checkpoint and chat template. Family-level model support is not proof of this exact configuration's compatibility.

6. **Validate attribution causally.** Attribution ranks candidate components; it does not by itself demonstrate that they cause refusal or unsafe assistance. Test selected components through activation patching, ablations and suitable negative/random controls. Evaluate on heldout seed groups and check collateral behavior. Token alignment across Japanese, English, mixed-language prompts and unequal-length responses requires an explicit design; start with within-language contrasts.

Known scientific limits remain relevant throughout: 498 identical original/twin control pairs affect 119 seeds; generated expectations are not human-validated; some twins or rewrites may change the task; missing variants are uneven; and guard reliability on the actual Japanese/mixed-language cohorts needs review. Current expected-correctness, attack-success and over-refusal fields remain unknown until validated expectations/outcome rules exist. Refusal and response-safety measurements remain separate.

Give the next researcher this document, both complete shard output directories, the merged artifacts, the original source database/manifest, the matching notebook release/shared modules and any independently collected human reviews. Ask them first to validate the candidate cohort and proposed objective. The remaining implementation is a validated adapter and causal experiment, not a direct import of every AuditKIT result into CircuitKIT.

Current backend qualification: discovery with a differentiable HF model is not automatically an explanation of the Q4_K_M GGUF model's behavior. Reproduce the selected contrasts with the intended CircuitKIT model first, or explicitly engineer and validate a matching differentiable quantized model. The default GGUF notebooks intentionally disable HF likelihood diagnostics and use separate `dataset2_eval_gguf/` output directories. Their saved exact native token IDs, model hashes and runtime provenance support this cross-backend validation; they do not replace it.
