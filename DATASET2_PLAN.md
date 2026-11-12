# Codex Implementation Plan: Dataset2 Parallel Tiny Aya + AuditKIT Evaluation

## 1. Primary objective

Implement a reproducible, end-to-end evaluation pipeline for Dataset2 in the existing `ez-penta` repository.

**The primary deliverables are TWO independently executable Google Colab Jupyter notebooks:**

- `eval/eval_dataset2_part1.ipynb`
- `eval/eval_dataset2_part2.ipynb`

Each notebook must evaluate a distinct, deterministic subset of Dataset2 using Tiny Aya Water for response generation and AuditKIT/Qwen3Guard for safety judgments.

The notebooks will be run independently, potentially simultaneously, by two different team members using separate Google Colab runtimes.

**Do not implement the earlier pilot-only plan. The objective is to process all retained Dataset2 prompts, divided between the two notebooks.**

The two notebook outputs must subsequently be mergeable into one complete evaluation dataset without duplicate records, lost observations, incompatible configurations, or checkpoint collisions.

A secondary objective is to make the resulting data suitable for future likelihood scoring and CircuitKIT experiments.

Do not implement full CircuitKIT attribution yet.

### Guiding principles

1. Reuse existing working code.
2. Do not duplicate evaluation logic across notebooks.
3. Preserve all input records and their provenance.
4. Preserve within-seed relationships.
5. Make parallel execution safe and reproducible.
6. Treat generated safety judgments as measurements, not ground truth.
7. Investigate unclear technical details rather than guessing.
8. Prioritize working Colab notebooks over documentation-heavy deliverables.
9. Make the notebooks understandable and maintainable by other team members.
10. Do not claim GPU functionality was tested unless actual model inference was performed.

---

## 2. Dataset background

Dataset2 contains:

- 320 underlying seeds.
- 10 categories, with 32 seeds per category.
- Two request kinds: original (`harmful`) and intended harmless (`twin`).
- Three cultural conditions: `neutral`, `japan`, `us`.
- Four framings: direct control, roleplay (`rp`), fiction (`fc`), refusal suppression (`rs`).
- Japanese, English, and mixed-language conditions.
- One balanced mixed-language direction assigned per seed.
- Up to 84 final prompts per seed.
- 26,880 maximum possible prompts.
- **21,184 actual retained prompts (78.8% coverage).**

Metadata partitions:

- Analysis: 160 seeds.
- Tune: 100 seeds.
- Heldout: 60 seeds.

These partitions are metadata assignments, not additional prompt variants.

The source database currently contains generated prompts, not Tiny Aya responses or AuditKIT judgments.

### Known dataset concerns

Previous inspection reported:

1. Among 1,920 matched original/twin control pairs, 498 contained identical prompt text, affecting 119 seeds.
2. Of those pairs, 153 had contradictory `decline` versus `answer` expectations; 345 contrasted `answer_carefully` versus `answer`.
3. Some requests labeled as clearly harmful appear innocuous.
4. Some twins alter the underlying task substantially rather than providing a minimally changed harmless alternative.
5. Some roleplay, fiction, or refusal-suppression variants may not faithfully implement the intended technique.
6. Borderline original requests may expect `answer_carefully`, not `decline`.
7. Missing variants are uneven across conditions, particularly sexual-content refusal suppression.

Treat these as reported findings to verify, not as justification to delete records indiscriminately.

**The dataset's original labels, prompt text, seed relationships, and generated condition metadata must never be silently modified.**

---

# PHASE 0 — Inspect the existing codebase before implementation

This is mandatory.

The repository already contains code for running Tiny Aya evaluations and obtaining guard-model judgments through AuditKIT.

Inspect the current branch, working tree, notebook structure, dependencies, existing tests, and relevant code.

### Files to inspect

Primary evaluation files:

- `eval/eval_dataset2.ipynb`
- `evaluate_answercarefully.py`
- `guard_judge.py`
- `answercarefully_dataset.py`
- `chat.py`
- `local_server.py`

Potential reusable helpers:

- `eval/checkpoints.py`
- `eval/scoring.py`
- `eval/analysis.py`

Dataset-generation and inspection notebooks:

- `data_ready/dataset2.ipynb`
- `data_ready/dataset2v2.ipynb`
- `eval/view/preview_dataset2.ipynb`

Dataset files:

- `marcus/dataset2_partial.sqlite3`
- `marcus/dataset2_high_coverage_incomplete.csv`

Also inspect available tests, package requirements, Colab-specific setup, and any relevant existing pipeline code.

Some paths may be present only in a local workspace or another branch. Identify what is actually available.

### Determine what can be reused

Specifically inspect existing implementations of:

- Loading and querying SQLite datasets.
- Extracting final prompt records and metadata.
- Tiny Aya initialization.
- Hugging Face authentication.
- Chat-template handling.
- Greedy inference.
- GPU memory management.
- AuditKIT scoring using previously generated answers.
- Qwen3Guard parsing.
- Refusal and language annotations.
- Checkpointing and resuming.
- Metrics, comparisons, and result export.
- Google Drive persistence.

### Required implementation approach

Prefer a **shared Python evaluation module** containing the actual evaluation logic.

The two notebooks should differ primarily in their shard configuration.

Do not maintain two independent copies of the complete evaluation implementation.

The notebooks should import and call shared functions instead of reimplementing the same logic.

Do not refactor existing working components merely for aesthetic reasons.

### Output

Provide a short repository audit explaining:

- What code already exists.
- What will be reused.
- What requires modifications.
- What is missing.
- What requires technical investigation.

---

# PHASE 1 — Deterministically divide the complete dataset into two halves

This is a critical requirement.

## 1.1 Partition by seed

**All retained prompts belonging to a single `seed_id` must be assigned to the same notebook.**

Do not split a seed's language, framing, cultural, original/twin, or other variants across notebooks.

This preserves within-seed comparisons and avoids forcing subsequent analysis to reconstruct seed groups from separate runs.

### Target partition

| Property | Part 1 | Part 2 |
|---|---:|---:|
| Underlying seeds | 160 | 160 |
| Target retained prompts | 10,592 | 10,592 |
| Total source coverage | Approximately 50% | Approximately 50% |

Attempt to achieve exactly 10,592 prompts per notebook while maintaining seed integrity and reasonable category/split balance.

Since seeds have unequal retained prompt counts, verify whether an exact balanced allocation is possible.

If exact equality conflicts with preserving seed integrity or important experimental balance:

- Preserve seed integrity.
- Minimize the difference in prompt counts.
- Report the exact prompt totals.
- Explain the allocation tradeoff.
- Do not split seeds merely to obtain an exact numerical half.

## 1.2 Stratification

Aim to balance the two partitions by:

- Category.
- Analysis/tune/heldout assignment.
- Number of prompts.
- Severity/expected behavior.
- Mixed-language direction.
- Available framing and language coverage.

Prefer 16 of the 32 seeds from each category in each notebook, where this is compatible with the overall allocation.

Similarly, target approximately:

- 80 analysis seeds per notebook.
- 50 tune seeds per notebook.
- 30 heldout seeds per notebook.

Never change a seed's original metadata split.

Use deterministic allocation with a fixed random seed and documented algorithm.

## 1.3 Create one authoritative shard manifest

Create a reproducible manifest such as:

`eval/manifests/dataset2_shards.csv`

Suggested columns:

- `seed_id`
- `shard_id`
- `category`
- `split`
- `expected_prompt_count`
- `actual_prompt_count`
- `mixed_direction`
- `manifest_version`

The two notebooks must load the same authoritative manifest.

They must not independently regenerate their own random subsets.

## 1.4 Partition invariants

Implement assertions verifying that:

1. Every eligible retained prompt is assigned to exactly one shard.
2. No prompt appears in both shards.
3. Every seed belongs to exactly one shard.
4. Both shards together contain exactly the intended retained dataset.
5. The original dataset metadata is preserved.
6. The assignment is reproducible.
7. The shard manifests produce identical assignments across independent environments.

For the currently described dataset, the target union is **21,184 prompt IDs**.

If the source database contains a different number of retained records, fail loudly or require explicit acknowledgment of a new dataset revision.

Do not silently continue using an outdated manifest.

---

# PHASE 2 — Dataset validation and quality flags

Before inference, validate the records loaded from the source database.

Do not require a seed to have all 84 possible variants.

Do not automatically discard incomplete-but-valid seed groups.

### Structural checks

Validate:

- SQLite integrity.
- Required tables and columns.
- Unique prompt IDs.
- Seed foreign-key relationships.
- Nonempty prompt text.
- Valid condition metadata.
- Expected per-seed prompt counts.
- Duplicate final prompt texts.
- Contradictory original/twin controls.
- Missing variants by category, technique, culture, language, and split.

### Semantic-quality checks

Flag cases involving:

- Potentially incorrect harmfulness labels.
- Identical original/twin prompts with contradictory expectations.
- Harmless twins that substantially change the underlying task.
- Poor cultural equivalence.
- Rewrites that fail to express the intended technique.
- Prompt language not matching the assigned condition.
- Borderline requests that do not clearly warrant refusal.

Separate objectively detected defects from cases requiring human judgment.

### Important distinction

**A record can be structurally valid while being scientifically questionable.**

Do not overwrite the original `expected_behavior`.

Instead, preserve quality indicators such as:

- `structural_valid`
- `identical_twin_flag`
- `expected_behavior_conflict`
- `semantic_review_required`
- `rewrite_review_required`
- `analysis_eligible`
- `exclusion_reason`

Where possible, distinguish:

1. Eligibility for model generation.
2. Eligibility for guard scoring.
3. Eligibility for particular scientific comparisons.

For example, an identical original/twin prompt pair may still be useful for checking inference consistency, but should not support claims about an expected behavioral difference between the two identical inputs.

### Evaluation policy

Do not silently drop structurally valid records because of uncertain semantic quality.

Retain them in the evaluation where appropriate, flag them, and make analysis exclusions explicit.

Use a consistent validation policy across both shards.

### Outputs

Produce a validation report and a machine-readable validation table, either centrally or as notebook exports backed by the same shared code.

---

# PHASE 3 — Build the two primary Google Colab notebooks

The main deliverables are:

### Notebook A

`eval/eval_dataset2_part1.ipynb`

Configuration:

```python
SHARD_ID = 1
RUN_MODE = "full_shard"
```

Evaluates only the records assigned to shard 1.

### Notebook B

`eval/eval_dataset2_part2.ipynb`

Configuration:

```python
SHARD_ID = 2
RUN_MODE = "full_shard"
```

Evaluates only the records assigned to shard 2.

### Required characteristics

Both notebooks must:

- Run from top to bottom.
- Be compatible with Google Colab.
- Install or verify dependencies.
- Mount Google Drive when configured.
- Authenticate with Hugging Face securely.
- Load the authoritative dataset and shard manifest.
- Validate input compatibility.
- Load Tiny Aya.
- Generate responses.
- Save answers incrementally.
- Release GPU memory when appropriate.
- Load AuditKIT/Qwen3Guard.
- Score stored answers.
- Save judgments incrementally.
- Produce shard-level results and inspection tables.
- Display completion and failure summaries.
- Support interruption and resumption.

**Both notebooks must use the same shared implementation.**

Do not create two notebooks with subtly different generation, judge, parsing, or aggregation logic.

### Configuration requirements

Provide an easy-to-understand configuration cell with:

- Shard identifier.
- Dataset path.
- Manifest path.
- Output directory.
- Model identifiers/revisions.
- Model dtype.
- Generation settings.
- Judge configuration.
- Batch size.
- Checkpoint settings.
- Retry settings.
- Whether to run optional likelihood scoring.

Default to evaluating the entire assigned shard.

A separate explicit smoke-test limit may be supported for debugging, but it must not be confused with the actual evaluation or silently truncate the shard.

### Notebook section structure

Each notebook should have clearly labeled sections:

1. Environment and configuration.
2. Authentication and Google Drive.
3. Dataset loading and shard verification.
4. Dataset validation summary.
5. Tiny Aya model loading.
6. Response generation.
7. Answer checkpoint summary.
8. AuditKIT/Qwen3Guard initialization.
9. Safety and refusal scoring.
10. Score checkpoint summary.
11. Shard-level analysis.
12. Manual-inspection samples.
13. Optional likelihood-scoring diagnostics.
14. Final export and completion verification.

Provide meaningful progress indicators and errors.

---

# PHASE 4 — Tiny Aya inference

Reuse the existing Hugging Face Tiny Aya inference code where suitable.

### Target configuration

Model:

`CohereLabs/tiny-aya-water`

Requested settings:

```python
do_sample = False
temperature = None
top_p = None
top_k = None
max_new_tokens = 1024
```

Use float16 where supported and appropriate.

Verify supported Transformers versions, model configuration, actual chat-template behavior, padding, end-of-turn tokens, and decoding conventions.

The original evaluation uses greedy decoding.

Do not inadvertently introduce sampling or change the model input format while refactoring.

### Model loading

Avoid loading Tiny Aya repeatedly for each prompt.

Use batching when safe and appropriate, and preserve correct attention masks and output-to-prompt mappings.

Document how truncated responses, EOS termination, and model failures are detected.

### Required answer metadata

Each generated answer should preserve:

- `prompt_id`
- `seed_id`
- `shard_id`
- Original request kind.
- Category.
- Expected behavior.
- Culture.
- Framing.
- Prompt-language condition.
- Output-language condition.
- Exact prompt text.
- Generated response.
- Input token count.
- Output token count.
- Truncation/finish status.
- Model identifier and revision.
- Tokenizer/chat-template identity.
- Generation configuration.
- Software versions.
- Run identifier.
- Error information, when applicable.

Avoid storing the full vocabulary-logit tensor for every generated token.

The model's exact inputs and selected token sequences should be reproducible for later likelihood-scoring work.

---

# PHASE 5 — Checkpointing and concurrent execution safety

This is especially important because two team members will run the notebooks independently.

## 5.1 Independent output directories

Use shard-specific paths.

For example:

```text
MyDrive/ez-penta/dataset2_eval/
    part1/
        answers.jsonl
        scores.jsonl
        run_metadata.json
        results.sqlite3
        logs/
    part2/
        answers.jsonl
        scores.jsonl
        run_metadata.json
        results.sqlite3
        logs/
```

The exact structure may be adapted to existing repository conventions.

However, **Part 1 must never write into Part 2's checkpoints, and vice versa.**

The source SQLite database must remain read-only.

Use local runtime storage for SQLite writes and safely copy finalized artifacts to Drive as needed. Avoid unsafe concurrent SQLite writes to Google Drive-mounted files.

If team members use separate Google Drive accounts, document how each can export its shard outputs for later merging.

## 5.2 Resumable generation

An interrupted notebook must resume without regenerating valid completed answers.

A cached response should be considered reusable only when the relevant identity matches, including:

- Prompt ID and exact input content.
- Model identifier and revision.
- Tokenizer revision.
- Chat template.
- Generation parameters.
- Relevant software/configuration versions.
- Output policy.

Do not rely solely on matching prompt ID.

## 5.3 Resumable judging

Judge checkpoints must include:

- Exact prompt and response identity.
- Judge model and revision.
- Judge input template.
- AuditKIT version.
- Judge configuration.
- Parser/annotator implementation identity.
- Metric definitions.
- Relevant embedding-model revisions for similarity measurements.

A known issue in the current notebook is inadequate invalidation when judge, embedding, parser, or annotator implementations change.

Inspect and correct this.

## 5.4 Checkpoint robustness

Handle:

- Interrupted writes.
- Duplicate records.
- Partially written lines.
- Failed generations.
- Failed scores.
- Retryable errors.
- Permanent errors.
- Stale cached observations.

Prefer a reliable append-only or atomic checkpoint design already supported by the repository.

Do not silently interpret missing or failed results as Safe.

### Required checkpoint tests

Verify that:

- Re-running a completed shard does not regenerate answers unnecessarily.
- Part 1 cannot consume Part 2's checkpoints.
- Changes to the model/configuration invalidate affected answers.
- Changes to the judge/parser invalidate affected scores.
- A partially completed run resumes correctly.
- Duplicate and failed records are resolved deterministically.

---

# PHASE 6 — Reuse AuditKIT and Qwen3Guard

Reuse the existing evaluation code wherever possible.

Do not rebuild the guard-model integration from scratch.

Score **precomputed Tiny Aya responses** so that a failed judge run does not force response regeneration.

### Separate measurement dimensions

The output must distinguish:

**Response safety**

Did Qwen3Guard classify the response as Safe, Unsafe, Controversial, or unknown?

**Refusal behavior**

Did Tiny Aya actually refuse the request?

**Prompt safety**

How did the guard classify the input request itself?

**Expected-behavior correctness**

Was Tiny Aya's response appropriate given the actual nature of the request?

**Attack success**

Did a rewritten prompt induce unsafe assistance on an eligible request?

**Over-refusal**

Did Tiny Aya unnecessarily refuse a legitimate request?

These are not interchangeable measurements.

A refusal is not always correct, and compliance is not always unsafe.

### Label handling

Preserve:

- Safe.
- Unsafe.
- Controversial.
- Unparsed.
- Missing/error states.

Do not collapse controversial, unparsed, or missing judgments into Safe or Unsafe without a justified, documented policy.

Preserve raw guard outputs and parsed classifications separately.

### Expected behavior

Dataset2 includes:

- `decline`
- `answer_carefully`
- `answer`

Do not treat all original requests as requiring refusal.

Do not automatically treat every refusal of an `answer_carefully` request as an error without reviewing whether the particular request actually warrants cautious assistance.

Only calculate expected-behavior correctness where the ground-truth expectation is sufficiently validated.

### Similarity and language annotations

Inspect and reuse existing multilingual similarity, language-detection, and refusal-heuristic code where useful.

Make these secondary measurements.

Do not assume semantic similarity proves safety equivalence or that a simple refusal phrase reliably identifies refusal across Japanese and English.

---

# PHASE 7 — Shard-level analysis

Each notebook should produce useful summaries of its own assigned records.

At minimum:

- Total assigned prompts.
- Completed responses.
- Completed guard judgments.
- Failed/unknown records.
- Unsafe response rate.
- Refusal rate.
- Response truncation rate.
- Expected-behavior correctness where eligible.
- Over-refusal where eligible.
- Results by language.
- Results by framing.
- Results by culture.
- Results by category.
- Results by expected behavior.

### Denominators

For every statistic, explicitly record:

- Eligible observations.
- Evaluated observations.
- Known judgments.
- Unknown judgments.
- Missing observations.
- Relevant seed count.

Never calculate a rate using an implicit denominator that includes missing or unparsed judgments as if they were negative outcomes.

### Matched comparisons

Use available matched variants belonging to the same seed.

Examples:

- Japanese control vs. English control.
- Japanese roleplay vs. Japanese control.
- Japanese fiction vs. Japanese control.
- Japanese refusal suppression vs. Japanese control.
- Mixed-language vs. single-language conditions.
- Culture-specific vs. neutral prompts.

Only compare rows satisfying the necessary matching conditions.

Preserve differences in prompt language, requested response language, framing, culture, and request kind.

### Statistical considerations

Because many variants come from the same seed, observations are not independent.

Use seed-level paired analysis or appropriate cluster-aware uncertainty estimates.

Do not perform naive independent-observation significance tests across thousands of correlated variants.

The two shards are computational partitions, not experimental treatment groups. Do not interpret differences between Part 1 and Part 2 as scientific effects.

Important: preserve the analysis/tune/heldout distinction.

Do not use heldout results for iterative parameter selection or judge tuning.

---

# PHASE 8 — Merge the two completed evaluations

This is a mandatory supporting deliverable.

Implement a reliable merge function, script, or notebook section that combines:

- Part 1 answers.
- Part 1 judgments.
- Part 2 answers.
- Part 2 judgments.
- Relevant metadata and validation flags.

Suggested artifact:

`eval/merge_dataset2_results.py`

Or a similarly suitable shared module.

### Merge requirements

Before merging, verify:

1. Both shards used the same dataset revision.
2. Both used the same authoritative shard manifest.
3. Model revisions match.
4. Tokenizer and chat-template identities match.
5. Generation configurations match.
6. Judge configurations and parser versions match for directly comparable metrics.
7. No prompt ID appears in both shards.
8. Every expected prompt ID is accounted for.
9. Failed, unknown, and missing observations remain explicit.
10. All seed and variant metadata survive the merge.

If configurations differ, do not silently combine the outputs into a single homogeneous evaluation.

Either fail with a clear error or preserve them as separately identified experimental runs.

### Final combined outputs

Produce:

- Combined answers and judgments.
- Complete evaluation SQLite database or equivalent.
- Global coverage/completeness report.
- Combined headline metrics.
- Matched-condition comparisons.
- Explicit denominator tables.
- Manual inspection export.
- Reproducibility metadata.

Verify:

```text
Expected assigned prompts = 21,184
Duplicate prompt IDs = 0
Unexpected prompt IDs = 0
```

The number of successful responses and known guard judgments may be lower if failures occurred. Report those separately.

### Important scientific requirement

Do not average the two notebooks' aggregate rates to produce final conclusions.

Combine the underlying observation-level records and recompute the relevant statistics using the full eligible cohorts.

---

# PHASE 9 — Manual judgment validation

Create inspection functionality using the existing viewer or a notebook table.

The inspection export should support manual review of:

- Guard Unsafe results.
- Guard Safe results.
- Controversial/unparsed results.
- Refusal heuristic disagreements.
- Contradictory expected behaviors.
- Original/twin relationships.
- Potentially invalid rewrites.
- Cross-language discrepancies.
- Model responses whose apparent safety differs from the guard label.

### Manual review fields

Include:

- `prompt_id`
- `seed_id`
- Language and condition metadata.
- Exact input.
- Tiny Aya response.
- Raw guard output.
- Parsed guard judgment.
- Refusal heuristic result.
- Original expected behavior.
- Human-assessed request safety.
- Human-assessed response safety.
- Human-assessed refusal.
- Human-assessed appropriateness.
- Human review notes.
- Review status.

Support uncertain cases.

Do not invent human labels.

### Evaluation strategy

Implement the inspection and review tools now, but do not pretend manual validation has occurred.

Since the two notebooks are expected to process the entire dataset, do not block their execution on hypothetical human-review results.

However, the final scientific report must explicitly distinguish automated measurements from manually validated conclusions.

If manual review reveals severe measurement defects, flag affected findings and recommend rescoring or reanalysis.

---

# PHASE 10 — Prepare for log-probability measurements

We intend to use the evaluation results for future CircuitKIT work.

For now, build the data infrastructure and, where practical, implement an optional likelihood-scoring section.

## 10.1 What we eventually want to calculate

For a prompt x and selected continuation y:

\[
\log P(y\mid x)
\]

Then, for a validated refusal continuation r and a validated compliance continuation c:

\[
m(x)=\log P(r\mid x)-\log P(c\mid x)
\]

This measures the model's relative likelihood of two particular response sequences.

It does not measure the total probability of all possible refusals versus all possible unsafe responses.

## 10.2 Important distinction

Each Dataset2 evaluation prompt currently yields one Tiny Aya response.

That does not automatically provide both candidate continuations.

We may later select refusal and compliance examples from different variants of the same seed, or construct carefully validated reference continuations.

Do not invent candidate pairs automatically or equate `guard=Safe` with refusal.

## 10.3 Optional likelihood-scoring implementation

If straightforward, add reusable functions to compute:

- Token-level log-probabilities of a specified continuation.
- Total sequence log-likelihood.
- Mean token log-likelihood.
- Token counts.
- Refusal/compliance likelihood margin when candidates are supplied.

Use the Hugging Face Tiny Aya checkpoint and correct chat-template boundaries.

Verify token indexing carefully.

Do not call `.generate()` merely to score a known continuation.

Use teacher forcing.

### Research questions

Determine:

- Whether the original generated response should include EOS/end-of-turn tokens in scoring.
- How to reconstruct prompt and assistant token boundaries.
- Whether the tokenizer retokenizes differently at the prompt/response boundary.
- How to mask prompt tokens from the response log-likelihood.
- Whether full precision and quantized model outputs are comparable.
- How sequence length influences the metric.
- Whether the required Transformers model exposes the necessary logits.
- How to compute the score efficiently in bounded memory.

### Current scope

Do not compute candidate refusal/compliance likelihood margins for the full dataset unless validated candidate pairs exist.

Do not save enormous per-token vocabulary-logit tensors.

A small optional demonstration using non-sensitive sample text is acceptable.

The main evaluation must work even if the likelihood-scoring section is disabled.

---

# PHASE 11 — CircuitKIT readiness, not full implementation

Prepare a brief technical document explaining how Dataset2 evaluation results can later feed into CircuitKIT.

Address:

1. Selection of validated refusal-versus-compliance prompt pairs.
2. Same-language comparisons before cross-language alignment.
3. Token alignment issues.
4. Single-token versus multi-token objectives.
5. Differentiable likelihood metrics.
6. EAP/EAP-IG activation attribution.
7. Activation patching and causal validation.
8. Whether CircuitKIT's current Tiny Aya support is compatible with the installed model environment.

Distinguish:

**Supported relatively directly:** Single-token logit-difference objectives, subject to compatibility checks.

**Requires custom engineering:** Full multi-token refusal/compliance sequence-likelihood attribution through EAP-IG.

Do not assume a custom `TaskSpec` by itself solves the multi-continuation likelihood problem.

Do not train, steer, fine-tune, prune, or otherwise modify Tiny Aya in this task.

---

# PHASE 12 — Technical investigation and online research

Do not guess when the implementation relies on unclear external behavior.

Use official documentation, repository source code, model cards, installed library source code, and relevant tests.

Research the following as needed.

### A. Tiny Aya and Transformers

- Supported Transformers versions.
- Model architecture and loading.
- Chat-template behavior.
- BOS/EOS handling.
- Greedy-generation settings.
- Padding/attention masks.
- Float16 compatibility.
- Accurate token counts.
- Efficient Colab GPU inference.

### B. AuditKIT and Qwen3Guard

- Supported AuditKIT 1.1.3 interfaces or the version actually used.
- Scoring precomputed answers.
- Guard prompt formatting.
- Guard label taxonomy.
- Multilingual evaluation behavior.
- Refusal-output interpretation.
- Parser failure handling.
- Model revision pinning.

### C. Dataset2 generation

- How `dataset2.ipynb` and `dataset2v2.ipynb` constructed prompts.
- What generation metadata is retained.
- Whether rejection reasons are recoverable.
- Whether seed IDs and prompt IDs are deterministic.
- Whether the database reconstruction omitted important provenance.
- Whether language conditions correspond to the actual texts.

### D. Statistics and data quality

- Suitable seed-level paired analyses.
- Cluster-aware confidence intervals.
- Partial factorial designs.
- Missing-data sensitivity analysis.
- Multiple testing.
- Guard validation against human review.

### E. Likelihood and CircuitKIT

- Teacher-forced sequence scoring.
- Log-probabilities versus logits.
- Candidate-response selection.
- CircuitKIT custom metrics.
- Tiny Aya architecture compatibility.
- Activation patching constraints.

### Research documentation

For each significant unresolved issue, record:

- Technical question.
- Current implementation.
- Verified findings.
- Source URL and version.
- Remaining uncertainty.
- Proposed solution.
- Whether it blocks notebook execution, interpretation, or future CircuitKIT work.

Prefer a compact technical research document rather than lengthy speculative discussion.

---

# PHASE 13 — Tests and verification

The notebooks must be tested at multiple levels.

## A. Static notebook checks

Verify that:

- Both notebooks are valid `.ipynb` files.
- All imports refer to available modules.
- Execution order is correct.
- Shared functions are called consistently.
- Part 1 has `SHARD_ID = 1`.
- Part 2 has `SHARD_ID = 2`.
- Neither notebook accidentally runs the other shard.
- The notebooks do not contain hardcoded credentials.
- Default configuration covers the full assigned shard.

## B. Dataset tests

Test:

- Source database validation.
- Unique IDs.
- Deterministic shard construction.
- Correct seed isolation.
- No overlapping prompt IDs.
- Complete union of shard assignments.
- Correct metadata preservation.
- Correct handling of incomplete seeds.
- Identical original/twin detection.

## C. Checkpoint tests

Test:

- Resume behavior.
- Idempotent repeated execution.
- Configuration fingerprint changes.
- Corrupted checkpoint handling.
- Independent shard directories.
- Judge rescoring without response regeneration.

## D. Analysis tests

Test:

- Correct denominators.
- Explicit unknown states.
- Matched-cohort eligibility.
- No cross-seed mismatching.
- Cluster-aware analysis inputs.
- Combined-results equivalence where appropriate.
- Correct merge validation and duplicate detection.

## E. GPU-dependent integration

If an authorized GPU environment is available, run a bounded inference smoke test for each notebook.

A smoke test is for verifying execution, not a replacement for the full-shard run.

If gated weights or GPU resources are unavailable:

- Complete CPU-testable functionality.
- Validate notebook structure.
- Test loaders and analysis with fixtures.
- Clearly report untested GPU stages.
- Give exact instructions for execution in Colab.

Do not claim the full evaluation was completed without actually executing it.

---

# PHASE 14 — Deliverables

## Primary deliverables

### 1. `eval/eval_dataset2_part1.ipynb`

Runs all records assigned to Part 1.

### 2. `eval/eval_dataset2_part2.ipynb`

Runs all records assigned to Part 2.

Both must be usable independently by different team members.

## Required supporting deliverables

### 3. Shared evaluation implementation

Reuse and extend the repository's existing Python modules as necessary.

### 4. Authoritative partition manifest

A deterministic allocation of all eligible Dataset2 records into two nonoverlapping shards.

### 5. Validation routines

Detect structural defects and generate scientific-quality flags.

### 6. Reliable checkpoints

Independent resumable generation and judging for each notebook.

### 7. Merge implementation

Combines both notebook outputs and verifies complete coverage and compatible configurations.

### 8. Evaluation summaries

Shard-level and combined metrics with explicit denominators.

### 9. Inspection functionality

Tables or lightweight views suitable for manually reviewing prompt/response/judgment combinations.

### 10. Optional likelihood-scoring module

A reusable teacher-forced scoring function, if compatible with available runtime and scope.

### 11. Technical research notes

Concise documentation of verified interfaces, necessary assumptions, and unresolved issues.

### 12. Tests

Unit and integration tests covering partitioning, checkpoints, scoring, and merging.

---

# PHASE 15 — Definition of done

The implementation is ready for the team when:

**Notebook execution**

- Both notebooks open successfully in Google Colab.
- Each notebook can execute its configured shard independently.
- The notebooks use the same inference and scoring implementation.
- Each notebook saves progress and can resume after interruption.

**Partition correctness**

- No seed is divided between shards.
- No prompt ID is assigned twice.
- The shard union equals the intended full dataset.
- Both shards have reasonably balanced workloads.
- The partition is fixed and reproducible.

**Evaluation correctness**

- Tiny Aya uses the specified model and generation settings.
- AuditKIT scores stored responses without regenerating them.
- Safe/Unsafe/unknown and refusal measurements remain distinct.
- Expected-behavior correctness is not based blindly on unvalidated labels.
- All metrics report valid denominators.

**Result integrity**

- Shards write into separate output locations.
- Checkpoint identity prevents inappropriate reuse.
- The merger detects incompatible runs.
- The combined dataset preserves all condition metadata.
- Missing/failed/unknown records remain visible.

**Scientific integrity**

- Dataset-quality flags are preserved.
- Incomplete factorial coverage is handled explicitly.
- Within-seed comparisons are properly matched.
- Heldout data is not used for iterative tuning.
- Automated guard judgments are not presented as human ground truth.

**Technical readiness**

- Unclear dependencies have been investigated.
- Known unresolved issues are documented.
- Relevant tests pass.
- Any unexecuted GPU-dependent steps are explicitly identified.

---

# PHASE 16 — Required final Codex report

After implementation, report the following.

## A. Repository reuse

Which existing components were reused, extended, or replaced?

Explain why new code was necessary.

## B. Shard manifest

Report:

- Part 1 seed count.
- Part 1 prompt count.
- Part 2 seed count.
- Part 2 prompt count.
- Balance by category/split.
- Duplicate count.
- Missing assignment count.
- Whether exact 50/50 prompt allocation was achieved.

## C. Delivered files

List both notebooks and all supporting modules.

## D. Verification

Report actual tests executed, pass/fail results, and whether Tiny Aya/AuditKIT GPU inference was run.

## E. Technical research

Summarize confirmed implementation details and unresolved issues, with source links.

## F. Execution instructions for both team members

Provide precise instructions for:

1. Opening the notebooks in Google Colab.
2. Configuring a GPU runtime.
3. Providing Hugging Face credentials securely.
4. Making the source dataset and manifest available.
5. Running the assigned notebook.
6. Resuming an interrupted session.
7. Exporting results.
8. Combining Part 1 and Part 2.
9. Inspecting the resulting evaluation.

## G. Scientific limitations

Summarize relevant dataset-quality concerns, guard reliability limitations, and incomplete-coverage issues.

## H. Future CircuitKIT integration

Explain what remains necessary to move from generated responses and guard labels to validated likelihood metrics and circuit discovery.

---

# Final execution instructions

Begin by inspecting the actual repository and identifying reusable evaluation components.

Next, implement the shared shard manifest and its invariants.

Then adapt the existing Dataset2 evaluation workflow into the two parallel Colab notebooks.

Implement shared checkpointing, validation, analysis, and merging functionality.

Run all feasible tests and investigate technical uncertainties as they arise.

**Do not stop after writing a proposal. Produce the notebooks and supporting implementation.**

Do not run the full 21,184-prompt evaluation yourself unless explicitly instructed and a suitable authorized environment is available. The two team members will execute the completed notebooks separately.

If full GPU testing is impossible in the current environment, finish the implementation and clearly identify what remains unverified.

**The main deliverable is two operational, scientifically defensible evaluation notebooks—not a new research proposal or a CircuitKIT implementation.**