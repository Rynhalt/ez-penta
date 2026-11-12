# New cohort plan after Dataset2 Tune exhaustion

**Status:** AnswerCarefully v2.2 snapshot downloaded locally for audit only; no examples have been scored.  
**Updated:** 2026-11-12

## Why a new cohort is needed

The aggregate-only audit of the two original Dataset2 response shards found 79 fully eligible matched Tune seeds and zero unexposed fully eligible Tune seeds under the recorded training, selection, and review exclusions. Two incomplete request keys from one seed could potentially be recovered, which cannot satisfy the current five-non-tied-seed metric gate. Reusing the already screened seeds after metric redesign would leak tuning decisions. See `outputs/lexsi/tune_pair_recovery_audit_20261112.json`.

Keep the existing seven-pair Phase 1 result as exploratory and failed/inadequate. Do not alter its metric or threshold to make it pass.

## Separate the two data needs

### A. Mechanism cohort: new matched roleplay contrasts

Create new source behaviors that have not appeared in Dataset2 or the previous Pipeline1111 work. For each behavior, prepare a same-intent control prompt and roleplay-wrapped prompt in the same language. A bilingual author/reviewer should verify that the underlying request and risk are equivalent and that the framing change is the intended difference. Collect model responses under the pinned Tiny Aya HF checkpoint and chat template, then label harmful intent, prompt equivalence, substantive compliance/refusal, and exact response spans without using guard labels as ground truth.

Before any scoring or selection, reserve unique source seeds into:

- discovery (analysis) only;
- metric development/validation (tune) only;
- fresh final evaluation only.

The existing CircuitKIT plan's floor is 32 independent discovery seeds; the causal notebook additionally requires at least 8 fresh reviewed seeds per language. Five non-tied Tune seeds is only the mathematical minimum for an exact one-sided sign test to reach p below .05 when all signs agree, not an adequately powered sample. Determine larger Tune and evaluation counts from an explicit power/precision target and estimated seed-level variance before collection.

For the current one-sided exact sign-test gate at alpha=.05, an exact binomial scenario calculation illustrates the distinction: if the true per-seed probability of the expected direction were .80, 5 non-tied seeds would have only .328 power (all five must agree). The first n reaching 80% power in this discrete test is 18 seeds (reject at 13/18; achieved power .867). At assumed success probabilities .70 and .75, the corresponding first n values are 37 and 23. These are **planning scenarios**, not estimates of this metric's accuracy; they ignore ties and the additional requirement that absolute refusal/compliance signs be correct on a majority. So they are lower-bound illustrations, not recommended final sample sizes. Reproduce with `python -m scripts.lexsi.metric_sign_test_power`; exact output is `outputs/lexsi/metric_sign_test_power_scenarios_20261112.json`.

Do not use the present seven Phase 1 probes for metric tuning, discovery, or final evaluation. Preserve Dataset2 as a separate historical/exploratory dataset and do not call it a new-seed test set.

### B. External behavior evaluation: answer safety and over-refusal

Use external datasets as separate, clearly named evaluation cohorts, with source licenses and test splits preserved:

- **AnswerCarefully (ACv2.2 / ACv3)** is a Japanese safety dataset. ACv2 has 1,800 question/answer pairs, including a 336-example test split; v2.2 adds English translations and translation/culture metadata. ACv3 adds 300 regionally sensitive examples, 100 test and 200 development. Its maintainers state that the samples are not definitive safety ground truth; AC data redistribution is prohibited. It may support a private, cited external Japanese safety/appropriateness evaluation, but should not be repackaged or used as a substitute for the roleplay-mechanism cohort. Sources: [NII dataset documentation](https://llmc.nii.ac.jp/en/answercarefully-dataset/) and [paper](https://arxiv.org/abs/2506.02372).
- **APTO-XSTest-JA** provides Japanese-localized XSTest prompts, reference responses and manual refusal reasoning, with safe prompts where over-refusal is a concern and unsafe prompts where refusal is expected. It is useful for over-refusal/under-refusal controls. Its card reports CC BY 4.0 for the Japanese translation/annotations, with caveats for model-generated completions. It does not test the same control-versus-roleplay manipulation. Source: [Hugging Face dataset card](https://huggingface.co/datasets/APTO-001/APTO-XSTest-JA).

An external test set is not “fresh” if any of its items are used to choose components, train adapters, tune the metric, or select checkpoints. Keep original test portions untouched and report the cohort separately from Dataset2.

### Current AnswerCarefully snapshot audit

A pinned local snapshot was fetched at revision `d1d4e57cd4d7aecb44f879e7488832011fcaddb0` and stored at the ignored path `data_ready/answercarefully.sqlite3` (SHA256 `5fd99d9c294f3435f771e0b1dae6d22d232e1b0892815fd31e6577c10acb549f`; 3,108,864 bytes). The aggregate audit is `outputs/lexsi/answercarefully_snapshot_audit_20261112.json`; its script emits no prompt/reference text or per-record IDs. SQLite integrity passed; there are 1,800 unique records (1,464 dev, 336 test), complete Japanese prompts/references and English prompt translations, but no English reference answers. Adaptation tag `0` occurs in 1,067 dev and 245 test rows; tags 1/2 and 349 nonempty translation notes mark cases needing special care in bilingual comparison. The metadata has risk categories but no explicit `refusal_required` label.

The repo contains an Aya AnswerCarefully evaluation notebook, but it has no saved execution counts or outputs. The prior project used AnswerCarefully-related generation/evaluation workflows, and accessible Drive search did not locate the referenced evaluation result files. Therefore prior Tiny Aya exposure of the 336-row source test split is unresolved: do not call it a fresh confirmatory test or expose it to component/metric selection. The `dev` rows may be screened for candidate seed construction only after language equivalence, request intent, and prompt safety are reviewed. Do not copy the non-redistributable dataset to Drive or Git.

## Metric protocol to freeze before data collection

1. Keep the current continuation-likelihood objective and its specified tokenization/boundary handling as the first candidate. The Phase 1 metric did not validate; do not assume the current objective works.
2. Define any candidate redesign only from a written hypothesis about the failure (e.g., full target text is semantically and stylistically different, and continuation lengths differ), without inspecting new Tune outcomes. List candidates and the primary metric before scoring.
3. Use Tune seeds only for a predeclared selection among those candidates. Freeze the selected definition, normalization, response spans, threshold, and success rule before accessing final-evaluation labels.
4. Require the frozen metric to distinguish adjudicated target behavior from negative controls (both-refuse, both-comply, benign answer) in each language. Report all score signs/magnitudes and seed-cluster uncertainty; a paired ordering alone is not enough to classify unsafe compliance.
5. Verify the CircuitKIT scalar and gradient positions against the independent Hugging Face teacher-forcing implementation on the same approved rows. This integration parity is separate from behavioral validity.

## Model-improvement evaluation

Once a suitable intervention/checkpoint is available, compare baseline, circuit-restricted, and broad-LoRA arms on the exact same untouched manifest and decoding settings. Evaluate substantive unsafe compliance on harmful prompts and over-refusal/helpfulness on benign controls. Keep Qwen3Guard as a secondary automated proxy, retain Controversial/Missing as unknown, and cluster uncertainty by source seed. Do not compare historical GGUF baseline outputs directly to new HF tuned outputs. Add a random-component training arm only if there is enough data and compute for the matched design.

## Immediate next actions

1. Confirm owner-selected source route: commission/author new bilingual matched roleplay seeds, or nominate a licensed external benchmark plus a new matched contrast set.
2. Write and freeze the seed-level split manifest and power/precision target before generation or annotation.
3. Obtain bilingual review/adjudication of prompt equivalence and response behavior; preserve LLM-assisted labels as such, not as independent human annotations.
4. Collect responses only for the new seeds, save privately with immutable hashes, and then evaluate the frozen metric on Tune seeds.
5. Proceed to Tiny Aya EAP-IG only if the metric, seed separation, data quality, and prompt/activation-alignment gates pass.
