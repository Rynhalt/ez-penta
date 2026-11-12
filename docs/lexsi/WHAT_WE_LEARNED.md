# What we learned so far

This is a plain-language interpretation of the current LEXSI Track-2 evidence, current as of 2026-11-12. It distinguishes established engineering facts from provisional behavioral signals and open hypotheses.

## What the dataset suggests

After reparsing raw safety-guard records, the project recovered 3,643 additional binary response labels without changing any of the 13,730 rows that already had binary labels. A seed-level analysis finds more Unsafe guard labels on US-framed than neutral harmful prompts in English, Japanese, and mixed-language data. The analysis and tune intervals are mostly above zero.

That is evidence of an association in this dataset and this automated guard. It does not prove the prompts were semantically interchangeable, that the model truly complied unsafely, or that culture framing caused the difference. The estimator averages different framing and category rows within each seed, so those factors can still contribute. The heldout split has already been seen in prior project work and cannot serve as fresh final confirmation.

The earlier row-level Mantel–Haenszel estimate reported higher unsafe-guard odds for US versus neutral framing, while Japan versus neutral was not significant. Its original-label cohort, stratum tables, MH odds ratios, corrected CMH tests, and RBG intervals have since been independently reproduced from the original shards. A separate corrected-label analysis also used seed-aware matched estimates. Both remain automated guard-proxy associations, not human-validated unsafe compliance or causal culture effects.

One earlier strict matching attempt appeared to find no English/Japanese matched rows. That conclusion was invalid: a code bug treated the dataset's NULL mix-direction value (not applicable to monolingual rows) as missing. After correction, the matched analysis ran on both original result shards; see `SOURCE_RECONCILIATION.md` for exact estimates, denominators, and limits.

## What the model experiments tell us

Tiny Aya Water's original Hugging Face checkpoint and CircuitKIT can produce closely matching log probabilities on the tested English, Japanese, and long-context inputs. This removes an important engineering uncertainty: CircuitKIT can represent this model closely enough for continued experiments.

It does **not** show which parts of the model cause a refusal. No Tiny Aya EAP-IG circuit has been discovered or causally tested. CPU tests of the adapter and position mapping used synthetic models/data.

The actual GPU smoke test ran baseline, restricted-SFT, and broad-SFT arms and got the same guard counts for each. With only four seeds, previously exposed prompts, and one unsafe judgment per arm, the interval collapses to zero and has essentially no ability to establish a useful effect. Equal counts here are not proof that the arms are equivalent. They also do not show that the earlier fine-tune failed.

## What is not ready to claim

- No human-approved refusal/compliance pair has been exported.
- Automated guard labels and automatically generated preference responses are not validated ground truth.
- The proposed likelihood margin has not yet been shown to distinguish reviewed refusal from reviewed compliance on Tiny Aya.
- Attribution scores, if later found, will nominate components for tests; they will not by themselves establish a mechanism.
- There is no validated before/after safety improvement, no causal circuit result, and no article-ready claim that a circuit-restricted change helped.
- The old training checkpoints are not recoverable as complete durable artifacts, so the historical tuned arms cannot currently be independently reproduced from their saved weights.

## The best next move

The critical next action is human review, not more model computation. Two independent reviewers should assess the prepared blinded packets, followed by adjudication. That creates a defensible set for checking whether prompt pairs really hold intent constant and whether one answer refuses while the other complies. Reviewers must also identify the exact continuation text to score.

Once that evidence exists, validate the continuation likelihood metric on tune seeds. Then use disjoint discovery seeds for CircuitKIT EAP-IG and separate evaluation seeds for patching/ablation. Preserve benign prompts, random-component controls, seed-level uncertainty, and all negative findings. Only then decide whether to train and run the larger paired before/after study needed for a Hugging Face article.

## Current blockers

1. The 87-row blinded reviewer packets still have no human judgments and the adjudication template remains blank. A separate one-pass LLM/Codex screen exists, but all 87 rows are provisional; it cannot satisfy human approval.
2. `outputs/lexsi/circuitkit_pairs.jsonl` is empty, correctly preventing unapproved data from entering real-data circuit execution.
3. Colab has 99.90 compute units and no active session or assignment. The next HF model load requires the owner to fetch the `HF_TOKEN` Colab Secret from the browser UI; CLI-only secret access times out. GPU quota is not the current blocker.
4. Restricted and broad training weights were not durably saved; Drive search found a restricted file only 0.7% of the previously observed runtime weight size; downloading it and opening it with `safetensors.safe_open` fails because its header metadata is incomplete. This confirms that the historical tuned checkpoints are unavailable from the inspected Drive location. The broad folder is empty.
5. The historical Pipeline3 accepted preference export was recoverable: 22 rows map to 21 distinct Dataset2 seeds through an exact prompt-ID join. The separate Pipeline2 mechanistic probe seed list was not preserved, so the seed registry for a fresh causal evaluation remains incomplete.
5. The existing “heldout” prompts were previously exposed; a fresh seed-disjoint confirmatory cohort must be designated or collected.

See [FINAL_REPORT.md](FINAL_REPORT.md) for numerical detail and [REPRODUCE.md](REPRODUCE.md) for the commands and artifact map.

## Latest execution and review status

The last saved Phase 1 L4 notebook did pass three HF↔CircuitKIT parity probes, but its final cell threw because the approved contrast export was empty; Colab CLI incorrectly returned exit code 0. A new auditor now detects that cell error and returns failure. The notebook has been changed to record the expected missing-approval gate cleanly, but it has not yet been rerun on Colab because the CLI cannot fetch `HF_TOKEN` without a browser-UI cell.

The 87 LLM/Codex annotations remain provisional. The approved export is still zero bytes. The Track-2 suite currently has 84 tests (83 pass, one expected CircuitKIT-local skip). Colab is at 99.90 CUs with no active runtime. No Tiny Aya behavioral metric, EAP-IG discovery, causal intervention, or validated model improvement has been established.

### CircuitKIT integration check

We found one additional engineering bug before discovery: the first token-position patch did not reach the CircuitKIT backend used by the real `discover_circuit()` function. We extended it and passed a 9-test synthetic Colab CPU suite. That makes the code path testable; it does not establish Tiny Aya behavior or a refusal mechanism. The public CircuitKIT API synthetic smoke outputs and full session history are also backed up in a new Drive folder: `https://drive.google.com/drive/folders/1p_tYjBiSIPuZ_d--RIJLZzRh3TaoUJxG`. Drive list-folder readback confirmed all uploaded names and byte sizes; local SHA256 values and Drive IDs are in `outputs/lexsi/circuitkit_public_api_smoke_20261112/drive_upload_receipt.json`. The discovery and validation notebooks are prepared but stop unless actual human-approved data, a validated metric, and fresh evaluation seeds are present.
# Update: strict matched guard-proxy analysis (2026-11-12)

Recovered and reconciled the original two response SQLite shards. In the exact seed/design-cell matched analysis, US-framed harmful rows had positive unsafe guard-proxy differences versus neutral in all nine split/language groups; some analysis/tune intervals include zero, and all heldout results are descriptive because those prompts were exposed. This pattern is consistent with an association, but says nothing causal and does not replace human response review. Exact counts, intervals, and limitations are in `SOURCE_RECONCILIATION.md`.

The wider research goal is still not met: no human-approved CircuitKIT contrasts, validated Tiny Aya behavior metric, real-model EAP-IG discovery, intervention/ablation result, or reliable before/after improvement estimate exists. The historical training checkpoints have not been durably recovered. See `STATUS.md` for the current gates.



Pipeline 1 was independently reproduced from the original result shards: all four archived MH comparisons, the 4,069-row cohort, language-specific tables, ORs, continuity-corrected CMH tests and RBG intervals match. This confirms the arithmetic for the historical automated guard analysis. It does not upgrade the result to a causal or human-validated finding.
