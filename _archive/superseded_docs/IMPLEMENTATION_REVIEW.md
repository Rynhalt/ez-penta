# Article implementation and completion review

## Gap assessment and changes

The prior article had numerical results but insufficient explanation of training units, toolkit attribution, cohorts and intervention identity. The revision documents frozen-base layer-18 LoRA, the language-composition versus projection-scope distinction, accepted aligned training items, the English×2 duration control, model-answer/guard-label separation, paired statistics and distinct evaluation populations. It corrects an ambiguous use of “seed”: a training seed controls an adapter run, while a source evaluation seed identifies a request and its prompt variants. The abstract and overview qualify the English×2 result: the supported bilingual advantage is on Japanese prompts.

## Strongest supported findings and sources

1. Bilingual versus English-only Unsafe-rate paired changes are −2.75 percentage points on English prompts and −6.00 on Japanese prompts. Against English×2, the Japanese change is −7.43 points; the English comparison is inconclusive. Source: `post_train_3/reports/source_data/bilingual_results.json`, `primary_g06` H1/H3 keys, with 4B robustness results. These are three-training-run paired analyses, unlike pooled descriptive counts.
2. Initial steering reduced English harmless refusal by 5.78 points while increasing Unsafe judgments and failing the prespecified preservation criteria. Source: `steering_results.json` H1/H2/H3; protocol and caveats in `technical_appendix.md`.
3. Original US prompts differ from constrained rewrites; matched UK–US intervals include zero. Source: `followups_results.json`, E4 cue-content and matched-cue contrasts. This supports controlling wording, not cultural equivalence or causal cultural claims.

## Recovered methodology and artifacts

Drive recovery located BI B1/B2/B4 source and completed notebook material, post_train_2 pipeline notebooks/report, steering aggregates/protocols, and XSTEER A/B/D reports and structured summaries. B1 verifies training hashes, accepted-item flags, language and refusal/helpful balance; B2 verifies prompt hashes and uses checkpointed evaluation with batch scheduling. The exact source URLs, local paths and hashes are in `EXPERIMENT_ARTIFACT_MAP.md` and `source_inventory.json`.

SafeTune and CuratorKIT support the earlier probing/data-generation pilot. AuditKit implements the separate repository Qwen3Guard interface. Main BI/XSTEER guard calls are direct. CircuitKIT preparation/synthetic checks do not establish a Tiny Aya circuit.

## Decomposition and steering status

B5 was **not independently validated**: a narrative was recovered, but not the source computation and complete row-level aggregation. Its numbers are explicitly provisional and excluded from the submission abstract and charts. XSTEER A/B diagnostics and D intervention results are incorporated with their distinct protocols. XSTEER C remains pending. Placeholder text has been removed from the submission abstract by including supported evidence and omitting unfinished findings; no one-to-one equivalence between every original prompt filename and A/B/D is assumed.

## Figures and validation

Six embedded figures use saved structured BI, steering, follow-up and XSTEER D results; PNG, SVG and plotted CSV outputs are supplied. No B5 or all-layer B chart is fabricated from narrative numbers. Figure-generation and report-validation commands completed successfully. The validator checks pooled counts, nine paired-statistic values, source/figure presence, abstract length and specific caveats; it does not independently rerun model judgments or every statistical test. Individual figures and the browser-rendered article opening were visually inspected. Local article links resolve. Exact GitHub rendering and anonymous Drive access were not verified.

Commands actually run:

```bash
python reports/figures/generate_figures.py
python reports/validate_report.py
```

A temporary Markdown installation and headless Chrome were also used for an HTML/PDF/screenshot preview under `/tmp`; no repository dependency change was required. No training, inference, destructive migration, commit or push was performed.

## Reproducibility and release

Chart regeneration is available locally. Full external experimental replication remains incomplete because immutable model revisions, reviewed prompts, complete adapter/input packages, environment manifests and some original notebooks/raw rows are not publicly bundled. XSTEER B's large activation/direction arrays are not uploaded. Human validation of automated labels is outstanding. These gaps are explicitly permitted as documented limitations of this evidence-recovery task; they are not claimed solved.

The proposed repository structure preserves current working paths: canonical report material in `reports/`, governance and reproduction documentation in `docs/`, and existing `data_ready/`, `eval/`, `notebooks/`, `post_train_3/`, `scripts/` and `tests/` namespaces. The migration plan identifies dependencies and release/storage concerns before any future move.

Next three public-release priorities:

1. Review and publish an eligible, versioned input/model/configuration bundle with hashes and immutable model revisions, preserving licensing and CircuitKIT approval gates.
2. Recover or recompute B5 from saved judgments and incorporate finalized C only after its protocol and outputs are audited.
3. Create a public repository entry-point/index, mirror sanitized source notebooks and verify reproduction from a clean checkout before implementing any path migration.
