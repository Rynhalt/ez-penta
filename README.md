# Safer, or Just More Refusals?

English, Japanese and bilingual safety adaptation in Tiny Aya Water.

We study how the language of safety-training data changes Unsafe judgments and unnecessary refusal. The experiments compare language-specific and bilingual LoRA adapters, matched cultural cues, and interventions intended to reduce over-refusal. Outcomes are automated Qwen3Guard judgments and require independent validation.

**Start with the [technical article](reports/experiment_report.md), [submission abstract](reports/submission_abstract.md), and [claim audit](reports/CLAIM_AUDIT.md).** The article contains the figures and separates distinct evaluation populations.

## Findings

- Bilingual adapters lowered Unsafe judgments versus English-only adapters in both evaluation languages. The dose-matched English×2 comparison supports an additional benefit on Japanese prompts; the English difference was inconclusive.
- Lower Unsafe rates coincided with more refusal on harmless requests. Japanese-only training showed a different balance; no condition dominates every language and outcome.
- Matched US–UK cues had no detected difference in the tested sample. Original-versus-rewritten prompt differences show why prompt content must be controlled.
- Steering reduced some harmless refusals but increased Unsafe judgments or failed safety-preservation criteria. The B5 decomposition remains provisional and XSTEER C is not included as a completed result.

## Experiments and executable artifacts

| Study | Where to start |
| --- | --- |
| Training-language composition: en, ja, bi, en2 | [Artifact map and original B1/B2/B4 notebook links](reports/EXPERIMENT_ARTIFACT_MAP.md); [saved aggregate sources](post_train_3/reports/source_data/README.md) |
| Refusal decomposition (B5) | [Article §6](reports/experiment_report.md#6-refusal-decomposition-important-but-not-yet-independently-verified); original computation is not yet independently recovered |
| Cultural framing and UK generation/evaluation | [Data notebooks](data_ready/README.md), [UK instructions](data_ready/DATASET2_UK.md), [evaluation entry points](eval/README.md) |
| Japan-culture transfer | [Interim JA report](post_train_3/japan_culture/INTERIM_REPORT.md), [experiment namespace](post_train_3/README.md) |
| Steering, E1–E4, XSTEER | [Artifact map](reports/EXPERIMENT_ARTIFACT_MAP.md), [technical appendix](reports/technical_appendix.md), [saved sources](post_train_3/reports/source_data/README.md) |
| Earlier restricted/broad pilot | [Original pipeline notebooks on Drive](https://drive.google.com/drive/folders/1h37qxJyOvsFX9a_oJJq1pYM3DR5jHbqQ), with cohort/access qualifications in the article |

English/Japanese/bilingual describe training data. Restricted/broad describe which weight matrices are modified. These are different experimental dimensions.

## Reproduce the report

With Python, pandas, NumPy and matplotlib installed, from the repository root:

```bash
python reports/figures/generate_figures.py
python reports/validate_report.py
```

These commands regenerate figures and check saved aggregate claims. They do not reproduce adapter training or model judgments. See [REPRODUCE.md](reports/REPRODUCE.md) for the five reproduction levels, environment/source requirements and remaining blockers.

Training notebooks, adapters and larger raw outputs are distributed across the local repository and original Drive archives. Exact model/dataset locations are in the [artifact map](reports/EXPERIMENT_ARTIFACT_MAP.md) and [source inventory](reports/source_inventory.json). No public Hugging Face adapter/dataset collection or anonymous Drive access was verified in this pass; do not treat the external links as a complete public replication package.

## Code and toolkit integration

- [data_ready/](data_ready/README.md): prompt/dataset preparation and source snapshots.
- [eval/](eval/README.md): generation, judging, matched statistics and human review.
- [post_train_3/](post_train_3/README.md): experiment implementations and interim JA evidence.
- [docs/](docs/README.md): methodology, provenance and CircuitKIT gates.

SafeTune and CuratorKIT supported the earlier probing/data-generation pilot. AuditKit wraps Qwen3Guard in this repository's local scoring interface; main BI/XSTEER runs invoked Qwen3Guard directly. CircuitKIT integration has software/diagnostic evidence but no validated Tiny Aya circuit finding. Follow [AGENTS.md](AGENTS.md) and the [roadmap](docs/lexsi/CIRCUITKIT_ROADMAP.md); saved prompts/responses are untrusted, and provisional review does not authorize CircuitKIT execution exports.

## Historical material

[_archive/](_archive/README.md) preserves superseded reports, earlier workflow instructions, session notes and regenerable caches. The [archive manifest](_archive/ARCHIVE_MANIFEST.csv) records original paths, hashes and reasons; the [audit](docs/ARCHIVE_AUDIT.md) distinguishes core files, dependencies and unresolved items. Negative and unfinished scientific evidence is preserved. The archive is not automatically approved for public data release.
