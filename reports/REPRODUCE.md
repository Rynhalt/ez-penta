# Reproducing the Tiny Aya report

This guide distinguishes regenerating a chart from reproducing a model result. Those are different levels of evidence.

## Reproducibility levels

| Level | What it reproduces | Current status |
|---|---|---|
| 1. Figures from saved aggregates | The report plots and CSV tables from saved machine-readable results. | **Runnable in this checkout** for BI, initial steering, E1–E4 and XSTEER D, after installing plotting dependencies. XSTEER A/B diagnostics are report/Drive-backed and not plotted by the local script. Figure generation does not rerun judgments or models. |
| 2. Metrics from per-example judgments | Descriptive rates and paired analyses from completed model answers and judge outputs. | **Partial.** Some raw BI, JA and XSTEER files are in Drive; BI B4 analysis source is available there. Raw rows are not fully mirrored into this checkout, and row-level B5 decomposition is not reconstructed. |
| 3. Evaluation from released prompts and models | Generate answers with exact model/adapters/settings, then judge and analyze. | **Blocked as a public reproducible run.** Prompts, full adapters, immutable base revision, complete package manifests and selected original notebooks are not in the public checkout. Drive access is account-dependent. No HF collection was verified. |
| 4. Training from released data/configs | Recreate adapters from training rows and pinned configurations. | **Partial / not yet externally reproducible.** BI B1 source and manifests were recovered in Drive; adapter weights and complete input package need validated public release. post_train_2 checkpoints were not durably verified in the searched archive. |
| 5. Historical exact reconstruction | Reproduce every reported experimental run and all selection decisions. | **Not currently possible from GitHub alone.** B5 source/output is missing; XSTEER A/D notebooks and some raw rows are absent from the inspected Drive archive; some Drive files are not publicly accessible; post_train_2 checkpoints are incomplete. |

## Regenerate figures

From the repository root, with Python and pandas, NumPy, matplotlib installed:

```bash
python reports/figures/generate_figures.py
python reports/validate_report.py
```

The generator reads the aggregate JSONs and XSTEER D structured results under `post_train_3/reports/source_data/`. It creates six PNG/SVG pairs and six plotted-data CSV tables. The report does not contain a B5 plot because the decomposition is not independently verified. `validate_report.py` checks saved pooled counts, paired estimates, figure/source presence, abstract word limit and toolkit mentions; visual inspection is still required for layout.

The main aggregate sources and hashes are listed in [`post_train_3/reports/source_data/README.md`](../post_train_3/reports/source_data/README.md). XSTEER A/D recovered result artifacts and hashes are also listed there. The chart command does not regenerate model answers, Qwen3Guard labels, bootstrap tests, training adapters or CircuitKIT results.

## Source notebooks and Drive artifacts

Original experiment code is currently split between GitHub and Drive:

- BI planning, training, generation and analysis: [bilingual Drive notebooks](https://drive.google.com/drive/folders/1vMX28_SSW1Rr9mzz-Ca65Thfvq9zVjh5); source scripts [B1](https://drive.google.com/file/d/1C8TSP0gzo6FVHdOUYqo266knJYLfbGqN/view), [B2](https://drive.google.com/file/d/1vJRTYPOlRgzSk_aZQzZ7SBpjV5FwbvR0/view), [B4](https://drive.google.com/file/d/1hOG8aiAv45D_FsOq40WZoLsnO1E1BDFI/view).
- post_train_2 pipeline and notebooks: [report](https://drive.google.com/file/d/1ZQ5UFfDdba4gjGiV2fvSqRUZ9s9rkpl7/view), [notebook directory](https://drive.google.com/drive/folders/1h37qxJyOvsFX9a_oJJq1pYM3DR5jHbqQ).
- Initial steering: [steering archive](https://drive.google.com/drive/folders/1b2NwRg4iFP8UmoJ636EMbrGK-OmE_Q1T).
- E1–E4: [November 15 run archive](https://drive.google.com/drive/folders/1bH59zkx1BlKXcqvR3L-NAbh3fbby3ltQ).
- XSTEER A/D reports: [A](https://drive.google.com/file/d/1_YNBp427gfJd4iDEFLf53P6-ZteQVzJj/view), [D](https://drive.google.com/file/d/1YZ3_lXGTlq7jvW0RZPQ1wU5NSKWPOXRJ/view), [artifact index](https://drive.google.com/file/d/1FlMshU2K7VEzAGE1ZmZkgumtN1Y8rRjH/view).
- XSTEER B all-layer diagnostic: [report](https://drive.google.com/file/d/1Cl04yW5MPBhQUVnykWh0O-ZUNhikkizl/view), [RESULTS](https://drive.google.com/file/d/1AaxY73_o8mC3FmF5GcrDv5nMkNphXdiZ/view), [structured JSON](https://drive.google.com/file/d/10Om79lHcquHI6V5hocD0NljcgIJllPih/view). Its aggregate conclusion is inspected, but the large activation and direction arrays were not uploaded. XSTEER C was still running and is not represented as a result here.
- Japan-culture extension: [interim local report](../post_train_3/japan_culture/INTERIM_REPORT.md); the chunk data and manifests are Drive-backed.

Connector access to these files was available to the report authoring session. Public link access, model redistribution rights, and all required input checksums have not been verified for an unauthenticated reader. Drive-only links must not be presented as a public replication package.

## Intended clean public workflow

1. Obtain the licensed, pinned base checkpoint and tokenizer.
2. Download versioned training/evaluation manifests and inputs from a public dataset repository, or a clearly documented controlled-access source.
3. Train adapters using the released notebook/config and record hardware, package versions, model revision, item IDs and adapter hashes.
4. Generate responses with the released prompt template, decoding config and base/adapter checkpoints.
5. Run the exact guard versions and parser, preserving Controversial/unknown outcomes as unknown.
6. Run paired statistics with the released analysis code and frozen seed mapping.
7. Generate figures and run the report validation.

## Not included / blockers

- No training or model evaluation is run by the figure script.
- B5 refusal-decomposition source and structured output must be recovered or re-derived from saved judgment rows before that result can be reproduced.
- XSTEER D’s report/results/manifest are available, but its original execution notebooks and answer/judge rows were not located in the inspected Drive index.
- XSTEER B’s report and structured aggregate are available in Drive, but the 2.28 GB activation archive and 20.7 MB all-layer direction file were not uploaded. The results can be checked from the saved summary, but full representation extraction cannot be rerun from the current public checkout.
- Exact immutable Tiny Aya model revisions are not recorded consistently in accessible source bundles.
- Historical post_train_2 restricted/broad checkpoints were not verified as complete and loadable in the searched Drive location.
- Prompt/response release requires license, privacy and safety review. Some raw outputs contain harmful content and must not be copied into a public repository without review.
- The CircuitKIT preparation tests validate software plumbing on synthetic/toy inputs; they do not reproduce any article claim and must remain clearly separate from research results.
- Human validation of the automated safety labels is outstanding.

## Final presentation check

The article was rendered to HTML with Python Markdown (tables and fenced-code extensions), then to a screenshot and PDF with headless Chrome on 2026-11-16. The opening page was visually inspected, and the six individual figure images were inspected separately. This is a local browser rendering check, not verification of GitHub’s exact renderer or anonymous Drive access. Temporary previews are in `/tmp/lexsi-article-preview.{html,png,pdf}` and are not release artifacts.
