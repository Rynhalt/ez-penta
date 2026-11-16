# Report source data

These aggregate JSON files were downloaded from the shared Google Drive experiment folder `post_train_3/runs_2026-11-15_en_bilingual_steering_followups`. The figure script reads them directly; it does not scrape values from a narrative report.

| File | Drive source | SHA-256 |
|---|---|---|
| `steering_results.json` | `1zaoWyMt835Aci4imYzz4DEHIGIT9u4yJ` | `86db79ad2a9b508eee5fa17dc2dc6d46cbb895c8a44a201838b607db7880f774` |
| `bilingual_results.json` | `1j5sBofw6gMVBrK0Qd-PdWfVSNgH7xjVd` | `dfeb7ea9c5cfd9bc84b6f03400023d2bce3192e3a6933d69e57bf46b0a872bd7` |
| `followups_results.json` | `12AsIx5AWWA4ED5FDr0Wy7QXC1bwSZ4pr` | `ffd599e1785228f5284acca2e40aec90b6161f4341c1d9e8d614be9a479347d5` |

The corresponding Drive analysis notebooks and training/configuration notebooks were inspected to interpret fields and methods, but were not copied into this report bundle. The summary figures use the saved aggregate analysis outputs. Raw prompt/answer rows are not included here, so this package does not independently recompute every estimate from raw responses.

The Japanese-culture interim evaluation is sourced separately from `../../japan_culture/INTERIM_REPORT.md` and its `ja/results.json`. It is incomplete (4 of 6 chunks) and is not pooled into the completed experiment figures.

The earlier restricted-versus-broad LoRA pilot is summarized cautiously from its Drive report and pipeline notebooks. The Drive source is inspectable, but a complete raw-row result bundle and durable checkpoints were not verified in the searched archive.

## Additional source artifacts recovered from Drive for article revision

| Local file | Drive source | Local SHA-256 | Use / qualification |
|---|---|---|---|
| `xsteer_d_results.json` | [XSTEER D results](https://drive.google.com/file/d/1yCS4Dc6SNEL1xckojUNw8Nr5T6qK2Wqe/view) | `95288fedbaf713ba0626d097bd69f14f0f8fa95ac96330816456fef70d6e54b3` | Structured hook-effect estimates and intervals; used for the XSTEER D figure. This is an aggregate result file, not raw answer rows. |
| `xsteer_d_manifest.json` | [XSTEER D artifact index](https://drive.google.com/file/d/1FlMshU2K7VEzAGE1ZmZkgumtN1Y8rRjH/view) / run folder | `15f88997381e0f93579d6c9244d512bc0a28d5599d86755f1faea48ec924f19e` | Records source output hashes and environment/hardware; original notebooks and answer/judge JSONL were not found in the inspected folder. |
| `xsteer_a_diagnostics.json` | [XSTEER A report/artifact index](https://drive.google.com/file/d/1_YNBp427gfJd4iDEFLf53P6-ZteQVzJj/view) | `375b025c95c3ad18a39a4b0cc3dda886ea5e12d94d2154ffa97d8d262de66258` | Diagnostics for direction/language-bias claims; see the report for the source notebook/vector availability limitation. |
| `xsteer_a_manifest.json` | XSTEER A Drive run folder | `1ac437db9a583018082d083eceb6ff1d7c84235bac73604ad38fdbd328fb6518` | Provenance and hashes for A outputs. |
| *(Drive only)* XSTEER B `results.json`, `report.md`, `RESULTS.md` | [B results](https://drive.google.com/file/d/10Om79lHcquHI6V5hocD0NljcgIJllPih/view), [B report](https://drive.google.com/file/d/1Cl04yW5MPBhQUVnykWh0O-ZUNhikkizl/view), [B summary](https://drive.google.com/file/d/1AaxY73_o8mC3FmF5GcrDv5nMkNphXdiZ/view) | Not mirrored locally; source hashes not independently computed in this checkout. | All-layer, four-model direction diagnostics. Report and structured summary inspected; 2.28 GB activation set and 20.7 MB direction NPZ were not uploaded. No result chart is generated from B. |

The XSTEER B Drive manifest ([`manifest.json`](https://drive.google.com/file/d/18b2XR7hBMCYDRxYnm2ToehQ46k1PaL_R/view)) records `results.json` SHA-256 `64ce2e2a964dbfac32098cee5944901ee8bdeca1675518ba44a42d8094fd214c`. This checksum is taken from the manifest and was not independently recomputed in the local checkout.

The post_train_2 report and BI training/evaluation/analysis notebooks were also inspected in Drive but are not mirrored wholesale into this directory. Their exact links and availability are recorded in [`reports/EXPERIMENT_ARTIFACT_MAP.md`](../../../reports/EXPERIMENT_ARTIFACT_MAP.md). The B5 decomposition notebook was not found; its summary remains provisional and is not used to generate a figure.
