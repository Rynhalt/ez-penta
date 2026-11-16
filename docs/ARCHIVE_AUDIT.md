# Article-driven archive audit — 2026-11-16

## Scope and inspection limits

This cleanup inventories **1,120 files** in the non-environment worktree before the archive moves. It excludes Git internals, active/offloaded virtual environments and symlink targets; it does not claim to inspect installed packages. Metadata was enumerated for every included file. **208 text/source/notebook files had content inspected by the inventory pass**; further article/dependency reads supplement that pass. Cloud placeholders and large raw files were not silently counted as content-reviewed. Generated reports and audit files created after the baseline are outside those baseline counts.

The per-file [inventory](REPOSITORY_FILE_INVENTORY.csv) records four classifications: **43 KEEP — Core**, **140 KEEP — Dependency**, **196 ARCHIVE — Safe**, and **741 REVIEW — Uncertain**. The uncertain group is deliberately broad: it includes unhydrated cloud files, raw execution artifacts and dynamically/externally invoked recovery code. It is not a claim that 741 files are obsolete. Keep/review files remain at their established paths.

**196 files were moved into `_archive/`.** Most of that count comprises regenerable caches or integration of previously archived material; it is not 196 discarded experiments. The [archive manifest](../_archive/ARCHIVE_MANIFEST.csv) distinguishes original paths, reasons, canonical sources, verification methods and Git/filesystem moves. The [article dependency map](ARTICLE_DEPENDENCY_MAP.md) supplements the [scientific artifact map](../reports/EXPERIMENT_ARTIFACT_MAP.md).

## What changed

- Integrate the former `archive/` into the required `_archive/`, retaining prior manifests and historical documentation.
- Preserve the long previous root README in `_archive/superseded_docs/`; replace the active root README with a concise article-driven entry point and experiment/artifact links.
- Archive the superseded root-layout audit and internal report-authoring review. Current claim/audit/reproduction documents remain active.
- Move regenerable Python/test caches and Finder metadata to ignored `_archive/regenerable/`; tests can regenerate caches, so their reappearance is not a failed scientific cleanup.
- Update current documentation links and private-artifact ignore rules. Historical Markdown and migration JSONs retain their original content and recorded paths; the archive index explains prefix resolution.
- Preserve earlier implementation relocations: AnswerCarefully selection is in `data_ready/`, evaluation/inspection/review in `eval/`, the dataset snapshot in `data_ready/dataset2_snapshot/`, and UI assets in `eval/review_ui/`.

Tracked current sources are moved with `git mv`; inherited archive paths and untracked/ignored files use filesystem moves. Existing uncommitted changes are preserved. No commit, push, history rewrite, GPU run or external Drive/HF cleanup is performed. Git staging from `git mv` is limited to moved tracked sources; unrelated changes are not committed.

## Active experiments and deliberate retention

All article experiment groups remain indexed: BI/en/ja/bi/en2, provisional B5, US/Japan/UK framing, original steering, E1–E4, XSTEER A/B/D and the earlier restricted/broad pilot. The checkout has no `EXPERIMENTS_SUMMARY.md`, `post_train_1/` or `post_train_2/`; relevant original Drive artifacts remain linked, not assumed absent from the project.

- Keep machine-readable figure sources, statistical outputs, model/configuration provenance and negative steering results. Article text and scientific aggregate hashes are protected before/after cleanup.
- Keep JA `analysis_inputs/`: the interim paired analysis consumes these rows/manifests. Keep `arm_ja/eval_work/`: `code/pt3_ja/evaluate.py` uses it. Neither is merely a redundant download folder.
- Keep shared chat/server/build-cache/guard helpers: Dataset2 and AnswerCarefully import them. File naming does not establish redundant functionality.
- Keep uncertain batch launch/recovery scripts, checkpoints, published manifests and external-notebook paths. No active-imported module or source dataset is moved into the archive.
- Keep gated CircuitKIT notebooks under REVIEW rather than presenting them as completed core findings or discarding them automatically. Executed diagnostic/patch provenance and approval rules remain available.
- Keep unique negative results, failed-run details and evidence needed to understand limitations. Automated reviews remain provisional; no CircuitKIT export is authorized by this cleanup.

## Validation

The final validation record is appended below after execution. The figure generator and report validator check saved aggregate-based reporting, not independent replication of model inference. Local test suites use simulated/local fixtures; no expensive experiment is rerun.

## Reproducibility and release limitations

Drive and Hugging Face files are untouched. Existing access to Drive is not evidence of anonymous/public accessibility. B5's original computation and finalized C remain unavailable/unverified for the canonical article; complete immutable model revisions, input/adapter bundles and human safety-label validation remain release gaps.

Large/cloud-placeholder inherited files were moved without content writes and checked for preserved inode/size; prior recorded hashes are carried forward where available. This is explicitly distinguished from fresh SHA-256 verification. Local ignored artifacts are not recoverable through Git; tracked files retain Git history after a reviewed commit, while ignored bundles require the private backup.

This repository is easier to navigate, but **not yet a complete public training/evaluation reproduction release**. Raw harmful content, model/data licensing, secret review and public model/dataset publication remain separate release tasks. No archive location is a publication approval.


### Executed validation results

- `python reports/figures/generate_figures.py`: six PNG/SVG pairs and six plotted-data CSVs regenerated from saved structured results.
- `python reports/validate_report.py`: passed; six figure pairs, nine paired-statistic claims and report/abstract caveats checked.
- Fourteen affected unit-test suites ran in fresh subprocesses: **110 tests discovered, 109 passed, one skipped**, zero suite failures. Covered dataset selection, AnswerCarefully runner, saved-result inspection, human-review server, shell launcher, Colab notebook, guard, chat/server/auth helpers, Dataset2 pipeline/GGUF, UK generation and UK HF evaluation. Test discovery was intentionally scoped to these affected workflows; this is not a claim of executing every test file in the inventory.
- Active Markdown links resolve in 14 entry-point/article/provenance documents.
- Twelve protected article, abstract and structured scientific-result files match their pre-cleanup SHA-256 values. No scientific claims/results were changed.
- All 196 archive destinations exist uniquely; fresh-content hash checks pass where performed. Large/cloud verification limitations are recorded per file.
- Six current chart images were visually inspected together after regeneration.
- `git diff --cached --check --no-renames`: passed. Full unstaged whitespace check status is recorded below once the cloud-object read finishes.

Machine-readable test/hash/link outcomes are in [ARCHIVE_VALIDATION.json](ARCHIVE_VALIDATION.json). The archive CSV is the source of truth for moves; the file inventory is a baseline classification snapshot rather than a current Git index.

- Final `git diff --check`: **passed, exit 0**. The same HEAD objects were fetched into a temporary object store to avoid cloud-file reads; original files/history were not replaced. Cached whitespace check also passed.
- After validation, the owner explicitly requested a reviewed commit and push. Publication scope excludes ignored private archives/caches, raw new experiment runs, and unrelated operational logs.
