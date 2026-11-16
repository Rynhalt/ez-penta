# Root layout audit

A dependency establishes that a file is needed; it does not establish that the file belongs at the repository root. The first cleanup was conservative. This follow-up removes two avoidable top-level directories and records why other items remain.

| Root item(s) | Evidence / current role | Disposition |
| --- | --- | --- |
| `marcus/` | Dataset snapshot used by validation, tests and Colab packaging | Moved to `data_ready/dataset2_snapshot/`; live consumers updated |
| `review_ui/` | Assets served by `review_results.py` | Moved to `eval/review_ui/`; server path updated |
| `codex.md`, `goal.md` | Historical task/session text | Moved to `archive/session_notes/` |
| `README.md`, `AGENTS.md`, `.gitignore`, `requirements.txt` | Repository entry point, contributor rules, exclusions and environment | Root conventions; retain |
| `DATASET2_PLAN.md` | Original Dataset2 design document | Useful documentation; root placement is historical, not technically required |
| `chat.py`, `guard_judge.py`, `local_server.py`, `colab_build_cache.py` | Imported by evaluation/runtime/scoring and packaged into Colab notebooks | Retain current import paths for this pass; future package extraction needs coordinated import and notebook-asset updates |
| `answercarefully_dataset.py`, `huggingface_auth.py` | Imported by the AnswerCarefully runner | Same compatibility constraint; not intrinsically root-only |
| `evaluate_answercarefully.py`, `inspect_evaluation.py`, `review_results.py`, `run.sh` | Documented executable entry points | Keep commands working; can later migrate with wrappers and updated instructions |
| `answercarefully_eval.sqlite3`, `eval_aya_answers.jsonl`, `eval_scores.jsonl` | Ignored saved-input defaults in inspection and saved-mode notebooks | Private local inputs; not public source files. Relocation requires a separate default/input-path migration |
| `.venv` | Active environment symlink | Retain; ignored |
| `.venv-offloaded-*`, `__pycache__/`, `.pytest_cache/`, `.DS_Store` | Ignored environment remnants/caches | Not scientific source; not part of a public release. No environment/cache deletion performed |
| `data_ready/`, `eval/`, `notebooks/`, `scripts/`, `tests/` | Working code/notebooks and regression tests | Keep functional paths; add navigation indexes |
| `post_train_3/`, `outputs/`, `runs/` | Active experiment/provenance/checkpoints and current local execution destinations | Preserve active paths and manifests; historical completed bundles selectively archived |
| `reports/`, `docs/`, `archive/` | Canonical article, methods/governance and historical artifacts | Public navigation structure, with private archive inputs ignored |

The repository is organized more clearly, but it has not been converted into a fully packaged `src/` architecture. Such conversion is broader than archival: imports, command wrappers, notebook templates, embedded payloads and reproduction instructions must change together. No assertion is made that every remaining root file is inherently required at that location.

Validation note: running the notebook-pipeline and GGUF suites in the same Python process exposed temporary-module leakage: a notebook setup test left `local_server.__file__` pointing into a removed temporary extraction. These suites are therefore checked in separate fresh processes; this does not require a runtime implementation change.

Follow-up checks passed in separate fresh processes: review UI 8/8; Dataset2 pipeline 35/35; GGUF 18 passed and 1 skipped (19 discovered). The regenerated Colab bundle uses asset hash `67fc2e257dab29ffd5b4b3b1719f1bc6b011e945da20de37c2cbcbb519ef58b3`. Report validation also passes. No GPU inference was run.


## AnswerCarefully implementation follow-up

Four more Python implementations were relocated rather than archived:

| Previous root path | Current module | Responsibility |
| --- | --- | --- |
| `answercarefully_dataset.py` | `data_ready/answercarefully_dataset.py` | Dataset loading and sample selection |
| `evaluate_answercarefully.py` | `eval/evaluate_answercarefully.py` | Model response generation and AuditKit judgment orchestration |
| `inspect_evaluation.py` | `eval/inspect_evaluation.py` | Read-only reconciliation/statistics of saved outputs |
| `review_results.py` | `eval/review_results.py` | Local human-review server |

These files have distinct behavior, not duplicate implementations. Existing shell launchers, imports, test mock targets, Colab source-upload mapping and current usage instructions were updated. Run them from the repository root as `python -m data_ready.answercarefully_dataset`, `python -m eval.evaluate_answercarefully`, `python -m eval.inspect_evaluation`, and `python -m eval.review_results`. No root forwarding copies were left. The shared inference/guard/server helpers remain source dependencies for both Dataset2 and AnswerCarefully, not duplicate AnswerCarefully runners.

Validation: 27 AnswerCarefully/dataset/inspection/review/launcher/notebook tests passed together after module relocation. Expected Controversial-label warnings remain explicit. The original code contents were hash-verified immediately after the moves; necessary import/path changes followed. See `archive/MIGRATION_ANSWERCAREFULLY_20261116.json`.
