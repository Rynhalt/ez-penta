# Article dependency audit

This supplements the [experiment artifact map](../reports/EXPERIMENT_ARTIFACT_MAP.md) with local dependency decisions for archival. Original Drive links/access limitations are preserved in that map; this cleanup does not relocate external artifacts.

| Article evidence | Dependency chain retained | Archival decision |
| --- | --- | --- |
| BI language composition | Article §5 → Drive B1/B2/B4 → PEFT training and direct Qwen3Guard → frozen 519-seed pool / 475 paired seeds → saved `post_train_3/reports/source_data/bilingual_results.json` → report figure generator → Figures 1–2 | Keep aggregate, source links, configurations and statistical provenance. Local absence of B1 is not permission to discard other evidence. |
| B5 decomposition | Article §6 → working narrative → unresolved original notebook/row-level aggregation | Keep provisional audit; do not fabricate a replacement or archive possible inputs as redundant. |
| E4 and cultural generation | Article §7 → Dataset2/Qwen source notebooks + UK extension builders → `eval` loaders/runtime/statistics → source snapshot and manifests → `followups_results.json` → Figure 3 | Keep notebooks/helpers, shared guard/server/cache utilities, tests and manifest paths. |
| Japan-culture transfer | Article §7 → JA J1/J2 notebooks → `analysis_inputs/` paired answer/judge rows and manifests → analysis outputs → interim report/results | Keep four-of-six interim study, including incomplete-status and hardware limitations. `analysis_inputs/` is an actual analysis dependency. |
| Steering / E1–E3 | Article §8 → original Drive training/eval/selection artifacts → saved steering/follow-up statistical JSON → Figures 4–5 | Keep negative safety-preservation results and original selection/fallback provenance. |
| XSTEER A/B/D | Article §8 → Drive diagnostics/hook reports and configs → local A/D JSON/manifests → Figure 6; B remains Drive-backed | Keep all available evidence; C pending. Do not call direction diagnostics a discovered circuit. |
| Earlier pipeline | Article §2 → Drive post_train_2 diagnosis/probe/CuratorKIT/training/validation → pilot report | Keep provenance links; no local post_train_1/post_train_2 directories exist in this checkout. |
| Toolkit demonstration | Article §3 → `guard_judge.py` → `chat.py`/`local_server.py`; Dataset2 `eval` modules → `colab_build_cache.py`; CircuitKIT patch/tests and roadmap | Keep shared code. Active imports demonstrate dependencies. Gated notebooks have uncertain publication/continuation relevance and remain under review, not automatically archived. |

## Transitive checks

Static inspection covered Python imports, package file lists, notebook source cells, shell launcher commands, Markdown and manifests where content was locally available. In particular, `eval/dataset2_gguf.py` imports shared chat/server/build helpers, `eval/scoring.py` imports the guard parser, Dataset2 Colab packaging embeds source/data paths, and JA `code/pt3_ja/evaluate.py` uses `arm_ja/eval_work/`. These are deliberately retained.

The prior root AnswerCarefully implementations were relocated to `data_ready/` and `eval/`, not archived. They handle selection, generation/scoring, saved-output inspection and human review respectively; filename overlap is not duplicated functionality. Updated module commands are in `eval/README.md`.

Static reference absence does not prove a dynamic or externally invoked helper is unused. Batch recovery scripts, unknown checkpoint/work folders and cloud-placeholder content therefore remain under review. Their metadata is recorded in the complete file inventory; no claim is made that all contents were read.
