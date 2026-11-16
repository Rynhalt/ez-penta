# Research archive

This directory preserves historical material moved during the owner-authorized repository reorganization on November 16, 2026. Files were moved, not deleted. The [migration manifest](MIGRATION_MANIFEST_20261116.json) records every original/destination path, file size and SHA-256 before the move; each hash was checked at its destination.

| Path | Contents | Status |
| --- | --- | --- |
| `reports/20261105T134711Z/` | Earlier AnswerCarefully report, analysis scripts, statistics, PDF and previews | Historical cohort; superseded as the main article |
| `runs/answercarefully-20261103T003252-655004Z/` | Completed AnswerCarefully response/judge run and server logs | Provenance, not a current experiment |
| `local_artifacts/evaluations/` | Completed local evaluation snapshots | Git-ignored private material |
| `local_artifacts/evaluation-review-20261105T132948-323308Z/` | Historical local review outputs | Git-ignored private material |
| `session_notes/` | Earlier `codex.md` and `goal.md` planning/session notes | Historical instructions, not current contributor guidance |

The current [article](../reports/experiment_report.md) and [reproduction guide](../reports/REPRODUCE.md) remain outside the archive. Current contributor rules are in [AGENTS.md](../AGENTS.md).

Historical outputs may contain raw model content. Moving them here does not authorize publication or satisfy CircuitKIT review gates. Previously tracked outputs remain tracked until a separate release review. Ignored local artifacts must remain ignored.

To restore a moved bundle, use the manifest's `from`/`to` pair, check that the original path is free, and move the bundle back. The archived report's scripts and reproduction commands were adjusted for the new location; the historical snapshots and saved result files were preserved.
