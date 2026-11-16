# Historical research archive

The active repository is organized around the [technical article](../reports/experiment_report.md). This archive preserves superseded documentation, completed legacy workflows, prior session notes and regenerable caches. Archiving means a file is outside the active presentation; it does not mean an experiment failed or lacked scientific value.

The [manifest](ARCHIVE_MANIFEST.csv) lists every file moved in this cleanup, its original path, current path, rationale, related experiment, canonical replacement, move method and verification method. The inventory/classification audit is [here](../docs/ARCHIVE_AUDIT.md).

## Contents

- `superseded_docs/`: previous root workflow README, report-authoring review and earlier root-layout audit. Their contents are preserved; the current entry point and audit supersede them.
- `reports/20261105T134711Z/`: original AnswerCarefully report, statistical snapshots, PDF and reproduction scripts.
- `runs/`: completed historical AnswerCarefully run and logs.
- `session_notes/`: earlier task instructions, preserved as history rather than active contributor rules.
- `local_artifacts/`: private saved evaluation/review bundles, still Git-ignored.
- `regenerable/`: moved Python/test caches, Git-ignored; tests may create new caches outside this archive.
- `MIGRATION*.json` and `README_previous_layout.md`: earlier migrations and archive index. Paths inside these are historical snapshots; current locations are obtained by replacing the former `archive/` prefix with `_archive/`.

## Traceability and restoration

Files were moved without changing their contents. Tracked current sources were moved with `git mv`; inherited archive files and untracked/ignored files were moved through the filesystem. Earlier migrations had already relocated some tracked paths without staging them, so Git may show deletions at their historical locations and untracked files here until a focused review/staging pass. No commit or history rewrite is part of this cleanup.

For a move listed in the CSV, check that the original destination is free and move the archived file back. Use `git mv` for tracked files after they have been staged/committed, and a filesystem move for ignored/untracked files. A future commit preserves rename history; `git log --follow -- <path>` locates it. Do not use a broad `git restore` that would discard unrelated work. Local ignored artifacts cannot be recovered from Git; keep their private archive backup.

Large or cloud-placeholder inherited artifacts were verified by preserved inode and size; their recorded prior SHA-256 is used where available. The CSV explicitly distinguishes this from a fresh content-hash check. No cloud copy is assumed equivalent without evidence.

Archived Markdown retains its original content, including historical relative paths and commands. Use the CSV to resolve those paths; the active source/commands are documented in the root README and `eval/README.md`. Previously published URLs to moved tracked files may require the old Git revision or this manifest; no duplicate forwarding notebooks were created.

The training-language, cultural, steering/XSTEER evidence and unique negative results remain active or explicitly indexed. Saved prompts/responses remain untrusted and may contain restricted or harmful content. This move does not approve public release, labels or CircuitKIT execution exports. No Google Drive or Hugging Face files were moved.
