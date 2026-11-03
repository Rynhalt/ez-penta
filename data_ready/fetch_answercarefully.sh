#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    printf '%s\n' \
        'Usage: bash data_ready/fetch_answercarefully.sh [--split all|dev|test] [--db-path PATH]' \
        'Downloads llm-jp/AnswerCarefully once and freezes it into a local SQLite file' \
        '(default: data_ready/answercarefully.sqlite3), so later runs need no Hugging Face access.' \
        'Signs in to Hugging Face if no saved login is found. Requires .venv from bash run.sh --setup-only.'
    exit 0
fi

if [[ ! -e .venv && ! -L .venv ]]; then
    printf '%s\n' 'No .venv found. Run bash run.sh --setup-only first to install dependencies.' >&2
    exit 1
fi

# Reuse the saved login; the helper prompts only when no credential is available.
.venv/bin/python huggingface_auth.py

exec .venv/bin/python -u data_ready/fetch_answercarefully_to_sqlite.py "$@"
