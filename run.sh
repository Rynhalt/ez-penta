#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    printf '%s\n' \
        'Usage: bash run.sh [--setup-only | --split all|dev|test --limit N --max-tokens 1024 --output-dir PATH]' \
        'By default, evaluates all 1,800 Japanese ACv2.2 dev and test prompts.' \
        'Installs missing macOS prerequisites, creates .venv, installs packages,' \
        'signs in to Hugging Face if needed, and runs the AnswerCarefully experiment.' \
        'Requires Homebrew and approved access to llm-jp/AnswerCarefully.'
    exit 0
fi

setup_only=false
if [[ "${1:-}" == "--setup-only" ]]; then
    setup_only=true
    shift
    if [[ "$#" -ne 0 ]]; then
        printf '%s\n' '--setup-only does not accept experiment options.' >&2
        exit 1
    fi
fi

if ! command -v python3.12 >/dev/null || ! command -v llama-server >/dev/null; then
    if [[ "$(uname -s)" != "Darwin" ]] || ! command -v brew >/dev/null; then
        printf '%s\n' 'Install Python 3.12 and llama-server first. On macOS, install Homebrew from https://brew.sh.' >&2
        exit 1
    fi
    command -v python3.12 >/dev/null || brew install python@3.12
    command -v llama-server >/dev/null || brew install llama.cpp
fi

if [[ ! -e .venv && ! -L .venv ]]; then
    if [[ "$(uname -s)" == "Darwin" ]]; then
        # Keep installed packages outside iCloud-managed Desktop/Documents folders.
        environment_root="${EZ_PENTA_ENV_ROOT:-$HOME/Library/Application Support/ez-penta/venvs}"
        workspace_id=$(python3.12 -c 'import hashlib, sys; print(hashlib.sha256(sys.argv[1].encode()).hexdigest()[:12])' "$PWD")
        environment_path="$environment_root/$workspace_id-python3.12"
        mkdir -p "$environment_root"
        if [[ ! -d "$environment_path" ]]; then
            python3.12 -m venv "$environment_path"
        fi
        ln -s "$environment_path" .venv
    else
        python3.12 -m venv .venv
    fi
fi
if ! .venv/bin/python -c 'import sys; raise SystemExit(sys.version_info[:2] != (3, 12))'; then
    printf '%s\n' 'Existing .venv is not Python 3.12. Rename it and rerun; it will not be deleted automatically.' >&2
    exit 1
fi

printf '%s\n' 'Installing project dependencies in .venv...'
.venv/bin/python -m pip install --disable-pip-version-check --no-input --only-binary=:all: -r requirements.txt
.venv/bin/python -m pip check
if [[ "$setup_only" == true ]]; then
    printf '%s\n' 'Setup complete. Run bash run.sh to start the experiment.'
    exit 0
fi

# Reuse the saved login; the helper prompts only when no credential is available.
.venv/bin/python huggingface_auth.py

exec .venv/bin/python -u evaluate_answercarefully.py run "$@"
