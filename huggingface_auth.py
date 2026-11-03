"""Reuse the saved Hugging Face login, or prompt securely when it is missing."""

import getpass
import os
import sys

from huggingface_hub import get_token, login
from huggingface_hub.constants import HF_HOME, HF_TOKEN_PATH


def preserve_login_location():
    # Changing AuditKIT's XDG cache must not relocate Hugging Face credentials.
    os.environ.setdefault("HF_HOME", HF_HOME)
    os.environ.setdefault("HF_TOKEN_PATH", HF_TOKEN_PATH)


def has_terminal():
    if sys.stdin.isatty():
        return True
    try:
        with open("/dev/tty") as terminal:
            return terminal.isatty()
    except OSError:
        return False


def ensure_login():
    if get_token():
        print("Using the existing Hugging Face login.", flush=True)
        return
    if not has_terminal():
        raise RuntimeError(
            "No saved Hugging Face login and no interactive terminal. "
            "Run .venv/bin/hf auth login in your terminal, then rerun bash run.sh."
        )
    print("No saved Hugging Face login. Enter a read token from your approved account.", flush=True)
    token = getpass.getpass("Hugging Face token (input hidden): ").strip()
    if not token:
        raise ValueError("No token entered; login cancelled")
    login(token=token, add_to_git_credential=False)


if __name__ == "__main__":
    try:
        ensure_login()
    except (EOFError, KeyboardInterrupt):
        sys.exit("\nLogin cancelled.")
    except Exception as error:
        # Login failures must never dump the entered token or request headers.
        sys.exit(f"Hugging Face login failed ({type(error).__name__}). Run .venv/bin/hf auth login to retry.")
