import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


FAKE_PYTHON = """#!/usr/bin/env bash
set -eu
printf '%s\\n' "$*" >> "$BOOTSTRAP_CALLS"
if [[ "${1:-}" == "-m" && "${2:-}" == "venv" ]]; then
    mkdir -p "$3/bin"
    cp "$0" "$3/bin/python"
elif [[ "${1:-}" == "-c" ]]; then
    if [[ "$2" == *"get_token"* ]]; then
        exit "${BOOTSTRAP_NO_TOKEN:-0}"
    fi
    exit "${BOOTSTRAP_BAD_VERSION:-0}"
elif [[ "${1:-}" == "-m" && "${2:-}" == "pip" && "${3:-}" == "install" ]]; then
    exit "${BOOTSTRAP_INSTALL_FAILURE:-0}"
fi
"""


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.workspace = tempfile.TemporaryDirectory()
        self.addCleanup(self.workspace.cleanup)
        self.root = Path(self.workspace.name)
        self.bin_dir = self.root / "bin"
        self.bin_dir.mkdir()
        shutil.copyfile(Path(__file__).resolve().parents[1] / "run.sh", self.root / "run.sh")
        (self.root / "requirements.txt").write_text("auditkit==1.1.3\n")
        for name, body in (("python3.12", FAKE_PYTHON), ("llama-server", "#!/bin/sh\nexit 0\n")):
            executable = self.bin_dir / name
            executable.write_text(body)
            executable.chmod(0o755)
        self.calls = self.root / "bootstrap-calls.txt"
        self.environment = {
            **os.environ, "PATH": f"{self.bin_dir}:{os.environ['PATH']}",
            "BOOTSTRAP_CALLS": str(self.calls),
        }

    def launch(self, *args, **environment):
        result = subprocess.run(
            ["bash", str(self.root / "run.sh"), *args], input="", capture_output=True,
            text=True, env={**self.environment, **environment}, timeout=10,
        )
        calls = self.calls.read_text() if self.calls.exists() else ""
        return result, calls

    def test_setup_creates_environment_and_checks_packages_without_running_models(self):
        result, calls = self.launch("--setup-only")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("-m venv .venv", calls)
        self.assertIn("--only-binary=:all:", calls)
        self.assertIn("-m pip check", calls)
        self.assertNotIn("evaluate_answercarefully", calls)

    def test_default_launch_forwards_experiment_options(self):
        result, calls = self.launch("--limit", "1", "--max-tokens", "256")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("-u evaluate_answercarefully.py run --limit 1 --max-tokens 256", calls)

    def test_failed_install_does_not_start_experiment(self):
        result, calls = self.launch(BOOTSTRAP_INSTALL_FAILURE="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("evaluate_answercarefully", calls)

    def test_wrong_interpreter_is_reported_without_deleting_existing_environment(self):
        result, calls = self.launch(BOOTSTRAP_BAD_VERSION="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("not Python 3.12", result.stderr)
        self.assertTrue((self.root / ".venv/bin/python").exists())
        self.assertNotIn("-m pip install", calls)

    def test_noninteractive_login_has_clear_instructions(self):
        result, calls = self.launch(BOOTSTRAP_NO_TOKEN="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("hf auth login", result.stderr)
        self.assertNotIn("evaluate_answercarefully", calls)
