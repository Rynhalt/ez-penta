import json
import tempfile
import unittest
from pathlib import Path

from scripts.lexsi.audit_notebook_execution import audit_notebook


class NotebookExecutionAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path("outputs/lexsi"))
        self.path = Path(self.temp.name) / "executed.ipynb"

    def tearDown(self):
        self.temp.cleanup()

    def write_notebook(self, outputs):
        notebook = {
            "nbformat": 4,
            "nbformat_minor": 5,
            "cells": [{
                "cell_type": "code", "execution_count": 3,
                "metadata": {}, "source": ["raise RuntimeError('example')"],
                "outputs": outputs,
            }],
        }
        self.path.write_text(json.dumps(notebook), encoding="utf-8")

    def test_clean_execution_passes(self):
        self.write_notebook([{"output_type": "stream", "name": "stdout", "text": ["ok"]}])
        report = audit_notebook(self.path)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["error_count"], 0)
        self.assertEqual(report["code_cell_count"], 1)

    def test_cell_exception_is_failure_even_if_cli_returned_zero(self):
        self.write_notebook([{
            "output_type": "error", "ename": "RuntimeError", "evalue": "blocked",
            "traceback": ["Traceback detail"],
        }])
        report = audit_notebook(self.path)
        self.assertEqual(report["status"], "CELL_ERRORS")
        self.assertEqual(report["errors"][0]["ename"], "RuntimeError")
        self.assertEqual(report["errors"][0]["traceback"], ["Traceback detail"])

    def test_rejects_non_notebook_json(self):
        self.path.write_text(json.dumps({"items": []}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Jupyter notebook"):
            audit_notebook(self.path)


if __name__ == "__main__":
    unittest.main()
