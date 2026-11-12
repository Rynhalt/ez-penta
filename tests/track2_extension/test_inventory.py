import json
import tempfile
import unittest
from pathlib import Path

from eval.track2_extension.inventory import inventory


class InventoryTests(unittest.TestCase):
    def test_inventory_summarizes_execution_without_serializing_prompt_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "Pipeline1111"
            combined = root / "combined"
            combined.mkdir(parents=True)
            (root / "pipeline1_stratified_diagnosis.ipynb").write_text(
                json.dumps(
                    {
                        "cells": [
                            {
                                "cell_type": "code",
                                "execution_count": 1,
                                "outputs": [{"output_type": "stream", "text": ["ok"]}],
                            },
                            {
                                "cell_type": "code",
                                "execution_count": 2,
                                "outputs": [
                                    {
                                        "output_type": "error",
                                        "ename": "ValueError",
                                        "evalue": "bad",
                                    }
                                ],
                            },
                        ],
                        "metadata": {"kernelspec": {"name": "python3"}},
                    }
                ),
                encoding="utf-8",
            )
            (combined / "pipeline3_safety_pairs.jsonl").write_text(
                json.dumps(
                    {
                        "seed_id": "seed-a",
                        "split": "analysis",
                        "clean_prompt": "private prompt",
                        "clean_answer": "private answer",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            output = Path(tmp) / "inventory.json"

            report = inventory(root, output)

            notebook = report["notebooks"]["pipeline1_stratified_diagnosis.ipynb"]
            self.assertEqual(notebook["executed_code_cell_count"], 2)
            self.assertEqual(notebook["code_cells_with_outputs_count"], 2)
            self.assertEqual(notebook["execution_errors"][0]["ename"], "ValueError")
            artifact = report["combined_artifacts"]["pipeline3_safety_pairs.jsonl"]
            self.assertEqual(artifact["row_count"], 1)
            self.assertEqual(artifact["unique_seed_count"], 1)
            serialized = output.read_text(encoding="utf-8")
            self.assertNotIn("private prompt", serialized)
            self.assertNotIn("private answer", serialized)

    def test_empty_approved_export_is_counted_as_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "Pipeline1111"
            (root / "combined").mkdir(parents=True)
            (root / "combined" / "pipeline3_safety_pairs.jsonl").write_text(
                "", encoding="utf-8"
            )
            report = inventory(root, Path(tmp) / "inventory.json")
            self.assertEqual(
                report["combined_artifacts"]["pipeline3_safety_pairs.jsonl"]["row_count"],
                0,
            )


if __name__ == "__main__":
    unittest.main()
