#!/usr/bin/env python3
"""Audit executed Jupyter notebook outputs for cell exceptions.

Colab CLI can return exit code 0 even when an executed notebook contains an
error output. This checker makes the notebook's cell-level result authoritative.
Keep generated reports private when they include tracebacks from private runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def audit_notebook(path: Path) -> dict[str, Any]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(notebook, dict) or not isinstance(notebook.get("cells"), list):
        raise ValueError("input is not a Jupyter notebook with a cells list")

    code_cells = 0
    errors = []
    for cell_index, cell in enumerate(notebook["cells"]):
        if not isinstance(cell, dict):
            raise ValueError(f"cell {cell_index} is not an object")
        if cell.get("cell_type") == "code":
            code_cells += 1
        for output_index, output in enumerate(cell.get("outputs", [])):
            if output.get("output_type") != "error":
                continue
            errors.append({
                "cell_index": cell_index,
                "execution_count": cell.get("execution_count"),
                "output_index": output_index,
                "ename": output.get("ename", ""),
                "evalue": output.get("evalue", ""),
                "traceback": output.get("traceback", []),
            })

    return {
        "notebook": path.name,
        "notebook_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "code_cell_count": code_cells,
        "error_count": len(errors),
        "status": "PASS" if not errors else "CELL_ERRORS",
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", type=Path)
    parser.add_argument("--output", type=Path, help="Write full JSON audit report here")
    args = parser.parse_args()
    report = audit_notebook(args.notebook)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "errors"}, indent=2))
    for error in report["errors"]:
        print(
            f"cell {error['cell_index']} output {error['output_index']}: "
            f"{error['ename']}: {error['evalue']}"
        )
    return 1 if report["error_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
