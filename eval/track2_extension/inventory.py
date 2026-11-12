"""Inventory Track-2 Drive artifacts without emitting prompt/response text.

This module is safe to run against the private Pipeline1111 Drive tree. It records
artifact metadata, hashes selected small source files, and summarizes executed
notebooks and JSON/JSONL outputs without copying their private contents to stdout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


NOTEBOOKS = (
    "pipeline1_stratified_diagnosis.ipynb",
    "pipeline2_circuit_discovery.ipynb",
    "pipeline3_curatorkit_data.ipynb",
    "pipeline4_circuit_restricted_fix.ipynb",
    "pipeline5_validation.ipynb",
)
SMALL_ARTIFACTS = (
    "pipeline1_baseline_mh_stats.json",
    "pipeline2_results.json",
    "pipeline3_safety_pairs.jsonl",
    "pipeline4_results.json",
    "pipeline5_results.json",
)
PROMPT_FILES = (
    "pipeline3_seed_prompts.json",
    "pipeline5_validation_prompts.json",
)
WEIGHT_SUFFIXES = {".safetensors", ".pt", ".pth", ".ckpt", ".bin"}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, hash_limit: int = 10 * 1024 * 1024) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": str(path),
        "exists": path.is_file(),
    }
    if path.is_file():
        size = path.stat().st_size
        record["size_bytes"] = size
        record["sha256"] = sha256_file(path) if size <= hash_limit else None
        record["hash_skipped_over_bytes"] = hash_limit if size > hash_limit else None
    return record


def notebook_record(path: Path) -> dict[str, Any]:
    record = file_record(path)
    if not path.is_file():
        return record
    notebook = json.loads(path.read_text(encoding="utf-8"))
    cells = notebook.get("cells", [])
    code_cells = [cell for cell in cells if cell.get("cell_type") == "code"]
    errors = []
    for index, cell in enumerate(cells):
        for output in cell.get("outputs", []):
            if output.get("output_type") == "error":
                errors.append(
                    {
                        "cell_index": index,
                        "ename": output.get("ename"),
                        "evalue": output.get("evalue"),
                    }
                )
    record.update(
        {
            "cell_count": len(cells),
            "code_cell_count": len(code_cells),
            "executed_code_cell_count": sum(
                cell.get("execution_count") is not None for cell in code_cells
            ),
            "code_cells_with_outputs_count": sum(
                bool(cell.get("outputs")) for cell in code_cells
            ),
            "code_cells_with_nonempty_outputs_count": sum(
                any(
                    output.get("output_type") == "error"
                    or bool(output.get("text"))
                    or bool(output.get("data"))
                    for output in cell.get("outputs", [])
                )
                for cell in code_cells
            ),
            "execution_errors": errors,
            "kernel": notebook.get("metadata", {}).get("kernelspec", {}).get("name"),
            "python_version": notebook.get("metadata", {}).get("language_info", {}).get(
                "version"
            ),
        }
    )
    return record


def jsonl_record(path: Path) -> dict[str, Any]:
    record = file_record(path)
    if not path.is_file():
        return record
    count = 0
    first_keys: list[str] = []
    seed_values: set[str] = set()
    split_counts: dict[str, int] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            count += 1
            if count == 1:
                first_keys = sorted(row)
            seed = row.get("seed_id") or row.get("source_seed_id")
            if seed is not None:
                seed_values.add(str(seed))
            split = row.get("split")
            if split is not None:
                split_counts[str(split)] = split_counts.get(str(split), 0) + 1
    record.update(
        {
            "row_count": count,
            "unique_seed_count": len(seed_values),
            "split_counts": split_counts,
            "first_row_keys": first_keys,
            "contains_prompt_text": any("prompt" in key for key in first_keys),
            "contains_response_text": any("answer" in key for key in first_keys),
        }
    )
    return record


def prompt_file_record(path: Path) -> dict[str, Any]:
    record = file_record(path)
    if not path.is_file():
        return record
    rows = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(rows, dict):
        # Pipeline2's extracted prompt bundle is a language-keyed mapping.
        record["top_level_keys"] = sorted(rows)
        record["row_count"] = sum(
            len(value)
            for value in rows.values()
            if isinstance(value, list)
        )
        record["contains_prompt_text"] = True
    elif isinstance(rows, list):
        record["row_count"] = len(rows)
        record["first_row_keys"] = sorted(rows[0]) if rows else []
        record["contains_prompt_text"] = any(
            key in (rows[0] if rows else {}) for key in ("prompt", "instruction")
        )
    else:
        record["json_type"] = type(rows).__name__
    return record


def find_checkpoint_candidates(roots: list[Path]) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path in seen:
                continue
            seen.add(path)
            name = path.name.lower()
            is_weight = path.is_file() and path.suffix.lower() in WEIGHT_SUFFIXES
            is_adapter_metadata = path.is_file() and name in {
                "adapter_config.json",
                "adapter_model.safetensors",
                "config.json",
            }
            if is_weight or is_adapter_metadata:
                found.append(
                    {
                        "path": str(path),
                        "size_bytes": path.stat().st_size if path.is_file() else None,
                    }
                )
    return sorted(found, key=lambda item: item["path"])


def inventory(pipeline_root: Path, output_path: Path) -> dict[str, Any]:
    combined = pipeline_root / "combined"
    result: dict[str, Any] = {
        "schema_version": 1,
        "pipeline_root": str(pipeline_root),
        "notebooks": {
            name: notebook_record(pipeline_root / name) for name in NOTEBOOKS
        },
        "combined_artifacts": {},
        "prompt_inputs": {},
        "checkpoint_candidates": find_checkpoint_candidates([pipeline_root]),
    }
    for name in SMALL_ARTIFACTS:
        path = combined / name
        result["combined_artifacts"][name] = (
            jsonl_record(path) if name.endswith(".jsonl") else file_record(path)
        )
        if path.suffix == ".json":
            result["combined_artifacts"][name]["json"] = json.loads(
                path.read_text(encoding="utf-8")
            ) if path.is_file() else None
    for name in PROMPT_FILES:
        result["prompt_inputs"][name] = prompt_file_record(pipeline_root / name)
    result["checkpoint_search_roots"] = [str(pipeline_root)]
    result["checkpoint_weights_found_in_search_roots"] = bool(
        result["checkpoint_candidates"]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(output_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = inventory(args.pipeline_root, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "notebooks": {
                    name: {
                        "exists": row["exists"],
                        "executed_code_cell_count": row.get("executed_code_cell_count"),
                        "code_cells_with_outputs_count": row.get(
                            "code_cells_with_outputs_count"
                        ),
                        "error_count": len(row.get("execution_errors", [])),
                    }
                    for name, row in result["notebooks"].items()
                },
                "generated_training_pair_rows": result["combined_artifacts"]
                .get("pipeline3_safety_pairs.jsonl", {})
                .get("row_count"),
                "human_approved_circuit_export_rows": None,
                "checkpoint_candidates": len(result["checkpoint_candidates"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
