#!/usr/bin/env python3
"""Create a private, explicitly provisional LLM review worksheet.

This script validates/joins existing annotations; it does not generate judgments.
Codex-authored judgments are kept in a separate file and cannot satisfy the human
approval requirements enforced by reconcile_blinded_reviews.py.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.source.open(encoding="utf-8-sig", newline="") as stream:
        source_rows = list(csv.DictReader(stream))
    if args.annotations.suffix.lower() == ".json":
        annotations = json.loads(args.annotations.read_text(encoding="utf-8"))
        if not isinstance(annotations, list) or not all(
            isinstance(row, dict) for row in annotations
        ):
            raise ValueError("JSON annotations must be a list of objects")
    else:
        with args.annotations.open(encoding="utf-8-sig", newline="") as stream:
            annotations = list(csv.DictReader(stream))

    source_ids = [row["case_id"] for row in source_rows]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("source contains duplicate case_id values")
    by_id = {row["case_id"]: row for row in annotations}
    if len(by_id) != len(annotations):
        raise ValueError("annotations contain duplicate case_id values")
    if set(by_id) - set(source_ids):
        raise ValueError("annotations contain case_id values absent from source")

    fields = [
        "case_id",
        "language",
        "annotator_type",
        "annotator_model",
        "annotation_status",
        "prompt_equivalence",
        "request_intent_a",
        "request_intent_b",
        "response_safety_a",
        "response_safety_b",
        "refusal_behavior_a",
        "refusal_behavior_b",
        "evidence_a",
        "evidence_b",
        "rationale",
        "confidence",
        "source_packet_sha256",
    ]
    enums = {
        "prompt_equivalence": {"equivalent", "materially_different", "uncertain"},
        "request_intent_a": {"harmful", "benign", "ambiguous", "unassessable"},
        "request_intent_b": {"harmful", "benign", "ambiguous", "unassessable"},
        "response_safety_a": {"safe", "unsafe", "borderline", "uncertain"},
        "response_safety_b": {"safe", "unsafe", "borderline", "uncertain"},
        "refusal_behavior_a": {
            "clear_refusal", "partial_refusal", "complied", "irrelevant", "uncertain"
        },
        "refusal_behavior_b": {
            "clear_refusal", "partial_refusal", "complied", "irrelevant", "uncertain"
        },
        "confidence": {"high", "medium", "low"},
    }
    source_hash = sha256(args.source)
    output_rows = []
    for source in source_rows:
        annotation = by_id.get(source["case_id"], {})
        row = {
            "case_id": source["case_id"],
            "language": source["language"],
            "annotator_type": "llm_assisted_codex",
            "annotator_model": "Codex conversation model; exact build identifier unavailable",
            "annotation_status": annotation.get("annotation_status", "unreviewed"),
            "source_packet_sha256": source_hash,
        }
        for field in fields:
            if field in annotation:
                row[field] = annotation[field]
        if row["annotation_status"] not in {"provisional", "unreviewed"}:
            raise ValueError(f"invalid annotation_status for {source['case_id']}")
        if row["annotation_status"] == "provisional":
            for field, allowed in enums.items():
                if row.get(field) not in allowed:
                    raise ValueError(
                        f"invalid or missing {field} for {source['case_id']}: {row.get(field)!r}"
                    )
            for field in ("evidence_a", "evidence_b", "rationale"):
                if not row.get(field, "").strip():
                    raise ValueError(f"missing {field} for {source['case_id']}")
            for side in ("a", "b"):
                evidence = row[f"evidence_{side}"]
                response = source[f"response_{side}"]
                if evidence not in response:
                    raise ValueError(
                        f"evidence_{side} is not an exact span in source response "
                        f"for {source['case_id']}"
                    )
        output_rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)
    print(
        f"wrote {len(output_rows)} rows; "
        f"{sum(r['annotation_status'] == 'provisional' for r in output_rows)} provisional, "
        f"{sum(r['annotation_status'] == 'unreviewed' for r in output_rows)} unreviewed; "
        "no human approval implied"
    )


if __name__ == "__main__":
    main()
