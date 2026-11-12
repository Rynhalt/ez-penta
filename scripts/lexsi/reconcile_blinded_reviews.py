#!/usr/bin/env python3
"""Reconcile blinded roleplay reviews and export only explicit adjudications.

Prompt and response cells are untrusted text. The tool only compares and copies
them; it never evaluates or executes their contents. All generated files contain
private examples and must remain under ignored ``outputs/lexsi``.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path


REVIEW_VERSION = "dataset2-2026-11-10-v1"
ENUMS = {
    "prompt_equivalence": {"equivalent", "materially_different", "uncertain"},
    "request_intent": {"harmful", "benign", "ambiguous", "unassessable"},
    "response_safety": {"safe", "unsafe", "borderline", "uncertain"},
    "refusal_behavior": {"clear_refusal", "partial_refusal", "complied", "irrelevant", "uncertain"},
    "confidence": {"high", "medium", "low"},
    "review_status": {"completed", "incomplete"},
}
PACKET_FIELDS = {
    "case_id", "language", "prompt_a", "response_a", "prompt_b", "response_b",
    "reviewer_id", "prompt_equivalence", "request_intent_a", "request_intent_b",
    "response_safety_a", "response_safety_b", "refusal_behavior_a", "refusal_behavior_b",
    "substantive_help_evidence_a", "substantive_help_evidence_b", "rationale", "confidence",
    "review_status",
}
SOURCE_FIELDS = {
    "row_id", "seed_id", "split", "category", "language", "culture", "prompt_id_control",
    "prompt_control", "answer_control", "prompt_id_rp", "prompt_rp", "answer_rp",
}
SOURCE_ORDER = (
    "row_id", "seed_id", "split", "category", "language", "culture", "prompt_id_control",
    "prompt_control", "answer_control", "prompt_id_rp", "prompt_rp", "answer_rp",
)
KEY_FIELDS = {
    "packet", "case_id", "source_row_id", "source_seed_id", "split", "language",
    "side_a_condition", "side_b_condition",
}
ADJUDICATION_FIELDS = {
    "adjudicator_id", "adjudication_status", "adjudication_prompt_equivalence",
    "adjudication_control_intent", "adjudication_roleplay_intent",
    "adjudication_control_safety", "adjudication_roleplay_safety",
    "adjudication_control_behavior", "adjudication_roleplay_behavior",
    "adjudication_clean_answer_prefix", "adjudication_corrupted_answer_prefix",
    "adjudication_rationale",
}
ADJUDICATION_ORDER = (
    "adjudicator_id", "adjudication_status", "adjudication_prompt_equivalence",
    "adjudication_control_intent", "adjudication_roleplay_intent",
    "adjudication_control_safety", "adjudication_roleplay_safety",
    "adjudication_control_behavior", "adjudication_roleplay_behavior",
    "adjudication_clean_answer_prefix", "adjudication_corrupted_answer_prefix",
    "adjudication_rationale",
)
WORKSHEET_ORDER = (
    *SOURCE_ORDER, "case_id",
    *(f"reviewer_{i}_id" for i in (1, 2)),
    *(f"reviewer_{i}_{field}" for i in (1, 2) for field in (
        "control_intent", "control_safety", "control_behavior", "control_evidence",
        "roleplay_intent", "roleplay_safety", "roleplay_behavior", "roleplay_evidence",
        "prompt_equivalence", "rationale", "confidence",
    )),
    *ADJUDICATION_ORDER, "review_version",
    "source_candidates_sha256", "reviewer_1_packet_sha256", "reviewer_2_packet_sha256", "private_key_sha256",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path: Path, required: set[str]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)):
            raise ValueError(f"Duplicate CSV column names in {path.name}")
        missing = required - set(fields)
        if missing:
            raise ValueError(f"{path.name} missing columns: {sorted(missing)}")
        rows = list(reader)
    if any(None in row for row in rows):
        raise ValueError(f"Malformed CSV row with extra columns in {path.name}")
    return rows


def _unique_index(rows: list[dict[str, str]], field: str, source: str) -> dict[str, dict[str, str]]:
    result = {}
    for row in rows:
        value = row.get(field, "").strip()
        if not value or value in result:
            raise ValueError(f"Missing or duplicate {field} in {source}")
        result[value] = row
    return result


def _validate_packet_row(row: dict[str, str], packet_name: str) -> list[str]:
    errors = []
    if row["review_status"] not in ENUMS["review_status"]:
        errors.append("invalid review_status")
        return errors
    if row["review_status"] != "completed":
        errors.append("review incomplete")
        return errors
    if not row["reviewer_id"].strip():
        errors.append("reviewer_id missing")
    enum_columns = {
        "prompt_equivalence": "prompt_equivalence",
        "request_intent_a": "request_intent",
        "request_intent_b": "request_intent",
        "response_safety_a": "response_safety",
        "response_safety_b": "response_safety",
        "refusal_behavior_a": "refusal_behavior",
        "refusal_behavior_b": "refusal_behavior",
        "confidence": "confidence",
    }
    for column, enum_name in enum_columns.items():
        if row[column] not in ENUMS[enum_name]:
            errors.append(f"invalid {column}")
    if not row["rationale"].strip():
        errors.append("rationale missing")
    for side in ("a", "b"):
        evidence = row[f"substantive_help_evidence_{side}"]
        if not evidence.strip() or evidence not in row[f"response_{side}"]:
            errors.append(f"evidence_{side} is not an exact response span")
    if row["language"] not in {"en", "ja"}:
        errors.append("unsupported language")
    if errors:
        return [f"{packet_name}/{row['case_id']}: {error}" for error in errors]
    return []


def reconcile(reviewer1: Path, reviewer2: Path, key_path: Path, candidates_path: Path):
    packets = {
        "reviewer_1": _unique_index(read_csv(reviewer1, PACKET_FIELDS), "case_id", reviewer1.name),
        "reviewer_2": _unique_index(read_csv(reviewer2, PACKET_FIELDS), "case_id", reviewer2.name),
    }
    candidates = _unique_index(read_csv(candidates_path, SOURCE_FIELDS), "row_id", candidates_path.name)
    key_rows = read_csv(key_path, KEY_FIELDS)
    key: dict[str, dict[str, dict[str, str]]] = {name: {} for name in packets}
    for item in key_rows:
        packet, case_id = item["packet"], item["case_id"]
        if packet not in key or not case_id or case_id in key[packet]:
            raise ValueError("Adjudication key has an unknown packet or duplicate/missing case ID")
        key[packet][case_id] = item

    for packet_name in packets:
        source_ids = [row["source_row_id"] for row in key[packet_name].values()]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError(f"{packet_name} private key maps multiple cases to one source row")

    errors = []
    ids1, ids2 = set(packets["reviewer_1"]), set(packets["reviewer_2"])
    if ids1 != ids2:
        errors.append(f"reviewer packet case-set mismatch: only_r1={len(ids1-ids2)}, only_r2={len(ids2-ids1)}")
    reviewer_ids = {}
    for packet_name, packet in packets.items():
        if set(packet) != set(key[packet_name]):
            errors.append(f"{packet_name} case IDs do not match its private key")
        seen_reviewer_ids = set()
        for case_id, row in packet.items():
            seen_reviewer_ids.add(row["reviewer_id"].strip())
            errors.extend(_validate_packet_row(row, packet_name))
            mapping = key[packet_name].get(case_id)
            if mapping is None:
                continue
            source = candidates.get(mapping["source_row_id"])
            if source is None:
                errors.append(f"{packet_name}/{case_id}: key points to unknown source row")
                continue
            if (mapping["source_seed_id"] != source["seed_id"] or mapping["split"] != source["split"]
                    or mapping["language"] != source["language"] or row["language"] != source["language"]):
                errors.append(f"{packet_name}/{case_id}: key/source metadata mismatch")
            side_a, side_b = mapping["side_a_condition"], mapping["side_b_condition"]
            if {side_a, side_b} != {"control", "rp"}:
                errors.append(f"{packet_name}/{case_id}: invalid A/B condition mapping")
                continue
            for side, condition in (("a", side_a), ("b", side_b)):
                if row[f"prompt_{side}"] != source[f"prompt_{condition}"] or row[f"response_{side}"] != source[f"answer_{condition}"]:
                    errors.append(f"{packet_name}/{case_id}: reviewer modified source text")
        if len(seen_reviewer_ids) == 1 and "" not in seen_reviewer_ids:
            reviewer_ids[packet_name] = next(iter(seen_reviewer_ids))
        else:
            errors.append(f"{packet_name} must contain one stable nonempty reviewer ID")
    if set(key) != set(packets) or len(key_rows) != sum(map(len, packets.values())):
        errors.append("private key row count does not match packets")
    if len(reviewer_ids) == 2 and reviewer_ids["reviewer_1"].casefold() == reviewer_ids["reviewer_2"].casefold():
        errors.append("reviewer IDs must identify two distinct people")
    if errors:
        raise ValueError(f"Review reconciliation failed ({len(errors)} issues):\n- " + "\n- ".join(errors[:30]))

    worksheet = []
    for case_id in sorted(ids1):
        mapped = {}
        source = None
        for packet_name in ("reviewer_1", "reviewer_2"):
            review = packets[packet_name][case_id]
            mapping = key[packet_name][case_id]
            source = candidates[mapping["source_row_id"]]
            for side in ("a", "b"):
                condition = mapping[f"side_{side}_condition"]
                mapped.setdefault(condition, {})[packet_name] = (review, side)
        record = {field: source[field] for field in SOURCE_ORDER}
        record["case_id"] = case_id
        for packet_name in ("reviewer_1", "reviewer_2"):
            reviewer = reviewer_ids[packet_name]
            record[f"{packet_name}_id"] = reviewer
            for condition, label in (("control", "control"), ("rp", "roleplay")):
                review, side = mapped[condition][packet_name]
                record[f"{packet_name}_{label}_intent"] = review[f"request_intent_{side}"]
                record[f"{packet_name}_{label}_safety"] = review[f"response_safety_{side}"]
                record[f"{packet_name}_{label}_behavior"] = review[f"refusal_behavior_{side}"]
                record[f"{packet_name}_{label}_evidence"] = review[f"substantive_help_evidence_{side}"]
            # Prompt equivalence and rationale/confidence are side-invariant.
            review = packets[packet_name][case_id]
            record[f"{packet_name}_prompt_equivalence"] = review["prompt_equivalence"]
            record[f"{packet_name}_rationale"] = review["rationale"]
            record[f"{packet_name}_confidence"] = review["confidence"]
        record.update({field: "" for field in ADJUDICATION_ORDER})
        record["adjudication_status"] = "pending"
        record["review_version"] = REVIEW_VERSION
        worksheet.append(record)

    disagreements = 0
    for row in worksheet:
        judgments = [row[f"reviewer_{i}_{field}"] for i in (1, 2)
                     for field in ("prompt_equivalence", "control_intent", "roleplay_intent", "control_safety", "roleplay_safety", "control_behavior", "roleplay_behavior")]
        if any(row[f"reviewer_1_{field}"] != row[f"reviewer_2_{field}"]
               for field in ("prompt_equivalence", "control_intent", "roleplay_intent", "control_safety", "roleplay_safety", "control_behavior", "roleplay_behavior")):
            disagreements += 1
    summary = {
        "source_candidate_rows": len(candidates),
        "reviewer_rows": {name: len(rows) for name, rows in packets.items()},
        "reviewer_ids": reviewer_ids,
        "reviewer_statuses": {name: "complete" for name in packets},
        "cases_with_any_label_disagreement": disagreements,
        "human_approved_cases": 0,
        "approval_state": "not adjudicated; no CircuitKIT export authorized",
        "input_sha256": {
            "reviewer_1": sha256_file(reviewer1), "reviewer_2": sha256_file(reviewer2),
            "private_key": sha256_file(key_path), "candidates": sha256_file(candidates_path),
        },
        "worksheet_rows": len(worksheet),
    }
    for row in worksheet:
        row.update({
            "source_candidates_sha256": summary["input_sha256"]["candidates"],
            "reviewer_1_packet_sha256": summary["input_sha256"]["reviewer_1"],
            "reviewer_2_packet_sha256": summary["input_sha256"]["reviewer_2"],
            "private_key_sha256": summary["input_sha256"]["private_key"],
        })
        # CSV field ordering is explicit so identical inputs produce byte-stable worksheets.
        ordered = {field: row[field] for field in WORKSHEET_ORDER}
        row.clear()
        row.update(ordered)
    return worksheet, summary


def _write_csv_atomic(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty adjudication worksheet")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    fd, temp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise


def _write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise


def _inside_private_outputs(path: Path) -> bool:
    root = Path("outputs/lexsi").resolve()
    return root == path.resolve().parent or root in path.resolve().parents


def build_adjudication(reviewer1: Path, reviewer2: Path, key: Path, candidates: Path,
                       output_csv: Path, summary_json: Path) -> dict:
    sources = {reviewer1.resolve(), reviewer2.resolve(), key.resolve(), candidates.resolve()}
    if output_csv.resolve() in sources or summary_json.resolve() in sources:
        raise ValueError("Refusing to overwrite source review or candidate files")
    if output_csv.resolve() == summary_json.resolve():
        raise ValueError("Adjudication CSV and summary JSON must use different paths")
    if not _inside_private_outputs(output_csv) or not _inside_private_outputs(summary_json):
        raise ValueError("Private adjudication outputs must be saved under outputs/lexsi")
    worksheet, summary = reconcile(reviewer1, reviewer2, key, candidates)
    _write_csv_atomic(output_csv, worksheet)
    summary["adjudication_worksheet_sha256"] = sha256_file(output_csv)
    _write_json_atomic(summary_json, summary)
    return summary


def approved_jsonl(adjudication_csv: Path, candidates_path: Path, reviewer1: Path, reviewer2: Path,
                   key_path: Path, output_jsonl: Path, manifest_json: Path) -> dict:
    """Validate explicit human decisions, then write approved CircuitKIT pairs."""
    if not _inside_private_outputs(output_jsonl) or not _inside_private_outputs(manifest_json):
        raise ValueError("Private CircuitKIT inputs and manifests must be saved under outputs/lexsi")
    source_paths = {adjudication_csv.resolve(), candidates_path.resolve(), reviewer1.resolve(), reviewer2.resolve(), key_path.resolve()}
    if (output_jsonl.resolve() in source_paths or manifest_json.resolve() in source_paths
            or manifest_json.resolve() == output_jsonl.resolve()):
        raise ValueError("Refusing to overwrite adjudication or candidate inputs")
    canonical_rows, _ = reconcile(reviewer1, reviewer2, key_path, candidates_path)
    canonical = _unique_index(canonical_rows, "row_id", "recomputed reviewer reconciliation")
    candidates = _unique_index(read_csv(candidates_path, SOURCE_FIELDS), "row_id", candidates_path.name)
    candidate_hash = sha256_file(candidates_path)
    reviewer_source_hashes = {1: sha256_file(reviewer1), 2: sha256_file(reviewer2)}
    key_source_hash = sha256_file(key_path)
    rows = read_csv(adjudication_csv, SOURCE_FIELDS | ADJUDICATION_FIELDS | {
        "case_id", "review_version", "reviewer_1_id", "reviewer_2_id",
        "reviewer_1_prompt_equivalence", "reviewer_2_prompt_equivalence",
        "reviewer_1_control_intent", "reviewer_2_control_intent", "reviewer_1_roleplay_intent", "reviewer_2_roleplay_intent",
        "reviewer_1_control_safety", "reviewer_2_control_safety", "reviewer_1_roleplay_safety", "reviewer_2_roleplay_safety",
        "reviewer_1_control_behavior", "reviewer_2_control_behavior", "reviewer_1_roleplay_behavior", "reviewer_2_roleplay_behavior",
        "reviewer_1_control_evidence", "reviewer_2_control_evidence", "reviewer_1_roleplay_evidence", "reviewer_2_roleplay_evidence",
        "reviewer_1_prompt_equivalence", "reviewer_2_prompt_equivalence", "reviewer_1_rationale", "reviewer_2_rationale",
        "reviewer_1_confidence", "reviewer_2_confidence", "source_candidates_sha256",
        "reviewer_1_packet_sha256", "reviewer_2_packet_sha256", "private_key_sha256",
    })
    worksheet_index = _unique_index(rows, "row_id", adjudication_csv.name)
    if set(worksheet_index) != set(canonical):
        raise ValueError("Adjudication worksheet row IDs differ from the rejoined reviewer packets")
    for rid, expected in canonical.items():
        actual = worksheet_index[rid]
        for field, value in expected.items():
            if field in ADJUDICATION_FIELDS:
                continue
            if actual.get(field) != value:
                raise ValueError(f"Reviewer/source field {field} was edited in adjudication worksheet for {rid}")
    seen_rows, seen_seeds, exported = set(), set(), []
    source_hashes = set()
    reviewer_hashes = {1: set(), 2: set()}
    key_hashes = set()
    for row in rows:
        rid = row["row_id"].strip()
        if not rid or rid in seen_rows:
            raise ValueError("Adjudication worksheet has missing or duplicate row IDs")
        seen_rows.add(rid)
        source_hashes.add(row["source_candidates_sha256"])
        reviewer_hashes[1].add(row["reviewer_1_packet_sha256"])
        reviewer_hashes[2].add(row["reviewer_2_packet_sha256"])
        key_hashes.add(row["private_key_sha256"])
        if row["reviewer_1_packet_sha256"] != reviewer_source_hashes[1]:
            raise ValueError("Adjudication worksheet was prepared from a different reviewer 1 packet")
        if row["reviewer_2_packet_sha256"] != reviewer_source_hashes[2]:
            raise ValueError("Adjudication worksheet was prepared from a different reviewer 2 packet")
        if row["private_key_sha256"] != key_source_hash:
            raise ValueError("Adjudication worksheet was prepared from a different private mapping key")
        if rid not in candidates:
            raise ValueError(f"Adjudication row is absent from the candidate source: {rid}")
        source = candidates[rid]
        if any(row[field] != source[field] for field in SOURCE_ORDER):
            raise ValueError(f"Adjudication worksheet source text/metadata changed for {rid}")
        for i in (1, 2):
            if not row[f"reviewer_{i}_id"].strip() or not row[f"reviewer_{i}_rationale"].strip():
                raise ValueError(f"Both completed reviewer records are required for {rid}")
            if row[f"reviewer_{i}_prompt_equivalence"] not in ENUMS["prompt_equivalence"]:
                raise ValueError(f"Invalid reviewer prompt-equivalence label for {rid}")
            for suffix, enum in (("intent", ENUMS["request_intent"]), ("safety", ENUMS["response_safety"]), ("behavior", ENUMS["refusal_behavior"])):
                for side in ("control", "roleplay"):
                    if row[f"reviewer_{i}_{side}_{suffix}"] not in enum:
                        raise ValueError(f"Invalid reviewer {side} {suffix} label for {rid}")
            for side, answer in (("control", row["answer_control"]), ("roleplay", row["answer_rp"])):
                evidence = row[f"reviewer_{i}_{side}_evidence"]
                if not evidence.strip() or evidence not in answer:
                    raise ValueError(f"Reviewer {i} evidence does not match source answer for {rid}")
            if row[f"reviewer_{i}_confidence"] not in ENUMS["confidence"]:
                raise ValueError(f"Invalid reviewer confidence for {rid}")
        status = row["adjudication_status"].strip()
        if status not in {"approved", "rejected", "pending"}:
            raise ValueError(f"Invalid adjudication_status for {rid}; use approved/rejected/pending")
        if status != "approved":
            continue
        reviewer_ids = [row["reviewer_1_id"].strip(), row["reviewer_2_id"].strip()]
        if (not all(reviewer_ids) or reviewer_ids[0].casefold() == reviewer_ids[1].casefold()
                or not row["adjudicator_id"].strip()
                or row["adjudicator_id"].strip().casefold() in {x.casefold() for x in reviewer_ids}):
            raise ValueError(f"Two distinct reviewers and a separate adjudicator are required for {rid}")
        if row["review_version"] != REVIEW_VERSION:
            raise ValueError(f"Unexpected review version for {rid}")
        if row["split"] not in {"analysis", "tune"} or row["language"] not in {"en", "ja"}:
            raise ValueError(f"Ineligible split or language for {rid}")
        if rid != hashlib.sha256(json.dumps([row["prompt_id_control"], row["prompt_id_rp"]], ensure_ascii=False).encode()).hexdigest()[:24]:
            raise ValueError(f"Source row ID does not match prompt IDs for {rid}")
        if row["seed_id"] in seen_seeds:
            raise ValueError(f"At most one approved pair per seed is allowed: {row['seed_id']}")
        labels = {
            "adjudication_prompt_equivalence": "equivalent",
            "adjudication_control_intent": "harmful",
            "adjudication_roleplay_intent": "harmful",
            "adjudication_control_safety": "safe",
            "adjudication_roleplay_safety": "unsafe",
            "adjudication_control_behavior": "clear_refusal",
            "adjudication_roleplay_behavior": "complied",
        }
        invalid = [field for field, expected in labels.items() if row[field].strip() != expected]
        if invalid:
            raise ValueError(f"Adjudicated labels do not establish the target contrast for {rid}: {invalid}")
        clean, corrupt = row["adjudication_clean_answer_prefix"], row["adjudication_corrupted_answer_prefix"]
        if (not clean.strip() or not corrupt.strip() or clean == corrupt
                or not row["answer_control"].startswith(clean) or not row["answer_rp"].startswith(corrupt)):
            raise ValueError(f"Continuation spans must be distinct exact source-answer prefixes for {rid}")
        if not row["adjudication_rationale"].strip():
            raise ValueError(f"Adjudication rationale is required for {rid}")
        if any(row[f"reviewer_{i}_{side}_evidence"].strip() not in row[f"answer_{'control' if side == 'control' else 'rp'}"]
               for i in (1, 2) for side in ("control", "roleplay")):
            raise ValueError(f"Reviewer evidence no longer matches the source answers for {rid}")
        exported.append({
            "row_id": rid, "seed_id": row["seed_id"], "split": row["split"], "language": row["language"],
            "category": row["category"], "culture": row["culture"],
            "prompt_id_control": row["prompt_id_control"], "prompt_id_rp": row["prompt_id_rp"],
            "clean_prompt": row["prompt_control"], "corrupt_prompt": row["prompt_rp"],
            "clean_answer": clean, "corrupted_answer": corrupt,
            "review_status": "human_approved", "review_version": REVIEW_VERSION,
            "reviewer_ids": reviewer_ids, "adjudicator_id": row["adjudicator_id"].strip(),
            "rationale": row["adjudication_rationale"].strip(),
        })
        seen_seeds.add(row["seed_id"])
    if len(source_hashes) != 1 or source_hashes != {candidate_hash}:
        raise ValueError("Adjudication worksheet does not match this exact candidate source file")
    if any(len(values) != 1 or not next(iter(values)) for values in (*reviewer_hashes.values(), key_hashes)):
        raise ValueError("Adjudication worksheet has inconsistent or missing reviewer/key hashes")
    if not exported:
        raise ValueError("No explicitly approved adjudications; no CircuitKIT export was written")
    _write_jsonl_atomic(output_jsonl, exported)
    manifest = {
        "schema": "lexsi.blinded_control_roleplay_approved_export/1",
        "review_version": REVIEW_VERSION,
        "approved_pairs": len(exported), "unique_seeds": len(seen_seeds),
        "candidate_source_sha256": candidate_hash,
        "adjudication_worksheet_sha256": sha256_file(adjudication_csv),
        "reviewer_1_packet_sha256": next(iter(reviewer_hashes[1])),
        "reviewer_2_packet_sha256": next(iter(reviewer_hashes[2])),
        "private_key_sha256": next(iter(key_hashes)),
        "approved_export_sha256": sha256_file(output_jsonl),
        "automatic_labels_used": False,
    }
    _write_json_atomic(manifest_json, manifest)
    return manifest


def _write_jsonl_atomic(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("prepare-adjudication")
    check.add_argument("--reviewer1", required=True, type=Path)
    check.add_argument("--reviewer2", required=True, type=Path)
    check.add_argument("--key", required=True, type=Path)
    check.add_argument("--candidates", required=True, type=Path)
    check.add_argument("--output-csv", required=True, type=Path)
    check.add_argument("--summary-json", required=True, type=Path)
    export = sub.add_parser("export-approved")
    export.add_argument("--adjudication-csv", required=True, type=Path)
    export.add_argument("--candidates", required=True, type=Path)
    export.add_argument("--reviewer1", required=True, type=Path)
    export.add_argument("--reviewer2", required=True, type=Path)
    export.add_argument("--key", required=True, type=Path)
    export.add_argument("--output-jsonl", required=True, type=Path)
    export.add_argument("--manifest-json", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "prepare-adjudication":
        result = build_adjudication(args.reviewer1, args.reviewer2, args.key, args.candidates,
                                    args.output_csv, args.summary_json)
    else:
        result = approved_jsonl(args.adjudication_csv, args.candidates, args.reviewer1, args.reviewer2,
                                args.key, args.output_jsonl, args.manifest_json)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
