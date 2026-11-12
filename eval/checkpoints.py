"""Shared case identities, completion rules, and append-only notebook checkpoints."""

import hashlib
import json
from pathlib import Path


ANSWER_KEY = ("source_id", "source_split", "variant", "language")
PAIR_KEY = ANSWER_KEY[:-1]


def case_id(row):
    return "|".join(str(row[field]) for field in ANSWER_KEY)


def fingerprint(prompt, answer, target):
    """Keep the historical text fingerprint so existing exports can be verified."""
    return hashlib.sha256(json.dumps([prompt, answer, target], ensure_ascii=False).encode()).hexdigest()[:16]


def config_id(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def register_configuration(checkpoint, configuration):
    """Keep recoverable settings beside a checkpoint, keyed by their full configuration hash."""
    identity = config_id(configuration)
    path = Path(checkpoint).with_suffix(".settings.json")
    existing = json.loads(path.read_text()) if path.exists() else {}
    if identity not in existing:
        existing[identity] = configuration
        path.parent.mkdir(parents=True, exist_ok=True)
        pending = path.with_suffix(".partial")
        pending.write_text(json.dumps(existing, ensure_ascii=False, indent=2) + "\n")
        pending.replace(path)
    return identity


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def generation_status(row):
    """Missing completion flags never count as completed; accept pandas scalar booleans."""
    error = row.get("error")
    if isinstance(error, str) and error.strip():
        return "error"
    answer = row.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        return "missing"
    try:
        if row.get("truncated") == True:  # value comparison, including numpy.bool_
            return "truncated"
        if row.get("truncated") == False:
            return "completed"
    except (TypeError, ValueError):  # e.g. pandas.NA cannot be converted to bool
        pass
    return "missing"


def load_jsonl(path, key_fields):
    """Latest record per key. Fail explicitly on damaged lines rather than losing data."""
    latest = {}
    if Path(path).exists():
        with Path(path).open(encoding="utf-8") as stream:
            for number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    latest[tuple(record[field] for field in key_fields)] = record
                except (ValueError, KeyError) as error:
                    raise ValueError(f"Invalid checkpoint {path}:{number}; preserve it before repair: {error}") from error
    return latest


def append_records(path, records):
    """Write complete JSON lines and flush each record, including on Drive mounts."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # A valid last record without a newline needs a separator before appending.
    if path.exists() and path.stat().st_size:
        with path.open("rb") as stream:
            stream.seek(-1, 2)
            needs_newline = stream.read(1) != b"\n"
    else:
        needs_newline = False
    with path.open("a", encoding="utf-8") as output:
        if needs_newline:
            output.write("\n")
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()


def compatible_score(record, sample, identity):
    return bool(record and record.get("config_id") == identity
                and record.get("fingerprint") == fingerprint(sample.input, sample.actual_output, sample.target)
                and record.get("scoring_status") == "completed")
