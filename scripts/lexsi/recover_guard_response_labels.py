"""Recover the historical Dataset2 guard verdicts from the saved GGUF shards."""

import json
from contextlib import closing
from pathlib import Path
import sqlite3

from guard_judge import parse_guard


BINARY = {"Safe", "Unsafe"}
FIELDS = """prompt_id, seed_id, source_id, split, category, kind, culture, framing,
    prompt_language, output_language, request, prompt, answer, generation_status,
    scoring_status, safety, prompt_safety, refused, identical_twin_flag, analysis_eligible,
    raw_guard_outputs, shard_id"""


def recover_guard(raw):
    if not raw or raw.get("status") != "completed":
        return "Missing"
    answer = raw.get("answer")
    return parse_guard(answer.replace("\r\n", "\n"))[1]["label"] if isinstance(answer, str) else "Unparsed"


def observations(paths):
    result = {}
    settings = []
    for shard, path in enumerate(paths, 1):
        uri = Path(path).resolve().as_uri() + "?mode=ro"
        with closing(sqlite3.connect(uri, uri=True)) as db:
            db.row_factory = sqlite3.Row
            if db.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                raise ValueError(f"Invalid SQLite export: {path}")
            metadata, = db.execute("SELECT json FROM run_metadata").fetchall()
            meta = json.loads(metadata[0])
            if meta["shard_id"] != shard:
                raise ValueError(f"Unexpected shard ID in {path}")
            settings.append(meta)
            expected = set(meta["prompt_ids"])
            actual = set()
            for source in db.execute(f"SELECT {FIELDS} FROM observations ORDER BY prompt_id"):
                row = dict(source)
                pid = row["prompt_id"]
                if pid in result or row["shard_id"] != shard or row["source_id"] != row["seed_id"]:
                    raise ValueError(f"Duplicate or inconsistent prompt {pid}")
                actual.add(pid)
                raw = json.loads(row.pop("raw_guard_outputs"))
                if not isinstance(raw, list) or len(raw) > 2:
                    raise ValueError(f"Unexpected raw guard record shape for {pid}")
                # The saved runtime invokes guard_response first, then guard_prompt.
                response, prompt = (raw + [None, None])[:2]
                row["guard_response_corrected"] = recover_guard(response)
                row["guard_prompt_corrected"] = recover_guard(prompt)
                if row["safety"] in BINARY and row["safety"] != row["guard_response_corrected"]:
                    raise ValueError(f"Recovery changed a published binary label for {pid}")
                result[pid] = row
            if actual != expected or len(expected) != len(meta["prompt_ids"]):
                raise ValueError(f"Observation inventory mismatch for {path}")
    if any(settings[0][key] != settings[1][key] for key in ("generation", "judge", "dataset")):
        raise ValueError("Shard configurations differ")
    return result, settings
