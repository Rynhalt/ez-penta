"""Recompute report evidence from local exports; write only into this report folder."""

from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sqlite3
import subprocess
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def exact_binomial(b, c):
    n = b + c
    if n == 0:
        return 1.0
    terms = [math.exp(math.lgamma(n + 1) - math.lgamma(i + 1)
                      - math.lgamma(n - i + 1) - n * math.log(2))
             for i in range(min(b, c) + 1)]
    return min(1.0, 2 * math.fsum(terms))


def main():
    inputs = [ROOT / name for name in ("eval_aya_answers.jsonl", "eval_scores.jsonl", "answercarefully_eval.sqlite3")]
    source_paths = [ROOT / name for name in git("ls-files").splitlines()]
    source_paths += [ROOT / "inspect_evaluation.py", ROOT / "tests/test_inspect_evaluation.py"]
    source_before = {str(path.relative_to(ROOT)): sha(path) for path in source_paths if path.is_file()}
    input_before = {path.name: {"bytes": path.stat().st_size, "sha256": sha(path)} for path in inputs}

    spec = importlib.util.spec_from_file_location("existing_audit", ROOT / "inspect_evaluation.py")
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    stats, rows = audit.analyze(inputs[2], inputs[0], inputs[1])
    assert stats["integrity"] == "ok"
    assert all(value == 0 for value in stats["validation"].values()), stats["validation"]

    # An independent SQL pass checks headline results, source coverage, and every
    # published variant rate without reading previous report statistics.
    db = sqlite3.connect(inputs[2].resolve().as_uri() + "?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA query_only=ON")
    integrity = db.execute("PRAGMA integrity_check").fetchone()[0]
    assert integrity == "ok"
    totals = dict(db.execute("""SELECT count(*) answers, sum(truncated) truncated,
        count(unsafe) judged, sum(unsafe) unsafe, count(refused) refusal_known, sum(refused) refused,
        sum(safety='Safe') safe, sum(safety='Controversial') controversial FROM answers""").fetchone())
    assert totals["answers"] == stats["answers"]
    assert totals["unsafe"] / totals["judged"] == stats["unsafe_rate"]
    assert totals["controversial"] == stats["labels"]["Controversial"]
    variant_sql = [dict(row) for row in db.execute("""SELECT variant,language,count(*) answers,
        count(unsafe) judged,sum(unsafe) unsafe,avg(unsafe) unsafe_rate,count(refused) refusal_known,
        sum(refused) refused,avg(refused) refusal_rate,sum(truncated) truncated
        FROM answers GROUP BY variant,language ORDER BY variant,language""")]
    for row in variant_sql:
        recomputed = next(item for item in stats["variant_language"]
                          if item["variant"] == row["variant"] and item["language"] == row["language"])
        for field in row:
            assert row[field] == recomputed[field], (field, row, recomputed)
    coverage = [dict(row) for row in db.execute("""SELECT source_id,source_split,count(*) n,
        count(distinct variant) variants,count(distinct language) languages
        FROM answers GROUP BY source_id,source_split""")]
    assert len(coverage) == 1800 and all(r["n"] == 16 and r["variants"] == 8 and r["languages"] == 2 for r in coverage)
    split_sources = dict(Counter(row["source_split"] for row in coverage))
    assert split_sources == {"dev": 1464, "test": 336}
    schema = {name: [dict(row) for row in db.execute(f'PRAGMA table_info("{name}")')]
              for name in ("answers", "pairs", "eval_info")}
    db.close()

    # Re-open both JSONL files to independently verify coverage, errors, and labels.
    seen = set()
    missing = errors = malformed = duplicates = 0
    model_counts, token_caps, labels, unknown_flags = Counter(), Counter(), Counter(), Counter()
    for line in inputs[0].open(encoding="utf-8"):
        try:
            r = json.loads(line)
        except ValueError:
            malformed += 1
            continue
        key = tuple(r[field] for field in ("source_id", "source_split", "variant", "language"))
        duplicates += key in seen
        seen.add(key)
        missing += not isinstance(r.get("answer"), str) or not r["answer"].strip()
        errors += bool(r.get("error"))
        model_counts[r["model"]] += 1
        token_caps[r["max_new_tokens"]] += 1
    for line in inputs[1].open(encoding="utf-8"):
        r = json.loads(line)
        if r["stage"] == "answer":
            meta = r["scores"]["guard_response"]["metadata"]
            labels[meta["label"]] += 1
            unknown_flags[meta["unknown"]] += 1
    assert len(seen) == 28800 and missing == errors == malformed == duplicates == 0
    assert labels == {"Safe": 25624, "Unsafe": 1731, "Controversial": 942}
    for item in stats["language_comparisons"]:
        independent = exact_binomial(item["first_only"], item["second_only"])
        assert math.isclose(independent, item["p_exact"], rel_tol=1e-10, abs_tol=1e-35)
    assert all(sha(path) == input_before[path.name]["sha256"] for path in inputs)
    assert all(sha(ROOT / name) == digest for name, digest in source_before.items())

    verification = {
        "verified_at": datetime.now(timezone.utc).isoformat(), "repository_head": git("rev-parse", "HEAD"),
        "initial_working_tree": git("status", "--short"),
        "recent_commits": git("log", "-10", "--format=%h %s").splitlines(),
        "inputs": input_before, "source_hashes": source_before,
        "sqlite_integrity_check": integrity, "independent_sql_totals": totals,
        "source_split_counts": split_sources, "model_counts": dict(model_counts),
        "max_new_token_counts": dict(token_caps), "response_label_counts": dict(labels),
        "missing_answers": missing, "recorded_errors": errors, "malformed_answers": malformed,
        "duplicate_answers": duplicates, "independent_variant_sql": variant_sql,
        "schema": schema, "source_and_input_hashes_unchanged": True,
        "methods": "Fresh JSONL reads and read-only SQLite; audit recomputation plus independent SQL and log-space exact binomial checks."
    }
    (HERE / "verification.json").write_text(json.dumps(verification, ensure_ascii=False, indent=2) + "\n")
    (HERE / "statistics.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"answers": len(seen), "sources": len(coverage), "labels": dict(labels),
                      "integrity": integrity, "reconciliation_mismatches": sum(stats["validation"].values())}, indent=2))


if __name__ == "__main__":
    main()
