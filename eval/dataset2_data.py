"""Read-only Dataset2 validation and deterministic seed-preserving allocation."""

from contextlib import closing
import itertools
import json
import re
import sqlite3
from pathlib import Path
import pandas as pd
from eval.checkpoints import config_id, file_sha256

CONDITION = ["kind", "culture", "framing", "prompt_language", "output_language"]
VERSION = "paired-dp-v1-seed42"


def load_dataset(path, expected_count=21184):
    with closing(
        sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
    ) as con:
        if con.execute("pragma integrity_check").fetchone()[0] != "ok":
            raise ValueError("SQLite integrity check failed")
        tables = {
            r[0]
            for r in con.execute("select name from sqlite_master where type='table'")
        }
        if (
            not {
                "prompts",
                "seeds",
                "seed_texts",
                "technique_outputs",
                "prompts_wide",
                "generation_info",
            }
            <= tables
        ):
            raise ValueError("Missing Dataset2 tables")
        schemas = {
            "prompts": {
                "prompt_id",
                "source_id",
                "request",
                "prompt",
                "source_split",
                "subtopic",
                "form",
            },
            "seeds": {
                "source_id",
                "source_split",
                "category",
                "subtopic",
                "form",
                "severity",
                "split",
                "mix_direction",
                "expected",
                "prompts",
                "complete",
            },
            "seed_texts": {"source_id", "culture", "kind", "ja", "en"},
            "technique_outputs": {
                "source_id",
                "culture",
                "kind",
                "technique",
                "ja",
                "en",
                "mix",
            },
            "prompts_wide": {"source_id", "kind", "n_variants"},
            "generation_info": {
                "source",
                "created_at",
                "techniques",
                "checks",
                "mix_english_share",
            },
        }
        for table, required_columns in schemas.items():
            actual = {r[1] for r in con.execute(f"pragma table_info({table})")}
            if not required_columns <= actual:
                raise ValueError(
                    f"Missing required {table} columns: {required_columns-actual}"
                )
        for table in ["prompts", "seed_texts", "technique_outputs", "prompts_wide"]:
            orphan_count = con.execute(
                f"SELECT COUNT(*) FROM {table} t LEFT JOIN seeds s ON t.source_id=s.source_id WHERE s.source_id IS NULL"
            ).fetchone()[0]
            if orphan_count:
                raise ValueError(f"Orphan seed relationship in {table}: {orphan_count}")
        wide_counts = dict(
            con.execute(
                "SELECT source_id,SUM(n_variants) FROM prompts_wide GROUP BY source_id"
            )
        )
        prompt_counts = dict(
            con.execute("SELECT source_id,COUNT(*) FROM prompts GROUP BY source_id")
        )
        if wide_counts != prompt_counts:
            raise ValueError("Wide-table variant counts differ from retained prompts")
        prompts = pd.read_sql_query("select * from prompts order by prompt_id", con)
        seeds = pd.read_sql_query("select * from seeds order by source_id", con)
    required = {
        "prompt_id",
        "source_id",
        "prompt",
        "request",
        "split",
        "category",
        "severity",
        "expected",
        "mix_direction",
        *CONDITION,
    }
    if not required <= set(prompts):
        raise ValueError("Missing required prompt columns")
    if (
        len(prompts) != expected_count
        or prompts.prompt_id.duplicated().any()
        or seeds.source_id.duplicated().any()
    ):
        raise ValueError("Dataset revision/count/ID mismatch")
    if (
        not prompts.source_id.isin(seeds.source_id).all()
        or not prompts.prompt.map(
            lambda x: isinstance(x, str) and bool(x.strip())
        ).all()
        or not prompts.request.map(
            lambda x: isinstance(x, str) and bool(x.strip())
        ).all()
    ):
        raise ValueError("Orphan seed or empty prompt")
    allowed = {
        "kind": ["harmful", "twin"],
        "culture": ["neutral", "japan", "us"],
        "framing": ["control", "rp", "fc", "rs"],
        "prompt_language": ["ja", "en", "mix"],
        "output_language": ["ja", "en"],
        "expected": ["decline", "answer_carefully", "answer"],
        "split": ["analysis", "tune", "heldout"],
        "severity": ["clear", "borderline"],
    }
    for col, values in allowed.items():
        if not prompts[col].isin(values).all():
            raise ValueError(f"Invalid {col}")
    if prompts.duplicated(["source_id"] + CONDITION).any():
        raise ValueError("Duplicate condition")
    counts = prompts.groupby("source_id").size()
    if not seeds.set_index("source_id").prompts.eq(counts).all():
        raise ValueError("Seed count mismatch")
    meta = seeds.set_index("source_id")
    for col in ["category", "split", "severity", "source_split", "subtopic", "form"]:
        if not prompts[col].eq(prompts.source_id.map(meta[col])).all():
            raise ValueError(f"Seed metadata mismatch: {col}")
    pure = prompts.prompt_language != "mix"
    if (
        not prompts.loc[pure, "prompt_language"]
        .eq(prompts.loc[pure, "output_language"])
        .all()
    ):
        raise ValueError("Invalid language pairing")
    if ((prompts.framing == "control") & ~pure).any():
        raise ValueError("Mixed control")
    rewritten = prompts.prompt_language == "mix"
    if (
        not prompts.loc[rewritten, "mix_direction"]
        .isin(["frame_en+request_ja", "frame_ja+request_en"])
        .all()
    ):
        raise ValueError("Invalid mixed direction")
    if (
        not prompts.loc[rewritten, "mix_direction"]
        .eq(prompts.loc[rewritten, "source_id"].map(meta.mix_direction))
        .all()
    ):
        raise ValueError("Mixed direction differs from assigned seed metadata")
    prompts["seed_id"] = prompts.source_id
    return prompts, seeds


def make_manifest(prompts, seeds):
    """Pair within category/split sorted by severity,direction,coverage,hash; exact subset DP.

    One from each pair goes to Part 1. DP minimizes total count imbalance, deterministic
    ordered first-path tie breaking. Category and original metadata split stay exactly balanced.
    """
    pairs = []
    for _, group in seeds.groupby(["category", "split"], sort=True):
        records = group.to_dict("records")
        records.sort(
            key=lambda r: (
                r["severity"],
                r["mix_direction"],
                r["prompts"],
                config_id([42, r["source_id"]]),
            )
        )
        if len(records) % 2:
            raise ValueError("Odd stratum needs explicit new manifest algorithm")
        pairs.extend(zip(records[::2], records[1::2]))
    dp = {0: ()}
    for a, b in pairs:
        nxt = {}
        for total, choices in sorted(dp.items()):
            for r in sorted([a, b], key=lambda r: r["source_id"]):
                nxt.setdefault(total + int(r["prompts"]), choices + (r["source_id"],))
        dp = nxt
    total = min(dp, key=lambda n: (abs(2 * n - len(prompts)), n))
    selected = set(dp[total])
    rows = []
    for r in seeds.to_dict("records"):
        rows.append(
            dict(
                seed_id=r["source_id"],
                shard_id=1 if r["source_id"] in selected else 2,
                category=r["category"],
                split=r["split"],
                severity=r["severity"],
                expected=r["expected"],
                expected_prompt_count=84,
                actual_prompt_count=r["prompts"],
                mixed_direction=r["mix_direction"],
                manifest_version=VERSION,
            )
        )
    return pd.DataFrame(rows).sort_values("seed_id").reset_index(drop=True)


def verify_manifest(prompts, seeds, manifest, metadata, dataset_path):
    if metadata["dataset_sha256"] != file_sha256(dataset_path):
        raise ValueError("Dataset revision hash mismatch")
    if config_id(manifest.to_dict("records")) != metadata["manifest_id"]:
        raise ValueError("Manifest hash mismatch")
    if (
        manifest.seed_id.duplicated().any()
        or set(manifest.seed_id) != set(seeds.source_id)
        or set(manifest.shard_id) != {1, 2}
    ):
        raise ValueError("Invalid seed assignments")
    expected = make_manifest(prompts, seeds)
    if not manifest.equals(expected):
        raise ValueError("Manifest is not the deterministic allocation")
    result = prompts.merge(
        manifest[["seed_id", "shard_id"]], on="seed_id", validate="many_to_one"
    )
    if len(result) != len(prompts) or result.prompt_id.duplicated().any():
        raise ValueError("Invalid prompt union")
    return result


def quality_flags(prompts):
    out = prompts.copy()
    keys = ["seed_id", "culture", "framing", "prompt_language", "output_language"]
    twin = out[out.kind == "twin"][keys + ["prompt", "expected"]]
    twin = twin.rename(columns={"prompt": "twin_prompt", "expected": "twin_expected"})
    flags = out.merge(twin, on=keys, how="left", validate="many_to_one")
    same = (flags.kind == "harmful") & flags.prompt.eq(flags.twin_prompt)
    identical_keys = set(map(tuple, flags.loc[same, keys].values))
    out["identical_twin_flag"] = [
        tuple(r[k] for k in keys) in identical_keys for r in out.to_dict("records")
    ]
    out["expected_behavior_conflict"] = out.identical_twin_flag
    out["duplicate_text_flag"] = out.prompt.duplicated(keep=False)
    out["structural_valid"] = True
    out["generation_eligible"] = True
    out["semantic_review_required"] = (
        True  # Generated expectations have not been human validated.
    )
    out["twin_task_review_required"] = True
    out["cultural_equivalence_review_required"] = out.culture != "neutral"
    out["borderline_expectation_review_required"] = out.expected == "answer_carefully"
    out["rewrite_review_required"] = out.framing != "control"

    def script_mismatch(row):
        japanese = bool(re.search(r"[\u3040-\u30ff\u4e00-\u9fff]", row["request"]))
        latin = bool(re.search(r"[A-Za-z]", row["request"]))
        return not (
            {"ja": japanese, "en": latin, "mix": japanese and latin}[
                row["prompt_language"]
            ]
        )

    out["language_review_required"] = [
        script_mismatch(r) for r in out.to_dict("records")
    ]
    out["analysis_eligible"] = ~out.identical_twin_flag
    out["expectation_validated"] = False
    out["exclusion_reason"] = out.identical_twin_flag.map(
        {True: "identical original/twin; exclude behavioral contrast", False: ""}
    )
    return out


def missing_variants(prompts, seeds):
    present = set(map(tuple, prompts[["seed_id"] + CONDITION].values))
    rows = []
    for s in seeds.to_dict("records"):
        for kind, culture, framing in itertools.product(
            ["harmful", "twin"],
            ["neutral", "japan", "us"],
            ["control", "rp", "fc", "rs"],
        ):
            for lang, output in [("ja", "ja"), ("en", "en")] + (
                [] if framing == "control" else [("mix", "ja"), ("mix", "en")]
            ):
                if (
                    s["source_id"],
                    kind,
                    culture,
                    framing,
                    lang,
                    output,
                ) not in present:
                    rows.append(
                        dict(
                            seed_id=s["source_id"],
                            category=s["category"],
                            split=s["split"],
                            kind=kind,
                            culture=culture,
                            framing=framing,
                            prompt_language=lang,
                            output_language=output,
                        )
                    )
    return pd.DataFrame(rows)


def build_artifacts(db, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    p, s = load_dataset(db)
    m = make_manifest(p, s)
    m.to_csv(directory / "dataset2_shards.csv", index=False)
    meta = {
        "dataset_sha256": file_sha256(db),
        "manifest_id": config_id(m.to_dict("records")),
        "algorithm": VERSION,
        "prompt_count": len(p),
    }
    (directory / "dataset2_shards.json").write_text(json.dumps(meta, indent=2) + "\n")
    return m


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="marcus/dataset2_partial.sqlite3")
    parser.add_argument("--output", default="eval/manifests")
    args = parser.parse_args()
    print(
        build_artifacts(args.dataset, args.output)
        .groupby("shard_id")
        .agg(seeds=("seed_id", "size"), prompts=("actual_prompt_count", "sum"))
    )
