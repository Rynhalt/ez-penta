"""UK variants of existing Dataset2 requests; no seed generation or inferred approvals."""

import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import tempfile

import pandas as pd

from eval.checkpoints import config_id, file_sha256, generation_status
from eval.dataset2_data import load_dataset
from data_ready.rewrite_helpers import run_rewrites


def load_source(path, seed_limit=None):
    """Keep neutral rows verbatim, with the original seed identities and partitions."""
    prompts, seeds = load_dataset(path, expected_count=None)
    with sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True) as con:
        texts = pd.read_sql_query("SELECT * FROM seed_texts WHERE culture='neutral'", con)
    ids = sorted(seeds.source_id)
    if seed_limit is not None:
        if not isinstance(seed_limit, int) or seed_limit < 1:
            raise ValueError("SEED_LIMIT must be a positive integer or None")
        ids = sorted(seeds.loc[seeds.split.eq("analysis"), "source_id"])[:seed_limit]
        if not ids:
            raise ValueError("No analysis seeds available for the pilot")
    seeds = seeds[seeds.source_id.isin(ids)].copy()
    prompts = prompts[prompts.source_id.isin(ids) & prompts.culture.eq("neutral")].copy()
    texts = texts[texts.source_id.isin(ids)].copy()
    if texts.duplicated(["source_id", "kind"]).any():
        raise ValueError("Duplicate neutral seed texts")
    if set(texts[texts.kind.eq("harmful")].source_id) != set(ids):
        raise ValueError("Every source seed needs an existing neutral harmful request")
    if not texts[["ja", "en"]].map(lambda v: isinstance(v, str) and bool(v.strip())).all().all():
        raise ValueError("Missing bilingual source text")
    return prompts, seeds, texts


def make_jobs(texts, seeds, *, stage, system, template, culture_rule, techniques=None,
              output_keys=("ja", "en")):
    """Adapt the existing rewrite runner to paired JSON jobs without losing source IDs."""
    meta = seeds.set_index("source_id")
    jobs = []
    for row in texts.to_dict("records"):
        for strategy, instruction in (techniques or {"base": ""}).items():
            seed = meta.loc[row["source_id"]]
            body = template.format(
                ja=row["ja"], en=row["en"], culture_rule=culture_rule,
                technique=instruction.strip(), category=seed["category"],
                mix_rule=("frame in English, underlying request in Japanese" if
                          seed["mix_direction"] == "frame_en+request_ja" else
                          "frame in Japanese, underlying request in English"),
                output_keys=", ".join(output_keys),
            )
            key = config_id([row["source_id"], row["kind"], stage, strategy])
            jobs.append(dict(source_id=key, source_split=seed["source_split"],
                             seed_id=row["source_id"], kind=row["kind"], strategy=strategy,
                             input=body, system=system))
    return pd.DataFrame(jobs)


def generate_jobs(jobs, checkpoint, configuration, *, model, tokenizer, batch_size,
                  max_batch_tokens, max_new_tokens):
    if jobs.empty:
        return {}
    system = jobs.system.iloc[0]
    return run_rewrites(
        jobs, {"paired": ("input", "Japanese and English")},
        lambda prompt, _: [{"role": "system", "content": system},
                           {"role": "user", "content": prompt}],
        checkpoint, configuration, model=model, tokenizer=tokenizer,
        batch_size=batch_size, max_batch_tokens=max_batch_tokens,
        max_new_tokens=max_new_tokens,
    )


def collect_jobs(jobs, records, keys=("ja", "en")):
    """No truncated, malformed, or empty outputs become evaluation prompts."""
    accepted, audit = [], []
    for job in jobs.to_dict("records"):
        record = records.get((job["source_id"], job["source_split"], "paired"), {})
        status = generation_status({"answer": record.get("output"),
                                    "truncated": record.get("truncated"),
                                    "error": record.get("error")})
        parsed, reason = None, None
        if status == "completed":
            text = record["output"].strip()
            if text.startswith("```") and text.endswith("```"):
                text = "\n".join(text.splitlines()[1:-1])
            try:
                parsed = json.loads(text)
                if not isinstance(parsed, dict) or any(
                    not isinstance(parsed.get(k), str) or not parsed[k].strip() for k in keys
                ):
                    raise ValueError("Missing required text fields")
            except (ValueError, TypeError) as error:
                parsed, reason = None, str(error)
        else:
            reason = status
        identity = {k: job[k] for k in ("seed_id", "kind", "strategy")}
        flags = ["automated generation; semantic and cultural equivalence unreviewed"]
        if parsed:
            if not re.search(r"[\u3040-\u30ff\u4e00-\u9fff]", parsed["ja"]):
                flags.append("Japanese script missing")
            if re.search(r"[\u3040-\u30ff]", parsed["en"]):
                flags.append("Japanese script in English")
            if "mix" in keys and not (re.search(r"[A-Za-z]", parsed["mix"]) and
                                       re.search(r"[\u3040-\u30ff\u4e00-\u9fff]", parsed["mix"])):
                parsed, reason = None, "Mixed output lacks one language"
        if parsed:
            accepted.append({**identity, **{k: parsed[k].strip() for k in keys},
                             "flags": "; ".join(flags)})
        audit.append({**identity, "accepted": parsed is not None,
                      "reason": reason, "flags": "; ".join(flags),
                      "raw_output": record.get("output"), "review_status": "provisional"})
    return accepted, pd.DataFrame(audit)


def export_dataset(destination, neutral, seeds, source_texts, bases, techniques, metadata,
                   include_mixed=True):
    """Export the familiar Dataset2 tables; missing variants stay visibly missing."""
    meta = seeds.set_index("source_id")
    base_map = {(r["seed_id"], r["kind"]): r for r in bases}
    rows = []
    instructions = {"ja": "日本語で答えてください。", "en": "Please answer in English."}
    for result in [*bases, *techniques]:
        sid, kind = result["seed_id"], result["kind"]
        seed = meta.loc[sid]
        base = base_map[(sid, kind)]
        framing = "control" if result["strategy"] == "base" else result["strategy"]
        languages = [("ja", "ja"), ("en", "en")]
        if framing != "control" and include_mixed:
            languages += [("mix", "ja"), ("mix", "en")]
        for language, output in languages:
            request_lang = ("ja" if seed["mix_direction"] == "frame_en+request_ja" else "en") if language == "mix" else language
            flags = base["flags"] + ("; " + result["flags"] if framing != "control" else "")
            row = {k: seed[k] for k in ["source_split", "split", "category", "subtopic", "form", "severity"]}
            row.update(source_id=sid, kind=kind, culture="uk", framing=framing,
                       expected="answer" if kind == "twin" else seed["expected"],
                       prompt_language=language, output_language=output,
                       mix_direction=seed["mix_direction"] if language == "mix" else None,
                       request=base[request_lang], prompt=result[language] + "\n\n" + instructions[output],
                       generation_flags=flags, n_generation_flags=len(flags.split("; ")),
                       review_status="provisional", expectation_validated=False)
            # Content identity prevents a changed rewrite from masquerading as the old prompt.
            row["prompt_id"] = "uk_" + config_id([sid, kind, framing, language, output, row["prompt"]])[:24]
            rows.append(row)
    if not rows:
        raise ValueError("No usable UK requests; inspect generation_review.csv and checkpoints")
    prompts = pd.concat([neutral, pd.DataFrame(rows)], ignore_index=True)
    prompts = prompts.drop(columns="seed_id", errors="ignore")
    counts = prompts.groupby("source_id").size()
    seed_table = seeds.copy()
    seed_table["prompts"] = seed_table.source_id.map(counts)
    expected = 56 if include_mixed else 32
    seed_table["complete"] = seed_table.prompts.eq(expected)
    texts = pd.concat([source_texts, pd.DataFrame([
        dict(source_id=r["seed_id"], culture="uk", kind=r["kind"], ja=r["ja"], en=r["en"])
        for r in bases])], ignore_index=True)
    technique_table = pd.DataFrame([
        dict(source_id=r["seed_id"], culture="uk", kind=r["kind"], technique=r["strategy"],
             ja=r["ja"], en=r["en"], mix=r.get("mix"), flags=r["flags"])
        for r in techniques], columns=["source_id", "culture", "kind", "technique", "ja", "en", "mix", "flags"])
    variant = (prompts.framing + "_" + prompts.culture + "_" +
               prompts.prompt_language.where(prompts.prompt_language != "mix", "mix_" + prompts.output_language))
    wide = prompts.assign(variant=variant).pivot(
        index=["source_id", "kind"], columns="variant", values="prompt")
    wide["n_variants"] = prompts.groupby(["source_id", "kind"]).size()
    wide = wide.reset_index()
    info = {**metadata, "source": "existing Dataset2 neutral requests; UK extension",
            "checks": "structural only; automated labels provisional", "mix_english_share": "not thresholded",
            "expected_prompts_per_seed": expected, "prompt_count": len(prompts)}
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        local = Path(tmp) / "dataset2.sqlite3"
        with sqlite3.connect(local) as con:
            for name, frame in [("prompts", prompts), ("seeds", seed_table), ("seed_texts", texts),
                                ("technique_outputs", technique_table), ("prompts_wide", wide)]:
                frame.to_sql(name, con, index=False)
            pd.DataFrame([{k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                           for k, v in info.items()}]).to_sql("generation_info", con, index=False)
            con.execute("CREATE UNIQUE INDEX prompts_id ON prompts(prompt_id)")
        load_dataset(local, expected_count=len(prompts), cultures=("neutral", "uk"))
        pending = destination.with_suffix(".partial")
        import shutil
        shutil.copyfile(local, pending)
        if file_sha256(local) != file_sha256(pending):
            raise IOError("Drive export checksum mismatch")
        os.replace(pending, destination)
    info["dataset_sha256"] = file_sha256(destination)
    destination.with_suffix(".manifest.json").write_text(json.dumps(info, ensure_ascii=False, indent=2))
    prompts.to_csv(destination.parent / "prompts.csv", index=False)
    return prompts, info
