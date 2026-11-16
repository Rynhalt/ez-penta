"""Build the Hugging Face dataset release for the post_train_3 multilingual study.

Reads the finalized post_train_3 working copy (read-only), validates it against the run's own
plan hashes and DONE markers, flattens it to parquet tables and writes one folder per HF dataset
repo with a dataset card and a SHA-256 manifest. Nothing is uploaded.

Repos built so far (see docs/HF_DATASET_RELEASE_PLAN.md):
  A. tiny-aya-safety-multilingual-prompts   evaluation inputs
  B. tiny-aya-safety-multilingual-training  LoRA training rows

Usage:
  .venv/bin/python scripts/build_hf_release.py            # build all
  .venv/bin/python scripts/build_hf_release.py --only prompts
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import glob
import hashlib
import json
import re
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SRC = Path.home() / "Library/CloudStorage/GoogleDrive-test-ezpenta/My Drive/ez-penta/post_train_3"
DEFAULT_OUT = REPO_ROOT / "outputs" / "hf_release"
NAMESPACE = "isaac369369"
BASE = "CohereLabs/tiny-aya-water"
LICENSE = "cc-by-nc-4.0"
DRIVE_SNAPSHOT = "https://drive.google.com/drive/folders/1bH59zkx1BlKXcqvR3L-NAbh3fbby3ltQ"
PILOT_COLLECTION = "https://huggingface.co/collections/isaac369369/tiny-aya-safety-datasets-6af671c33b325a1767adb0a5"

REPO_PROMPTS = "tiny-aya-safety-multilingual-prompts"
REPO_TRAINING = "tiny-aya-safety-multilingual-training"

# Strings that must never reach a public file.
SECRET_PATTERNS = [
    (re.compile(r"hf_[A-Za-z0-9]{30,}"), "huggingface token"),
    (re.compile(r"sk-[A-Za-z0-9_-]{20,}"), "api key"),
    (re.compile(r"/Users/[A-Za-z0-9_.-]+"), "local macOS path"),
    (re.compile(r"/content/drive"), "colab drive path"),
    (re.compile(r"Library/CloudStorage"), "local drive path"),
    (re.compile(r"[A-Za-z0-9._%+-]+@(gmail|googlemail|icloud|outlook|yahoo)\.com"), "personal email"),
]


# ----------------------------------------------------------------------------- helpers

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(p: Path) -> list[dict]:
    with open(p, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def check_done(folder: Path) -> None:
    """Every run folder publishes manifest.json then DONE = sha256(manifest.json)."""
    done = folder / "DONE"
    if not done.exists():
        raise SystemExit(f"not finalized (no DONE): {folder}")
    if done.read_text().strip() != sha256_file(folder / "manifest.json"):
        raise SystemExit(f"DONE does not match manifest.json: {folder}")


def js(v) -> str:
    """Nested values are stored as JSON strings so the parquet schema stays flat and stable."""
    return json.dumps(v, ensure_ascii=False, sort_keys=True)


class Release:
    """Collects tables for one HF repo, then writes parquet, card and manifest."""

    def __init__(self, name: str, src: Path, out: Path):
        self.name, self.src, self.dir = name, src, out / name
        self.tables: dict[str, pd.DataFrame] = {}
        self.sources: dict[str, str] = {}
        self.checks: list[str] = []

    def source(self, p: Path) -> Path:
        self.sources[str(p.relative_to(self.src))] = sha256_file(p)
        return p

    def check(self, cond: bool, msg: str) -> None:
        if not cond:
            raise SystemExit(f"[{self.name}] CHECK FAILED: {msg}")
        self.checks.append(msg)

    def add(self, config: str, df: pd.DataFrame) -> None:
        self.tables[config] = df.reset_index(drop=True)

    def scan(self) -> None:
        hits = []
        for config, df in self.tables.items():
            for col in df.columns:
                if df[col].dtype != object:
                    continue
                text = "\n".join(df[col].dropna().astype(str))
                for rx, what in SECRET_PATTERNS:
                    m = rx.search(text)
                    if m:
                        hits.append(f"{config}.{col}: {what} ({m.group(0)[:40]!r})")
        if hits:
            raise SystemExit(f"[{self.name}] secret/path scan failed:\n  " + "\n  ".join(hits))
        self.checks.append("secret/path scan: no tokens, keys, local paths or personal emails in any text column")

    def write(self, card: str) -> None:
        self.scan()
        data = self.dir / "data"
        data.mkdir(parents=True, exist_ok=True)
        for old in data.glob("*.parquet"):
            old.unlink()
        files = {}
        for config, df in self.tables.items():
            p = data / f"{config}.parquet"
            df.to_parquet(p, index=False, compression="zstd")
            files[f"data/{config}.parquet"] = {"rows": len(df), "sha256": sha256_file(p)}
        (self.dir / "README.md").write_text(card, encoding="utf-8")
        manifest = {
            "repo": f"{NAMESPACE}/{self.name}",
            "built_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "builder": "scripts/build_hf_release.py",
            "files": files,
            "sources_sha256": dict(sorted(self.sources.items())),
            "checks_passed": self.checks,
        }
        (self.dir / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
        print(f"\n[{self.name}] -> {self.dir}")
        for f, m in files.items():
            print(f"  {f:45s} {m['rows']:>7,} rows")
        print(f"  {len(self.checks)} checks passed")


def configs_yaml(configs: list[str], default: str) -> str:
    out = ["configs:"]
    for c in configs:
        out.append(f"- config_name: {c}")
        if c == default:
            out.append("  default: true")
        out.append(f"  data_files: data/{c}.parquet")
    return "\n".join(out)


def table_md(rel: Release, descr: dict[str, str]) -> str:
    rows = ["| Config | Rows | Contents |", "|---|---:|---|"]
    for c, df in rel.tables.items():
        rows.append(f"| `{c}` | {len(df):,} | {descr[c]} |")
    return "\n".join(rows)


# ----------------------------------------------------------------------------- A. prompts

def build_prompts(src: Path, out: Path) -> Release:
    rel = Release(REPO_PROMPTS, src, out)
    bplan = json.loads(rel.source(src / "bilingual/plan.json").read_text())
    fplan = json.loads(rel.source(src / "followups/plan.json").read_text())
    jplan = json.loads(rel.source(src / "japan_culture/plan.json").read_text())
    ok_bi = set(bplan["eval"]["paired_ok_seeds"])
    ok_half = set(fplan["paired_ok_half"])

    # --- frozen pool: seeds and US prompts per batch
    seeds, prompts, rejects = [], [], []
    for b in sorted((src / "eval_pool/batches").iterdir()):
        check_done(b)
        rel.source(b / "manifest.json")
        for r in load_jsonl(rel.source(b / "seeds.jsonl")):
            seeds.append({
                "batch": r["batch"], "row": r["row"], "seed_id": r["seed_id"], "cell_id": r["cell_id"],
                "category": r["category"], "subtopic": r["subtopic"], "form": r["form"], "severity": r["severity"],
                "paired_ok": r["seed_id"] in ok_bi,
                "texts_json": js(r["texts"]), "versions_json": js(r["versions"]),
                "twin_versions_json": js(r["twin_versions"]), "checks_json": js(r["checks"]),
            })
        prompts += load_jsonl(rel.source(b / "prompts.jsonl"))
        for r in load_jsonl(rel.source(b / "rejects.jsonl")):
            rejects.append({"batch": int(b.name[1:]), "key": str(r.get("key")), "stage": str(r.get("stage")),
                            "attempt": r.get("attempt"), "reason": str(r.get("reason")), "raw": js(r.get("raw"))})
    seeds_df = pd.DataFrame(seeds)
    # seed_id is reused by 7 seeds across batches with different text; the eval uses the batch it first appears in.
    first = bplan["eval"]["first_batch_of_seed"]
    seeds_df["in_bilingual_eval"] = [int(first.get(s, -1)) == int(b) for s, b in zip(seeds_df.seed_id, seeds_df.batch)]
    rel.check(len(seeds_df) == 600, "pool_seeds: 6 frozen batches x 100 generated seeds = 600 rows")
    rel.check(seeds_df.in_bilingual_eval.sum() == bplan["eval"]["n_eval_seeds"] == 519,
              "pool_seeds: 519 seeds enter the bilingual evaluation (first batch of each seed_id)")
    rel.check(seeds_df.paired_ok.sum() >= 475 and len(ok_bi) == 475, "pool_seeds: 475 paired_ok seeds (plan.json)")
    rel.add("pool_seeds", seeds_df)

    pool = pd.DataFrame(prompts)
    pool["in_bilingual_eval"] = [int(first.get(s, -1)) == int(b) for s, b in zip(pool.seed_id, pool.batch)]
    rel.check(len(pool) == 7200 and not pool.duplicated(["batch", "prompt_id"]).any(),
              "pool_prompts_us: 7,200 rows, unique per (batch, prompt_id); 7 reused seed_ids repeat ids across batches")
    rel.check(pool[pool.in_bilingual_eval].prompt_id.is_unique, "pool_prompts_us: prompt_id unique among rows used in evaluation")
    rel.add("pool_prompts_us", pool)
    rel.add("pool_rejects", pd.DataFrame(rejects))

    # --- bilingual evaluation set (US + Japan culture, EN + JA)
    p = rel.source(src / "bilingual/inputs/eval_prompts.jsonl")
    rel.check(sha256_file(p) == bplan["inputs"]["eval_prompts.jsonl"]["sha256"],
              "bilingual_eval: file hash equals plan.json inputs hash")
    bi = pd.DataFrame(load_jsonl(p))
    bi["paired_ok"] = bi.seed_id.isin(ok_bi)
    rel.check(len(bi) == 12426 and bi.prompt_id.is_unique, "bilingual_eval: 12,426 unique prompts")
    rel.check(bi.seed_id.nunique() == 519, "bilingual_eval: 519 source seeds")
    rel.check(len(bi[bi.culture == "us"]) == bplan["eval"]["n_us"] and len(bi[bi.culture == "japan"]) == bplan["eval"]["n_japan"],
              "bilingual_eval: US/Japan counts match plan.json")
    rel.add("bilingual_eval", bi)

    # --- half pool (E1-E3)
    half = pd.DataFrame(load_jsonl(rel.source(src / "followups/inputs/eval_half.jsonl")))
    half["paired_ok"] = half.seed_id.isin(ok_half)
    rel.check(len(half) == int(fplan["n_eval_half"]) == 6598 and half.seed_id.nunique() == 276,
              "half_pool: 6,598 prompts on 276 seeds (batches 1, 3, 5)")
    rel.check(set(half.prompt_id) <= set(bi.prompt_id), "half_pool: every prompt is in bilingual_eval")
    rel.add("half_pool", half)

    # --- E4 matched US/UK
    e4dir = src / "followups/e4/prompts"
    check_done(e4dir)
    rel.source(e4dir / "manifest.json")
    e4 = pd.DataFrame(load_jsonl(rel.source(e4dir / "prompts_matched.jsonl")))
    sc = pd.DataFrame(load_jsonl(rel.source(e4dir / "seed_checks.jsonl")))
    sc["paired_ok"] = sc.paired_ok.astype(bool)
    both = sc.groupby("seed_id").paired_ok.agg(lambda s: bool(s.all()) and len(s) == 2)
    e4["matched_both_cultures"] = e4.seed_id.map(both).fillna(False).astype(bool)
    rel.check(len(e4) == 5776 and e4.prompt_id.is_unique, "e4_us_uk_matched: 5,776 unique prompts")
    rel.check(int(both.sum()) == 150, "e4_us_uk_matched: 150 of 276 seeds pass matching checks in both cultures")
    rel.add("e4_us_uk_matched", e4)
    sc["reasons"] = sc.reasons.map(js)
    sc["checks"] = sc.checks.map(js)
    rel.add("e4_seed_checks", sc)

    # --- Japan-culture prompts (6 chunks)
    jc, jcs = [], []
    for c in sorted((src / "japan_culture/chunks").iterdir()):
        d = c / "prompts"
        check_done(d)
        rel.source(d / "manifest.json")
        for r in load_jsonl(rel.source(d / "prompts_japan.jsonl")):
            jc.append({**r, "chunk": int(c.name[1:])})
        for r in load_jsonl(rel.source(d / "seed_checks.jsonl")):
            jcs.append({"chunk": int(c.name[1:]), "seed_id": r["seed_id"], "paired_ok": bool(r["paired_ok"]),
                        "reasons": js(r["reasons"]), "checks": js(r["checks"])})
    jc, jcs = pd.DataFrame(jc), pd.DataFrame(jcs)
    rel.check(len(jcs) == int(jplan["n_seeds"]) == 519, "japan_culture_seed_checks: 519 seeds across 6 chunks")
    rel.check(set(jc.prompt_id) == set(bi[bi.culture == "japan"].prompt_id),
              "japan_culture: chunk prompts are exactly the Japan-culture half of bilingual_eval")
    rel.add("japan_culture", jc)
    rel.add("japan_culture_seed_checks", jcs)

    descr = {
        "pool_seeds": "Generated Dataset2-v3 source seeds of the frozen pool (6 batches). `texts_json` holds every harmful/twin x framing x language text; `paired_ok` = used in paired analyses.",
        "pool_prompts_us": "US-culture prompts built from the pool seeds (harmful + harmless twin, 3 framings, EN + JA).",
        "pool_rejects": "Generation attempts rejected by the pool's quality gates, with reasons.",
        "bilingual_eval": "**Main evaluation set** of the language-composition study: US- and Japan-framed prompts, EN + JA, 519 seeds.",
        "half_pool": "Half of the pool (batches 1, 3, 5; 276 seeds) used by follow-ups E1-E3.",
        "e4_us_uk_matched": "E4 rewritten US (`us_m`) and UK (`uk_m`) cue prompts with matched content. `matched_both_cultures` marks the 150 analysed seeds.",
        "e4_seed_checks": "Per-seed, per-culture E4 matching checks (EN/JA cosine, reasons for exclusion).",
        "japan_culture": "Japan-framed prompts per chunk with the `us_prompt_id` of their US counterpart (same seed, kind, framing, language).",
        "japan_culture_seed_checks": "Per-seed checks for the Japan-culture prompts.",
    }
    card = f"""---
license: {LICENSE}
language: [en, ja]
pretty_name: Tiny Aya multilingual safety evaluation prompts
task_categories: [text-generation]
tags: [safety, jailbreak, refusal, over-refusal, multilingual, japanese, cultural-framing, red-teaming, lexsi-hackathon]
size_categories: [10K<n<100K]
{configs_yaml(list(rel.tables), "bilingual_eval")}
---

# Tiny Aya multilingual safety evaluation prompts

> **Content warning.** This dataset contains requests written to elicit harmful content (violence, crime,
> self-harm and other categories), in English and Japanese. It is released for safety research only.

These are the frozen evaluation inputs of the study *"Safer, or Just More Refusals? English, Japanese and
Bilingual Safety Fine-Tuning of Tiny Aya"*. Each **source seed** is a request expanded into:

- a **harmful** prompt (`kind=harmful`, `expected=decline`) and a **harmless twin** with the same surface form
  (`kind=twin`, `expected=answer`) used to measure over-refusal;
- three **framings**: `control` (plain request), `rp` (role-play) and `fc` (fiction);
- two **prompt languages** (`lang` = `en`, `ja`), each ending with an instruction to answer in that language;
- **cultural cues** (`culture`): `us`, `japan`, and in E4 rewritten `us_m` / `uk_m`. Prompt language and cultural
  cue are separate factors: an English prompt can carry a Japan cue.

Model answers and Qwen3Guard labels for these prompts are in the companion results dataset; training rows are in
[`{NAMESPACE}/{REPO_TRAINING}`](https://huggingface.co/datasets/{NAMESPACE}/{REPO_TRAINING}).

## Configs

{table_md(rel, descr)}

## How the sets relate

- `pool_prompts_us` + the Japan-culture prompts = `bilingual_eval` (519 seeds; 475 `paired_ok` seeds are used for
  paired statistics).
- `half_pool` is the subset on batches 1, 3, 5 (276 seeds) used by follow-ups E1-E4.
- `e4_us_uk_matched` rewrites the `half_pool` seeds with matched US and UK cues; only seeds with
  `matched_both_cultures=True` (150/276) enter the US-vs-UK comparison. The rewrites change wording as well as the
  cue, so original-vs-rewrite differences are not a pure culture effect.
- Evaluation seeds (`d3-*`) are disjoint from training seeds (`d2-*`).

## Key fields

`prompt_id` (unique), `seed_id`, `us_prompt_id` (US counterpart of a non-US prompt), `kind`, `expected`, `framing`,
`lang`, `culture`, `category`, `subtopic`, `form`, `severity` (`clear` / borderline), `text` (request only) and
`prompt` (exact model input, including the answer-language instruction).

## Provenance

Generated and filtered by the post_train_3 pipeline (prompt writer: `{BASE}`; checks include EN/JA embedding
cosine and gate stages listed in `pool_rejects` / `*_seed_checks`). Every source file was checked against the
run's own `plan.json` hashes and `DONE` markers; source and output SHA-256 values are in `manifest.json`.
Snapshot of the run folder: [Google Drive]({DRIVE_SNAPSHOT}) (access may be restricted).
Earlier pilot datasets: [collection]({PILOT_COLLECTION}).

## Limitations

Prompts are machine-generated and machine-filtered; no human review of every prompt was performed. Translations are
checked by embedding similarity, not by native speakers. Cultural cues are stereotyped proxies (e.g. "samurai",
"cowboy") and do not represent any culture. License follows the base model's CC BY-NC 4.0 terms.
"""
    rel.write(card)
    return rel


# ----------------------------------------------------------------------------- B. training

def build_training(src: Path, out: Path) -> Release:
    rel = Release(REPO_TRAINING, src, out)
    tdir = src / "arm_en/train"
    bplan = json.loads(rel.source(src / "bilingual/plan.json").read_text())
    fplan = json.loads(rel.source(src / "followups/plan.json").read_text())
    rel.source(tdir / "manifest.json")

    ip = rel.source(tdir / "items.jsonl")
    rel.check(sha256_file(ip) == bplan["hashes"]["items.jsonl"] == fplan["hashes"]["items.jsonl"],
              "items.jsonl hash equals bilingual and follow-up plan hashes")
    items = load_jsonl(ip)
    train_ids = json.loads(rel.source(tdir / "train_item_ids.json").read_text())
    rel.check(train_ids == bplan["train_item_ids"], "train_item_ids.json equals bilingual plan order")
    tid = set(train_ids)
    half = fplan["item_half"]

    flat = []
    for r in items:
        for lang in ("en", "ja"):
            flat.append({
                "row_id": f"{r['id']}|{lang}", "item_id": r["id"], "seed_id": r["seed_id"], "lang": lang,
                "kind": r["kind"], "framing": r["framing"], "item_kind": r["item_kind"], "split": r["split"],
                "category": r["category"], "subtopic": r["subtopic"],
                "prompt_id": r["prompt_id"][lang], "prompt": r["prompt"][lang], "target": r["answer"][lang],
                "gate_pass": bool(r["pass"][lang]), "attempt_passed": r["attempt_passed"][lang],
                "item_survives": bool(r["survives"]), "in_main_training": r["id"] in tid,
                "followup_half": half.get(r["id"]),
            })
    flat = pd.DataFrame(flat)
    rel.check(len(items) == 823 and len(flat) == 1646, "items: 823 aligned items -> 1,646 (item, language) rows")
    main = flat[flat.in_main_training]
    kinds = main.drop_duplicates("item_id").item_kind.value_counts().to_dict()
    rel.check(len(tid) == 568 and kinds == {"refusal": 284, "helpful": 284},
              "items: 568 main training items, 284 refusal + 284 helpful")
    rel.check(bool(main.item_survives.all()) and bool(main.gate_pass.all()),
              "items: every main training item survived the gate in both languages")
    rel.add("items", flat)

    # Exact training streams, one row per optimizer step (batch size 1).
    lookup = flat.set_index("row_id")
    streams = []
    for study, plan in (("bilingual", bplan), ("followups", fplan)):
        for cond, spec in plan["rows"].items():
            rel.check(len(spec) == plan["row_counts"][cond], f"conditions: {cond} has {plan['row_counts'][cond]} rows")
            for step, (item_id, lang) in enumerate(spec):
                r = lookup.loc[f"{item_id}|{lang}"]
                streams.append({"study": study, "condition": cond, "step": step, "item_id": item_id, "lang": lang,
                                "item_kind": r.item_kind, "kind": r.kind, "framing": r.framing,
                                "prompt": r.prompt, "target": r.target})
    streams = pd.DataFrame(streams)
    expect = {"en": 568, "ja": 568, "bi": 1136, "en2": 1136,
              "bi_h2": 1704, "ja2": 1136, "en_A": 333, "ja_A": 333, "en_B": 235, "ja_B": 235}
    rel.check(streams.condition.value_counts().to_dict() == expect,
              "conditions: row counts " + ", ".join(f"{k}={v}" for k, v in expect.items()))
    rel.check(set(streams[streams.condition == "ja"].lang) == {"ja"} and set(streams[streams.condition.isin(["en", "en2"])].lang) == {"en"},
              "conditions: en/en2 rows are English only, ja rows Japanese only")
    rel.add("conditions", streams)

    gen = pd.DataFrame(load_jsonl(rel.source(tdir / "gen.jsonl")))
    gen["categories"] = gen["categories"].map(lambda v: None if v is None else js(v))
    rel.add("generation_attempts", gen)

    gs = json.loads(rel.source(tdir / "gate_stats.json").read_text())
    rel.add("gate_stats", pd.DataFrame([{"group": k, **{kk: (vv if not isinstance(vv, (dict, list)) else js(vv)) for kk, vv in v.items()}}
                                        for k, v in gs.items()]))

    # Leakage: training seeds and texts must not appear in any evaluation prompt.
    ev = pd.DataFrame(load_jsonl(src / "bilingual/inputs/eval_prompts.jsonl"))
    e4 = pd.DataFrame(load_jsonl(src / "followups/e4/prompts/prompts_matched.jsonl"))
    rel.check(not (set(flat.seed_id) & (set(ev.seed_id) | set(e4.seed_id))), "leakage: 0 training seeds in evaluation sets")
    rel.check(not (set(flat.prompt) & (set(ev.prompt) | set(e4.prompt))), "leakage: 0 training prompt texts in evaluation sets")

    descr = {
        "items": "Every aligned training item in both languages: prompt, target answer, gate result, `in_main_training` (the 568 used), `followup_half` (A/B split for E3).",
        "conditions": "The exact training stream of every adapter condition, one row per optimizer step (batch size 1), in training order.",
        "generation_attempts": "All target-answer generation attempts with the Qwen3Guard gate label that accepted or rejected them.",
        "gate_stats": "Gate pass rates per item kind and language.",
    }
    card = f"""---
license: {LICENSE}
language: [en, ja]
pretty_name: Tiny Aya multilingual safety LoRA training data
task_categories: [text-generation]
tags: [safety, refusal, sft, lora, multilingual, japanese, lexsi-hackathon]
size_categories: [1K<n<10K]
{configs_yaml(list(rel.tables), "conditions")}
---

# Tiny Aya multilingual safety LoRA training data

> **Content warning.** Prompts include harmful requests (paired with refusal targets) in English and Japanese.

Training data for the LoRA adapters in *"Safer, or Just More Refusals? English, Japanese and Bilingual Safety
Fine-Tuning of Tiny Aya"*. 568 aligned items (284 refusal targets for harmful prompts, 284 helpful targets for
harmless twins) are presented in different languages and orders:

| Condition | Rows (steps) | Stream |
|---|---:|---|
| `en` | 568 | English row per item |
| `ja` | 568 | Japanese row per item |
| `bi` | 1,136 | English row then Japanese row per item |
| `en2` | 1,136 | English row twice per item (dose control for `bi`) |
| `bi_h2` | 1,704 | `bi`, with helpful items shown twice (follow-up E2) |
| `ja2` | 1,136 | Japanese row twice (follow-up) |
| `en_A`, `ja_A`, `en_B`, `ja_B` | 333 / 235 | Item halves A and B (follow-up E3) |

Each main condition was trained with 3 seeds (0, 1, 2). An **item** is an aligned prompt/target example; a **row**
presents it in one language; a **training seed** controls one adapter run.

## Training setup

Base model [`{BASE}`](https://huggingface.co/{BASE}) (3.35B, frozen). PEFT LoRA rank 16, alpha 32, dropout 0.05,
bias `none`, target **only** `model.layers.18.mlp.down_proj` (208,896 trainable parameters). AdamW, learning rate
1e-4, one epoch, batch size 1, no scheduler, no gradient accumulation, max sequence length 768, chat template of the
base model. Targets were generated by the base model and accepted only if Qwen3Guard-Gen-0.6B judged them Safe with
the expected refusal behaviour (see `generation_attempts`, `gate_stats`).

## Configs

{table_md(rel, descr)}

## Checks run at build time

{chr(10).join('- ' + c for c in rel.checks)}

## Provenance and limitations

Item and plan hashes match the training run's `plan.json`; SHA-256 values are in `manifest.json`. Targets are
model-generated and automatically gated, not human-written or human-reviewed. Japanese targets passed the gate less
often than English ones on the first attempt (see `gate_stats`). License follows the base model's CC BY-NC 4.0 terms.
"""
    rel.write(card)
    return rel


# ----------------------------------------------------------------------------- main

BUILDERS = {"prompts": build_prompts, "training": build_training}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC, help="post_train_3 working copy (read-only)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--only", choices=sorted(BUILDERS), action="append")
    a = ap.parse_args()
    if not a.src.is_dir():
        sys.exit(f"source not found: {a.src}")
    for name in a.only or BUILDERS:
        BUILDERS[name](a.src, a.out)


if __name__ == "__main__":
    main()
