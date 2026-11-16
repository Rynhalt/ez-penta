# %% [markdown]
# # B0_plan (local CPU): verify inputs, build training rows for ja / bi / en2, build the frozen evaluation set
# Writes `PT3/bilingual/plan.json`, `inputs/` (eval prompts, reuse-check targets), then `plan.DONE`.

# %% tags=["parameters"]
FORCE = False

# %%
import glob, hashlib, json, os, time, shutil, re as _re
from pathlib import Path

MYDRIVE = Path(glob.glob(os.path.expanduser("~/Library/CloudStorage/GoogleDrive-*"))[0]) / "My Drive"
PT3 = MYDRIVE / "ez-penta" / "post_train_3"
ARM = PT3 / "arm_en"
JC = PT3 / "japan_culture"
POOL = PT3 / "eval_pool" / "batches"
OUT = PT3 / "bilingual"
INP = OUT / "inputs"
INP.mkdir(parents=True, exist_ok=True)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_jsonl(p):
    p = Path(p)
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []


def write_jsonl(path, rs):
    with open(path, "w", encoding="utf-8") as f:
        for r in rs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def folder_ok(d):
    if not (d / "DONE").exists() or (d / "DONE").read_text().strip() != sha256_file(d / "manifest.json"):
        return False
    man = json.loads((d / "manifest.json").read_text())
    return all(sha256_file(d / f) == m["sha256"] for f, m in man["files"].items())


if (OUT / "plan.DONE").exists() and (OUT / "plan.DONE").read_text().strip() == sha256_file(OUT / "plan.json") and not FORCE:
    print("plan.DONE checks out; B0 already done")
    SKIP = True
else:
    SKIP = False

# %%
# 1. Hashes: training items (computed as in 01_training_items / 02_train), frozen pool, pool batches, Japan chunks, EN adapters.
hashes = {}
train_ids = json.loads((ARM / "train" / "train_item_ids.json").read_text())
EXPECTED_TRAIN_SHA = "a856a60d0ca5bccecbc56cfdb301c5986f92472d2d37211fff37100e0e658b6d"
got = hashlib.sha256("\n".join(sorted(train_ids)).encode()).hexdigest()
assert got == (ARM / "train" / "train_items_sha256.txt").read_text().strip() == EXPECTED_TRAIN_SHA, got
tman = json.loads((ARM / "train" / "manifest.json").read_text())
assert tman["train_items_sha256"] == EXPECTED_TRAIN_SHA
for f, m in tman["files"].items():
    if (ARM / "train" / f).exists():
        assert sha256_file(ARM / "train" / f) == m["sha256"], f"train file hash mismatch: {f}"
hashes["train_items_sha256"] = got
hashes["items.jsonl"] = sha256_file(ARM / "train" / "items.jsonl")
hashes["train_item_ids.json"] = sha256_file(ARM / "train" / "train_item_ids.json")

jplan = json.loads((JC / "plan.json").read_text())
assert (JC / "plan.DONE").read_text().strip() == sha256_file(JC / "plan.json")
hashes["frozen_pool.json"] = sha256_file(ARM / "frozen_pool.json")
assert hashes["frozen_pool.json"] == jplan["frozen_pool_sha256"], "frozen_pool.json hash differs from the Japan-culture plan"
frozen = json.loads((ARM / "frozen_pool.json").read_text())
assert frozen["batches"] == [1, 3, 5, 7, 9, 11] and frozen["n_seeds"] == 519

hashes["batches"] = {}
for b in frozen["batches"]:
    bd = POOL / f"b{b:04d}"
    assert (bd / "DONE").read_text().strip() == sha256_file(bd / "manifest.json"), f"{bd.name} DONE mismatch"
    bman = json.loads((bd / "manifest.json").read_text())
    for f, meta in bman["files"].items():
        assert sha256_file(bd / f) == meta["sha256"], f"hash mismatch: {bd.name}/{f}"
    assert sha256_file(bd / "manifest.json") == jplan["batch_manifest_sha256"][str(b)]
    hashes["batches"][bd.name] = sha256_file(bd / "manifest.json")

hashes["japan_chunks"] = {}
for c in jplan["chunks"]:
    cd = JC / "chunks" / f"c{c:04d}"
    for sub in ("prompts", "base", "en_s0"):
        assert folder_ok(cd / sub), f"{cd.name}/{sub} fails its DONE/manifest check"
        hashes["japan_chunks"][f"{cd.name}/{sub}"] = sha256_file(cd / sub / "manifest.json")

amani = json.loads((ARM / "adapters" / "manifest.json").read_text())
hashes["en_adapters"] = {}
for s in ("0", "1", "2"):
    for f, h in amani["seeds"][s]["files"].items():
        assert sha256_file(ARM / "adapters" / f"seed{s}" / f) == h, f"EN adapter seed{s}/{f} changed"
    hashes["en_adapters"][f"seed{s}"] = amani["seeds"][s]["files"]
hashes["arm_en_eval"] = {}
for b in frozen["batches"]:
    em = json.loads((ARM / "eval" / f"b{b:04d}.manifest.json").read_text())
    assert sha256_file(ARM / "eval" / f"b{b:04d}.jsonl") == em["sha256"], f"arm_en eval b{b:04d} hash mismatch"
    hashes["arm_en_eval"][f"b{b:04d}"] = em["sha256"]
print(json.dumps({k: v for k, v in hashes.items() if isinstance(v, str)}, indent=1))
print("all input hashes verified")

# %%
# 2. Training rows, in the order of train_item_ids.json. Only the rows differ between conditions.
items = {r["id"]: r for r in load_jsonl(ARM / "train" / "items.jsonl")}
assert len(items) == 823 and len(train_ids) == 568 and len(set(train_ids)) == 568
for i in train_ids:
    it = items[i]
    assert it["survives"] is True, i
    for l in ("en", "ja"):
        assert isinstance(it["prompt"].get(l), str) and it["prompt"][l].strip(), (i, "prompt", l)
        assert isinstance(it["answer"].get(l), str) and it["answer"][l].strip(), (i, "answer", l)
rows = {
    "en": [[i, "en"] for i in train_ids],
    "ja": [[i, "ja"] for i in train_ids],
    "bi": [[i, l] for i in train_ids for l in ("en", "ja")],
    "en2": [[i, l] for i in train_ids for l in ("en", "en")],
}
assert len(rows["en"]) == 568 and len(rows["ja"]) == 568 and len(rows["bi"]) == 1136 and len(rows["en2"]) == 1136
assert [r[0] for r in rows["bi"][::2]] == train_ids == [r[0] for r in rows["en2"][1::2]]
print({k: len(v) for k, v in rows.items()})

# %%
# 3. Evaluation set. US: frozen-pool seeds, batches ascending, first occurrence of each seed. Japan: all prompts of the 6 chunks.
seed_set = set(frozen["seed_ids"])
us, seen_seed, first_batch = [], set(), {}
for b in frozen["batches"]:
    bps = load_jsonl(POOL / f"b{b:04d}" / "prompts.jsonl")
    batch_seeds = []
    for p in bps:
        if p["seed_id"] in seed_set and p["seed_id"] not in first_batch:
            batch_seeds.append(p["seed_id"])
    new = set(batch_seeds)
    for sid in new:
        first_batch[sid] = b
    us += [p for p in bps if p["seed_id"] in new]
assert set(first_batch) == seed_set and len(seed_set) == 519
assert len(us) == 6228 and len({p["prompt_id"] for p in us}) == 6228, len(us)
# Cross-check against the Japan-culture plan's US prompt IDs.
plan_us = {pid for s in jplan["us_prompt_ids"].values() for pid in s.values()}
assert plan_us == {p["prompt_id"] for p in us}
jp, checks = [], []
for c in jplan["chunks"]:
    pd_ = JC / "chunks" / f"c{c:04d}" / "prompts"
    jp += load_jsonl(pd_ / "prompts_japan.jsonl")
    checks += load_jsonl(pd_ / "seed_checks.jsonl")
assert len(jp) == 6198 and len({p["prompt_id"] for p in jp}) == 6198, len(jp)
paired_ok = sorted(c["seed_id"] for c in checks if c["paired_ok"])
assert len(paired_ok) == 475, len(paired_ok)
assert {p["seed_id"] for p in jp} <= seed_set

META = ["seed_id", "kind", "framing", "lang", "severity", "category", "expected", "subtopic", "form"]
evp = []
for p in us:
    evp.append({"prompt_id": p["prompt_id"], "culture": "us", "us_prompt_id": p["prompt_id"], **{m: p[m] for m in META}, "prompt": p["prompt"]})
for p in jp:
    evp.append({"prompt_id": p["prompt_id"], "culture": "japan", "us_prompt_id": p["us_prompt_id"], **{m: p[m] for m in META}, "prompt": p["prompt"]})
assert len({p["prompt_id"] for p in evp}) == len(evp) == 12426

train_seeds = {i.split("|")[0] for i in train_ids}
assert all(items[i]["seed_id"] == i.split("|")[0] for i in train_ids)
assert not (train_seeds & seed_set), "a training seed is in the evaluation set"
assert not (train_seeds & {p["seed_id"] for p in evp})
print(f"eval prompts: US {len(us)}, Japan {len(jp)}; paired_ok seeds {len(paired_ok)}; training seeds {len(train_seeds)} disjoint from {len(seed_set)} eval seeds")

# %%
# 4. Reuse-check targets (4.2): stored rows where int(prompt_id, 16) % 10 == 0, per model x set.
us_ids = {p["prompt_id"] for p in us}
stored_us = {}
for b in frozen["batches"]:
    for r in load_jsonl(ARM / "eval" / f"b{b:04d}.jsonl"):
        if r["prompt_id"] in us_ids:
            stored_us[(r["model"], r["prompt_id"])] = r
stored_jp = {}
for c in jplan["chunks"]:
    for m in ("base", "en_s0"):
        for r in load_jsonl(JC / "chunks" / f"c{c:04d}" / m / "eval.jsonl"):
            if r["culture"] == "japan":
                stored_jp[(m, r["prompt_id"])] = r
targets = []
for m in ("base", "en_s0", "en_s1", "en_s2"):
    assert all((m, pid) in stored_us for pid in us_ids), f"stored US rows missing for {m}"
    for p in us:
        if int(p["prompt_id"], 16) % 10 == 0:
            r = stored_us[(m, p["prompt_id"])]
            targets.append({"model": m, "set": "us", "prompt_id": p["prompt_id"], "g06_unsafe": r["g06_unsafe"], "g06_refused": r["g06_refused"],
                            "g06_label": r["g06_label"]})
for m in ("base", "en_s0"):
    assert all((m, p["prompt_id"]) in stored_jp for p in jp), f"stored Japan rows missing for {m}"
    for p in jp:
        if int(p["prompt_id"], 16) % 10 == 0:
            r = stored_jp[(m, p["prompt_id"])]
            targets.append({"model": m, "set": "japan", "prompt_id": p["prompt_id"], "g06_unsafe": r["g06_unsafe"], "g06_refused": r["g06_refused"],
                            "g06_label": r["g06_label"]})
from collections import Counter
print("reuse-check targets:", Counter((t["model"], t["set"]) for t in targets))

# Stored manifests for the session-match part of the reuse rule.
stored_manifests = {"us": {f"b{b:04d}": {k: json.loads((ARM / "eval" / f"b{b:04d}.manifest.json").read_text())[k] for k in ("hardware", "dtype", "versions", "serving")}
                           for b in frozen["batches"]},
                    "japan": {f"c{c:04d}/{m}": {k: json.loads((JC / "chunks" / f"c{c:04d}" / m / "manifest.json").read_text())[k] for k in ("hardware", "dtype", "versions")}
                              for c in jplan["chunks"] for m in ("base", "en_s0")}}

# %%
# 5. Write inputs/ and plan.json, then plan.DONE.
if not SKIP:
    write_jsonl(INP / "eval_prompts.jsonl", evp)
    write_jsonl(INP / "reuse_targets.jsonl", targets)
    plan = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "conditions": ["en", "ja", "bi", "en2"], "train_seeds": [0, 1, 2],
            "rows": rows, "row_counts": {k: len(v) for k, v in rows.items()},
            "train_item_ids": train_ids, "train_seed_ids": sorted(train_seeds),
            "eval": {"us_prompt_ids": [p["prompt_id"] for p in us], "japan_prompt_ids": [p["prompt_id"] for p in jp],
                     "n_us": len(us), "n_japan": len(jp), "paired_ok_seeds": paired_ok, "n_eval_seeds": len(seed_set),
                     "first_batch_of_seed": first_batch},
            "hashes": hashes, "stored_manifests": stored_manifests,
            "inputs": {"eval_prompts.jsonl": {"rows": len(evp), "sha256": sha256_file(INP / "eval_prompts.jsonl")},
                       "reuse_targets.jsonl": {"rows": len(targets), "sha256": sha256_file(INP / "reuse_targets.jsonl")}}}
    (OUT / "plan.json").write_text(json.dumps(plan, indent=1, ensure_ascii=False))
    (OUT / "plan.DONE").write_text(sha256_file(OUT / "plan.json") + "\n")
    print("wrote plan.json + plan.DONE")
plan = json.loads((OUT / "plan.json").read_text())
for f, m in plan["inputs"].items():
    assert sha256_file(INP / f) == m["sha256"], f
print(plan["row_counts"], plan["eval"]["n_us"], plan["eval"]["n_japan"], len(plan["eval"]["paired_ok_seeds"]))
