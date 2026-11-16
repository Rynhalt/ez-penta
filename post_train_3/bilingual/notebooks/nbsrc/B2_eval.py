# %% [markdown]
# # B2_eval (Colab GPU): answer and judge the frozen evaluation set with one merged model
# `MODEL` = output name. `<m>` answers all 12,426 prompts (US + Japan); `<m>_jp` the 6,198 Japanese-culture prompts only; `<m>_us` the 6,228 US prompts only.
# `<m>` is `base`, an EN-arm adapter `en_s{0,1,2}` (existing, unchanged), or a new adapter `{ja,bi,en2}_s{0,1,2}`.
# Helpers, environment, torchao removal, merge-and-load, answer and judge loops are copied from the EN arm notebooks (0.6).

# %% tags=["parameters"]
MODEL = "bi_s0"
WORK = "/content/pt3bi"
MERGED_ROOT = "/content/merged"
ANSWER_BATCH = 400
JUDGE_BATCH = 800

# %% COPY japan_culture/en/notebooks/J1_chunk_en.ipynb 2

# %% COPY japan_culture/en/notebooks/J1_chunk_en.ipynb 3

# %% COPY japan_culture/en/notebooks/J1_chunk_en.ipynb 4

# %% COPY japan_culture/en/notebooks/J1_chunk_en.ipynb 5

# %%
# free_llm, copied verbatim from 04_eval_batch (cell 5) / J1_chunk_en (cell 9).
def free_llm(llm):
    eng = getattr(llm, "llm_engine", None)
    core = getattr(eng, "engine_core", None)
    if core is not None and hasattr(core, "shutdown"):
        core.shutdown()
    del llm
    gc.collect()
    torch.cuda.empty_cache()

# %%
# Inputs, prompt set, adapter. Resumes from STAGE checkpoints; skips if the output folder's DONE checks out.
IN = Path(WORK) / "inputs"
assert (IN / "plan.DONE").read_text().strip() == sha256_file(IN / "plan.json"), "plan.DONE mismatch"
plan = json.loads((IN / "plan.json").read_text())
assert sha256_file(IN / "eval_prompts.jsonl") == plan["inputs"]["eval_prompts.jsonl"]["sha256"]
EVP = load_jsonl(IN / "eval_prompts.jsonl")
assert len(EVP) == 12426
if MODEL.endswith("_jp"):
    SRC, SETS = MODEL[:-3], ("japan",)
elif MODEL.endswith("_us"):
    SRC, SETS = MODEL[:-3], ("us",)
else:
    SRC, SETS = MODEL, ("us", "japan")
PROMPTS = [p for p in EVP if p["culture"] in SETS]
assert len(PROMPTS) == sum({"us": 6228, "japan": 6198}[s] for s in SETS)
OUTD = Path(WORK) / "out" / "eval" / MODEL
STAGE = Path(WORK) / "work" / MODEL
OUTD.mkdir(parents=True, exist_ok=True)
STAGE.mkdir(parents=True, exist_ok=True)


def dir_done(d):
    if not (d / "DONE").exists() or (d / "DONE").read_text().strip() != sha256_file(d / "manifest.json"):
        return False
    man = json.loads((d / "manifest.json").read_text())
    return all((d / f).exists() and sha256_file(d / f) == m["sha256"] for f, m in man["files"].items())


SKIP = dir_done(OUTD)
if SRC == "base":
    ADIR, adapter_sha, adapter_source = None, None, None
elif SRC.startswith("en_s"):
    ADIR = IN / "en_adapters" / f"seed{int(SRC[-1])}"
    adapter_source = "arm_en/adapters (existing EN adapter, unchanged)"
    exp = plan["hashes"]["en_adapters"][f"seed{int(SRC[-1])}"]
    adapter_sha = {f: sha256_file(ADIR / f) for f in ("adapter_config.json", "adapter_model.safetensors")}
    assert all(adapter_sha[f] == exp[f] for f in adapter_sha), "EN adapter differs from arm_en/adapters/manifest.json"
else:
    ADIR = Path(WORK) / "out" / "adapters" / SRC
    adapter_source = f"bilingual/adapters/{SRC}"
    assert dir_done(ADIR), f"adapter {SRC} has no valid DONE"
    am = json.loads((ADIR / "manifest.json").read_text())
    adapter_sha = {f: sha256_file(ADIR / f) for f in ("adapter_config.json", "adapter_model.safetensors")}
    assert all(adapter_sha[f] == am["files"][f]["sha256"] for f in adapter_sha)
print(f"MODEL {MODEL}: source {SRC}, sets {SETS}, {len(PROMPTS)} prompts; skip={SKIP}; {HARDWARE} {DTYPE}")

# %%
# Merge the adapter into bf16 weights on the VM's local disk (merge code from 04_eval_batch / J1_chunk_en), then answer in batches of 400.
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
TORCH_DTYPE = torch.bfloat16 if DTYPE == "bfloat16" else torch.float16
ANS = STAGE / "answers.jsonl"
ans = {r["prompt_id"]: r for r in load_jsonl(ANS)}
todo = [p for p in PROMPTS if p["prompt_id"] not in ans]
print(f"answers: {len(ans)} done, {len(todo)} to go")
if not SKIP and todo:
    if SRC == "base":
        MPATH = TINY_AYA
    else:
        mp = Path(MERGED_ROOT) / SRC
        if not (mp / "config.json").exists():
            b = AutoModelForCausalLM.from_pretrained(TINY_AYA, dtype=TORCH_DTYPE, device_map="cpu")
            PeftModel.from_pretrained(b, str(ADIR)).merge_and_unload().save_pretrained(mp)
            AutoTokenizer.from_pretrained(TINY_AYA).save_pretrained(mp)
            del b
            gc.collect()
        MPATH = str(mp)
    llm = load_llm(MPATH, DTYPE, gpu_memory_utilization=0.40)
    t0 = time.time()
    for i in range(0, len(todo), ANSWER_BATCH):
        ch = todo[i:i + ANSWER_BATCH]
        outs = answer(llm, [p["prompt"] for p in ch])
        rs = [{"model": MODEL, "prompt_id": p["prompt_id"], **o} for p, o in zip(ch, outs)]
        append_jsonl(ANS, rs)
        ans.update({r["prompt_id"]: r for r in rs})
        print(f"  {MODEL}: {i + len(ch)}/{len(todo)} ({time.time() - t0:.0f}s)")
    free_llm(llm)
assert SKIP or all(p["prompt_id"] in ans for p in PROMPTS)

# %%
# Judge every answer with both guards, one loaded at a time (loop from 04_eval_batch / J1_chunk_en).
by_pid = {p["prompt_id"]: p for p in PROMPTS}
verdicts = {}
if not SKIP:
    for tag, gid, util in [("g06", GUARD_06, 0.15), ("g4b", GUARD_4B, 0.30)]:
        path = STAGE / f"judge_{tag}.jsonl"
        v = {r["prompt_id"]: r for r in load_jsonl(path)}
        todo_j = [p["prompt_id"] for p in PROMPTS if p["prompt_id"] not in v]
        print(f"{tag}: {len(v)} judged, {len(todo_j)} to go")
        if todo_j:
            g = load_llm(gid, DTYPE, gpu_memory_utilization=util, max_model_len=8192)
            for i in range(0, len(todo_j), JUDGE_BATCH):
                ch = todo_j[i:i + JUDGE_BATCH]
                js = judge(g, [by_pid[k]["prompt"] for k in ch], [ans[k]["answer"] for k in ch])
                rs = [{"prompt_id": k, **j} for k, j in zip(ch, js)]
                append_jsonl(path, rs)
                v.update({r["prompt_id"]: r for r in rs})
                print(f"  {tag}: {i + len(ch)}/{len(todo_j)}")
            free_llm(g)
        verdicts[tag] = v

# %%
# Write eval.jsonl, then manifest.json, then DONE.
def write_jsonl(path, rs):
    with open(path, "w", encoding="utf-8") as f:
        for r in rs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if not SKIP:
    META = ["culture", "us_prompt_id", "seed_id", "kind", "framing", "lang", "severity", "category", "expected", "subtopic", "form"]
    out_rows = []
    for p in PROMPTS:
        a = ans[p["prompt_id"]]
        row = {"model": MODEL, "prompt_id": p["prompt_id"], **{m: p[m] for m in META}, "answer": a["answer"], "tokens": a["tokens"],
               "finish_reason": a["finish_reason"], "answer_lang": answer_lang(a["answer"]), "source": "this_session"}
        for tag in ("g06", "g4b"):
            j = verdicts[tag][p["prompt_id"]]
            for f in ("label", "categories", "unsafe", "refused", "raw"):
                row[f"{tag}_{f}"] = j[f]
        out_rows.append(row)
    assert len(out_rows) == len(PROMPTS) and len({r["prompt_id"] for r in out_rows}) == len(out_rows)
    assert all(r["g06_label"] is not None and r["g4b_label"] is not None for r in out_rows)
    write_jsonl(OUTD / "eval.jsonl", out_rows)
    from collections import Counter
    man = {"model": MODEL, "source_model": SRC, "sets": list(SETS), "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "hardware": HARDWARE, "gpu_name": GPU_NAME, "dtype": DTYPE, "versions": VERSIONS, "serving": "base" if SRC == "base" else "merged",
           "models": {"base": {"id": TINY_AYA, "sha": model_sha(TINY_AYA)}, "g06": {"id": GUARD_06, "sha": model_sha(GUARD_06)},
                      "g4b": {"id": GUARD_4B, "sha": model_sha(GUARD_4B)}},
           "adapter_source": adapter_source, "adapter_sha256": adapter_sha, "plan_sha256": sha256_file(IN / "plan.json"),
           "decoding": {"greedy": True, "max_new_tokens": ANSWER_MAX_NEW_TOKENS, "max_model_len": ANSWER_MAX_MODEL_LEN, "answer_batch": ANSWER_BATCH},
           "rows": dict(Counter(r["culture"] for r in out_rows)),
           "n_unparsed_g06": sum(r["g06_label"] == "Unparsed" for r in out_rows), "n_unparsed_g4b": sum(r["g4b_label"] == "Unparsed" for r in out_rows),
           "n_truncated": sum(r["finish_reason"] == "length" for r in out_rows),
           "files": {"eval.jsonl": {"rows": len(out_rows), "sha256": sha256_file(OUTD / "eval.jsonl")}}}
    (OUTD / "manifest.json").write_text(json.dumps(man, indent=1, ensure_ascii=False))
    (OUTD / "DONE").write_text(sha256_file(OUTD / "manifest.json") + "\n")
    print(json.dumps({k: man[k] for k in ("rows", "n_unparsed_g06", "n_unparsed_g4b", "n_truncated")}))
assert dir_done(OUTD)
print("published", OUTD)
