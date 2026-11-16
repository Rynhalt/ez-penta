# %% [markdown]
# # B3_reuse (Colab GPU): reuse checks for the stored base and `en` rows (AGENT_PROMPT_BILINGUAL 4.2)
# For every model × set (US: base, en_s0, en_s1, en_s2; Japan: base, en_s0): session match (hardware, dtype, vLLM, transformers) and
# re-answer the stored prompts with `int(prompt_id, 16) % 10 == 0`, judge with the 0.6B judge, agreement ≥ 95% on `unsafe` and `refused`.
# Writes `eval/reuse/` (answers, labels) and `eval/reuse_checks.json`. No condition comparison is computed here.

# %% tags=["parameters"]
WORK = "/content/pt3bi"
MERGED_ROOT = "/content/merged"
ANSWER_BATCH = 400

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
IN = Path(WORK) / "inputs"
assert (IN / "plan.DONE").read_text().strip() == sha256_file(IN / "plan.json")
plan = json.loads((IN / "plan.json").read_text())
for f in ("eval_prompts.jsonl", "reuse_targets.jsonl"):
    assert sha256_file(IN / f) == plan["inputs"][f]["sha256"], f
by_pid = {p["prompt_id"]: p for p in load_jsonl(IN / "eval_prompts.jsonl")}
targets = load_jsonl(IN / "reuse_targets.jsonl")
OUTD = Path(WORK) / "out" / "eval" / "reuse"
OUTD.mkdir(parents=True, exist_ok=True)
ANS = OUTD / "answers.jsonl"
JUD = OUTD / "judge_g06.jsonl"
ans = {(r["model"], r["prompt_id"]): r for r in load_jsonl(ANS)}

# Session match against every stored manifest of each set.
def match(m):
    return {"hardware": m["hardware"] == HARDWARE, "dtype": m["dtype"] == DTYPE, "vllm": m["versions"]["vllm"] == VERSIONS["vllm"],
            "transformers": m["versions"]["transformers"] == VERSIONS["transformers"]}


session = {}
for s, mans in plan["stored_manifests"].items():
    ms = [match(m) for m in mans.values()]
    session[s] = {k: all(x[k] for x in ms) for k in ms[0]}
print("session:", HARDWARE, DTYPE, VERSIONS["vllm"], VERSIONS["transformers"], session)

# %%
# Re-answer the sample with each model (merged bf16 for adapters, as in B2), one vLLM model at a time.
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
TORCH_DTYPE = torch.bfloat16 if DTYPE == "bfloat16" else torch.float16
for m in ("base", "en_s0", "en_s1", "en_s2"):
    todo = sorted({t["prompt_id"] for t in targets if t["model"] == m and (m, t["prompt_id"]) not in ans})
    print(m, "to answer:", len(todo))
    if not todo:
        continue
    if m == "base":
        MPATH = TINY_AYA
    else:
        ADIR = IN / "en_adapters" / f"seed{int(m[-1])}"
        exp = plan["hashes"]["en_adapters"][f"seed{int(m[-1])}"]
        assert all(sha256_file(ADIR / f) == exp[f] for f in ("adapter_config.json", "adapter_model.safetensors"))
        mp = Path(MERGED_ROOT) / m
        if not (mp / "config.json").exists():
            b = AutoModelForCausalLM.from_pretrained(TINY_AYA, dtype=TORCH_DTYPE, device_map="cpu")
            PeftModel.from_pretrained(b, str(ADIR)).merge_and_unload().save_pretrained(mp)
            AutoTokenizer.from_pretrained(TINY_AYA).save_pretrained(mp)
            del b
            gc.collect()
        MPATH = str(mp)
    llm = load_llm(MPATH, DTYPE, gpu_memory_utilization=0.40)
    for i in range(0, len(todo), ANSWER_BATCH):
        ch = todo[i:i + ANSWER_BATCH]
        outs = answer(llm, [by_pid[k]["prompt"] for k in ch])
        rs = [{"model": m, "prompt_id": k, **o} for k, o in zip(ch, outs)]
        append_jsonl(ANS, rs)
        ans.update({(r["model"], r["prompt_id"]): r for r in rs})
        print(f"  {m}: {i + len(ch)}/{len(todo)}")
    free_llm(llm)

# %%
# Judge with the 0.6B judge and compare with the stored labels.
v = {(r["model"], r["prompt_id"]): r for r in load_jsonl(JUD)}
keys = sorted({(t["model"], t["prompt_id"]) for t in targets})
todo = [k for k in keys if k not in v]
if todo:
    g = load_llm(GUARD_06, DTYPE, gpu_memory_utilization=0.15, max_model_len=8192)
    for i in range(0, len(todo), 800):
        ch = todo[i:i + 800]
        js = judge(g, [by_pid[k[1]]["prompt"] for k in ch], [ans[k]["answer"] for k in ch])
        rs = [{"model": k[0], "prompt_id": k[1], **j} for k, j in zip(ch, js)]
        append_jsonl(JUD, rs)
        v.update({(r["model"], r["prompt_id"]): r for r in rs})
    free_llm(g)
res = {}
for m, s in sorted({(t["model"], t["set"]) for t in targets}):
    ts = [t for t in targets if t["model"] == m and t["set"] == s]
    au = float(np.mean([v[(m, t["prompt_id"])]["unsafe"] == t["g06_unsafe"] for t in ts]))
    ar = float(np.mean([v[(m, t["prompt_id"])]["refused"] == t["g06_refused"] for t in ts]))
    same_text_n = None
    sm = session[s]
    passed = bool(all(sm.values()) and au >= 0.95 and ar >= 0.95)
    res[f"{m}|{s}"] = {"model": m, "set": s, "n": len(ts), "agree_unsafe": au, "agree_refused": ar, "session_match": sm, "passed": passed,
                       "decision": "reuse stored rows" if passed else "re-answer and re-judge all prompts of this model x set in this session (B2)"}
    print(f"{m:6s} {s:6s} n={len(ts)} agree_unsafe={au:.3f} agree_refused={ar:.3f} session={all(sm.values())} -> {'PASS' if passed else 'FAIL'}")

# %%
def write_json(p, o):
    Path(p).write_text(json.dumps(o, indent=1, ensure_ascii=False))


rc = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "hardware": HARDWARE, "gpu_name": GPU_NAME, "dtype": DTYPE, "versions": VERSIONS,
      "rule": "session (G4, bf16, vLLM, transformers) matches every stored manifest AND 0.6B-judge agreement >= 0.95 on unsafe and refused over prompts with int(prompt_id,16) % 10 == 0",
      "checks": res}
write_json(OUTD.parent / "reuse_checks.json", rc)
files = ["answers.jsonl", "judge_g06.jsonl"]
man = {"created_utc": rc["created_utc"], "hardware": HARDWARE, "dtype": DTYPE, "versions": VERSIONS, "checks": res,
       "files": {f: {"rows": sum(1 for _ in open(OUTD / f, encoding="utf-8")), "sha256": sha256_file(OUTD / f)} for f in files}}
write_json(OUTD / "manifest.json", man)
(OUTD / "DONE").write_text(sha256_file(OUTD / "manifest.json") + "\n")
print(json.dumps({k: (r["passed"], round(r["agree_unsafe"], 3), round(r["agree_refused"], 3)) for k, r in res.items()}))
