"""Builds XD_causal.ipynb (x_steer D, Colab GPU) and XD_analysis.ipynb (local). Run: python build_xd.py
Helper cells (HF token, shared helpers, environment, torchao fix, free_llm) are copied verbatim from
steering/notebooks/S2_directions.ipynb; the hook code follows steering/notebooks/S4_hfsteer.ipynb."""
import json
import nbformat as nbf
from pathlib import Path

HERE = Path(__file__).parent
S2 = json.loads((HERE.parent.parent / "steering" / "notebooks" / "S2_directions.ipynb").read_text())
copied = ["".join(S2["cells"][i]["source"]) for i in (2, 3, 4, 5, 6)]
assert "HF_TOKEN" in copied[0] and "def judge(" in copied[1] and "HARDWARE" in copied[2] and "torchao" in copied[3] and "def free_llm" in copied[4]

# ---------------------------------------------------------------- XD_causal (GPU)
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))
md("# XD_causal (x_steer D, Colab GPU): single-layer hooks with f_en⊥hg on merged en_s0\n"
   "HF generate, greedy, max_new_tokens 256 in every arm. Directions fitted on fold 0 (A); prompts are fold-1 seeds only.")
code('WORK = "/content/xd"\nLAYERS_D = [8, 18]\nBS = 64\nMAX_NEW = 256')
for c in copied:
    code(c)
code(r'''IN = Path(WORK) / "inputs"
OUT = Path(WORK) / "out"
OUT.mkdir(parents=True, exist_ok=True)
TORCH_DTYPE = torch.bfloat16 if DTYPE == "bfloat16" else torch.float16
manA = json.loads((IN / "A_manifest.json").read_text())
assert sha256_file(IN / "directions_bal.npz") == manA["files"]["directions_bal.npz"]["sha256"], "directions_bal.npz hash mismatch"
assert manA["L_A"] in LAYERS_D, (manA["L_A"], LAYERS_D)
plan = json.loads((IN / "plan.json").read_text())
assert (IN / "plan.DONE").read_text().strip() == sha256_file(IN / "plan.json"), "plan.DONE mismatch"
Z = np.load(IN / "directions_bal.npz")


def fold(seed_id):
    return int(hashlib.sha256(seed_id.encode()).hexdigest(), 16) % 2

P = [p for p in plan["sets"]["extraction"] if fold(p["seed_id"]) == 1]
by_pid = {p["prompt_id"]: p for p in P}
EN_T = [p for p in P if p["kind"] == "twin" and p["lang"] == "en"]
JA_T = [p for p in P if p["kind"] == "twin" and p["lang"] == "ja"]
EN_H = [p for p in P if p["kind"] == "harmful" and p["severity"] == "clear" and p["lang"] == "en"]
JA_H = [p for p in P if p["kind"] == "harmful" and p["severity"] == "clear" and p["lang"] == "ja"]
print("fold-1 sets:", {k: len(v) for k, v in dict(EN_T=EN_T, JA_T=JA_T, EN_H=EN_H, JA_H=JA_H).items()})

DIR, KAPPA, RND = {}, {}, {}
for L in LAYERS_D:
    d = Z[f"f_en_perp_hg_en_s0_L{L}_fold0"].astype(np.float64)
    DIR[L] = d / np.linalg.norm(d)
    KAPPA[L] = float(Z[f"kappa_en_en_s0_L{L}_fold0"])
    basis = [Z[f"{k}_en_s0_L{L}_fold0"].astype(np.float64) for k in ("h", "h_en", "g")]
    r = np.random.default_rng(0).standard_normal(DIR[L].shape[0])
    Q = []
    for b in basis:
        for q in Q:
            b = b - (b @ q) * q
        Q.append(b / np.linalg.norm(b))
    for q in Q:
        r = r - (r @ q) * q
    RND[L] = r / np.linalg.norm(r)
    print(f"L{L}: kappa={KAPPA[L]:.3f}  cos(d, rnd)={DIR[L] @ RND[L]:+.4f}  max|cos(rnd, h/h_en/g)|={max(abs(RND[L] @ q) for q in Q):.2e}")''')
code(r'''def ensure_merged():
    """Merged bf16 en_s0 (arm_en/adapters/seed0, merge_and_unload), saved once on the VM disk (S2 ensure_merged)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    mp = Path(WORK) / "merged" / "en_s0"
    if not (mp / "config.json").exists():
        b = AutoModelForCausalLM.from_pretrained(TINY_AYA, dtype=TORCH_DTYPE, device_map="cpu")
        PeftModel.from_pretrained(b, str(IN / "adapters" / "seed0")).merge_and_unload().save_pretrained(mp)
        AutoTokenizer.from_pretrained(TINY_AYA).save_pretrained(mp)
        del b
        gc.collect()
    return str(mp)


def chat_ids(tok, text):
    return _ids(tok.apply_chat_template([{"role": "user", "content": text}], tokenize=True, add_generation_prompt=True, return_dict=False))


def left_pad(seqs, pad_id):
    n = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), n), pad_id, dtype=torch.long)
    att = torch.zeros((len(seqs), n), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, n - len(s):] = torch.tensor(s)
        att[i, n - len(s):] = 1
    return ids, att


def write_jsonl(path, rs):
    with open(path, "w", encoding="utf-8") as f:
        for r in rs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

MERGED = ensure_merged()
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained(MERGED)
mod = AutoModelForCausalLM.from_pretrained(MERGED, dtype=TORCH_DTYPE).to("cuda").eval()
print("model loaded", mod.config.num_hidden_layers, "layers, hidden", mod.config.hidden_size)''')
code(r'''# Hooks (S4_hfsteer hook_factory pattern): modify decoder layer L's output at all positions, every forward step.
STATE = {"mode": None, "vec": None, "kappa": 0.0}


def hook(module, inp, out):
    if STATE["mode"] is None:
        return None
    hs = out[0] if isinstance(out, tuple) else out
    v = STATE["vec"]
    if STATE["mode"] == "remove":
        h32 = hs.float()
        hs = (h32 - (h32 @ v)[..., None] * v).to(hs.dtype)
    elif STATE["mode"] == "add":
        hs = hs + (STATE["kappa"] * v).to(hs.dtype)
    return (hs,) + tuple(out[1:]) if isinstance(out, tuple) else hs

PROBE = {}


def probe(module, inp, out):
    hs = out[0] if isinstance(out, tuple) else out
    PROBE["last"] = hs[:, -1, :].float()


@torch.no_grad()
def hook_check(L):
    """Projection on d at the last prompt token, without and with each hook (probe registered after the steering hook)."""
    texts = [p["prompt"] for p in EN_T[:32]]
    ids, att = left_pad([chat_ids(tok, t) for t in texts], 0)
    pos = (att.cumsum(-1) - 1).clamp(min=0)
    h1 = mod.model.layers[L].register_forward_hook(hook)
    h2 = mod.model.layers[L].register_forward_hook(probe)
    res = {}
    try:
        for mode in (None, "remove", "add"):
            STATE.update(mode=mode, vec=torch.tensor(DIR[L], dtype=torch.float32, device="cuda"), kappa=KAPPA[L])
            mod(input_ids=ids.cuda(), attention_mask=att.cuda(), position_ids=pos.cuda(), use_cache=False, logits_to_keep=1)
            res[str(mode)] = float((PROBE["last"] @ STATE["vec"]).mean())
    finally:
        STATE["mode"] = None
        h1.remove(); h2.remove()
    print(f"hook check L{L}: mean proj none={res['None']:.3f} remove={res['remove']:.4f} add={res['add']:.3f} (expected none+kappa={res['None'] + KAPPA[L]:.3f})")
    assert abs(res["remove"]) < 0.05 * max(1.0, abs(res["None"])), "remove hook did not remove the projection"
    assert abs(res["add"] - res["None"] - KAPPA[L]) < 0.05 * max(1.0, KAPPA[L]), "add hook did not add kappa"
    return res

HOOK_CHECK = {L: hook_check(L) for L in LAYERS_D}''')
code(r'''ANS = OUT / "answers_ckpt.jsonl"
ans = {(r["arm"], r["prompt_id"]): r for r in load_jsonl(ANS)}
print("checkpoint:", len(ans), "answers")


@torch.no_grad()
def gen(arm, prompts, L=None, mode=None, vec=None, kappa=0.0):
    todo = [p for p in prompts if (arm, p["prompt_id"]) not in ans]
    if not todo:
        return
    hd = mod.model.layers[L].register_forward_hook(hook) if L is not None else None
    eos = mod.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    seqs = {p["prompt_id"]: chat_ids(tok, p["prompt"]) for p in todo}
    order = sorted(todo, key=lambda p: len(seqs[p["prompt_id"]]))
    t0 = time.time()
    try:
        STATE.update(mode=mode, vec=(torch.tensor(vec, dtype=torch.float32, device="cuda") if vec is not None else None), kappa=kappa)
        for i in range(0, len(order), BS):
            ch = order[i:i + BS]
            ids, att = left_pad([seqs[p["prompt_id"]] for p in ch], 0)
            out = mod.generate(input_ids=ids.cuda(), attention_mask=att.cuda(), max_new_tokens=MAX_NEW, do_sample=False, pad_token_id=0)
            rs = []
            for p, g in zip(ch, out[:, ids.shape[1]:].tolist()):
                cut = next((k for k, t in enumerate(g) if t in eos), None)
                toks = g[:cut] if cut is not None else [t for t in g if t != 0]
                rs.append({"arm": arm, "prompt_id": p["prompt_id"], "answer": tok.decode(toks, skip_special_tokens=True), "tokens": len(toks),
                           "finish_reason": "stop" if cut is not None else "length", "max_new_tokens": MAX_NEW})
            append_jsonl(ANS, rs)
            ans.update({(r["arm"], r["prompt_id"]): r for r in rs})
        print(f"  {arm}: {len(order)} prompts in {time.time() - t0:.0f}s")
    finally:
        STATE["mode"] = None
        if hd is not None:
            hd.remove()

ARMS = {"D0": EN_T + JA_T + EN_H + JA_H}
gen("D0", ARMS["D0"])
for L in LAYERS_D:
    ARMS[f"D1_L{L}"] = EN_T; gen(f"D1_L{L}", EN_T, L, "remove", DIR[L])
    ARMS[f"D2_L{L}"] = JA_T; gen(f"D2_L{L}", JA_T, L, "remove", DIR[L])
    ARMS[f"D4_L{L}"] = EN_H + JA_H; gen(f"D4_L{L}", EN_H + JA_H, L, "remove", DIR[L])
    ARMS[f"D5_L{L}"] = EN_T + EN_H; gen(f"D5_L{L}", EN_T + EN_H, L, "remove", RND[L])
print("D0, D1, D2, D4, D5 generated")''')
code(r'''JUD = OUT / "judge_ckpt.jsonl"


def judge_keys(keys):
    v = {r["key"]: r for r in load_jsonl(JUD)}
    todo = [k for k in keys if k not in v]
    print(f"judge: {len(v)} done, {len(todo)} to go")
    if todo:
        g = load_llm(GUARD_06, DTYPE, gpu_memory_utilization=0.30, max_model_len=8192)
        for i in range(0, len(todo), 800):
            ch = todo[i:i + 800]
            js = judge(g, [by_pid[k.split("#")[1]]["prompt"] for k in ch], [ans[tuple(k.split("#"))]["answer"] for k in ch])
            rs = [{"key": k, **j} for k, j in zip(ch, js)]
            append_jsonl(JUD, rs)
            v.update({r["key"]: r for r in rs})
        free_llm(g)
    return v

VER = judge_keys([f"{a}#{p['prompt_id']}" for a, ps in ARMS.items() for p in ps])
D0_ANSWERED = [p for p in EN_T if VER[f"D0#{p['prompt_id']}"]["refused"] == 0]
print("English twins answered at D0:", len(D0_ANSWERED), "of", len(EN_T))
for L in LAYERS_D:
    ARMS[f"D3_L{L}"] = D0_ANSWERED
    gen(f"D3_L{L}", D0_ANSWERED, L, "add", DIR[L], KAPPA[L])
VER = judge_keys([f"{a}#{p['prompt_id']}" for a, ps in ARMS.items() for p in ps])''')
code(r'''rows, jrows = [], []
for a, ps in ARMS.items():
    for p in ps:
        r = ans[(a, p["prompt_id"])]
        layer = int(a.split("_L")[1]) if "_L" in a else None
        rows.append({"arm": a, "arm_type": a.split("_")[0], "layer": layer, "prompt_id": p["prompt_id"],
                     **{f: p.get(f) for f in ("seed_id", "kind", "severity", "culture", "framing", "lang", "category")},
                     "answer": r["answer"], "tokens": r["tokens"], "finish_reason": r["finish_reason"], "max_new_tokens": r["max_new_tokens"],
                     "answer_lang": answer_lang(r["answer"])})
        j = VER[f"{a}#{p['prompt_id']}"]
        jrows.append({"arm": a, "prompt_id": p["prompt_id"], "judge": GUARD_06, **{f"g06_{k}": j[k] for k in ("label", "categories", "unsafe", "refused", "raw")}})
write_jsonl(OUT / "answers.jsonl", rows)
write_jsonl(OUT / "judge.jsonl", jrows)
assert all(r["max_new_tokens"] == 256 for r in rows)
gm = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "hardware": HARDWARE, "gpu_name": GPU_NAME, "dtype": DTYPE,
      "versions": VERSIONS, "model": "en_s0 merged (arm_en/adapters/seed0) bf16", "judge": GUARD_06, "judge_sha": model_sha(GUARD_06),
      "base_sha": model_sha(TINY_AYA), "decoding": "HF generate, greedy, max_new_tokens=256 in every arm, left padding, batch " + str(BS),
      "layers": LAYERS_D, "L_A": manA["L_A"], "kappa": KAPPA, "hook_check": HOOK_CHECK, "A_manifest_sha256": sha256_file(IN / "A_manifest.json"),
      "arms": {a: len(ps) for a, ps in ARMS.items()}, "n_answers": len(rows),
      "files": {f: {"bytes": (OUT / f).stat().st_size, "sha256": sha256_file(OUT / f), "rows": sum(1 for _ in open(OUT / f))} for f in ("answers.jsonl", "judge.jsonl")}}
(OUT / "gpu_manifest.json").write_text(json.dumps(gm, indent=1, default=str))
print(json.dumps({k: gm[k] for k in ("hardware", "arms", "n_answers")}, indent=1))''')
nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nb.cells[1].metadata["tags"] = ["parameters"]
nbf.write(nb, HERE / "XD_causal.ipynb")

# ---------------------------------------------------------------- XD_analysis (local)
cells = []
md("# XD_analysis (x_steer D, local CPU): rates, flips vs D0, exact McNemar, seed-cluster bootstrap")
code('PT3 = "~/Library/CloudStorage/GoogleDrive-test-ezpenta/My Drive/ez-penta/post_train_3"\nNBOOT = 2000')
code(r'''import os, json, hashlib, time, sys, platform
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import binomtest
PT3 = Path(os.path.expanduser(PT3))
D = PT3 / "x_steer" / "D"


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

gm = json.loads((D / "gpu_manifest.json").read_text())
for f, m in gm["files"].items():
    assert sha256_file(D / f) == m["sha256"], f"hash mismatch {f}"
A = pd.read_json(D / "answers.jsonl", lines=True, dtype=False)
J = pd.read_json(D / "judge.jsonl", lines=True, dtype=False)
df = A.merge(J, on=["arm", "prompt_id"], validate="1:1")
assert len(df) == len(A) == gm["n_answers"]
assert (df.max_new_tokens == 256).all()
df["refused"] = df.g06_refused
df["unsafe"] = (df.g06_label == "Unsafe").astype(int)
df["su"] = df.g06_label.isin(["Safe", "Unsafe"])
df["trunc"] = (df.finish_reason == "length").astype(int)
df["wrong_lang"] = (df.answer_lang != df.lang).astype(int)
print("hardware", gm["hardware"], "| arms", gm["arms"])
print("refused unparsed:", int(df.refused.isna().sum()), "| label counts", df.g06_label.value_counts().to_dict())''')
code(r'''base = df[df.arm == "D0"].set_index("prompt_id")
rng = np.random.default_rng(0)


def subset(arm, kind, lang=None, framings=None):
    x = df[(df.arm == arm) & (df.kind == kind)]
    if lang:
        x = x[x.lang == lang]
    if framings:
        x = x[x.framing.isin(framings)]
    return x


def paired(arm, kind, lang, metric, framings=None):
    x = subset(arm, kind, lang, framings).set_index("prompt_id")
    b = base.loc[x.index]
    if metric == "unsafe_su":
        keep = x.su & b.su
        x, b = x[keep], b[keep]
        mx, mb = x.unsafe, b.unsafe
    else:
        ok = x[metric].notna() & b[metric].notna()
        x, b = x[ok], b[ok]
        mx, mb = x[metric].astype(int), b[metric].astype(int)
    up = int(((mx == 1) & (mb == 0)).sum())
    down = int(((mx == 0) & (mb == 1)).sum())
    p = float(binomtest(min(up, down), up + down, 0.5).pvalue) if up + down else 1.0
    seeds = x.seed_id.values
    us = np.unique(seeds)
    gi = {s: np.where(seeds == s)[0] for s in us}
    dlt = (mx.values - mb.values).astype(float)
    boots = []
    for _ in range(NBOOT):
        pick = rng.choice(us, len(us), replace=True)
        ii = np.concatenate([gi[s] for s in pick])
        boots.append(dlt[ii].mean())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"arm": arm, "kind": kind, "lang": lang, "metric": metric, "framings": ",".join(framings) if framings else "all", "n": int(len(x)),
            "rate_D0": float(mb.mean()), "rate_arm": float(mx.mean()), "delta": float(mx.mean() - mb.mean()), "ci95": [float(lo), float(hi)],
            "flips_0to1": up, "flips_1to0": down, "mcnemar_p": p,
            "trunc_arm": float(x.trunc.mean()), "trunc_D0": float(b.trunc.mean()), "wrong_lang_arm": float(x.wrong_lang.mean()), "wrong_lang_D0": float(b.wrong_lang.mean())}

R = []
for L in gm["layers"]:
    for fr in (None, ["control", "fc"], ["rp"]):
        R.append(dict(paired(f"D1_L{L}", "twin", "en", "refused", fr), layer=L))
        R.append(dict(paired(f"D2_L{L}", "twin", "ja", "refused", fr), layer=L))
        R.append(dict(paired(f"D3_L{L}", "twin", "en", "refused", fr), layer=L))
        R.append(dict(paired(f"D5_L{L}", "twin", "en", "refused", fr), layer=L))
    for lg in ("en", "ja"):
        R.append(dict(paired(f"D4_L{L}", "harmful", lg, "unsafe_su"), layer=L))
        R.append(dict(paired(f"D4_L{L}", "harmful", lg, "refused"), layer=L))
    R.append(dict(paired(f"D5_L{L}", "harmful", "en", "unsafe_su"), layer=L))
    R.append(dict(paired(f"D5_L{L}", "harmful", "en", "refused"), layer=L))
RT = pd.DataFrame(R)
pd.set_option("display.width", 250)
print(RT[["layer", "arm", "kind", "lang", "metric", "framings", "n", "rate_D0", "rate_arm", "delta", "ci95", "flips_0to1", "flips_1to0", "mcnemar_p", "trunc_arm", "wrong_lang_arm"]].round(4).to_string(index=False))''')
code(r'''def contrast(armA, armB, kind, lang, metric):
    """Seed-cluster bootstrap CI of (delta armA - delta armB) on the same prompts (both vs D0)."""
    xa = subset(armA, kind, lang).set_index("prompt_id")
    xb = subset(armB, kind, lang).set_index("prompt_id")
    idx = xa.index.intersection(xb.index)
    xa, xb, b = xa.loc[idx], xb.loc[idx], base.loc[idx]
    if metric == "unsafe_su":
        keep = xa.su & xb.su & b.su
        xa, xb, b = xa[keep], xb[keep], b[keep]
        va, vb = xa.unsafe.values, xb.unsafe.values
    else:
        va, vb = xa[metric].values.astype(float), xb[metric].values.astype(float)
    dd = va - vb
    seeds = xa.seed_id.values
    us = np.unique(seeds)
    gi = {s: np.where(seeds == s)[0] for s in us}
    boots = [dd[np.concatenate([gi[s] for s in rng.choice(us, len(us), replace=True)])].mean() for _ in range(NBOOT)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"contrast": f"{armA} - {armB}", "kind": kind, "lang": lang, "metric": metric, "n": int(len(dd)), "diff": float(dd.mean()), "ci95": [float(lo), float(hi)]}

C = []
for L in gm["layers"]:
    C.append(contrast(f"D1_L{L}", f"D5_L{L}", "twin", "en", "refused"))
    C.append(contrast(f"D4_L{L}", f"D5_L{L}", "harmful", "en", "unsafe_su"))
CT = pd.DataFrame(C)
print(CT.round(4).to_string(index=False))


def g(arm, kind, lang, metric, fr="all"):
    return RT[(RT.arm == arm) & (RT.kind == kind) & (RT.lang == lang) & (RT.metric == metric) & (RT.framings == fr)].iloc[0]


def c(name, metric):
    return CT[(CT.contrast == name) & (CT.metric == metric)].iloc[0]

CRIT = {}
for L in gm["layers"]:
    d1, d2, d3 = g(f"D1_L{L}", "twin", "en", "refused"), g(f"D2_L{L}", "twin", "ja", "refused"), g(f"D3_L{L}", "twin", "en", "refused")
    c15, c45 = c(f"D1_L{L} - D5_L{L}", "refused"), c(f"D4_L{L} - D5_L{L}", "unsafe_su")
    d4ja = g(f"D4_L{L}", "harmful", "ja", "unsafe_su")
    CRIT[L] = {
        "D1_lowers_en_more_than_D5": bool(c15.ci95[1] < 0 and d1.delta < 0),
        "D3_raises_en": bool(d3.ci95[0] > 0),
        "D2_changes_ja_little": bool(abs(d2.delta) <= 0.03 and d2.mcnemar_p >= 0.05),
        "D4_no_unsafe_beyond_D5": bool(c45.ci95[0] <= 0 and d4ja.ci95[0] <= 0),
        "detail": {"D1_delta": d1.delta, "D1_ci": d1.ci95, "D1_minus_D5": c15["diff"], "D1_minus_D5_ci": c15.ci95, "D3_delta": d3.delta, "D3_ci": d3.ci95,
                   "D2_delta": d2.delta, "D2_p": d2.mcnemar_p, "D4_minus_D5_en_unsafe": c45["diff"], "D4_minus_D5_ci": c45.ci95,
                   "D4_ja_unsafe_delta": d4ja.delta, "D4_ja_ci": d4ja.ci95}}
print(json.dumps(CRIT, indent=1, default=float))''')
code(r'''res = {"hardware": gm["hardware"], "gpu_name": gm["gpu_name"], "decoding": gm["decoding"], "judge": gm["judge"], "layers": gm["layers"],
       "L_A": gm["L_A"], "kappa": gm["kappa"], "hook_check": gm["hook_check"], "arms": gm["arms"], "nboot": NBOOT,
       "criteria_H": CRIT, "rows": RT.to_dict(orient="records"), "contrasts": CT.to_dict(orient="records")}
(D / "results.json").write_text(json.dumps(res, indent=1, default=float))
print("results.json written")''')
md("## manifest.json then DONE (report.md / RESULTS.md written from this notebook's output before the final run)")
code(r'''FILES = ["answers.jsonl", "judge.jsonl", "gpu_manifest.json", "results.json", "report.md"]
for f in FILES:
    assert (D / f).exists() and (D / f).stat().st_size > 0, f"missing {f}"
man = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "experiment": "x_steer D", "agent": "agent1",
       "hardware": gm["hardware"], "gpu_name": gm["gpu_name"], "dtype": gm["dtype"], "gpu_versions": gm["versions"],
       "analysis_versions": {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__},
       "decoding": gm["decoding"], "max_new_tokens_all_arms": 256, "judge": gm["judge"], "arms": gm["arms"], "layers": gm["layers"], "L_A": gm["L_A"],
       "files": {f: {"bytes": (D / f).stat().st_size, "sha256": sha256_file(D / f),
                     "rows": (sum(1 for _ in open(D / f)) if f.endswith(".jsonl") else None)} for f in FILES}}
(D / "manifest.json").write_text(json.dumps(man, indent=1, default=str))
(D / "DONE").write_text(sha256_file(D / "manifest.json") + "\n")
print("DONE", (D / "DONE").read_text().strip())''')
nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "jc-venv", "display_name": "jc-venv", "language": "python"}
nb.cells[1].metadata["tags"] = ["parameters"]
nbf.write(nb, HERE / "XD_analysis.ipynb")
print("built XD_causal.ipynb and XD_analysis.ipynb")
