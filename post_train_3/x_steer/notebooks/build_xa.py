"""Builds XA_balanced_directions.ipynb (x_steer experiment A, local CPU). Run: python build_xa.py"""
import nbformat as nbf
from pathlib import Path

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md("# XA: language-balanced over-refusal directions on the saved layers (x_steer A, local CPU)\n"
   "Definitions follow `post_train_3/XSTEER_SHARED.md` §2; thresholds §3.")
code('''PT3 = "~/Library/CloudStorage/GoogleDrive-test-ezpenta/My Drive/ez-penta/post_train_3"''')
code(r'''import os, json, hashlib, time, platform, sys
from pathlib import Path
import numpy as np, pandas as pd, sklearn, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
PT3 = Path(os.path.expanduser(PT3))
S2 = PT3 / "steering" / "s2_directions"
S1 = PT3 / "steering" / "s1_labels"
OUT = PT3 / "x_steer" / "A"
(OUT / "figures").mkdir(parents=True, exist_ok=True)
MODELS = ["base", "en_s0"]
LAYERS = [8, 12, 16, 18, 20, 24, 28]
LANGS = ["en", "ja"]
FRAMINGS = ["control", "fc", "rp"]
AUC_MIN, COS_MAX, MIN_REF = 0.80, 0.30, 30


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def check_folder(d, only=None):
    man = json.loads((d / "manifest.json").read_text())
    assert (d / "DONE").read_text().strip() == sha256_file(d / "manifest.json"), f"DONE mismatch in {d}"
    for f, m in man["files"].items():
        if only is None or f in only:
            assert sha256_file(d / f) == m["sha256"], f"hash mismatch {d / f}"
    return man

man_s2 = check_folder(S2)
man_s1 = check_folder(S1)
print("inputs hash-checked:", len(man_s2["files"]), "s2 files;", len(man_s1["files"]), "s1 files")
INPUT_HASHES = {"s2_directions/manifest.json": sha256_file(S2 / "manifest.json"), "s1_labels/manifest.json": sha256_file(S1 / "manifest.json")}''')

code(r'''idx = pd.read_json(S2 / "acts" / "index.jsonl", lines=True, dtype=False)
lab = pd.read_json(S1 / "answers.jsonl", lines=True, dtype=False)
assert len(idx) == 3864 and idx.prompt_id.is_unique
REF = {}
for m in MODELS:
    r = lab[lab.model == m].set_index("prompt_id").g06_refused
    assert r.index.is_unique and len(r) == 3864, (m, len(r))
    REF[m] = idx.prompt_id.map(r).values.astype(float)
    assert not np.isnan(REF[m][(idx.kind == "twin").values]).any(), f"missing twin labels {m}"
FOLD = idx.seed_id.map(lambda s: int(hashlib.sha256(s.encode()).hexdigest(), 16) % 2).values
TWIN = (idx.kind == "twin").values
LANG = idx.lang.values
FRAM = idx.framing.values
print("rows", len(idx), "| twins", TWIN.sum(), "| fold sizes", np.bincount(FOLD))''')

md("## Counts: refused/answered twins per model × language × framing × culture")
code(r'''rows = []
for m in MODELS:
    t = idx[TWIN].assign(refused=REF[m][TWIN])
    for (lg, fr, cu), g in t.groupby(["lang", "framing", "culture"]):
        rows.append({"model": m, "lang": lg, "framing": fr, "culture": cu, "refused": int(g.refused.sum()), "answered": int((g.refused == 0).sum())})
COUNTS = pd.DataFrame(rows)
print(COUNTS.pivot_table(index=["lang", "framing", "culture"], columns="model", values=["refused", "answered"]).to_string())
tot = COUNTS.groupby(["model", "lang"])[["refused", "answered"]].sum()
print(tot)
LOW_N = {(m, lg): bool(tot.loc[(m, lg), "refused"] < MIN_REF) for m in MODELS for lg in LANGS}
print("low_n cells:", [k for k, v in LOW_N.items() if v])''')

md("## Directions (§2) — full data and per fold")
code(r'''def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v * np.nan


def orth(x, basis):
    """x minus its projection on span(basis) via Gram-Schmidt, renormalised."""
    Q = []
    for b in basis:
        b = b.copy()
        for q in Q:
            b = b - (b @ q) * q
        if np.linalg.norm(b) > 1e-8:
            Q.append(b / np.linalg.norm(b))
    y = x.copy()
    for q in Q:
        y = y - (y @ q) * q
    return unit(y)

KEY = ["seed_id", "culture", "framing", "lang"]
H_ = idx[(idx.kind == "harmful") & (idx.severity == "clear")].reset_index().set_index(KEY)["index"]
T_ = idx[idx.kind == "twin"].reset_index().set_index(KEY)["index"]
PAIR = H_.index.intersection(T_.index)
PH, PT = H_.loc[PAIR].values, T_.loc[PAIR].values
PLANG = np.array([k[3] for k in PAIR])
TK = idx[TWIN].reset_index()
en_t = TK[TK.lang == "en"].set_index(["seed_id", "culture", "framing"])["index"]
ja_t = TK[TK.lang == "ja"].set_index(["seed_id", "culture", "framing"])["index"]
GP = en_t.index.intersection(ja_t.index)
GE, GJ = en_t.loc[GP].values, ja_t.loc[GP].values
print("harm/twin pairs:", len(PAIR), {lg: int((PLANG == lg).sum()) for lg in LANGS}, "| en/ja twin pairs:", len(GP))


def fit(X, ref, rowmask):
    """All directions from rows in rowmask (bool over the 3,864 rows). Returns dict name -> unit vector, plus meta."""
    pm = rowmask[PH] & rowmask[PT]
    D, meta = {}, {}
    D["h"] = unit((X[PH[pm]] - X[PT[pm]]).mean(0))
    for lg in LANGS:
        q = pm & (PLANG == lg)
        D[f"h_{lg}"] = unit((X[PH[q]] - X[PT[q]]).mean(0))
    gm = rowmask[GE] & rowmask[GJ]
    D["g"] = unit((X[GE[gm]] - X[GJ[gm]]).mean(0))
    for lg in LANGS:
        num, wsum, used = 0, 0, []
        for s in FRAMINGS:
            base_m = rowmask & TWIN & (LANG == lg) & (FRAM == s)
            rf, an = base_m & (ref == 1), base_m & (ref == 0)
            if rf.sum() >= 5 and an.sum() >= 5:
                num = num + rf.sum() * (X[rf].mean(0) - X[an].mean(0))
                wsum += rf.sum()
                used.append(s)
        D[f"f_{lg}"] = unit(num / wsum) if wsum else np.full(X.shape[1], np.nan)
        meta[f"strata_{lg}"] = used
    D["f_bal"] = unit(unit(D["f_en"]) + unit(D["f_ja"]))
    rf, an = rowmask & TWIN & (ref == 1), rowmask & TWIN & (ref == 0)
    D["f_pooled_old"] = unit(X[rf].mean(0) - X[an].mean(0))
    HL = {"f_en": [D["h"], D["h_en"]], "f_ja": [D["h"], D["h_ja"]],
          "f_bal": [D["h"], D["h_en"], D["h_ja"]], "f_pooled_old": [D["h"], D["h_en"], D["h_ja"]]}
    for f, B in HL.items():
        D[f"{f}_perp_h"] = orth(D[f], B)
        D[f"{f}_perp_hg"] = orth(D[f], B + [D["g"]])
    # kappa for D3: mean projection of refused minus answered English twins on f_en_perp_hg (this fit's rows)
    for lg in LANGS:
        rf, an = rowmask & TWIN & (LANG == lg) & (ref == 1), rowmask & TWIN & (LANG == lg) & (ref == 0)
        meta[f"kappa_{lg}"] = float((X[rf] @ D["f_en_perp_hg"]).mean() - (X[an] @ D["f_en_perp_hg"]).mean()) if lg == "en" else \
            float((X[rf] @ D["f_ja_perp_hg"]).mean() - (X[an] @ D["f_ja_perp_hg"]).mean())
    return D, meta


def auc(y, s):
    y = np.asarray(y)
    if y.sum() == 0 or (1 - y).sum() == 0 or np.isnan(s).any():
        return np.nan
    return float(roc_auc_score(y, s))

SCOPES = {"en": lambda: TWIN & (LANG == "en"), "ja": lambda: TWIN & (LANG == "ja"), "pooled": lambda: TWIN}
AUC_DIRS = ["f_en", "f_ja", "f_bal", "f_pooled_old"]
AUC_DIRS = AUC_DIRS + [f"{d}_perp_h" for d in AUC_DIRS] + [f"{d}_perp_hg" for d in AUC_DIRS] + ["h", "h_en", "h_ja", "g"]
ALL = np.ones(len(idx), bool)''')

code(r'''DIRS_OUT, DIAG, FOLD0 = {}, [], {}
old = np.load(S2 / "directions.npz")
for m in MODELS:
    ref = REF[m]
    for L in LAYERS:
        X = np.load(S2 / "acts" / f"acts_{m}_L{L}.npy").astype(np.float32)
        assert X.shape[0] == 3864
        Dfull, meta = fit(X, ref, ALL)
        Dfold = [fit(X, ref, FOLD == k) for k in (0, 1)]
        for k, v in Dfull.items():
            DIRS_OUT[f"{k}_{m}_L{L}"] = v.astype(np.float32)
        for k, v in Dfold[0][0].items():
            DIRS_OUT[f"{k}_{m}_L{L}_fold0"] = v.astype(np.float32)
        for k, v in Dfold[1][0].items():
            DIRS_OUT[f"{k}_{m}_L{L}_fold1"] = v.astype(np.float32)
        DIRS_OUT[f"kappa_en_{m}_L{L}_fold0"] = np.float32(Dfold[0][1]["kappa_en"])
        c = lambda a, b: float(Dfull[a] @ Dfull[b])
        base = {"model": m, "layer": L,
                "cos_f_pooled_old_vs_steering_f": float(Dfull["f_pooled_old"] @ old[f"f_{m}_L{L}"]),
                "strata_en": ",".join(meta["strata_en"]), "strata_ja": ",".join(meta["strata_ja"]),
                "strata_en_fold0": ",".join(Dfold[0][1]["strata_en"]), "strata_en_fold1": ",".join(Dfold[1][1]["strata_en"]),
                "kappa_en_fold0": Dfold[0][1]["kappa_en"], "kappa_en_full": meta["kappa_en"],
                "cos_f_en_f_ja": c("f_en", "f_ja"), "cos_f_en_perp_hg_f_ja_perp_hg": c("f_en_perp_hg", "f_ja_perp_hg"),
                "cos_f_pooled_old_g": c("f_pooled_old", "g"), "cos_f_pooled_old_f_bal": c("f_pooled_old", "f_bal"),
                "cos_h_g": c("h", "g"), "cos_h_en_h_ja": c("h_en", "h_ja"),
                "low_n_en": LOW_N[(m, "en")], "low_n_ja": LOW_N[(m, "ja")]}
        for d in AUC_DIRS:
            row = dict(base, direction=d)
            v = Dfull[d]
            row["cos_h"] = float(v @ Dfull["h"])
            hl = "h_en" if d.startswith("f_en") else ("h_ja" if d.startswith("f_ja") else None)
            row["h_lang"] = hl
            row["cos_h_lang"] = float(v @ Dfull[hl]) if hl else float(max(abs(v @ Dfull["h_en"]), abs(v @ Dfull["h_ja"])) * np.sign(v @ Dfull["h_en"]))
            row["cos_h_en"] = float(v @ Dfull["h_en"])
            row["cos_h_ja"] = float(v @ Dfull["h_ja"])
            row["cos_g"] = float(v @ Dfull["g"])
            for sc, fm in SCOPES.items():
                msk = fm()
                row[f"auc_in_{sc}"] = auc(ref[msk], X[msk] @ v)
                ho = []
                for k in (0, 1):
                    te = msk & (FOLD == 1 - k)
                    ho.append(auc(ref[te], X[te] @ Dfold[k][0][d]))
                row[f"auc_ho_{sc}"] = float(np.mean(ho))
                row[f"auc_ho_{sc}_fold0fit"] = ho[0]
            DIAG.append(row)
        g = DIAG[-1]
        fe = [r for r in DIAG if r["model"] == m and r["layer"] == L and r["direction"] == "f_en_perp_hg"][0]
        fj = [r for r in DIAG if r["model"] == m and r["layer"] == L and r["direction"] == "f_ja_perp_hg"][0]
        gg = [r for r in DIAG if r["model"] == m and r["layer"] == L and r["direction"] == "g"][0]
        print(f"{m} L{L}: cos(f_old,g)={base['cos_f_pooled_old_g']:+.3f} cos(f_old,f_bal)={base['cos_f_pooled_old_f_bal']:+.3f} "
              f"| g-only AUC pooled ho={gg['auc_ho_pooled']:.3f} | AUC_ho f_en⊥hg(en)={fe['auc_ho_en']:.3f} f_ja⊥hg(ja)={fj['auc_ho_ja']:.3f} "
              f"| cos(f_old, steering f)={base['cos_f_pooled_old_vs_steering_f']:+.4f}")
DG = pd.DataFrame(DIAG)''')

md("## Bias check and §3 qualification")
code(r'''def get(m, L, d, col):
    return float(DG[(DG.model == m) & (DG.layer == L) & (DG.direction == d)][col].iloc[0])

BIAS, QUAL = [], []
for m in MODELS:
    for L in LAYERS:
        BIAS.append({"model": m, "layer": L,
                     "cos_f_pooled_old_g": get(m, L, "f_pooled_old", "cos_g"),
                     "cos_f_pooled_old_f_bal": get(m, L, "f_pooled_old", "cos_f_pooled_old_f_bal"),
                     "lang_only_auc_ho_pooled": get(m, L, "g", "auc_ho_pooled"),
                     "lang_only_auc_in_pooled": get(m, L, "g", "auc_in_pooled"),
                     "cos_f_pooled_old_h": get(m, L, "f_pooled_old", "cos_h"),
                     "cos_f_bal_h": get(m, L, "f_bal", "cos_h"),
                     "cos_f_en_h_en": get(m, L, "f_en", "cos_h_lang"), "cos_f_ja_h_ja": get(m, L, "f_ja", "cos_h_lang"),
                     "auc_ho_pooled_f_pooled_old": get(m, L, "f_pooled_old", "auc_ho_pooled"),
                     "auc_ho_pooled_f_bal_perp_hg": get(m, L, "f_bal_perp_hg", "auc_ho_pooled")})
        for lg in LANGS:
            a = get(m, L, f"f_{lg}_perp_hg", f"auc_ho_{lg}")
            cs = get(m, L, f"f_{lg}", "cos_h_lang")
            QUAL.append({"model": m, "lang": lg, "layer": L, "auc_ho": a, "auc_in": get(m, L, f"f_{lg}_perp_hg", f"auc_in_{lg}"),
                         "cos_f_lang_h_lang": cs, "cos_f_lang_h": get(m, L, f"f_{lg}", "cos_h"), "cos_f_lang_g": get(m, L, f"f_{lg}", "cos_g"),
                         "low_n": LOW_N[(m, lg)], "qualifies": bool(a >= AUC_MIN and abs(cs) <= COS_MAX and not LOW_N[(m, lg)]),
                         "score": a - abs(cs)})
BIAS, QUAL = pd.DataFrame(BIAS), pd.DataFrame(QUAL)
pd.set_option("display.width", 250)
print(BIAS.round(3).to_string(index=False))
print(QUAL.round(3).to_string(index=False))


def pick(m, lg):
    q = QUAL[(QUAL.model == m) & (QUAL.lang == lg)].sort_values("layer")
    ok = q[q.qualifies]
    if len(ok):
        best = ok[ok.auc_ho == ok.auc_ho.max()].iloc[0]
        return int(best.layer), "qualifies"
    best = q[q.score == q.score.max()].iloc[0]
    return int(best.layer), "fallback_max(auc_ho-|cos|)"

PICKS = {f"{m}|{lg}": dict(zip(("layer", "rule"), pick(m, lg))) for m in MODELS for lg in LANGS}
L_A, L_A_RULE = PICKS["en_s0|en"]["layer"], PICKS["en_s0|en"]["rule"]
print("picks:", PICKS)
print("L_A (en_s0, English) =", L_A, L_A_RULE)''')

md("## Figures")
code(r'''fig, ax = plt.subplots(1, 2, figsize=(11, 4))
sty = {("base", "en"): ("#1f77b4", "-"), ("base", "ja"): ("#1f77b4", "--"), ("en_s0", "en"): ("#d62728", "-"), ("en_s0", "ja"): ("#d62728", "--")}
for (m, lg), (col, ls) in sty.items():
    q = QUAL[(QUAL.model == m) & (QUAL.lang == lg)].sort_values("layer")
    lbl = f"{m} {lg}" + (" (low_n)" if LOW_N[(m, lg)] else "")
    ax[0].plot(q.layer, q.cos_f_lang_h_lang, ls, color=col, marker="o", label=lbl)
    ax[1].plot(q.layer, q.auc_ho, ls, color=col, marker="o", label=lbl)
ax[0].axhline(COS_MAX, color="gray", lw=0.8, ls=":"); ax[0].axhline(-COS_MAX, color="gray", lw=0.8, ls=":")
ax[1].axhline(AUC_MIN, color="gray", lw=0.8, ls=":")
ax[0].set_title("cos(f_lang, h_lang)"); ax[1].set_title("held-out within-language AUC of f_lang⊥hg")
for a in ax:
    a.set_xlabel("layer"); a.set_xticks(LAYERS); a.legend(fontsize=8); a.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "figures" / "cos_auc_vs_layer.png", dpi=130)
fig, a = plt.subplots(figsize=(5.5, 4))
for m, col in (("base", "#1f77b4"), ("en_s0", "#d62728")):
    b = BIAS[BIAS.model == m]
    a.plot(b.layer, b.cos_f_pooled_old_g, "-o", color=col, label=f"{m} cos(f_pooled_old, g)")
    a.plot(b.layer, b.lang_only_auc_ho_pooled, "--s", color=col, label=f"{m} language-only AUC (g)")
a.set_xticks(LAYERS); a.set_xlabel("layer"); a.grid(alpha=0.3); a.legend(fontsize=8); a.set_title("Bias of the old pooled direction")
fig.tight_layout(); fig.savefig(OUT / "figures" / "bias_vs_layer.png", dpi=130)
print("figures saved")''')

md("## Outputs")
code(r'''np.savez(OUT / "directions_bal.npz", **DIRS_OUT)
DG.to_csv(OUT / "diagnostics.csv", index=False)
COUNTS.to_csv(OUT / "counts.csv", index=False)
out = {"definitions": "XSTEER_SHARED.md §2; f_bal/f_pooled_old orthogonalised against span{h,h_en,h_ja}(+g)",
       "thresholds": {"auc_ho_min": AUC_MIN, "abs_cos_max": COS_MAX, "min_refused": MIN_REF},
       "n_pairs_harm_twin": int(len(PAIR)), "n_pairs_en_ja": int(len(GP)), "fold_sizes_rows": np.bincount(FOLD).tolist(),
       "low_n": {f"{m}|{lg}": v for (m, lg), v in LOW_N.items()},
       "picks": PICKS, "L_A": L_A, "L_A_rule": L_A_RULE,
       "bias": BIAS.to_dict(orient="records"), "qualification": QUAL.to_dict(orient="records"),
       "counts": COUNTS.to_dict(orient="records"), "diagnostics": DG.to_dict(orient="records")}
(OUT / "diagnostics.json").write_text(json.dumps(out, indent=1, default=float))
print("written", sorted(p.name for p in OUT.iterdir()))''')

md("## manifest.json then DONE (report.md is written from this notebook's printed numbers before the final run)")
code(r'''FILES = ["directions_bal.npz", "diagnostics.csv", "diagnostics.json", "counts.csv", "report.md",
         "figures/cos_auc_vs_layer.png", "figures/bias_vs_layer.png"]
for f in FILES:
    assert (OUT / f).exists() and (OUT / f).stat().st_size > 0, f"missing {f}"
man = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "experiment": "x_steer A", "agent": "agent1",
       "hardware": f"local CPU ({platform.machine()}, {platform.system()} {platform.release()})", "gpu": None,
       "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__},
       "inputs": INPUT_HASHES, "n_prompts": int(len(idx)), "n_twins": int(TWIN.sum()),
       "counts_refused_twins": {f"{m}|{lg}": int(tot.loc[(m, lg), "refused"]) for m in MODELS for lg in LANGS},
       "L_A": L_A, "L_A_rule": L_A_RULE, "any_cell_qualifies": bool(QUAL.qualifies.any()),
       "files": {f: {"bytes": (OUT / f).stat().st_size, "sha256": sha256_file(OUT / f),
                     "rows": (sum(1 for _ in open(OUT / f)) - 1 if f.endswith(".csv") else None)} for f in FILES}}
(OUT / "manifest.json").write_text(json.dumps(man, indent=1))
(OUT / "DONE").write_text(sha256_file(OUT / "manifest.json") + "\n")
print("DONE", (OUT / "DONE").read_text().strip())''')

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "jc-venv", "display_name": "jc-venv", "language": "python"}
code_tags = nb.cells[1].setdefault("metadata", {}).setdefault("tags", [])
code_tags.append("parameters")
nbf.write(nb, Path(__file__).parent / "XA_balanced_directions.ipynb")
print("built")
