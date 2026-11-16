# Builds post_train_3/article/notebooks/A1_figures.ipynb (writes the .ipynb only; all data work runs inside the notebook).
from pathlib import Path
import nbformat as nbf

OUT = Path("/Users/isaac/App_Files/Code/ez-penta/post_train_3/article/notebooks")
OUT.mkdir(parents=True, exist_ok=True)

C = []
C.append(("md", "# A1: figures and numbers for the v2 article (local CPU)\n"
          "Reads only saved, hash-published outputs (bilingual, B5 decomposition logic, follow-ups, steering, x_steer). "
          "Writes `article/figures/*.png|svg` and `article/data/numbers.json` (every number the article cites from these sources)."))
C.append(("code", 'SAVE = True\n', ["parameters"]))
C.append(("code", r'''import glob, json, os, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path("/Users/isaac/App_Files/Code/ez-penta")
ART = REPO / "post_train_3" / "article"
FIG, DATA = ART / "figures", ART / "data"
FIG.mkdir(parents=True, exist_ok=True); DATA.mkdir(parents=True, exist_ok=True)
PT3 = Path(glob.glob(os.path.expanduser("~/Library/CloudStorage/GoogleDrive-*"))[0]) / "My Drive" / "ez-penta" / "post_train_3"
BI = json.loads((PT3 / "bilingual" / "results.json").read_text())
FU = json.loads((PT3 / "followups" / "results.json").read_text())
ST = json.loads((PT3 / "steering" / "results.json").read_text())
XC = json.loads((PT3 / "x_steer" / "C" / "results.json").read_text())
XD = json.loads((PT3 / "x_steer" / "D" / "results.json").read_text())
assert BI["final"] is True
N = {}   # every cited number goes here

# Style: reference palette (validated); Japanese-only is the highlighted series.
COL = {"base": "#52514e", "en": "#2a78d6", "ja": "#eb6834", "bi": "#1baf7a", "en2": "#a8a7a2"}
NAME = {"base": "Base", "en": "English-only", "ja": "Japanese-only", "bi": "Bilingual", "en2": "English ×2"}
MARK = {"base": "s", "en": "o", "ja": "o", "bi": "o", "en2": "D"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#8a8984",
                     "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.grid": True,
                     "grid.color": "#e6e5e1", "grid.linewidth": 0.6, "figure.dpi": 150, "savefig.bbox": "tight",
                     "font.family": "DejaVu Sans"})
def save(fig, name):
    if SAVE:
        fig.savefig(FIG / f"{name}.png", dpi=200); fig.savefig(FIG / f"{name}.svg")
    plt.close(fig)
'''))
C.append(("md", "## Language-pooled rates per training run (both cultural framings pooled; 0.6B judge)"))
C.append(("code", r'''def pooled(metric):
    """{cond: {'runs': {model: [k, n]}, 'k': K, 'n': N}} per prompt language, summing US and Japan framings."""
    out = {"en": {}, "ja": {}}
    for row in BI["headline_g06"][metric]:
        lang = row["lang"]
        for cond, d in row["raw"].items():
            o = out[lang].setdefault(cond, {"runs": {}})
            for m, (k, n) in d["per_seed"].items():
                r = o["runs"].setdefault(m, [0, 0]); r[0] += k; r[1] += n
    for lang in out:
        for cond, o in out[lang].items():
            o["k"] = sum(r[0] for r in o["runs"].values()); o["n"] = sum(r[1] for r in o["runs"].values())
            o["rate"] = o["k"] / o["n"]; o["run_rates"] = [k / n for k, n in o["runs"].values()]
    return out
SAF, OVR = pooled("safety"), pooled("over_refusal")
tab = []
for lang in ("en", "ja"):
    for c in ("base", "en", "ja", "bi", "en2"):
        s, o = SAF[lang][c], OVR[lang][c]
        tab.append({"prompt_lang": lang, "condition": c, "unsafe": f"{s['k']}/{s['n']}", "unsafe_pct": round(100 * s["rate"], 2),
                    "over_refusal": f"{o['k']}/{o['n']}", "over_refusal_pct": round(100 * o["rate"], 2)})
T1 = pd.DataFrame(tab); print(T1.to_string(index=False))
# all prompts pooled
allp = {}
for c in ("base", "en", "ja", "bi", "en2"):
    ks = SAF["en"][c]["k"] + SAF["ja"][c]["k"]; ns = SAF["en"][c]["n"] + SAF["ja"][c]["n"]
    ko = OVR["en"][c]["k"] + OVR["ja"][c]["k"]; no = OVR["en"][c]["n"] + OVR["ja"][c]["n"]
    allp[c] = {"unsafe": [ks, ns], "unsafe_pct": round(100 * ks / ns, 2), "over_refusal": [ko, no], "over_refusal_pct": round(100 * ko / no, 2)}
print(json.dumps(allp, indent=0))
N["pooled_by_language"] = tab; N["pooled_all"] = allp
'''))
C.append(("md", "## Figure 1: Unsafe rate vs over-refusal, by prompt language"))
C.append(("code", r'''fig, axes = plt.subplots(1, 2, figsize=(10, 4.3), sharey=True)
for ax, lang, title in zip(axes, ("en", "ja"), ("English prompts", "Japanese prompts")):
    for c in ("base", "en2", "en", "bi", "ja"):
        s, o = SAF[lang][c], OVR[lang][c]
        xs = [100 * r for r in s["run_rates"]]; ys = [100 * r for r in o["run_rates"]]
        if c != "base":
            ax.scatter(xs, ys, s=14, color=COL[c], alpha=0.45, edgecolor="none", zorder=2)
        hollow = c == "en2"
        ax.scatter([100 * s["rate"]], [100 * o["rate"]], s=95 if c == "ja" else 70, marker=MARK[c], zorder=4,
                   facecolor="white" if hollow else COL[c], edgecolor=COL[c] if hollow else "white", linewidth=1.6)
        OFFS = {"en": {"base": (0.35, 1.0), "en": (0.35, 1.0), "ja": (0.35, -2.8), "bi": (0.35, 1.0), "en2": (0.35, 1.0)},
                "ja": {"base": (-2.6, -2.6), "en": (-4.6, 1.6), "ja": (-4.9, 1.6), "bi": (0.4, -2.8), "en2": (0.35, 1.6)}}
        dx, dy = OFFS[lang][c]
        ax.annotate(NAME[c], (100 * s["rate"] + dx, 100 * o["rate"] + dy), fontsize=9,
                    color="#0b0b0b", fontweight="bold" if c == "ja" else "normal")
    ax.set_title(title, fontsize=11, loc="left"); ax.set_xlabel("Unsafe answers to harmful prompts (%)  ← safer")
    ax.set_xlim(0, 19); ax.set_ylim(0, 40)
axes[0].set_ylabel("Refused harmless prompts (%)  ← more helpful")
fig.text(0.0, -0.04, "Large markers: three training runs pooled (base: one run). Small dots: individual training runs. "
         "US- and Japan-framed prompts pooled. Judge: Qwen3Guard-Gen-0.6B.", fontsize=8, color="#52514e")
save(fig, "fig1_safety_vs_overrefusal")
'''))
C.append(("md", "## Figure 2: paired change from the base model, by training language and prompt language"))
C.append(("code", r'''def sec(key):
    v = BI["secondary"][key]
    return {"estimate_pp": round(100 * v["estimate"], 2), "ci95_pp": [round(100 * v["ci95"][0], 2), round(100 * v["ci95"][1], 2)],
            "p_boot": v.get("p_boot"), "n": v["n"], "n_seeds": v["n_seeds"], "rate_A_pct": round(100 * v["rate_A"], 2),
            "rate_B_pct": round(100 * v["rate_B"], 2)}
VS_BASE = {f"{m}|{c}|{l}": sec(f"{m}|{c}-base|{l}") for m in ("safety", "over_refusal") for c in ("en", "ja", "bi", "en2") for l in ("en", "ja")}
N["vs_base"] = VS_BASE
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
conds = ["en", "ja", "bi", "en2"]
for ax, m, title, xl in zip(axes, ("safety", "over_refusal"), ("Unsafe answers (change from base)", "Refused harmless prompts (change from base)"),
                            ("percentage points  ← safer", "percentage points  ← fewer refusals")):
    for i, c in enumerate(conds):
        for j, (l, mk) in enumerate((("en", "o"), ("ja", "s"))):
            v = VS_BASE[f"{m}|{c}|{l}"]; y = i + (-0.17 if l == "en" else 0.17)
            ax.errorbar(v["estimate_pp"], y, xerr=[[v["estimate_pp"] - v["ci95_pp"][0]], [v["ci95_pp"][1] - v["estimate_pp"]]],
                        fmt=mk, ms=7, color=COL[c], mfc=COL[c] if l == "en" else "white", mew=1.6, elinewidth=1.4, capsize=0, zorder=3)
    ax.axvline(0, color="#8a8984", lw=1); ax.set_title(title, fontsize=11, loc="left"); ax.set_xlabel(xl)
axes[0].set_yticks(range(4)); axes[0].set_yticklabels([NAME[c] for c in conds]); axes[0].invert_yaxis()
from matplotlib.lines import Line2D
fig.legend(handles=[Line2D([], [], marker="o", ls="", color="#52514e", ms=7, label="English prompts"),
                    Line2D([], [], marker="s", ls="", mfc="white", color="#52514e", mew=1.6, ms=7, label="Japanese prompts")],
           loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.12))
fig.text(0.0, -0.17, "Paired estimates on identical prompts (475 seeds; US and Japan framings pooled); bars are two-level bootstrap 95% CIs "
         "(evaluation seeds × three training runs). Judge: Qwen3Guard-Gen-0.6B.", fontsize=8, color="#52514e")
save(fig, "fig2_change_from_base")
'''))
C.append(("md", "## Main contrasts (primary hypotheses and Japanese-only secondary contrasts)"))
C.append(("code", r'''def pri(key, j="g06"):
    v = BI[f"primary_{j}"][key]
    return {"estimate_pp": round(100 * v["estimate"], 2), "ci95_pp": [round(100 * v["ci95"][0], 2), round(100 * v["ci95"][1], 2)],
            "p_holm": v.get("p_holm"), "rate_A_pct": round(100 * v["rate_A"], 2), "rate_B_pct": round(100 * v["rate_B"], 2)}
N["primary"] = {k: {"g06": pri(k), "g4b": pri(k, "g4b")} for k in BI["primary_g06"] if not k.startswith("H4")}
N["ja_contrasts"] = {k: sec(k) for k in ["safety|ja-en|en", "over_refusal|ja-en|en", "fix_rate|ja-en|en", "safety|ja-en|ja", "over_refusal|ja-en|ja",
                                         "fix_rate|ja-en|ja", "safety|bi-ja|en", "over_refusal|bi-ja|en", "safety|bi-ja|ja", "over_refusal|bi-ja|ja",
                                         "fix_rate|bi-ja|ja", "safety|en2-en|ja", "fix_rate|bi-en|ja"]}
print(json.dumps(N["primary"], indent=0)); print(json.dumps(N["ja_contrasts"], indent=0))
'''))
C.append(("md", "## Figure 3: refusal decomposition (same computation as `bilingual/notebooks/B5_refusal_decomposition.ipynb`)"))
C.append(("code", r'''# Load answers exactly as B5 does (it executes the head of B4_analysis), then recompute the two-factor Shapley split.
cwd = os.getcwd(); os.chdir(REPO / "post_train_3" / "bilingual" / "notebooks")
src = Path("nbsrc/B4_analysis.py").read_text(); head = src[:src.index("MODELS = sorted(ROWS)")].replace("N_BOOT = 10000", "N_BOOT = 0")
g = {}; exec(compile(head, "B4_analysis_head", "exec"), g); os.chdir(cwd)
E, ROWS = g["E"], g["ROWS"]
H = E[E.kind == "harmful"].reset_index(drop=True)
def lab(m):
    d = ROWS[m]
    u = np.array([{"Unsafe": 1.0, "Safe": 0.0}.get(d[p]["g06_label"], np.nan) if p in d else np.nan for p in H.prompt_id])
    r = np.array([np.nan if (p not in d or d[p]["g06_refused"] is None) else float(d[p]["g06_refused"]) for p in H.prompt_id])
    return u, r
def stats(m, mask):
    u, r = lab(m); k = mask & ~np.isnan(u) & ~np.isnan(r); u, r = u[k], r[k]
    return {"n": int(k.sum()), "unsafe": u.mean(), "not_refused": (r == 0).mean(), "ucond": u[r == 0].mean(),
            "unsafe_refused": int(((u == 1) & (r == 1)).sum()), "k_unsafe_answered": int(((u == 1) & (r == 0)).sum()), "n_answered": int((r == 0).sum())}
GROUPS = {"en": ["en_s0", "en_s1", "en_s2"], "ja": ["ja_s0", "ja_s1", "ja_s2"], "bi": ["bi_s0", "bi_s1", "bi_s2"], "en2": ["en2_s0", "en2_s1", "en2_s2"]}
DEC = {}
for lang, mask in (("en", (H.lang == "en").values), ("ja", (H.lang == "ja").values), ("all", np.ones(len(H), bool))):
    b = stats("base", mask)
    DEC[f"base|{lang}"] = {"unsafe_pct": round(100 * b["unsafe"], 2), "refused_pct": round(100 * (1 - b["not_refused"]), 2),
                           "unsafe_given_answered_pct": round(100 * b["ucond"], 2), "answered_unsafe": f"{b['k_unsafe_answered']}/{b['n_answered']}"}
    for c, ms in GROUPS.items():
        per = []
        for m in ms:
            s = stats(m, mask)
            dA, dB = s["not_refused"] - b["not_refused"], s["ucond"] - b["ucond"]
            per.append({"d": s["unsafe"] - b["unsafe"], "ref": dA * (b["ucond"] + s["ucond"]) / 2, "ans": dB * (b["not_refused"] + s["not_refused"]) / 2,
                        "unsafe": s["unsafe"], "refused": 1 - s["not_refused"], "ucond": s["ucond"], "unsafe_refused": s["unsafe_refused"],
                        "ka": s["k_unsafe_answered"], "na": s["n_answered"]})
        P = pd.DataFrame(per)
        DEC[f"{c}|{lang}"] = {"d_unsafe_pp": round(100 * P.d.mean(), 2), "from_refusing_pp": round(100 * P.ref.mean(), 2),
                              "from_answers_pp": round(100 * P.ans.mean(), 2), "unsafe_pct": round(100 * P.unsafe.mean(), 2),
                              "refused_harmful_pct": round(100 * P.refused.mean(), 2), "unsafe_given_answered_pct": round(100 * P.ucond.mean(), 2),
                              "answered_unsafe_pooled": f"{int(P.ka.sum())}/{int(P.na.sum())}", "refused_and_unsafe_rows": int(P.unsafe_refused.sum()),
                              "per_run_from_answers_pp": [round(100 * x, 2) for x in P.ans]}
N["decomposition"] = DEC
print(pd.DataFrame(DEC).T.to_string())
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True, sharex=True)
for ax, lang, title in zip(axes, ("en", "ja"), ("English prompts", "Japanese prompts")):
    for i, c in enumerate(["en", "ja", "bi", "en2"]):
        d = DEC[f"{c}|{lang}"]
        ax.barh(i - 0.18, d["from_refusing_pp"], height=0.34, color="#8a8984", edgecolor="white", linewidth=1)
        ax.barh(i + 0.18, d["from_answers_pp"], height=0.34, color="#6250d6", edgecolor="white", linewidth=1)
        ax.plot([d["d_unsafe_pp"]], [i], marker="D", color=COL[c], ms=8, mec="white", mew=1.2, zorder=4)
    ax.axvline(0, color="#52514e", lw=1); ax.set_title(title, fontsize=11, loc="left")
    ax.set_xlabel("Change in Unsafe rate vs base (pp)  ← safer")
axes[0].set_yticks(range(4)); axes[0].set_yticklabels([NAME[c] for c in ["en", "ja", "bi", "en2"]]); axes[0].invert_yaxis()
from matplotlib.patches import Patch
fig.legend(handles=[Patch(color="#8a8984", label="from refusing more often"), Patch(color="#6250d6", label="from answering more (or less) safely"),
                    Line2D([], [], marker="D", ls="", color="#52514e", ms=7, label="total change")],
           loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.14))
fig.text(0.0, -0.2, "Two-factor Shapley split of P(Unsafe) = P(answered) × P(Unsafe | answered), mean of three training runs; harmful prompts, "
         "0.6B judge. Models answer different prompts, so the answer component compares different answered sets.", fontsize=8, color="#52514e")
save(fig, "fig3_refusal_decomposition")
'''))
C.append(("md", "## Figure 4: culture vs language: base model by cultural cue, and the matched-cue test (E4)"))
C.append(("code", r'''base_cult = {}
for row in BI["headline_g06"]["safety"]:
    k, n = row["raw"]["base"]["per_seed"]["base"]; base_cult[f"{row['culture']}|{row['lang']}"] = {"k": k, "n": n, "pct": round(100 * k / n, 2)}
N["base_by_culture"] = base_cult
E4 = {}
for k, v in FU["tests"]["E4"].items():
    if not v.get("n"): continue
    E4[k] = {"estimate_pp": round(100 * v["estimate"], 2), "ci95_pp": [round(100 * v["ci95"][0], 2), round(100 * v["ci95"][1], 2)],
             "p_holm": v.get("p_holm"), "confirmed_4b": v.get("confirmed_4b"), "g4b_pp": None if v["g4b"].get("estimate") is None else round(100 * v["g4b"]["estimate"], 2),
             "rates_pct": {a: round(100 * b, 2) for a, b in v["rates"].items()}, "n": v["n"], "n_seeds": v["n_seeds"]}
N["E4"] = E4; N["E4_note"] = FU["notes"].get("E4")
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1, 1.1], "wspace": 0.75})
x = np.arange(2)
for j, (cu, lab_, colr) in enumerate((("us", "US cue (cowboy, mafia, Hollywood…)", "#2a78d6"), ("japan", "Japan cue (ninja, samurai, anime…)", "#eb6834"))):
    vals = [base_cult[f"{cu}|{l}"]["pct"] for l in ("en", "ja")]
    bars = a1.bar(x + (j - 0.5) * 0.36, vals, width=0.34, color=colr, edgecolor="white", linewidth=1.5, label=lab_)
    for xi, v, l in zip(x + (j - 0.5) * 0.36, vals, ("en", "ja")):
        a1.text(xi, v + 0.4, f"{v:.1f}%", ha="center", fontsize=8.5, color="#0b0b0b")
a1.set_xticks(x); a1.set_xticklabels(["English prompts", "Japanese prompts"]); a1.set_ylabel("Base model: Unsafe answers (%)")
a1.set_title("Original Dataset2 cues", fontsize=11, loc="left"); a1.legend(frameon=False, fontsize=8, loc="upper left"); a1.set_ylim(0, 29)
rowsE4 = [("E4|safety|base: us-us_m (cue content)|en", "Original US − rewritten US\n(English)"), ("E4|safety|base: us-us_m (cue content)|ja", "Original US − rewritten US\n(Japanese)"),
          ("E4|safety|base: uk_m-us_m|en", "Rewritten UK − rewritten US\n(English)"), ("E4|safety|base: uk_m-us_m|ja", "Rewritten UK − rewritten US\n(Japanese)")]
for i, (k, lab_) in enumerate(rowsE4):
    v = E4[k]
    a2.errorbar(v["estimate_pp"], i, xerr=[[v["estimate_pp"] - v["ci95_pp"][0]], [v["ci95_pp"][1] - v["estimate_pp"]]], fmt="o", ms=7,
                color="#2a78d6", elinewidth=1.4, capsize=0, zorder=3)
    a2.plot(v["g4b_pp"], i + 0.22, marker="s", ms=5.5, color="#52514e", mfc="white", mew=1.3, ls="")
a2.axvline(0, color="#8a8984", lw=1); a2.set_yticks(range(4)); a2.set_yticklabels([r[1] for r in rowsE4], fontsize=8.5); a2.invert_yaxis()
a2.set_xlabel("Difference in Unsafe rate, base model (pp)"); a2.set_title("Matched-cue rewrite (E4, 150 seeds)", fontsize=11, loc="left")
a2.legend(handles=[Line2D([], [], marker="o", ls="-", color="#2a78d6", label="0.6B judge, 95% CI"),
                   Line2D([], [], marker="s", ls="", mfc="white", color="#52514e", label="4B judge")], frameon=False, fontsize=8, loc="lower right")
fig.text(0.0, -0.06, "Left: base model, full pool (475 seeds). Right: paired seed-cluster bootstrap on the 150 seeds whose US and UK rewrites passed all checks; "
         "the rewrite can change wording beyond the cue.", fontsize=8, color="#52514e")
save(fig, "fig4_culture_vs_language")
'''))
C.append(("md", "## Figure 5: interventions that reduce over-refusal (English prompts)"))
C.append(("code", r'''def stp(key, j="g06"):
    v = ST[f"primary_{j}"][key]; return round(100 * v["estimate"], 2), [round(100 * v["ci95"][0], 2), round(100 * v["ci95"][1], 2)]
def fu(e, key):
    v = FU["tests"][e][key]; return round(100 * v["estimate"], 2), [round(100 * v["ci95"][0], 2), round(100 * v["ci95"][1], 2)]
def drow(arm, kind, lang, metric):
    r = next(r for r in XD["rows"] if r["arm"] == arm and r["kind"] == kind and r["lang"] == lang and r["metric"] == metric and r.get("framings") == "all")
    return round(100 * r["delta"], 2), [round(100 * r["ci95"][0], 2), round(100 * r["ci95"][1], 2)]
ck_or = XC["tests"]["P|over_refusal|us|en"]
ck_saf = next(v for k, v in XC["tests"].items() if "safety" in k and "|us|en" in k)
INT = {
 "Weight edit, English adapter (steering)": {"or": stp("H1|en"), "unsafe": stp("H2|en"), "cohort": "full pool, 519 seeds, en_s0, US+Japan"},
 "Weight edit, bilingual adapter (E1)": {"or": fu("E1", "E1|over_refusal|A*bi-bi|us|en"), "unsafe": fu("E1", "E1|safety|A*bi-bi|us|en"), "cohort": "half pool, US cue, 3 runs"},
 "Helpful examples ×2, bilingual (E2)": {"or": fu("E2", "E2|over_refusal|bi_h2-bi|us|en"), "unsafe": fu("E2", "E2|safety|bi_h2-bi|us|en"), "cohort": "half pool, US cue, 3 runs"},
 "Hook removal at layer 18 (x_steer D)": {"or": drow("D1_L18", "twin", "en", "refused"), "unsafe": drow("D4_L18", "harmful", "en", "unsafe_su"), "cohort": "fold-1 seeds, en_s0, 256 tokens"},
 "Harm-independent direction, layer 1 (x_steer C)": {"or": (round(100 * ck_or["estimate"], 2), [round(100 * x, 2) for x in ck_or["ci95"]]),
                                                    "unsafe": (round(100 * ck_saf["estimate"], 2), [round(100 * x, 2) for x in ck_saf["ci95"]]), "cohort": "half pool, US cue, en_s0"},
}
N["interventions_en"] = {k: {"over_refusal_pp": v["or"][0], "over_refusal_ci": v["or"][1], "unsafe_pp": v["unsafe"][0], "unsafe_ci": v["unsafe"][1], "cohort": v["cohort"]} for k, v in INT.items()}
N["xsteer_D_ja"] = {"over_refusal": drow("D2_L18", "twin", "ja", "refused"), "unsafe": drow("D4_L18", "harmful", "ja", "unsafe_su")}
print(json.dumps(N["interventions_en"], indent=0)); print(N["xsteer_D_ja"])
fig, ax = plt.subplots(figsize=(8.2, 4.6))
ax.axhspan(-6, 0, xmin=0, xmax=1, color="#eef6ee", zorder=0); ax.text(-19, -0.7, "safer than unedited", fontsize=8, color="#008300")
OFF = {"Weight edit, English adapter (steering)": (0.5, -0.75), "Weight edit, bilingual adapter (E1)": (0.5, 0.55),
       "Helpful examples ×2, bilingual (E2)": (-3.2, 1.35), "Hook removal at layer 18 (x_steer D)": (0.6, 0.55),
       "Harm-independent direction, layer 1 (x_steer C)": (-10.2, -0.95)}
for name, v in INT.items():
    (x0, xc), (y0, yc) = v["or"], v["unsafe"]
    ax.errorbar(x0, y0, xerr=[[x0 - xc[0]], [xc[1] - x0]], yerr=[[y0 - yc[0]], [yc[1] - y0]], fmt="o", ms=7, color="#2a78d6",
                elinewidth=1.2, capsize=0, mec="white", mew=1, zorder=3)
    ax.annotate(name, (x0 + OFF[name][0], y0 + OFF[name][1]), fontsize=8.5, color="#0b0b0b")
ax.axhline(0, color="#8a8984", lw=1); ax.axvline(0, color="#8a8984", lw=1)
ax.set_xlim(-19.5, 4); ax.set_ylim(-2.3, 7.5)
ax.set_xlabel("Change in refused harmless English prompts (pp)  ← fewer refusals")
ax.set_ylabel("Change in Unsafe English answers (pp)  ↑ less safe")
ax.set_title("Every intervention that cut over-refusal also raised Unsafe answers", fontsize=11, loc="left")
fig.text(0.0, -0.08, "Each point is a paired change against its own unedited model on its own cohort (see labels in the article); bars are 95% CIs. "
         "Cohorts differ, so compare directions, not sizes. Judge: Qwen3Guard-Gen-0.6B.", fontsize=8, color="#52514e")
save(fig, "fig5_interventions")
'''))
C.append(("md", "## Other cited numbers: steering hypotheses, follow-ups E1–E3 (both languages)"))
C.append(("code", r'''def fut(v):
    g = v.get("g4b") or {}
    return {"estimate_pp": round(100 * v["estimate"], 2), "ci95_pp": [round(100 * v["ci95"][0], 2), round(100 * v["ci95"][1], 2)],
            "p_holm": v.get("p_holm"), "confirmed_4b": v.get("confirmed_4b"), "g4b_pp": None if g.get("estimate") is None else round(100 * g["estimate"], 2),
            "rates_pct": {a: round(100 * b, 2) for a, b in v["rates"].items()}, "n": v["n"], "n_seeds": v["n_seeds"]}
N["followups"] = {k: fut(v) for e in ("E1", "E2", "E3") for k, v in FU["tests"][e].items() if v.get("n")}
N["steering_primary"] = {}
for j in ("g06", "g4b"):
    for k, v in ST[f"primary_{j}"].items():
        N["steering_primary"][f"{k}|{j}"] = {"estimate_pp": round(100 * v["estimate"], 2), "ci95_pp": [round(100 * v["ci95"][0], 2), round(100 * v["ci95"][1], 2)],
                                             "a_rate": v.get("a_rate"), "b_rate": v.get("b_rate"), "supported": v.get("supported"), "p_holm": v.get("p_holm")}
print(len(N["followups"]), "follow-up tests;", list(N["steering_primary"])[:12])
'''))
C.append(("md", "## Save numbers"))
C.append(("code", r'''if SAVE:
    (DATA / "numbers.json").write_text(json.dumps(N, indent=1, default=float))
print("figures:", sorted(p.name for p in FIG.glob("*.png")))
'''))

nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
cells = []
for kind, src, *tags in C:
    c = nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src)
    if tags:
        c.metadata["tags"] = tags[0]
    cells.append(c)
nb.cells = cells
nbf.write(nb, OUT / "A1_figures.ipynb")
print("wrote", OUT / "A1_figures.ipynb")
