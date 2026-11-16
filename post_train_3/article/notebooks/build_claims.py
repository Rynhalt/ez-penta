# Writes post_train_3/article/data/claims.json and notebooks/A2_validate.ipynb (the notebook does the checking).
import json
from pathlib import Path
import nbformat as nbf

ART = Path("/Users/isaac/App_Files/Code/ez-penta/post_train_3/article")
NJ = "article/data/numbers.json (from bilingual/results.json, followups/results.json, steering/results.json, x_steer/{C,D}/results.json)"

# kind "num": value at numbers.json path must equal `value` (to `dp` decimals) and `text` must appear in ARTICLE.md.
# kind "src": `pattern` must appear in the named source file and `text` in ARTICLE.md.
claims = [
 # Pooled rates (Section 4 table, abstract)
 ("num", "Base Unsafe, all prompts", ["pooled_all", "base", "unsafe_pct"], 13.29, 2, "13.3% (702/5,282)", "bilingual/results.json headline_g06"),
 ("num", "Bilingual Unsafe, all prompts", ["pooled_all", "bi", "unsafe_pct"], 6.85, 2, "6.9%\n(1,147/16,743)", "bilingual/results.json headline_g06"),
 ("num", "Base over-refusal, all prompts", ["pooled_all", "base", "over_refusal_pct"], 3.79, 2, "3.8% (144/3,800)", "bilingual/results.json headline_g06"),
 ("num", "Bilingual over-refusal, all prompts", ["pooled_all", "bi", "over_refusal_pct"], 25.68, 2, "25.7% (2,927/11,400)", "bilingual/results.json headline_g06"),
 ("num", "Japanese-only Unsafe, English prompts", ["ja_contrasts", "safety|ja-en|en", "rate_A_pct"], 6.45, 2, "6.45% (528/8,183)", "bilingual/results.json secondary"),
 ("num", "English-only Unsafe, English prompts", ["ja_contrasts", "safety|ja-en|en", "rate_B_pct"], 6.78, 2, "6.78% (557/8,211)", "bilingual/results.json secondary"),
 ("num", "Japanese-only over-refusal, English prompts", ["ja_contrasts", "over_refusal|ja-en|en", "rate_A_pct"], 13.16, 2, "13.16% (750/5,700)", "bilingual/results.json secondary"),
 ("num", "English-only over-refusal, English prompts", ["ja_contrasts", "over_refusal|ja-en|en", "rate_B_pct"], 20.39, 2, "20.39% (1,162/5,700)", "bilingual/results.json secondary"),
 ("num", "Bilingual over-refusal, English prompts", ["ja_contrasts", "over_refusal|bi-ja|en", "rate_A_pct"], 36.0, 2, "36.00% (2,052/5,700)", "bilingual/results.json secondary"),
 ("num", "Japanese-only Unsafe, Japanese prompts", ["ja_contrasts", "safety|ja-en|ja", "rate_A_pct"], 8.33, 2, "8.33% (696/8,359)", "bilingual/results.json secondary"),
 ("num", "English-only Unsafe, Japanese prompts", ["ja_contrasts", "safety|ja-en|ja", "rate_B_pct"], 15.68, 2, "15.68% (1,297/8,273)", "bilingual/results.json secondary"),
 ("num", "Bilingual Unsafe, Japanese prompts", ["ja_contrasts", "safety|bi-ja|ja", "rate_A_pct"], 9.68, 2, "9.68% (809/8,360)", "bilingual/results.json secondary"),
 ("num", "English ×2 Unsafe, Japanese prompts", ["ja_contrasts", "safety|en2-en|ja", "rate_A_pct"], 17.11, 2, "17.11% Unsafe (1,412/8,253", "bilingual/results.json secondary"),
 ("num", "Base Unsafe, Japanese prompts", ["vs_base", "safety|ja|ja", "rate_B_pct"], 15.37, 2, "15.37% (414/2,694)", "bilingual/results.json secondary"),
 ("num", "Japanese-only over-refusal, Japanese prompts", ["ja_contrasts", "over_refusal|ja-en|ja", "rate_A_pct"], 17.12, 2, "17.12% (976/5,700)", "bilingual/results.json secondary"),
 ("num", "Bilingual over-refusal, Japanese prompts", ["ja_contrasts", "over_refusal|bi-ja|ja", "rate_A_pct"], 15.35, 2, "15.35% (875/5,700)", "bilingual/results.json secondary"),
 ("num", "English-only over-refusal, Japanese prompts", ["ja_contrasts", "over_refusal|ja-en|ja", "rate_B_pct"], 6.39, 2, "6.39% (364/5,700)", "bilingual/results.json secondary"),
 # Japanese-only contrasts (Section 4.1)
 ("num", "ja − en Unsafe, English prompts", ["ja_contrasts", "safety|ja-en|en", "estimate_pp"], -0.33, 2, "−0.33 pp, 95% CI −1.09 to +0.45", "bilingual/results.json secondary safety|ja-en|en"),
 ("num", "ja − en over-refusal, English prompts", ["ja_contrasts", "over_refusal|ja-en|en", "estimate_pp"], -7.23, 2, "7.2 pp fewer harmless English prompts (CI −8.7 to −5.8)", "bilingual/results.json secondary"),
 ("num", "ja − en fix rate, English prompts", ["ja_contrasts", "fix_rate|ja-en|en", "estimate_pp"], -3.82, 2, "67.0% vs 70.8%; −3.8 pp, CI −7.7 to +0.03", "bilingual/results.json secondary"),
 ("num", "ja − base Unsafe, Japanese prompts", ["vs_base", "safety|ja|ja", "estimate_pp"], -7.04, 2, "−7.04 pp from base (CI −8.72 to −5.42)", "bilingual/results.json secondary"),
 ("num", "ja − en Unsafe, Japanese prompts", ["ja_contrasts", "safety|ja-en|ja", "estimate_pp"], -7.35, 2, "−7.35 pp versus English-only", "bilingual/results.json secondary"),
 ("num", "bi − ja Unsafe, Japanese prompts", ["ja_contrasts", "safety|bi-ja|ja", "estimate_pp"], 1.35, 2, "1.35 pp below bilingual (CI 0.47 to 2.29)", "bilingual/results.json secondary"),
 ("num", "ja − en fix rate, Japanese prompts", ["ja_contrasts", "fix_rate|ja-en|ja", "estimate_pp"], 30.22, 2, "+30.2 pp, CI 25.4 to 35.0", "bilingual/results.json secondary"),
 ("num", "bi − ja over-refusal, Japanese prompts", ["ja_contrasts", "over_refusal|bi-ja|ja", "estimate_pp"], -1.77, 2, "−1.77 pp, CI −2.86 to −0.72", "bilingual/results.json secondary"),
 # English transfer (4.2)
 ("num", "en − base Unsafe, English prompts", ["vs_base", "safety|en|en", "estimate_pp"], -4.34, 2, "lowered English Unsafe by 4.34 pp", "bilingual/results.json secondary"),
 ("num", "en − base Unsafe, Japanese prompts", ["vs_base", "safety|en|ja", "estimate_pp"], 0.31, 2, "+0.31 pp, CI −1.04 to\n+1.66", "bilingual/results.json secondary"),
 ("num", "en2 − base Unsafe, Japanese prompts", ["vs_base", "safety|en2|ja", "estimate_pp"], 1.74, 2, "+1.74 pp, CI +0.40 to +3.09", "bilingual/results.json secondary"),
 # Primary hypotheses (4.3)
 ("num", "H1 bi − en, English (0.6B)", ["primary", "H1|safety|bi-en|en", "g06", "estimate_pp"], -2.75, 2, "−2.75 (−3.56 to −1.95)", "bilingual/results.json primary_g06"),
 ("num", "H1 bi − en, Japanese (0.6B)", ["primary", "H1|safety|bi-en|ja", "g06", "estimate_pp"], -6.0, 2, "−6.00 (−7.38 to −4.65)", "bilingual/results.json primary_g06"),
 ("num", "H3 bi − en2, English (0.6B)", ["primary", "H3|safety|bi-en2|en", "g06", "estimate_pp"], -0.63, 2, "−0.63 (−1.26 to +0.02)", "bilingual/results.json primary_g06"),
 ("num", "H3 bi − en2, Japanese (0.6B)", ["primary", "H3|safety|bi-en2|ja", "g06", "estimate_pp"], -7.43, 2, "−7.43 (−8.91 to −5.99)", "bilingual/results.json primary_g06"),
 ("num", "H1 bi − en, English (4B)", ["primary", "H1|safety|bi-en|en", "g4b", "estimate_pp"], -1.68, 2, "| −1.68 |", "bilingual/results.json primary_g4b"),
 ("num", "H1 bi − en, Japanese (4B)", ["primary", "H1|safety|bi-en|ja", "g4b", "estimate_pp"], -4.42, 2, "| −4.42 |", "bilingual/results.json primary_g4b"),
 ("num", "H3 bi − en2, English (4B)", ["primary", "H3|safety|bi-en2|en", "g4b", "estimate_pp"], -0.25, 2, "| −0.25 |", "bilingual/results.json primary_g4b"),
 ("num", "H3 bi − en2, Japanese (4B)", ["primary", "H3|safety|bi-en2|ja", "g4b", "estimate_pp"], -5.69, 2, "| −5.69 |", "bilingual/results.json primary_g4b"),
 # Decomposition (Section 5)
 ("num", "ja answer component, Japanese", ["decomposition", "ja|ja", "from_answers_pp"], -2.05, 2, "**−2.05**", "B5 logic re-run in A1 (matches B5_refusal_decomposition_out.ipynb)"),
 ("num", "ja answer component, English", ["decomposition", "ja|en", "from_answers_pp"], -1.48, 2, "**−1.48**", "B5 logic re-run in A1"),
 ("num", "bi answer component, Japanese", ["decomposition", "bi|ja", "from_answers_pp"], -0.35, 2, "| −5.34 | −0.35 |", "B5 logic re-run in A1"),
 ("num", "bi answer component, English", ["decomposition", "bi|en", "from_answers_pp"], -0.12, 2, "| −6.94 | −0.12 |", "B5 logic re-run in A1"),
 ("num", "ja Unsafe among answered, Japanese", ["decomposition", "ja|ja", "unsafe_given_answered_pct"], 17.24, 2, "20.57% → 17.24%", "B5 logic re-run in A1"),
 ("num", "bi Unsafe among answered, all prompts", ["decomposition", "bi|all", "unsafe_given_answered_pct"], 18.09, 2, "(18.1% vs 17.8% Unsafe)", "B5 logic re-run in A1"),
 ("num", "base Unsafe among answered, all prompts", ["decomposition", "base|all", "unsafe_given_answered_pct"], 17.84, 2, "(18.1% vs 17.8% Unsafe)", "B5 logic re-run in A1"),
 ("num", "ja rows refused and Unsafe", ["decomposition", "ja|all", "refused_and_unsafe_rows"], 2, 0, "only 2 of about 16,500", "B5 logic re-run in A1"),
 # Culture (Section 6)
 ("num", "Base Unsafe, US cue, English", ["base_by_culture", "us|en", "pct"], 15.86, 2, "15.9% (199/1,255)", "bilingual/results.json headline_g06 base"),
 ("num", "Base Unsafe, Japan cue, English", ["base_by_culture", "japan|en", "pct"], 6.68, 2, "6.7% (89/1,333)", "bilingual/results.json headline_g06 base"),
 ("num", "Base Unsafe, US cue, Japanese", ["base_by_culture", "us|ja", "pct"], 20.23, 2, "20.2% (268/1,325)", "bilingual/results.json headline_g06 base"),
 ("num", "Base Unsafe, Japan cue, Japanese", ["base_by_culture", "japan|ja", "pct"], 10.66, 2, "10.7% (146/1,369)", "bilingual/results.json headline_g06 base"),
 ("num", "E4 UK − US matched, English", ["E4", "E4|safety|base: uk_m-us_m|en", "estimate_pp"], -0.25, 2, "UK − US −0.25 pp, CI −2.93 to +2.46", "followups/results.json tests.E4"),
 ("num", "E4 UK − US matched, Japanese", ["E4", "E4|safety|base: uk_m-us_m|ja", "estimate_pp"], 1.43, 2, "+1.43 pp, CI −2.14 to +4.96", "followups/results.json tests.E4"),
 ("num", "E4 original − matched US, English", ["E4", "E4|safety|base: us-us_m (cue content)|en", "estimate_pp"], 8.06, 2, "+8.06 pp in English (CI +3.42 to +12.74)", "followups/results.json tests.E4"),
 ("num", "E4 original − matched US, Japanese", ["E4", "E4|safety|base: us-us_m (cue content)|ja", "estimate_pp"], 13.17, 2, "+13.17 pp in\nJapanese (CI +7.97 to +18.76)", "followups/results.json tests.E4"),
 # E3 (4.4)
 ("num", "E3 ja2 − ja Unsafe, English", ["followups", "E3|safety|ja2-ja|us|en", "estimate_pp"], -0.64, 2, "−0.64 pp, CI −1.63 to +0.32", "followups/results.json tests.E3"),
 ("num", "E3 half A ja − en Unsafe, English", ["followups", "E3|safety|ja_A-en_A|us|en", "estimate_pp"], 1.86, 2, "+1.9 and +2.0 pp", "followups/results.json tests.E3"),
 ("num", "E3 half A ja − en over-refusal, English", ["followups", "E3|over_refusal|ja_A-en_A|us|en", "estimate_pp"], -7.61, 2, "−7.6 and −6.2 pp", "followups/results.json tests.E3"),
 ("num", "E3 half B ja − en Unsafe, Japanese", ["followups", "E3|safety|ja_B-en_B|us|ja", "estimate_pp"], -2.74, 2, "−7.6 and −2.7 pp", "followups/results.json tests.E3"),
 # Interventions (Section 7)
 ("num", "Steering H1 English", ["steering_primary", "H1|en|g06", "estimate_pp"], -5.78, 2, "−5.78 (−6.89 to −4.76)", "steering/results.json primary_g06"),
 ("num", "Steering H2 English", ["steering_primary", "H2|en|g06", "estimate_pp"], 1.46, 2, "+1.46 (+0.89 to +2.10)", "steering/results.json primary_g06"),
 ("num", "Steering H3 English", ["steering_primary", "H3|en|g06", "estimate_pp"], -5.19, 2, "−5.19 pp (222/308 → 206/308)", "steering/results.json primary_g06"),
 ("num", "E1 over-refusal, English", ["followups", "E1|over_refusal|A*bi-bi|us|en", "estimate_pp"], -5.68, 2, "−5.68 (−8.39 to −3.02)", "followups/results.json tests.E1"),
 ("num", "E1 over-refusal, Japanese", ["followups", "E1|over_refusal|A*bi-bi|us|ja", "estimate_pp"], 3.26, 2, "+3.26 pp", "followups/results.json tests.E1"),
 ("num", "E2 over-refusal, English", ["followups", "E2|over_refusal|bi_h2-bi|us|en", "estimate_pp"], -7.91, 2, "−7.91 (−11.29 to −4.47)", "followups/results.json tests.E2"),
 ("num", "E2 Unsafe, Japanese", ["followups", "E2|safety|bi_h2-bi|us|ja", "estimate_pp"], 2.82, 2, "+2.82 pp", "followups/results.json tests.E2"),
 ("num", "x_steer D layer 18 over-refusal, English", ["interventions_en", "Hook removal at layer 18 (x_steer D)", "over_refusal_pp"], -14.97, 2, "−14.97 (−18.58 to −11.42)", "x_steer/D/results.json rows"),
 ("num", "x_steer D layer 18 Unsafe, English", ["interventions_en", "Hook removal at layer 18 (x_steer D)", "unsafe_pp"], 3.75, 2, "+3.75 (+0.93 to +6.73)", "x_steer/D/results.json rows"),
 ("num", "x_steer C over-refusal, English", ["interventions_en", "Harm-independent direction, layer 1 (x_steer C)", "over_refusal_pp"], 0.72, 2, "+0.72 (−1.09 to +2.72)", "x_steer/C/results.json tests"),
 # Text-sourced claims (reports)
 ("src", "x_steer A: cos(f_pooled_old, g) = 0.60 at L18", None, None, None, "cosine 0.60", "post_train_3/x_steer/A/report.md", "at the steering layer L18 cos(f, g) = 0.60"),
 ("src", "x_steer A: f_bal vs h 0.78 at L18", None, None, None, "cosine 0.78 at layer 18", "post_train_3/x_steer/A/report.md", "| en_s0 | **f_bal vs h** | +0.28 | +0.47 | +0.74 | +0.78 |"),
 ("src", "x_steer B: no layer qualifies", None, None, None, "No layer, model or language met our separability rule", "post_train_3/x_steer/B/RESULTS.md", "no layer qualifies at any of the 36 layers"),
 ("src", "x_steer B: ja_s0 English twin refusals 99 → 285", None, None, None, "(99 → 285 of 975)", "post_train_3/x_steer/B/RESULTS.md", "99 → 285 refused English twins"),
 ("src", "Steering quality check", None, None, None, "118 edited wins, 104 unedited, 178 ties", "post_train_3/steering/report.md", "edited wins 118, unedited wins 104, ties 178"),
 ("src", "CAST on base, Unsafe", None, None, None, "10.0% (131/1,314) to 6.8% (90/1,332)", "post_train_3/steering/report.md", "131/1314 (10.0%) | 90/1332 (6.8%)"),
 ("src", "CAST on base, over-refusal", None, None, None, "4.4% (42/950) to 6.9% (66/950)", "post_train_3/steering/report.md", "42/950 (4.4%) | 66/950 (6.9%)"),
 ("src", "Pilot Unsafe counts", None, None, None, "12/145 (8.3%)", "post_train_2/report.md", "8.3% (12/145)"),
 ("src", "Pilot over-refusal counts", None, None, None, "29/80 (36.3%)", "post_train_2/report.md", "36.3% (29/80)"),
 ("src", "Japan cue: ninja share", None, None, None, "ninja (25% of Japan-framed English prompts)", "japan_culture/chunks/*/prompts (count in this session: 387/1,543)", None),
]

out = []
for c in claims:
    kind, name, path, value, dp, text, source = c[:7]
    out.append({"kind": kind, "claim": name, "path": path, "value": value, "dp": dp, "text": text, "source": source,
                "pattern": c[7] if len(c) > 7 else None})
(ART / "data").mkdir(parents=True, exist_ok=True)
(ART / "data" / "claims.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))

cells = [
 nbf.v4.new_markdown_cell("# A2: validate every cited number\nChecks each claim in `data/claims.json`: the value at its `numbers.json` path (written by A1 from the saved results), "
                          "that the cited text appears in `ARTICLE.md`, and for report-sourced claims that the pattern appears in the source report. Writes `CLAIMS.md`."),
 nbf.v4.new_code_cell(r'''import glob, json, os, re
from pathlib import Path
ART = Path("/Users/isaac/App_Files/Code/ez-penta/post_train_3/article")
REPO = ART.parents[1]
PT3D = Path(glob.glob(os.path.expanduser("~/Library/CloudStorage/GoogleDrive-*"))[0]) / "My Drive" / "ez-penta" / "post_train_3"
N = json.loads((ART / "data" / "numbers.json").read_text())
CL = json.loads((ART / "data" / "claims.json").read_text())
TXT = (ART / "ARTICLE.md").read_text()
norm = lambda s: re.sub(r"\s+", " ", s)
TXTn = norm(TXT)
def src_text(source):
    for base in (REPO, PT3D):
        p = base / source.replace("post_train_3/", "", 1) if base == PT3D else base / source
        if p.exists():
            return p.read_text()
    return None
rows, bad = [], 0
for c in CL:
    in_article = norm(c["text"]) in TXTn
    if c["kind"] == "num":
        v = N
        for k in c["path"]:
            v = v[k]
        ok_val = round(float(v), c["dp"]) == round(float(c["value"]), c["dp"])
        status = "ok" if (ok_val and in_article) else f"FAIL (value {v}, in article {in_article})"
        src = f"`numbers.json` → {' → '.join(c['path'])} (from {c['source']})"
    else:
        if c["pattern"] is None:
            ok_src, status_note = True, "counted in session, no saved file"
        else:
            t = src_text(c["source"]); ok_src = t is not None and norm(c["pattern"]) in norm(t)
        status = "ok" if (ok_src and in_article) else f"FAIL (source match {ok_src}, in article {in_article})"
        src = f"{c['source']}" + (f': "{c["pattern"]}"' if c["pattern"] else " (manual count)")
    bad += status != "ok"
    rows.append(f"| {c['claim']} | {c['text'].replace(chr(10), ' ').replace('|', '¦')} | {src.replace('|', '¦')} | {status} |")
md = ["# Claim audit: v2 article", "",
      f"Generated by `notebooks/A2_validate.ipynb`. {len(CL)} claims checked, {bad} failed. Numerical claims are compared with "
      "`data/numbers.json`, which `notebooks/A1_figures.ipynb` builds from the saved results; text claims are matched in the source report.", "",
      "| Claim | Text in article | Source | Check |", "|---|---|---|---|"] + rows
(ART / "CLAIMS.md").write_text("\n".join(md) + "\n")
print(f"{len(CL)} claims, {bad} failed")
for r in rows:
    if "FAIL" in r: print(r)
assert bad == 0
'''),
]
nb = nbf.v4.new_notebook(); nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}; nb.cells = cells
nbf.write(nb, ART / "notebooks" / "A2_validate.ipynb")
print("wrote claims.json and A2_validate.ipynb;", len(out), "claims")
