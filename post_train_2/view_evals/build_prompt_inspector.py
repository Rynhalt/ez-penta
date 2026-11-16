"""Build inspect_decline_vs_careful.ipynb and the small data extract it reads.

    python post_train_2/view_evals/build_prompt_inspector.py

The extract (post_train_2/notebooks/combined/dataset2_prompts_extract.csv.gz, about 10 MB) holds
dataset2's prompts, metadata and the original evaluation's answers and labels, so the notebook
does not need the 412 MB observations.csv (which is gitignored and not on Drive).
"""
import json
import uuid
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXTRACT = HERE.parent / "notebooks" / "combined" / "dataset2_prompts_extract.csv.gz"
COLS = ["prompt_id", "seed_id", "split", "category", "subtopic", "form", "severity", "kind", "expected",
        "culture", "framing", "prompt_language", "mix_direction", "output_language", "request", "prompt",
        "analysis_eligible", "safety", "refused", "answer"]


def build_extract():
    src = ROOT / "notebooks" / "combined" / "observations.csv"
    df = pd.read_csv(src, low_memory=False, usecols=COLS)[COLS]
    df.to_csv(EXTRACT, index=False, compression="gzip")
    print(f"wrote {EXTRACT.relative_to(ROOT)} ({EXTRACT.stat().st_size / 1e6:.1f} MB, {len(df)} rows)")


def md(src):
    return {"cell_type": "markdown", "id": uuid.uuid4().hex[:8], "metadata": {},
            "source": src.strip("\n").splitlines(keepends=True)}


def code(src):
    return {"cell_type": "code", "execution_count": None, "id": uuid.uuid4().hex[:8], "metadata": {},
            "outputs": [], "source": src.strip("\n").splitlines(keepends=True)}


CELLS = [
    md("""
# Inspect prompts: "decline" vs "answer carefully"

dataset2 has two kinds of harmful prompt:

| Label (`expected`) | Severity | Ideal behaviour | Seeds in dataset2 |
|---|---|---|---|
| `decline` | clear | refuse | 167 |
| `answer_carefully` | borderline | answer, with care and caveats | 153 |

**Why there are no same-seed pairs.** Severity belongs to the *seed* (the base request), so every
version of a seed shares one label. No seed has both a "decline" and an "answer carefully" version.
The closest match is two different seeds on the **same subtopic**: all 41 subtopics (in 10
categories) contain both kinds (38 of 41 once the heldout split is left out). This notebook pairs them that way, and always compares the **same
rendering** (culture, jailbreak technique, language) on both sides.

Each seed is rendered 24–84 ways: harmful + harmless twin × culture (US / Japan / neutral) ×
technique (direct, role-play, fiction, refusal suppression) × language (English / Japanese / mixed).
The `answer` / `safety` / `refused` columns are from the **original dataset2 evaluation** (Tiny Aya
Water GGUF Q4_K_M, temperature 0, judged by Qwen3Guard).

**Sections**
1. Settings and data
2. Overview: seeds per category
3. Side by side by subtopic (all seeds on one row)
4. Matched pairs: one "decline" seed next to one "answer carefully" seed, same rendering
5. One pair across every rendering
6. How the original model treated each kind
7. Free browsing with filters

Runs on any runtime (no GPU). On Colab it reads the data from Google Drive
(`MyDrive/ez-penta/post_train_2/notebooks/combined/dataset2_prompts_extract.csv.gz`).
Prompts and answers are untrusted text: they are shown escaped, as plain text.
"""),
    md("## 1. Settings and data"),
    code("""
# Which rendering to show in sections 3-4 (the plainest is neutral / control / en).
CULTURE = "neutral"     # "neutral", "us" or "japan"
FRAMING = "control"     # "control" (direct), "rp" (role-play), "fc" (fiction), "rs" (refusal suppression)
LANGUAGE = "en"         # "en", "ja" or "mix"
# Only used when LANGUAGE = "mix": mixed prompts come in two directions and two answer languages.
MIX_DIRECTION = "frame_ja+request_en"   # or "frame_en+request_ja"
OUTPUT_LANGUAGE = "en"                  # language the prompt asks the answer in: "en" or "ja"

# The heldout split is reserved for the final evaluation. Reading it is harmless, but leave it out
# if you might use what you see here to change the fix.
INCLUDE_HELDOUT = False
SHOW_ANSWERS = False    # True: add the original model's answers to the matched-pairs table

DRIVE_PROJECT_DIR = "/content/drive/MyDrive/ez-penta"
"""),
    code("""
import html
from pathlib import Path

import pandas as pd
from IPython.display import HTML, display

FILE = "dataset2_prompts_extract.csv.gz"
try:
    import google.colab  # noqa: F401
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    project = Path(DRIVE_PROJECT_DIR)
    candidates = [project / "post_train_2" / "notebooks" / "combined" / FILE,
                  project / "notebooks" / "combined" / FILE, Path("/content") / FILE,
                  Path("/content/drive/MyDrive") / FILE, Path("/content/drive/MyDrive/Colab Notebooks") / FILE]
    # Also any other copy of post_train_2 inside the project folder (e.g. a duplicate folder).
    candidates += sorted(project.glob(f"*/notebooks/combined/{FILE}")) + sorted(project.glob(f"*/*/notebooks/combined/{FILE}"))
else:
    candidates = [Path.cwd().parent / "notebooks" / "combined" / FILE]   # repo: post_train_2/view_evals
path = next((p for p in candidates if p.exists()), None)

if path is None and IN_COLAB:
    print(f"{FILE} is not on your Drive. Either copy it (from the repo's post_train_2/notebooks/combined/)")
    print(f"into {project / 'post_train_2' / 'notebooks' / 'combined'} and rerun, or upload it now:")
    from google.colab import files
    uploaded = files.upload()          # shows an upload button under this cell
    if FILE in uploaded:
        path = Path("/content") / FILE
        path.write_bytes(uploaded[FILE])
assert path, f"{FILE} not found; looked in: {[str(p) for p in candidates]}"

data = pd.read_csv(path, low_memory=False)
if not INCLUDE_HELDOUT:
    data = data[data["split"] != "heldout"]
data["unsafe"] = data["safety"].map({"Unsafe": 1, "Safe": 0})
TECHNIQUE = {"control": "direct", "rp": "role-play", "fc": "fiction", "rs": "refusal suppression"}
data["technique"] = data["framing"].map(TECHNIQUE)
data["mix_direction"] = data["mix_direction"].fillna("-")
harmful = data[data["kind"] == "harmful"]
print(f"loaded {len(data):,} prompts from {path}")
print(f"harmful: {harmful['seed_id'].nunique()} seeds, {len(harmful):,} renderings "
      f"({'incl.' if INCLUDE_HELDOUT else 'excl.'} heldout)")

STYLE = ("<style>.pi td{white-space:pre-wrap;vertical-align:top;text-align:left;max-width:520px;"
         "font-size:12px} .pi th{text-align:left}</style>")


def show(df, max_rows=None):
    df = df.head(max_rows) if max_rows else df
    # Escape every value ourselves (prompts and answers are untrusted text), then turn newlines into
    # line breaks; pandas' own escaping would print them as a literal backslash-n.
    safe = df.apply(lambda col: col.map(lambda v: "-" if pd.isna(v) else html.escape(str(v)).replace("\\n", "<br>")))
    table = safe.to_html(index=False, escape=False).replace("<table", '<table class="pi"', 1)
    display(HTML(STYLE + table))


def verdict(row):
    # The original evaluation's labels for one answer, in words.
    if row["safety"] not in ("Safe", "Unsafe", "Controversial"):
        return f"not scored ({str(row['safety']).lower()})"
    ref = {1: "refused", 0: "answered"}.get(row["refused"], "refusal unclear")
    return f"{row['safety']}, {ref}"
"""),
    md("## 2. Overview: seeds per category"),
    code("""
seeds = harmful.drop_duplicates("seed_id")
overview = pd.crosstab([seeds["category"]], seeds["expected"], margins=True, margins_name="total")
overview.columns.name = None
display(overview)
print(f"{seeds['subtopic'].nunique()} subtopics; "
      f"{(seeds.groupby('subtopic')['expected'].nunique() == 2).sum()} contain both kinds")
"""),
    md("""
## 3. Side by side by subtopic

One row per subtopic: every "decline" seed on the left and every "answer carefully" seed on the
right, all shown in the rendering chosen in the settings (default: neutral, direct, English).
"""),
    code("""
view = harmful[(harmful["culture"] == CULTURE) & (harmful["framing"] == FRAMING)
               & (harmful["prompt_language"] == LANGUAGE)]
if LANGUAGE == "mix":
    view = view[(view["mix_direction"] == MIX_DIRECTION) & (view["output_language"] == OUTPUT_LANGUAGE)]
assert not view["seed_id"].duplicated().any(), "rendering settings matched more than one prompt per seed"
rows = []
for (cat, sub), g in view.groupby(["category", "subtopic"]):
    def listing(kind):
        items = g[g["expected"] == kind].sort_values("form")
        return "\\n\\n".join(f"• [{r.form}] {r.prompt.strip()}" for r in items.itertuples())
    rows.append({"category": cat, "subtopic": sub,
                 "decline (clearly harmful)": listing("decline"),
                 "answer carefully (borderline)": listing("answer_carefully")})
print(f"rendering: culture={CULTURE}, technique={TECHNIQUE[FRAMING]}, language={LANGUAGE}")
show(pd.DataFrame(rows))
"""),
    md("""
## 4. Matched pairs

Within each subtopic, each "decline" seed is paired with an "answer carefully" seed, same question
or instruction form where possible. Both are shown in the same rendering, with the original model's
verdict on each. `pair` numbers are used in section 5.
"""),
    code("""
pairs = []
for (cat, sub), g in seeds.groupby(["category", "subtopic"]):
    dec = g[g["expected"] == "decline"].sort_values(["form", "seed_id"])
    car = g[g["expected"] == "answer_carefully"].sort_values(["form", "seed_id"])
    car_left = list(car.itertuples())
    for d in dec.itertuples():
        if not car_left:
            break
        same_form = [c for c in car_left if c.form == d.form]
        c = (same_form or car_left)[0]
        car_left.remove(c)
        pairs.append({"category": cat, "subtopic": sub, "decline_seed": d.seed_id, "careful_seed": c.seed_id,
                      "same_form": c.form == d.form})
pairs = pd.DataFrame(pairs)
pairs.insert(0, "pair", range(1, len(pairs) + 1))
print(f"{len(pairs)} matched pairs ({pairs['same_form'].mean():.0%} with the same form)")

rend = view.set_index("seed_id")
table = []
for p in pairs.itertuples():
    if p.decline_seed not in rend.index or p.careful_seed not in rend.index:
        continue
    d, c = rend.loc[p.decline_seed], rend.loc[p.careful_seed]
    row = {"pair": p.pair, "category": p.category, "subtopic": p.subtopic,
           "decline prompt": d["prompt"], "decline: original verdict": verdict(d),
           "answer-carefully prompt": c["prompt"], "careful: original verdict": verdict(c)}
    if SHOW_ANSWERS:
        row["decline: answer"], row["careful: answer"] = d["answer"], c["answer"]
    table.append(row)
print(f"rendering: culture={CULTURE}, technique={TECHNIQUE[FRAMING]}, language={LANGUAGE}")
show(pd.DataFrame(table))
"""),
    md("""
## 5. One pair across every rendering

Set `PAIR` to a number from section 4. Each row is one rendering (culture × technique × language)
that exists for both seeds, so you can see how the framing changes each prompt and how the original
model answered.
"""),
    code("""
PAIR = 1

p = pairs[pairs["pair"] == PAIR].iloc[0]
print(f"pair {PAIR}: {p['category']} / {p['subtopic']}")
key = ["culture", "framing", "prompt_language", "mix_direction", "output_language"]
d = harmful[harmful["seed_id"] == p["decline_seed"]].set_index(key)
c = harmful[harmful["seed_id"] == p["careful_seed"]].set_index(key)
both = d.index.intersection(c.index)
out = pd.DataFrame({
    "decline prompt": d.loc[both, "prompt"],
    "decline verdict": [verdict(r) for _, r in d.loc[both].iterrows()],
    "answer-carefully prompt": c.loc[both, "prompt"],
    "careful verdict": [verdict(r) for _, r in c.loc[both].iterrows()],
}).reset_index().sort_values(key)
out["framing"] = out["framing"].map(TECHNIQUE)
show(out.rename(columns={"framing": "technique", "prompt_language": "language",
                         "mix_direction": "mix direction", "output_language": "answer in"}))
"""),
    md("""
## 6. How the original model treated each kind

From the original dataset2 evaluation, analysis-eligible answers only.
- **Unsafe rate** should be low for both kinds.
- **Refusal rate** should be high for "decline" and low for "answer carefully": refusing a
  borderline prompt is over-caution.
"""),
    code("""
# "Error" / "Missing" mean the answer was never scored; leave those out.
scored = harmful[(harmful["analysis_eligible"] == True)
                 & harmful["safety"].isin(["Safe", "Unsafe", "Controversial"])]


def rates(by):
    g = scored.groupby([by, "expected"]).agg(n=("prompt_id", "size"), unsafe=("unsafe", "mean"),
                                              refused=("refused", "mean")).unstack("expected")
    g.columns = [f"{m} ({e})" for m, e in g.columns]
    for col in g.columns:
        if not col.startswith("n ("):
            g[col] = g[col].map(lambda v: f"{v:.0%}" if pd.notna(v) else "-")
    return g


overall = scored.groupby("expected").agg(n=("prompt_id", "size"), unsafe=("unsafe", "mean"),
                                         refused=("refused", "mean"))
display(overall.style.format({"unsafe": "{:.0%}", "refused": "{:.0%}"}))
for by in ["culture", "technique", "prompt_language", "category"]:
    display(rates(by))
"""),
    md("""
## 7. Browse with filters

`browse(...)` takes any of: `category`, `subtopic`, `expected` ("decline" / "answer_carefully"),
`culture`, `framing` ("control", "rp", "fc", "rs"), `prompt_language`, `mix_direction`,
`output_language`, `form`, `safety`, and
`text` (a word that must appear in the prompt). Values can be a single value or a list.
"""),
    code("""
def browse(text=None, max_rows=200, **filters):
    df = harmful
    for k, v in filters.items():
        df = df[df[k].isin(v if isinstance(v, (list, tuple, set)) else [v])]
    if text:
        df = df[df["prompt"].str.contains(text, case=False, na=False, regex=False)]
    df = df.sort_values(["category", "subtopic", "expected", "seed_id", "culture", "framing"])
    print(f"{len(df)} prompts" + (f" (showing first {max_rows})" if len(df) > max_rows else ""))
    show(df[["category", "subtopic", "expected", "seed_id", "culture", "technique", "prompt_language",
             "mix_direction", "output_language", "prompt", "safety", "refused"]], max_rows=max_rows)


browse(subtopic=seeds["subtopic"].iloc[0], culture="neutral", prompt_language="en")
"""),
]


def build_notebook():
    nb = {"cells": CELLS, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}, "colab": {"provenance": []}},
        "nbformat": 4, "nbformat_minor": 5}
    path = HERE / "inspect_decline_vs_careful.ipynb"
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote", path.name, f"({len(CELLS)} cells)")


if __name__ == "__main__":
    build_extract()
    build_notebook()
