"""Build the bilingual notebooks from percent-format sources.

A cell header `# %% COPY <notebook> <cell index>` inserts that cell's source verbatim from an existing notebook
(AGENT_PROMPT_BILINGUAL 0.6). Any text after the header line is ignored for COPY cells.
"""
import json, re, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]  # post_train_3/


def src_cell(nb, idx):
    return "".join(json.loads((REPO / nb).read_text())["cells"][int(idx)]["source"])


def build(name):
    text = (HERE / f"{name}.py").read_text()
    parts = re.split(r"^# %%(.*)$", text, flags=re.M)
    cells = []
    for hdr, body in zip(parts[1::2], parts[2::2]):
        hdr = hdr.strip()
        body = body.strip("\n")
        if hdr.startswith("COPY"):
            _, nb, idx = hdr.split()
            cells.append({"cell_type": "code", "metadata": {"copied_from": f"{nb}#{idx}"}, "execution_count": None, "outputs": [],
                          "source": src_cell(nb, idx)})
        elif hdr.startswith("[markdown]"):
            md = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in body.splitlines())
            cells.append({"cell_type": "markdown", "metadata": {}, "source": md})
        else:
            meta = {"tags": ["parameters"]} if 'tags=["parameters"]' in hdr else {}
            cells.append({"cell_type": "code", "metadata": meta, "execution_count": None, "outputs": [], "source": body})
    nb = {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    for c in nb["cells"]:
        c["id"] = f"c{nb['cells'].index(c):02d}"
    (HERE.parent / f"{name}.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    print("built", name, len(cells), "cells")


for n in sys.argv[1:]:
    build(n)
