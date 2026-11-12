"""Embed self-contained Colab assets and optionally package the source worktree."""

import argparse
from pathlib import Path
import zipfile

FILES = [
    "eval/DATASET2_VERIFICATION.md",
    "eval/validate_dataset2.py",
    "eval/DATASET2_RUN.md",
    "eval/DATASET2_RESEARCH.md",
    "eval/DATASET2_HANDOFF.md",
    "guard_judge.py",
    "chat.py",
    "eval/__init__.py",
    "eval/checkpoints.py",
    "eval/scoring.py",
    "eval/dataset2_data.py",
    "eval/dataset2_runtime.py",
    "eval/dataset2_environment.py",
    "eval/dataset2_gguf.py",
    "local_server.py",
    "colab_build_cache.py",
    "eval/requirements-dataset2-gguf.txt",
    "eval/dataset2_analysis.py",
    "eval/dataset2_likelihood.py",
    "eval/merge_dataset2_results.py",
    "eval/requirements-dataset2.txt",
    "eval/manifests/dataset2_validation_report.json",
    "eval/manifests/dataset2_shards.csv",
    "eval/manifests/dataset2_shards.json",
    "eval/eval_dataset2_part1.ipynb",
    "eval/eval_dataset2_part2.ipynb",
    "marcus/dataset2_partial.sqlite3",
]


def package(output):
    root = Path(__file__).resolve().parents[1]
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as z:
        for name in FILES:
            z.write(root / name, "ez-penta/" + name)
    return Path(output)


def embed_notebook_assets():
    """Make both Colab notebooks independently runnable without external asset uploads.

    Both carry the same deterministic compressed snapshot of the shared files.
    Notebook files are excluded from the snapshot to avoid recursive packaging.
    """
    import base64
    import hashlib
    import io
    import json

    root = Path(__file__).resolve().parents[1]
    payload = io.BytesIO()
    with zipfile.ZipFile(
        payload, "w", zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for name in sorted(FILES):
            if name.endswith(".ipynb"):
                continue
            info = zipfile.ZipInfo("ez-penta/" + name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, (root / name).read_bytes(), compresslevel=9)
    raw = payload.getvalue()
    digest = hashlib.sha256(raw).hexdigest()
    encoded = base64.b64encode(raw).decode("ascii")
    bootstrap = r"""#@title Automatic setup (dataset, shared code, dependencies)
# The dataset and shared implementation are embedded in this notebook.
# No repository clone, ZIP upload or separate dataset upload is required.
import os, sys, io, base64, hashlib, zipfile, subprocess, importlib, json
from pathlib import Path
REPO_DIR = Path(globals().get("REPO_DIR", "/content/ez-penta"))
ASSETS_SHA256 = DIGEST_PLACEHOLDER
# Bundled bytes live in notebook metadata, not in the code editor.
notebook_file = globals().get("DATASET2_NOTEBOOK_PATH")
if notebook_file is not None:
    notebook_document = json.loads(Path(notebook_file).read_text())
else:
    try:
        from google.colab import _message
    except ImportError as error:
        raise RuntimeError("Open this notebook in Colab, or set DATASET2_NOTEBOOK_PATH to its local .ipynb file.") from error
    print("Reading bundled dataset and code from notebook metadata...", flush=True)
    reply = _message.blocking_request("get_ipynb", timeout_sec=30)
    if isinstance(reply, str):
        reply = json.loads(reply)
    if not isinstance(reply, dict):
        raise RuntimeError("Colab could not read the notebook metadata. Reopen the updated notebook and rerun setup.")
    notebook_document = reply.get("ipynb", reply)
assets = notebook_document.get("metadata", {}).get("dataset2_assets", {})
if assets.get("sha256") != ASSETS_SHA256 or not isinstance(assets.get("payload"), str):
    raise RuntimeError("Bundled metadata is missing or from another release. Open the updated Part 1 or Part 2 notebook file.")
import re
for cell in notebook_document.get("cells", []):
    if cell.get("cell_type") == "code":
        text = cell.get("source", "")
        text = "".join(text) if isinstance(text, list) else text
        if re.search(r"(?m)^\s*[!%]pip\s+install[^\n]*--upgrade[^\n]*(?:huggingface_hub|transformers)", text):
            raise RuntimeError("Remove the extra '!pip install --upgrade huggingface_hub transformers' line from the model-loading cell. It overrides this notebook's tested versions and causes the restart loop. Then rerun setup.")
EMBEDDED_ASSETS = assets["payload"]
asset_bytes = base64.b64decode(EMBEDDED_ASSETS, validate=True)
if hashlib.sha256(asset_bytes).hexdigest() != ASSETS_SHA256:
    raise ValueError("Embedded assets failed integrity verification; reopen the original notebook")
with zipfile.ZipFile(io.BytesIO(asset_bytes)) as archive:
    for member in archive.infolist():
        relative = Path(member.filename)
        if relative.parts[0] != "ez-penta" or ".." in relative.parts or relative.is_absolute():
            raise ValueError("Unsafe embedded asset path")
        destination = REPO_DIR.joinpath(*relative.parts[1:])
        if not destination.resolve().is_relative_to(REPO_DIR.resolve()):
            raise ValueError("Embedded asset path escapes the runtime folder")
        content = archive.read(member)
        if not destination.is_file() or destination.read_bytes() != content:
            destination.parent.mkdir(parents=True, exist_ok=True)
            pending = destination.with_name(destination.name + ".partial")
            pending.write_bytes(content)
            os.replace(pending, destination)
del asset_bytes, EMBEDDED_ASSETS, content, assets, notebook_document
if "reply" in globals():
    del reply
os.chdir(REPO_DIR)
sys.path.insert(0, str(REPO_DIR.resolve()))
importlib.invalidate_caches()
# Refresh old backend imports without restarting the kernel.
for runner_name in ("aya", "guard_runner"):
    runner = globals().pop(runner_name, None)
    if runner is not None:
        runner.close()
for module_name in ("local_server", "eval.dataset2_runtime", "eval.dataset2_gguf"):
    if module_name in sys.modules:
        importlib.reload(sys.modules[module_name])
print("Dataset and shared code extracted; checking dependencies:", REPO_DIR)
from eval import dataset2_environment as setup_environment
importlib.reload(setup_environment)
setup_environment.install_gguf_dependencies(REPO_DIR / "eval/requirements-dataset2-gguf.txt")
""".replace(
        "DIGEST_PLACEHOLDER", repr(digest)
    )
    for part in [1, 2]:
        path = root / f"eval/eval_dataset2_part{part}.ipynb"
        notebook = json.loads(path.read_text())
        notebook["cells"][3]["source"] = bootstrap.splitlines(True)
        notebook["cells"][3]["metadata"].update(
            {"cellView": "form", "jupyter": {"source_hidden": True}}
        )
        import_setup = """# Restore package imports independently of setup-cell kernel state.
import sys, importlib
from pathlib import Path
if "REPO_DIR" not in globals():
    raise RuntimeError("Run the Environment and configuration cell first, then Automatic setup.")
repo_import_path = str(Path(REPO_DIR).resolve())
if not (Path(repo_import_path) / "eval" / "dataset2_data.py").is_file():
    raise RuntimeError("Shared code is not extracted yet. Run the Automatic setup cell near the top, wait for it to finish, then rerun this cell. Runtime > Run all also executes setup in order.")
if repo_import_path not in sys.path:
    sys.path.insert(0, repo_import_path)
importlib.invalidate_caches()
# End import-path recovery.
"""
        load_source = "".join(notebook["cells"][7]["source"])
        marker = "# End import-path recovery.\n"
        if marker in load_source:
            load_source = load_source.split(marker, 1)[1]
        notebook["cells"][7]["source"] = (import_setup + load_source).splitlines(True)
        for cell in notebook["cells"]:
            text = "".join(cell.get("source", []))
            if cell["cell_type"] != "code":
                continue
            if "from eval.dataset2_runtime import init_run, atomic_json, stage, make_judge" in text:
                text = text.replace("init_run, atomic_json, stage, make_judge",
                                    "init_run, atomic_json, stage, make_judge, checkpoint_summary")
            if text.startswith("try:\n    answers = stage("):
                text = ('checkpoint_summary(rows, RUN_DIR / "answers.jsonl", SHARD_ID, generation_config, retry_errors=RETRY_ERRORS)\n' + text)
            if text.startswith("try:\n    scores = stage("):
                text = ('checkpoint_summary(rows, RUN_DIR / "scores.jsonl", SHARD_ID, judge_config, answers=answers, retry_errors=RETRY_ERRORS)\n' + text)
            cell["source"] = text.splitlines(True)
        notebook["cells"][0]["source"] = (
            f"""# Dataset2 Part {part}: Tiny Aya + AuditKIT
Evaluates its full **160 seed / 10,592 prompt** shard. Both notebooks import the same implementation. No seed crosses parts. Automated guard measurements are not human ground truth.

Open this notebook in Colab, select **Runtime → Change runtime type → GPU**, and run the cells in order. The notebook includes its dataset and shared code and prepares them automatically. No ZIP or separate dataset upload is needed. Each team member runs only their assigned notebook. Keep the same notebook release, revisions and settings for both parts. Default GGUF checkpoints persist in your own Google Drive under `dataset2_eval_gguf/part1` or `part2`. Models and the CUDA server build are reused from Drive cache; GGUF model downloads are public and need no gated HF checkpoint.
"""
        ).splitlines(True)
        for cell in notebook["cells"]:
            text = "".join(cell.get("source", []))
            if cell["cell_type"] == "code" and text.startswith("aya = HFRunner("):
                cell["source"] = (
                    "from eval.dataset2_environment import require_clean_environment\nrequire_clean_environment()\n"
                    + text
                ).splitlines(True)
        for cell in notebook["cells"]:
            text = "".join(cell.get("source", []))
            if cell["cell_type"] != "code":
                continue
            for variable, assignment in [
                ("aya", "aya = GGUFRunner("),
                ("guard_runner", "guard_runner = GGUFRunner("),
            ]:
                cleanup = f'if "{variable}" in globals():\n    {variable}.close()\n    del {variable}\n'
                if assignment in text and cleanup not in text:
                    text = text.replace(assignment, cleanup + assignment, 1)
            cell["source"] = text.splitlines(True)
        notebook["metadata"]["dataset2_assets_sha256"] = digest
        notebook["metadata"]["dataset2_assets"] = {
            "encoding": "base64+zip",
            "sha256": digest,
            "payload": encoded,
        }
        path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n")
    return digest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="/tmp/dataset2_colab_bundle.zip")
    parser.add_argument(
        "--embed-notebooks",
        action="store_true",
        help="Refresh the identical self-contained assets in both notebooks before packaging",
    )
    args = parser.parse_args()
    if args.embed_notebooks:
        print("Embedded asset revision:", embed_notebook_assets())
    print(package(args.output))
