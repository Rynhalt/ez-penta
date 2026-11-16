# %% [markdown]
# # B1_train (Colab GPU): train one layer-18 LoRA adapter for condition COND (ja / bi / en2) and training seed SEED
# The training cell is copied verbatim from `arm_en/notebooks/02_train.ipynb`; only the training rows differ (AGENT_PROMPT_BILINGUAL 0.6, 2).
# The copied cell saves to `ADAPTERS / f"seed{SEED}"`; here `ADAPTERS = raw/{COND}`, and the publish cell copies it to `adapters/{COND}_s{SEED}/`.

# %% tags=["parameters"]
COND = "bi"
SEED = 0
WORK = "/content/pt3bi"

# %% COPY japan_culture/en/notebooks/J1_chunk_en.ipynb 2

# %% COPY arm_en/notebooks/02_train.ipynb 1

# %% COPY arm_en/notebooks/02_train.ipynb 3

# %% COPY arm_en/notebooks/02_train.ipynb 4

# %%
# Training rows for this condition, built from plan.json (B0) and items.jsonl. Everything else is the EN arm's setup.
assert COND in ("ja", "bi", "en2") and SEED in (0, 1, 2), (COND, SEED)
IN = Path(WORK) / "inputs"
TRAIN = IN / "train"
assert (IN / "plan.DONE").read_text().strip() == sha256_file(IN / "plan.json"), "plan.DONE mismatch"
plan = json.loads((IN / "plan.json").read_text())
train_ids = json.loads((TRAIN / "train_item_ids.json").read_text())
assert hashlib.sha256("\n".join(sorted(train_ids)).encode()).hexdigest() == (TRAIN / "train_items_sha256.txt").read_text().strip() == plan["hashes"]["train_items_sha256"], "train_item_ids hash mismatch"
assert sha256_file(TRAIN / "items.jsonl") == plan["hashes"]["items.jsonl"]
assert train_ids == plan["train_item_ids"]
items = {r["id"]: r for r in load_jsonl(TRAIN / "items.jsonl")}
row_spec = plan["rows"][COND]
assert len(row_spec) == plan["row_counts"][COND] == {"ja": 568, "bi": 1136, "en2": 1136}[COND]
rows = [(items[i]["prompt"][l], items[i]["answer"][l]) for i, l in row_spec]
assert all(items[i]["survives"] for i in train_ids)
assert all(answer_lang(a) == l for (_, a), (_, l) in zip(rows, row_spec)), "a training answer is not in its row's language"
N = len(train_ids) // 2
assert sum(items[i]["item_kind"] == "refusal" for i in train_ids) == N == sum(items[i]["item_kind"] == "helpful" for i in train_ids)
from collections import Counter
print(f"{COND}: {len(rows)} training rows (N = {N} per kind); languages {dict(Counter(l for _, l in row_spec))}")
TARGET = "model.layers.18.mlp.down_proj"
TORCH_DTYPE = torch.bfloat16 if DTYPE == "bfloat16" else torch.float16
OUTD = Path(WORK) / "out" / "adapters" / f"{COND}_s{SEED}"
ADAPTERS = Path(WORK) / "out" / "raw" / COND
ADAPTERS.mkdir(parents=True, exist_ok=True)


def dir_done(d):
    if not (d / "DONE").exists() or (d / "DONE").read_text().strip() != sha256_file(d / "manifest.json"):
        return False
    man = json.loads((d / "manifest.json").read_text())
    return all((d / f).exists() and sha256_file(d / f) == m["sha256"] for f, m in man["files"].items())


MODE = "skip" if dir_done(OUTD) else "train"
print("MODE:", MODE)

# %% COPY arm_en/notebooks/02_train.ipynb 6

# %%
# Publish: adapter files + train_log.json + checks.json, then manifest.json, then DONE. No sanity gate stops the run (0.5).
if MODE == "train":
    import shutil
    OUTD.mkdir(parents=True, exist_ok=True)
    src = ADAPTERS / f"seed{SEED}"
    for f in src.iterdir():
        if f.is_file():
            shutil.copy(f, OUTD / f.name)
    tl = json.loads((OUTD / "train_log.json").read_text())
    tl.update({"condition": COND, "rows": len(rows), "row_languages": dict(Counter(l for _, l in row_spec))})
    (OUTD / "train_log.json").write_text(json.dumps(tl, indent=1))
    steps_logged = [r["step"] for r in tl["loss_log"]]
    expected_steps = [s for s in range(25, len(rows) + 1, 25)] + ([len(rows)] if len(rows) % 25 else [])
    checks = {"condition": COND, "seed": SEED, "trainable_params": tl["trainable_params"], "trainable_params_ok": tl["trainable_params"] == 208_896,
              "lora_B_abs_sum": tl["lora_B_abs_sum"], "lora_B_nonzero": tl["lora_B_nonzero"], "nan_loss": tl["nan_loss"],
              "steps": tl["steps"], "steps_ok": tl["steps"] == len(rows), "loss_logged_every_25": steps_logged == expected_steps,
              "n_truncated": tl["n_truncated"], "first_loss_25": tl["loss_log"][0]["mean_loss_last_25"], "last_loss_25": tl["loss_log"][-1]["mean_loss_last_25"],
              "hardware": HARDWARE, "dtype": DTYPE, "versions": VERSIONS}
    (OUTD / "checks.json").write_text(json.dumps(checks, indent=1))
    files = sorted(f.name for f in OUTD.iterdir() if f.is_file() and f.name not in ("manifest.json", "DONE"))
    man = {"condition": COND, "seed": SEED, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "hardware": HARDWARE, "gpu_name": GPU_NAME,
           "dtype": DTYPE, "versions": VERSIONS, "base_model": {"id": TINY_AYA, "sha": model_sha(TINY_AYA)},
           "lora": {"r": 16, "lora_alpha": 32, "lora_dropout": 0.05, "bias": "none", "target_modules": [TARGET]},
           "training": {"optimizer": "AdamW", "lr": 1e-4, "epochs": 1, "batch_size": 1, "scheduler": None, "grad_accum": 1, "max_len": 768,
                        "rows": len(rows), "row_order": "train_item_ids.json order; " + {"ja": "Japanese row per item", "bi": "English row then Japanese row per item",
                                                                                          "en2": "English row twice per item"}[COND]},
           "train_items_sha256": plan["hashes"]["train_items_sha256"], "plan_sha256": sha256_file(IN / "plan.json"), "checks": checks,
           "files": {f: {"sha256": sha256_file(OUTD / f), "bytes": (OUTD / f).stat().st_size} for f in files}}
    (OUTD / "manifest.json").write_text(json.dumps(man, indent=1))
    (OUTD / "DONE").write_text(sha256_file(OUTD / "manifest.json") + "\n")
    print(json.dumps(checks, indent=1))
assert dir_done(OUTD)
print("published", OUTD)
