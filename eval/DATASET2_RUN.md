Running the two Dataset2 GGUF notebooks
======================================

Open the updated `eval/eval_dataset2_part1.ipynb` or `eval/eval_dataset2_part2.ipynb` in Colab. Each notebook contains the source dataset, authoritative manifest and shared Python implementation in metadata. No repository clone or separate ZIP upload is needed. Each independent shard contains 160 complete seed groups / 10,592 retained prompts; together they cover all 21,184 prompt IDs without overlap.

The current default reuses the earlier AnswerCarefully local GGUF/llama.cpp workflow:

| Component | Repository / format |
|---|---|
| Tiny Aya | `CohereLabs/tiny-aya-water-GGUF`, Q4_K_M |
| Qwen3Guard | `QuantFactory/Qwen3Guard-Gen-0.6B-GGUF`, Q8_0 |
| Inference | Pinned llama.cpp CUDA server, one model at a time |
| Decoding | Greedy temperature 0.0, 1024 Aya / 128 guard token caps |
| Context | 8192 tokens, one slot, no context shifting |

Both GGUF file endpoints were verified publicly accessible without credentials. The gated original fp16 checkpoint is not used. Default Python setup omits Hugging Face Hub, Transformers and Torch, and retains compatible Colab NumPy/Pandas versions. Remove manual upgrade/repair cells carried over from earlier notebook copies.

1. Assign Part 1 and Part 2 to different team members. Use the same notebook release/model settings for both. Open through **Colab → File → Upload notebook**.
2. Select **Runtime → Change runtime type → GPU**, with an NVIDIA GPU and nvcc. A T4 is the intended minimum. Run the configuration and small Automatic setup cell. Setup reads embedded notebook metadata, verifies/extracts files, and installs only missing/incompatible lightweight Python dependencies. A real loaded-library mismatch may still require one session restart; matching repeat setup skips pip.
3. Run the Google Drive cell and authorize the mount. No Hugging Face login or HF_TOKEN is required for the public GGUF path.
4. Keep `RUN_MODE="full_shard"`, `SMOKE_SEED_LIMIT=None` and your assigned `SHARD_ID`. The dataset/hash/manifest checks run before models load. All retained prompts stay in generation; semantic and missing-variant flags remain visible.
5. The Tiny Aya loading cell restores or compiles a pinned CUDA llama-server. First compilation may take several minutes; progress and build-log location are printed. The build is headless: unused web-UI provisioning is disabled. It then reuses or downloads/checksums the Aya GGUF and starts the owned local server.
6. Generation checkpoints each answer, including exact native input/output token IDs, rendered chat input, stop status, model revision/hash, quantization and backend. EOS is completed; token-cap output is truncated and excluded from guard scoring. Context truncation is an error, never a valid safety measurement.
7. Aya's server stops before the guard loads. AuditKIT scores saved answers through the local Qwen3Guard server, using separate prompt/response assessments and preserving raw outputs. Judging failure does not require response regeneration.
8. Run the final export cell and inspect coverage/failure counts. Download/share each entire part directory for merging. Completed guard judgments can still be Controversial/Unparsed; they remain outside binary safety denominators.

Default output directories are:

```
My Drive/ez-penta/dataset2_eval_gguf/part1/
My Drive/ez-penta/dataset2_eval_gguf/part2/
```

These are deliberately separate from earlier `dataset2_eval/` fp16 results. Change OUTPUT_ROOT only if needed, consistently preserving shard ownership. With USE_DRIVE=False, outputs/cache exist only in runtime storage and must be downloaded/copied before deleting the runtime.

Model reuse and resumption
--------------------------

- Runtime GGUF locations: `/content/models/aya/` and `/content/models/guard/`. These also match the earlier AnswerCarefully notebook's model directories.
- Persistent verified models: `My Drive/ez-penta-cache/models/<checksum-prefix>/`.
- Persistent GPU/CPU/compiler-specific CUDA builds: `My Drive/ez-penta-cache/llama-cuda/`.
- AYA_LOCAL_GGUF and GUARD_LOCAL_GGUF optionally point to already-downloaded files accessible to Colab/Drive. Files remaining only on your Mac are not automatically visible to Colab. Overrides must match the pinned revision/checksum; arbitrary quantizations are not silently substituted.
- Missing models download from immutable public URLs and are checked against publisher LFS SHA256/size. Downloads now write `.partial` files directly in the persistent model cache, so a replacement runtime can resume via byte ranges when supported. Completed verified files are copied to runtime storage for inference. Existing local models are also published to Drive if absent there. A checksum failure remains explicit.
- On interruption, reconnect a GPU and run the same notebook release/settings from the top. Matching answers/judgments are reused. Generation and score journals flush/fsync each record, recover a backed-up damaged final line and reject interior corruption. Do not run two writers for the same shard/output directory simultaneously.
- The compilation log is `/content/llama.cpp/dataset2-build.log`; model-server logs are under each part's `logs/` directory. Only processes owned by this notebook are stopped.
- Ports are allocated automatically; a collision during startup is retried with another port. Setup refreshes previously imported backend modules and closes its own earlier runners, so updated code takes effect without a kernel restart. Remove manual `fuser`/`pkill` cells.
- Generation and scoring print compatible checkpoint counts before running. Running cells again reloads Python state and checks cached assets; it does not repeat compatible saved answers or scores. A request interrupted before its record is saved must run again. A completed CUDA build is cached; an interrupted compilation itself must restart. The exact GPU/compiler cache identity must match to reuse a build.

For a bounded deployment check, set RUN_MODE="smoke" and SMOKE_SEED_LIMIT=1. It processes one complete seed and writes to a separate `smoke/partN` directory. Reset to full defaults for the actual evaluation; the merger rejects smoke runs.

Merging and inspection
-----------------------

Bring both complete part directories into one environment with this repository (or use `/content/ez-penta` after notebook setup):

```bash
python -m eval.merge_dataset2_results \
  --part1 /path/to/part1 \
  --part2 /path/to/part2 \
  --output /path/to/combined
```

Use `--dataset` and `--manifest` if those files live elsewhere. The merger validates source/manifest/model/backend/quantization/settings/parser identities, inventories every assigned ID, retains explicit missing/failed observations and recomputes aggregates from observations. It does not average shard rates. Interrupted runs without final current exports must resume/export first. GGUF and fp16 configurations cannot be merged as one homogeneous experiment.

Inspect `coverage.json`, `answers_current.jsonl`, `scores_current.jsonl`, `observations.csv`, `headline.csv`, `denominators.csv`, `comparisons.csv`, `matched_observations.csv`, `sensitivity.csv`, `manual_inspection.csv`, and `results.sqlite3`. SQLite is written to local temporary storage, then finalized snapshots are copied to output. Journals write independently into each part directory.

Human review fields start blank. Generated expectations are not ground truth; expected correctness, attack success and over-refusal remain unknown without validated expectations/outcome rules. Keep heldout data out of iterative tuning, and treat bootstrap intervals as descriptive/unadjusted. `DATASET2_HANDOFF.md` explains the next CircuitKIT steps and why quantized screening behavior must be reproduced on the intended differentiable model before attribution.

Developer verification and optional HF workflow
----------------------------------------------

```bash
python -m pip install -r eval/requirements-dataset2-gguf.txt
python -m unittest discover -s tests -v
```

Optional HF tensor/likelihood tests require the separate historical `requirements-dataset2.txt` and nbformat. Default GGUF evaluation does not install them or run HF likelihood diagnostics. `eval/dataset2_likelihood.py` remains a separate diagnostic for an authorized HF checkpoint; it is not a differentiable attribution implementation. Do not combine likelihood claims across backends/precision without validation.

Maintainers refresh the identical embedded asset snapshots with `python -m eval.package_dataset2_colab --embed-notebooks`. The optional source ZIP is for source transfer/local merging, not a required notebook upload. If an old notebook copy's huge setup cell froze the page, close it and open the updated file; current payloads live in metadata and code cells remain small. Colab uses its private notebook-metadata bridge; local Jupyter can set DATASET2_NOTEBOOK_PATH to the notebook file instead.

Port handling: the GGUF runners now ask the OS for available loopback ports, rather than assuming 8080/8081 are free. They retry a detected automatic-port collision without stopping the existing listener. Explicit port overrides still fail clearly if busy. Port numbers are transport details and do not change a run's generation configuration. For an older notebook copy, pass a free port via the runner's `port=` argument and continue; no model re-download is needed.
