# ez-penta

Local Japanese-English safety experiments with Tiny Aya, AnswerCarefully, and AuditKIT.

## Run on a Colab GPU

Open [notebooks/answercarefully_colab.ipynb](notebooks/answercarefully_colab.ipynb)
in [Google Colab](https://colab.research.google.com/) using **File → Upload notebook**,
or select this repository from Colab's GitHub tab after these changes are pushed.
The notebook imports this repository's code. Before these Python changes are
pushed, set `UPLOAD_LOCAL_FILES = True` in section 2 and upload the eight small
source files listed there. After pushing, use the default GitHub checkout.

1. Select **Runtime → Change runtime type → GPU**.
2. Run cells in order: install Python packages, restore/build CUDA llama-server, log into
   Hugging Face, load data, download models, generate answers, then score them.
3. Start with `LIMIT = 5`. Set `LIMIT = None` for the whole selected split;
   use `SPLIT = "dev"` to keep test held out. The output limit remains 1,024 tokens.
   Choose `PROMPT_LANGUAGE = "ja"` for Japanese or `"en"` for the dataset's
   `meta-mlmc["text-English"]` translation. The reference stays Japanese and is
   retained only for inspection. Run each language separately with the same settings
   and pair examples using `metadata.source_id`.
4. Download the results ZIP before the runtime ends, or set `SAVE_TO_DRIVE = True`
   to write results into a private Drive folder. Unzip into local `runs/` to use
   the review viewer below.

The notebook uses the same GGUF quantizations, HTTP client, and AuditKIT guard
scoring as macOS. Aya and the guard run sequentially on Colab's GPU. Each run
records model snapshot hashes, dataset revision, llama.cpp/project commits,
settings, and phase durations. First setup includes a source build and downloads.
Your Mac's `.venv` and `run.sh` are not used in Colab.

Section 3 defaults to `CACHE_BUILD_IN_DRIVE = True`. It saves the completed server,
shared libraries, and build log in **My Drive/ez-penta-cache/llama-cuda/**.
Fresh runtimes restore a matching build and validate file hashes and CUDA device
detection before using it. The cache key includes the source commit, build options,
GPU, CUDA/compiler versions, CPU, and Linux environment. A mismatch or failed check
rebuilds automatically. `FORCE_REBUILD = True` requests a clean build; disable it
afterward to reuse the cache. Compilation still happens the first time, and partial
compilation is not persisted. A previously completed local build can be cached by
rerunning section 3; CMake reuses unchanged outputs. This is separate from
`SAVE_TO_DRIVE`, which controls experiment results.

Notebook code and mocked workflow tests are checked locally; NVIDIA execution
must be verified in Colab. GPU availability and runtime limits vary. Clear outputs
before sharing the notebook; AnswerCarefully records must remain private.

## Run everything (macOS)

Install [Homebrew](https://brew.sh/) and accept access to
[llm-jp/AnswerCarefully](https://huggingface.co/datasets/llm-jp/AnswerCarefully)
using your own Hugging Face account.

```sh
git clone https://github.com/Rynhalt/ez-penta.git
cd ez-penta
bash run.sh
```

The launcher installs missing Python 3.12 / llama.cpp through Homebrew, creates
`.venv`, installs pinned dependencies, and prompts for Hugging Face login if needed.
Use a read-capable token; do not put it in source files.
An existing Hugging Face login is reused. If none is found, the launcher prompts
for a token with hidden input and saves it using Hugging Face's normal login store.

On macOS, `.venv` links to a per-project environment under
`~/Library/Application Support/ez-penta/venvs/`. This keeps installed packages out
of iCloud-managed Desktop/Documents folders, where offloaded files can stall Python.

It then loads all 1,800 Japanese ACv2.2 examples (1,464 dev + 336 test), starts Tiny Aya, saves its
answers, stops Aya, starts Qwen3Guard, evaluates the saved answers with AuditKIT,
and stops the guard. Models download automatically on first use.
Expect several GB of disk space and a slower first run.

Stop other model servers first. The launcher refuses occupied ports 8080/8081
instead of terminating processes it does not own. Only one experiment model is
loaded at a time.

Options:

```sh
bash run.sh --setup-only
bash run.sh --split dev
bash run.sh --limit 1 --max-tokens 256
bash run.sh --limit 3 --max-tokens 1024 --output-dir runs/my-experiment
```

Aya's output limit defaults to 1,024 tokens; use `--max-tokens` to override it.
Without `--limit`, every row in the selected split is evaluated, including repeated
risk categories. `--limit N` selects the first N rows, not one row per category.
Use `--split dev` to keep test data held out while developing the evaluation.
This runs ACv2.2, not the overlapping older versions or separate ACv3/Borderline sets.
Full runs can take hours; generation and guard progress include completed/total counts.

Each teammate creates their own `.venv`; do not copy it. Python 3.12 is required
by the launcher (tested with 3.12.9). Activation selects that environment's
interpreter; it does not install Python.

## Program flow

```text
bash run.sh
  |
  +-- Install dependencies and reuse/prompt for Hugging Face login
  |
  +-- AnswerCarefully v2.2 -> all 1,800 Japanese dev + test examples
  |     ak.Sample(input=question, target=reference, metadata=source categories)
  |
  +-- Start Tiny Aya through llama-server (localhost:8080)
  |     AuditKIT: ak.evaluate(samples, model=answer_prompts, scorers=[])
  |       -> chat.complete([{role: "user", content: question}])
  |       -> HTTP POST /v1/chat/completions -> Tiny Aya generates an answer
  |       <- JSON response: choices[0].message.content
  |     Save question + reference + actual answer -> tiny-aya-responses.jsonl
  |
  +-- Stop Tiny Aya; exclude incomplete answers from judging
  |
  +-- Start Qwen3Guard through llama-server (localhost:8081)
  |     AuditKIT: ak.evaluate(saved_samples, model="precomputed", scorers=[guard])
  |       -> GuardJudge -> our LocalGuard HTTP adapter
  |       -> Send [user: original question, assistant: saved Aya answer]
  |       <- Qwen3Guard verdict: Safety / Categories / Refusal
  |     Parse verdict and aggregate -> AuditKIT RunResult
  |
  +-- Save auditkit-guard-results.json + summary.json; stop the guard
```

AuditKIT coordinates both phases. The first generates answers without scoring;
the second scores those exact saved answers without calling Tiny Aya again.
Each question starts a fresh conversation. The reference and source metadata are
retained for inspection but are sent to neither model. The guard judges the
question/answer pair, not whether Aya's wording matches the reference.

## Results

Each run creates a new `runs/answercarefully-<UTC timestamp>/` directory:

| File | Contents |
| --- | --- |
| `tiny-aya-responses.jsonl` | Prompts, references, Aya answers, source metadata, generation settings |
| `auditkit-guard-results.json` | Individual verdicts, raw guard replies, AuditKIT statistics and errors |
| `summary.json` | Safe/unsafe counts, unsafe rate, unscored and excluded counts |
| `run-settings.json` | Dataset revision, selected IDs, models and settings |
| `tiny-aya-server.log` | Aya startup and inference logs |
| `qwen3guard-server.log` | Guard startup and inference logs |

Start with `summary.json`, then inspect the responses and detailed report.
Existing output directories are never overwritten. Completed responses survive
an interrupted generation.

Safe = 0, Unsafe = 1. The unsafe rate uses only valid binary verdicts; it is
`null` if none were scored. Controversial/malformed verdicts are unscored.
Truncated responses are excluded and recorded separately. These are automated
guard judgments, not verified safety labels or helpfulness scores.
Limited runs take the first N rows and are not representative benchmarks.
Sampling uses temperature 0.2 without a fixed seed.

The guard is the community `QuantFactory/Qwen3Guard-Gen-0.6B-GGUF:Q8_0`
(about 805 MB), wrapped by a custom AuditKIT profile. No hosted API, PyTorch,
or separate GPU runtime is needed.

## Review results in a browser

```sh
python3 review_results.py
```

Open **http://127.0.0.1:8765**. This reads existing runs; it does not start models,
install packages, or upload data. No virtual environment is required for the viewer.

- Select a run and search Japanese/English prompts, answers, references, or IDs.
- Filter by guard verdict, detected refusal, risk area, dev/test split, or human label.
- Select a row to inspect the full prompt, Aya answer, reference, raw guard verdict,
  token usage, and generation settings. Excluded/truncated answers remain visible.
- Enter your reviewer name, choose a human label, add notes, and save. Use
  **Over-refusal** only when you determine that a legitimate request was refused;
  the guard's refusal flag alone does not establish over-refusal.

Human reviews persist in git-ignored `runs/review-annotations.sqlite3`, separately from the
original experiment files. Each reviewer has their own assessment. Back up this
file or export reviews before deleting old runs.

**Export reviews / Import reviews** lets teammates merge annotations for the same
saved prompt/answer pairs. Exports contain IDs, response hashes, reviewer labels,
and notes, not original prompts/references/answers. Check notes before sharing.
Imports reject mismatched responses and preserve conflicting existing reviews.

This version runs on localhost. Teammates can run it against their own authorized
local files and exchange annotation exports for matching transcripts; it is not a
shared hosted service. Do not publish AnswerCarefully records or original outputs
through a public website or online SQL explorer. Stop the viewer with Ctrl+C.

Use `--port 8766` if the default port is occupied, or `--runs-dir /path/to/runs`
to inspect another local results directory.

## Other commands

To inspect the expanded evaluation exported from `eval/eval_jp_en_variants.ipynb`,
place `answercarefully_eval.sqlite3`, `eval_aya_answers.jsonl`, and `eval_scores.jsonl`
in the project root, then run:

```sh
python3 inspect_evaluation.py
```

This opens SQLite read-only and writes a new private `runs/evaluation-review-*/`
folder with `report.md`, exact statistics/CSV tables, a searchable `review.html`
containing 32 example pairs, and `inspect.sql` for querying the full database locally.
The examples include preliminary audit notes and additional cases needing review;
they are not human gold labels. No models run and no files are uploaded.

After setup, activate the environment:

```sh
source .venv/bin/activate
python answercarefully_dataset.py --limit 10
python chat.py --stats --save
python -m unittest discover -s tests -v
```

- `answercarefully_dataset.py`: previews original ACv2.2 development records.
- `chat.py`: interactive chat; `/reset` clears history, `/exit` quits.
- `evaluate_answercarefully.py`: experiment commands, without reinstalling packages.

Rerun the experiment or judge previously saved responses:

```sh
python evaluate_answercarefully.py run --limit 3
python evaluate_answercarefully.py score --responses runs/EXISTING-RUN/tiny-aya-responses.jsonl
```

The score command also accepts the older `runs/ac-pilot.jsonl` format.
Both commands manage their own servers and require the selected ports to be free.
Scoring alone creates a new directory with the guard log, detailed report, and summary.
Tests use simulated responses and localhost sockets; they need no model or dataset.

## Troubleshooting

- Dataset 401/403: run `.venv/bin/hf auth login` with the account granted access.
- Wrong existing Python environment: rename `.venv` and rerun `bash run.sh`.
- Port occupied: stop that server or use `--aya-port` / `--guard-port`.
- Slow startup: inspect the current run's server log for download/load progress.
- Truncated answers: increase `--max-tokens` and create a new run.
- Nonzero exit after judging: inspect report errors and unscored counts.

`.venv`, model files, and the review database are git-ignored. Keep tokens and
AnswerCarefully records private; do not publish reports or notebook outputs containing original data.
