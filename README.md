# ez-penta

A small Python project exploring whether Tiny Aya makes consistent safety decisions
for equivalent English, Japanese, and mixed Japanese-English requests. We intend to
measure both unsafe compliance and over-refusal. No safety results have been
established by this scaffold.

## Components

- **Tiny Aya Water** supplies the trained model weights.
- **llama.cpp / llama-server** loads the weights, runs inference, and serves HTTP.
- **chat.py** is a standard-library Python HTTP client with an interactive loop.
  It does not train or implement the model.
- **tests/test_chat.py** uses a temporary localhost HTTP server to test the client
  without downloading weights or running inference.

Data flows from terminal input to a Python messages list, then UTF-8 JSON over HTTP,
then model inference, then a JSON response, and finally terminal output and optional
local JSONL logging. Each request includes recent conversation messages; conversation
memory is maintained by Python. Retaining three completed exchanges limits message
count, but does not guarantee that the next request fits the token context window.

## Run locally

Prerequisites: Python 3.9+ and a working local llama.cpp installation. No Python
packages are required. On the project's macOS setup:

```sh
brew install llama.cpp
python3 chat.py --stats --save
```

The client checks `http://127.0.0.1:8080/health`. It reuses a ready server, waits for
a loading server, or launches this command when the endpoint is offline:

```sh
llama-server -hf CohereLabs/tiny-aya-water-GGUF:Q4_K_M --host 127.0.0.1 --port 8080 --ctx-size 2048
```

You can also start that command yourself in a separate terminal. Model download may
be necessary on first startup. Automatically started server output goes to
`runs/llama-server.log`. The client stops only a server it started itself.
Keep the service bound to localhost.

| Option | Purpose | Default |
| --- | --- | --- |
| `--stats` | Display token counts and throughput when supplied | Off |
| `--save` | Save local UTF-8 JSONL records | Off |
| `--model` | Hugging Face selector when launching a server | `CohereLabs/tiny-aya-water-GGUF:Q4_K_M` |
| `--port` | Local server port | `8080` |
| `--ctx-size` | Context setting passed to a newly launched server | `2048` |
| `--max-tokens` | Maximum response tokens per request | `256` |

Temperature is fixed at `0.2`. `--model` and `--ctx-size` do not reconfigure an
existing server. The model ID comes from `/v1/models`; the current model mismatch
notice compares repository names, so it cannot verify quantization or revision.
Actual per-slot token capacity depends on the server's context and parallel-slot
configuration; inspect server startup output rather than inferring it from the
client option alone.

Commands: `/help`, `/reset` (clear Python history), `/exit`, `/quit`.
EOF and Ctrl+C also exit; a full context may require `/reset` or shorter inputs.

## Understand the HTTP boundary

In `chat.py`:

1. `complete()` builds a dictionary with `messages`, `temperature`, and `max_tokens`.
2. `read_json()` calls `json.dumps(..., ensure_ascii=False).encode("utf-8")`.
3. `Request(url, data=body, headers=...)` selects POST when a body is provided;
   `urlopen()` sends it to `/v1/chat/completions`.
4. `json.load(response)` decodes the response into Python dictionaries and lists.
5. `main()` extracts `result["choices"][0]["message"]["content"]`.
6. `print()` displays the answer; `--save` serializes a record as one JSON line.

This is the same HTTP exchange as the earlier curl experiment. Python additionally
manages conversation history, server ownership, error handling, statistics, and logs.
For GET requests such as health checks, `read_json()` omits the request body.

## Verify a conversation

Run `python3 chat.py --stats --save`, then enter:

```text
日本語で短く答えてください。私の合言葉は青い月です。
What is my passphrase? Answer briefly in English.
/reset
What passphrase did I give you? If none is in this conversation, say so briefly.
/exit
```

Check readable Japanese, an English follow-up recalling the passphrase, no prior
passphrase after reset, token statistics, and a saved session path. Model answers
can vary; tests of request history establish reset behavior independently of what
the model claims. To exercise automatic server startup while preserving an existing
8080 server, use an unused port, for example `--port 8081`.

## Inspect local logs

`--save` creates `runs/session-<timestamp>.jsonl`. Each line records UTC time,
reported model ID, current prompt, answer, input messages (including retained
history), token usage, timing metadata, finish reason, temperature, and output limit.
`finish_reason: "length"` can indicate a truncated response. Logging errors are
reported in the terminal. Logs contain conversation text and are excluded from Git;
ignoring files does not encrypt them. Older records may omit newer metadata fields.

Inspect a selected session (replace the example path):

```sh
python3 -c 'import json, sys; [print(json.dumps(json.loads(line), ensure_ascii=False, indent=2)) for line in sys.stdin if line.strip()]' < runs/session-EXAMPLE.jsonl
git status --short --ignored
```

These are exploratory chat logs, not complete reproducibility records: actual
server context, model revision/hash, llama.cpp version, and sampling seed are not
yet captured. Model IDs alone do not pin model weights.

## Tests

```sh
python3 -m unittest discover -s tests -v
```

Tests cover HTTP paths/payloads, Japanese round trips, health states, malformed JSON
and HTTP errors, reuse of a ready server, and the CLI's history/reset/stats/logging
behavior. They need permission to bind an ephemeral localhost socket. They do not
prove model quality, automatic real-model startup, or safety performance.

## Next experiment

The next small addition should be `baseline.py`: validate a JSONL dataset before
inference, send each prompt with a fresh one-message list, and write one output
record per prompt. Record prompt/intent IDs, language, expected behavior, response,
usage, model/settings, and errors. Preserve the dataset and server configuration
for reproducibility, and review responses manually rather than scoring keywords.

Start with 10–20 original intents and equivalent English/Japanese/mixed variants,
balancing benign requests, benign sensitive vocabulary, requests warranting refusal,
and ambiguity. This is a workflow pilot, not a validated benchmark. `complete()` can
be reused directly initially; a separate client module is only needed when shared
behavior becomes awkward. Later investigate CuratorKIT for dataset preparation and
AuditKIT for evaluation; verify actual interfaces and hackathon rules at that time.
Safety interventions and weight modifications are later milestones.
