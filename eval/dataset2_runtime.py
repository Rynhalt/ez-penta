"""Shared generation, durable checkpoints and AuditKIT precomputed scoring for Dataset2."""

from datetime import datetime, timezone
import gc
from importlib.metadata import version, PackageNotFoundError
import inspect
import json
import os
import sys
from pathlib import Path
import shutil
import uuid

from eval.checkpoints import config_id, file_sha256

GENERATION = dict(
    do_sample=False, temperature=None, top_p=None, top_k=None, max_new_tokens=1024
)


def software_versions():
    result = {}
    for name in [
        "torch",
        "transformers",
        "tokenizers",
        "auditkit",
        "huggingface_hub",
        "accelerate",
    ]:
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = "unavailable"
    return result


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def implementation_id(stage_name="generation"):
    """Hash only code that can change this stage's persisted observation.

    Guard parsing and derived analysis never change an Aya completion. Keeping
    those out of the generation identity permits independent judge revisions.
    """
    sources = [implementation_id, HFRunner, Journal, stage, response_identity]
    if stage_name == "judge":
        from guard_judge import parse_guard
        from eval.scoring import guard_messages

        sources.extend(
            [GuardTransport, make_judge, parse_guard, guard_messages, score_identity]
        )
    elif stage_name != "generation":
        raise ValueError(f"Unknown implementation stage: {stage_name}")
    return config_id(
        {
            "stage": stage_name,
            "sources": [inspect.getsource(source) for source in sources],
        }
    )


class Journal:
    """One writer per shard. Fsync each observation; recover ONLY a damaged final line.

    Interior corruption fails. Discarded tail is preserved in a unique backup before
    truncation. Latest compatible completed observation wins over later failures.
    """

    def __init__(self, path, shard_id):
        self.path = Path(path)
        self.shard_id = shard_id
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.records = []
        self.index = {}
        configurations = {}
        if self.path.exists():
            size = self.path.stat().st_size
            # Stream large journals instead of retaining a second byte/line copy.
            with self.path.open("rb") as stream:
                number = 0
                while True:
                    offset = stream.tell()
                    line = stream.readline()
                    if not line:
                        break
                    number += 1
                    try:
                        record = json.loads(line)
                        if record["shard_id"] != shard_id:
                            raise ValueError("Foreign shard checkpoint")
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        if stream.tell() != size:
                            raise ValueError(
                                f"Interior checkpoint corruption: {self.path}:{number}"
                            )
                        backup = self.path.with_name(
                            self.path.name + ".damaged-" + uuid.uuid4().hex
                        )
                        shutil.copyfile(self.path, backup)
                        with self.path.open("r+b") as output:
                            output.truncate(offset)
                            output.flush()
                            os.fsync(output.fileno())
                        print(
                            f"Recovered interrupted final line; backup {backup}",
                            flush=True,
                        )
                        break
                    if "configuration" in record:
                        identity = config_id(record["configuration"])
                        record["configuration"] = configurations.setdefault(
                            identity, record["configuration"]
                        )
                    self.records.append(record)
                    self.index.setdefault(
                        (record["prompt_id"], record["identity"]), []
                    ).append(record)

    def append(self, record):
        if record["shard_id"] != self.shard_id:
            raise ValueError("Foreign shard write")
        separator = b""
        if self.path.exists() and self.path.stat().st_size:
            with self.path.open("rb") as f:
                f.seek(-1, 2)
                separator = b"" if f.read(1) == b"\n" else b"\n"
        with self.path.open("ab") as f:
            f.write(
                separator
                + json.dumps(record, ensure_ascii=False, allow_nan=False).encode()
                + b"\n"
            )
            f.flush()
            os.fsync(f.fileno())
        self.records.append(record)
        self.index.setdefault((record["prompt_id"], record["identity"]), []).append(
            record
        )

    def get(self, prompt_id, identity):
        matches = self.index.get((prompt_id, identity), [])
        good = [r for r in matches if r["status"] in ("completed", "truncated")]
        return (good or matches or [None])[-1]


def init_run(root, shard_id, dataset_meta, run_mode="full_shard"):
    if shard_id not in (1, 2):
        raise ValueError("Shard must be 1 or 2")
    path = Path(root) / f"part{shard_id}"
    path.mkdir(parents=True, exist_ok=True)
    identity = {"shard_id": shard_id, "dataset": dataset_meta, "run_mode": run_mode}
    p = path / "ownership.json"
    if p.exists() and json.loads(p.read_text()) != identity:
        raise ValueError("Output directory ownership/revision mismatch")
    atomic_json(p, identity)
    (path / "logs").mkdir(exist_ok=True)
    return path


def response_identity(row, configuration):
    return config_id(
        {
            "prompt_id": row["prompt_id"],
            "prompt": row["prompt"],
            "configuration": configuration,
        }
    )


def score_identity(row, answer, configuration):
    return config_id(
        {
            "prompt_id": row["prompt_id"],
            "prompt": row["prompt"],
            "answer": answer["answer"],
            "answer_identity": answer["identity"],
            "configuration": configuration,
        }
    )


class StageFailure(RuntimeError):
    """Retryable stage failure retaining partial diagnostic evidence."""

    def __init__(self, message, **details):
        super().__init__(message)
        self.details = details


def checkpoint_summary(rows, path, shard_id, configuration, answers=None, retry_errors=True):
    """Report reusable observations using the same identity/retry rules as stage."""
    journal = Journal(path, shard_id)
    reused = pending = ineligible = 0
    for row in rows:
        answer = None if answers is None else answers.get(row["prompt_id"])
        if answers is not None and (not answer or answer["status"] != "completed"):
            ineligible += 1
            continue
        identity = (response_identity(row, configuration) if answers is None
                    else score_identity(row, answer, configuration))
        cached = journal.get(row["prompt_id"], identity)
        if cached and (cached["status"] in ("completed", "truncated")
                       or not retry_errors or cached.get("permanent_error")):
            reused += 1
        else:
            pending += 1
    report = dict(reused=reused, pending=pending, ineligible=ineligible)
    print(f"Checkpoint {path}: {reused} reused, {pending} pending, "
          f"{ineligible} ineligible. Each new record is flushed and fsynced.", flush=True)
    return report


def stage(
    rows,
    path,
    shard_id,
    configuration,
    worker,
    answers=None,
    max_retries=2,
    retry_errors=True,
):
    if max_retries < 0:
        raise ValueError("max_retries must be nonnegative")
    journal = Journal(path, shard_id)
    result = {}
    run_id = uuid.uuid4().hex
    from tqdm.auto import tqdm

    for number, row in enumerate(tqdm(rows, desc=Path(path).name), 1):
        answer = None if answers is None else answers.get(row["prompt_id"])
        if answers is not None and (not answer or answer["status"] != "completed"):
            result[row["prompt_id"]] = None
            continue
        identity = (
            response_identity(row, configuration)
            if answers is None
            else score_identity(row, answer, configuration)
        )
        cached = journal.get(row["prompt_id"], identity)
        if cached and (
            cached["status"] in ("completed", "truncated")
            or not retry_errors
            or cached.get("permanent_error")
        ):
            result[row["prompt_id"]] = {**cached, **row}
            continue
        for attempt in range(max_retries + 1):
            record = {
                **row,
                "prompt_id": row["prompt_id"],
                "seed_id": row["seed_id"],
                "shard_id": shard_id,
                "identity": identity,
                "configuration": configuration,
                "run_id": run_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "attempt": attempt + 1,
            }
            try:
                payload = worker(row, answer) if answers is not None else worker(row)
                record.update(payload)
                record.setdefault("status", "completed")
                record.setdefault("error", None)
            except Exception as error:
                record.update(getattr(error, "details", {}))
                record.update(
                    status="error",
                    error=f"{type(error).__name__}: {error}",
                    permanent_error=isinstance(
                        error, (ValueError, KeyError, TypeError)
                    ),
                )
                gc.collect()
                try:
                    torch = sys.modules.get("torch")
                    if torch is not None and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
            journal.append(record)
            result[row["prompt_id"]] = record
            if record["status"] != "error" or record.get("permanent_error"):
                break
        if number % 25 == 0 or number == len(rows):
            print(f"{Path(path).name}: {number}/{len(rows)} visited", flush=True)
    return result


class HFRunner:
    """Existing notebook's HF chat input convention, with exact unpadded token traces."""

    def __init__(self, model_id, revision, dtype="float16", judge=False):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError("Select a Colab GPU runtime before model loading")
        if len(revision) != 40:
            raise ValueError("Use an immutable Hugging Face commit SHA")
        self.torch = torch
        self.judge = judge
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, revision=revision, padding_side="left"
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, revision=revision, dtype=getattr(torch, dtype), device_map="cuda"
        )
        self.model.eval()
        self.configuration = {
            "model_id": model_id,
            "revision": revision,
            "tokenizer_revision": revision,
            "chat_template": self.tokenizer.chat_template,
            "dtype": dtype,
            "generation": dict(GENERATION, max_new_tokens=128 if judge else 1024),
            "software": software_versions(),
            "implementation": implementation_id("judge" if judge else "generation"),
            "output_policy": "eos_only_complete_v1",
            "batch_size": 1,
            "add_generation_prompt": not judge,
            "model_generation_defaults": self.model.generation_config.to_dict(),
            "eos_token_id": self.model.generation_config.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }

    def generate(self, messages, max_new_tokens=None):
        t = self.tokenizer
        torch = self.torch
        rendered = t.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=not self.judge
        )
        encoded = t(rendered, add_special_tokens=False, return_tensors="pt").to(
            self.model.device
        )
        settings = dict(self.configuration["generation"])
        if max_new_tokens is not None:
            settings["max_new_tokens"] = max_new_tokens
        if (
            encoded["input_ids"].shape[1] + settings["max_new_tokens"]
            > self.model.config.max_position_embeddings
        ):
            raise ValueError(
                "Prompt plus response exceeds model context; no input truncation permitted"
            )
        with torch.inference_mode():
            output = self.model.generate(
                **encoded, **settings, pad_token_id=t.pad_token_id
            )
        ids = output[0, encoded["input_ids"].shape[1] :].tolist()
        eos = self.model.generation_config.eos_token_id
        eos = set(eos if isinstance(eos, list) else [eos])
        stop = next((i for i, x in enumerate(ids) if x in eos), None)
        selected = ids if stop is None else ids[: stop + 1]
        text = t.decode(selected, skip_special_tokens=True).strip()
        if not text:
            raise ValueError("Empty model completion")
        return {
            "answer": text,
            "input_ids": encoded["input_ids"][0].tolist(),
            "output_ids": selected,
            "input_tokens": encoded["input_ids"].shape[1],
            "output_tokens": len(selected),
            "status": "completed" if stop is not None else "truncated",
            "finish_reason": "eos" if stop is not None else "length",
            "truncated": stop is None,
            "rendered_input": rendered,
        }

    def answer(self, row):
        return self.generate([{"role": "user", "content": row["prompt"]}])

    def close(self):
        del self.model
        gc.collect()
        self.torch.cuda.empty_cache()


class GuardTransport:
    """AuditKIT Model-compatible transport, preserving full raw guard completions."""

    def __init__(self, runner):
        self.runner = runner
        self.name = runner.configuration["model_id"]
        self.raw = []

    def generate(self, requests):
        from auditkit.model import Generated, Result_

        result = []
        for request in requests:
            record = self.runner.generate(
                request.params["messages"], request.params["max_tokens"]
            )
            self.raw.append(record)
            if record["status"] != "completed":
                raise RuntimeError("Guard completion truncated; no verdict accepted")
            result.append(Result_(completions=[Generated(text=record["answer"])]))
        return result


def make_judge(runner):
    import auditkit as ak
    from eval.scoring import guard_messages
    from guard_judge import parse_guard

    transport = GuardTransport(runner)
    profile = {
        "id": "dataset2_qwen3guard",
        "build_messages": guard_messages,
        "parse": parse_guard,
    }
    scorers = [
        ak.GuardJudge(
            profile=profile,
            judge_model=transport,
            assess=assess,
            name="guard_" + assess,
            max_tokens=128,
        )
        for assess in ["response", "prompt"]
    ]
    config = {
        "runner": runner.configuration,
        "parser": inspect.getsource(parse_guard),
        "template": inspect.getsource(guard_messages),
        "auditkit": version("auditkit"),
        "metrics": [s.identity() for s in scorers],
        "implementation": implementation_id("judge"),
    }

    def score(row, answer):
        transport.raw = []
        sample = ak.Sample(
            id=row["prompt_id"], input=row["prompt"], actual_output=answer["answer"]
        )
        try:
            result = ak.evaluate(
                [sample],
                model="precomputed",
                scorers=scorers,
                config=ak.RunConfig(
                    max_retries=0, concurrency=1, extra={"attempt": uuid.uuid4().hex}
                ),
            )
        except Exception as error:
            raise StageFailure(
                f"AuditKIT evaluation raised: {type(error).__name__}: {error}",
                raw_guard_outputs=transport.raw,
            ) from error
        if result.errors or len(result.predictions) != 1:
            raise StageFailure(
                f"AuditKIT scoring failed: {result.errors}",
                raw_guard_outputs=transport.raw,
            )
        scores = {
            s["name"]: s for s in result.predictions[0].metadata.get("scores", [])
        }
        if set(scores) != {"guard_response", "guard_prompt"}:
            raise StageFailure(
                "Missing guard metrics",
                raw_guard_outputs=transport.raw,
                partial_scores=scores,
            )
        if len(transport.raw) != 2:
            raise StageFailure(
                "Guard transport did not produce both assessments",
                raw_guard_outputs=transport.raw,
                partial_scores=scores,
            )
        return {
            "status": "completed",
            "scores": scores,
            "raw_guard_outputs": transport.raw,
            "answer_identity": answer["identity"],
        }

    return score, config


def resolve_revisions(
    aya="CohereLabs/tiny-aya-water", guard="Qwen/Qwen3Guard-Gen-0.6B"
):
    from huggingface_hub import HfApi

    api = HfApi()
    return {model: api.model_info(model).sha for model in [aya, guard]}


def sync_artifacts(source, destination):
    """Publish durable journals and finalized files; SQLite is only written locally."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for p in Path(source).iterdir():
        if p.is_file() and not p.name.endswith(".partial"):
            temp = destination / (p.name + ".partial")
            shutil.copyfile(p, temp)
            os.replace(temp, destination / p.name)
