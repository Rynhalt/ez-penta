"""Shared rewrite batching/checkpoint mechanics; strategy templates stay in the notebook."""

import hashlib
import json

from eval.checkpoints import append_records, config_id, generation_status, load_jsonl, register_configuration


def load_rewrites(path, configuration):
    identity = config_id(configuration)
    return {key: record for key, record in load_jsonl(path, ("source_id", "source_split", "language")).items()
            if record.get("config_id") == identity}


def token_count(tokenizer, messages):
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def generate_batch(model, tokenizer, messages, max_tokens):
    import torch
    rendered = [tokenizer.apply_chat_template(turns, tokenize=False, add_generation_prompt=True) for turns in messages]
    encoded = tokenizer(rendered, add_special_tokens=False, padding=True, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output = model.generate(**encoded, max_new_tokens=max_tokens, do_sample=False,
                                temperature=None, top_p=None, top_k=None, pad_token_id=tokenizer.pad_token_id)
    replies = output[:, encoded["input_ids"].shape[1]:]
    eos = model.generation_config.eos_token_id
    stop_ids = set(eos if isinstance(eos, list) else [eos]) | {tokenizer.pad_token_id}
    cuts = [len(reply) == max_tokens and int(reply[-1]) not in stop_ids for reply in replies]
    return tokenizer.batch_decode(replies, skip_special_tokens=True), cuts


def run_rewrites(rows, languages, messages_for, checkpoint, configuration, *, model=None, tokenizer=None,
                 batch_size=16, max_batch_tokens=24000, max_new_tokens=512, generate=None, count_tokens=None):
    """Return compatible records, retrying failed cases and preserving cut outputs as cut."""
    if min(batch_size, max_batch_tokens, max_new_tokens) <= 0:
        raise ValueError("Batch and token limits must be positive")
    configuration = {**configuration, "batch_size": batch_size, "max_batch_tokens": max_batch_tokens,
                     "max_new_tokens": max_new_tokens}
    identity = register_configuration(checkpoint, configuration)
    finished = load_rewrites(checkpoint, configuration)
    oom_error, cleanup = (), lambda: None
    if generate is None:
        import torch
        generate = lambda messages: generate_batch(model, tokenizer, messages, max_new_tokens)
        oom_error, cleanup = torch.cuda.OutOfMemoryError, torch.cuda.empty_cache
    count_tokens = count_tokens or (lambda messages: token_count(tokenizer, messages))
    pending = []
    for row in rows.itertuples(index=False):
        for code, (field, language) in languages.items():
            prompt = getattr(row, field)
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"Missing {code} source prompt: {row.source_id}")
            messages = messages_for(prompt, language)
            input_hash = hashlib.sha256(json.dumps(messages, ensure_ascii=False).encode()).hexdigest()
            key = (row.source_id, row.source_split, code)
            existing = finished.get(key, {})
            status = generation_status({"answer": existing.get("output"), "truncated": existing.get("truncated"),
                                        "error": existing.get("error")})
            if existing.get("input_hash") == input_hash and status in ("completed", "truncated"):
                continue
            pending.append({"key": key, "messages": messages, "input_hash": input_hash,
                            "tokens": count_tokens(messages)})
    pending.sort(key=lambda row: row["tokens"])
    batches, batch = [], []
    for job in pending:
        if batch and (len(batch) == batch_size or (len(batch) + 1) * (job["tokens"] + max_new_tokens) > max_batch_tokens):
            batches.append(batch)
            batch = []
        batch.append(job)
    if batch:
        batches.append(batch)

    def attempt(batch):
        failed = False
        try:
            replies, cuts = generate([job["messages"] for job in batch])
        except oom_error:
            failed = True
        if failed:
            cleanup()
            if len(batch) == 1:
                return [(None, False, "OutOfMemoryError")]
            midpoint = len(batch) // 2
            return attempt(batch[:midpoint]) + attempt(batch[midpoint:])
        if len(replies) != len(batch) or len(cuts) != len(batch):
            raise ValueError("Generator reply count differs from the batch size")
        return [(reply.strip(), bool(cut), None) for reply, cut in zip(replies, cuts)]

    print(f"{len(pending)} rewrites pending; completed compatible outputs will be reused.", flush=True)
    for number, batch in enumerate(batches, 1):
        records = []
        for job, (output, cut, error) in zip(batch, attempt(batch)):
            source_id, split, language = job["key"]
            record = {"source_id": source_id, "source_split": split, "language": language,
                      "config_id": identity, "input_hash": job["input_hash"], "output": output,
                      "truncated": cut, "error": error}
            records.append(record)
            finished[job["key"]] = record
        append_records(checkpoint, records)
        print(f"Rewrite batch {number}/{len(batches)} checkpointed", flush=True)
    return finished
