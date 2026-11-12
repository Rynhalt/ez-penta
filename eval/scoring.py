"""AuditKIT scoring mechanics; the notebook chooses and explains the scorers."""

from datetime import datetime, timezone
import hashlib
import inspect
from importlib.metadata import version
import uuid

import auditkit as ak

from eval.checkpoints import ANSWER_KEY, append_records, case_id, compatible_score, config_id, fingerprint, generation_status, load_jsonl, register_configuration
from guard_judge import parse_guard


def guard_messages(prompt, output, assess):
    if assess == "prompt":
        return [{"role": "user", "content": prompt}]
    if assess != "response":
        raise ValueError(f"Unsupported assessment: {assess}")
    return [{"role": "user", "content": prompt}, {"role": "assistant", "content": output}]


QWEN3GUARD = {"id": "qwen3guard", "build_messages": guard_messages, "parse": parse_guard, "max_tokens": 128}
JAPANESE_RUN = r"[\u3040-\u30ff\u4e00-\u9fff]{8,}"


def scoring_manifest(configuration, scorers, annotators):
    """Changes to models, parsing, metric settings, or annotations invalidate cached scores."""
    parser_code = inspect.getsource(parse_guard) + inspect.getsource(guard_messages)
    return {"schema_version": 2, "configuration": configuration,
                      "auditkit_version": version("auditkit"),
                      "parser_sha256": hashlib.sha256(parser_code.encode()).hexdigest(),
                      "scorers": [scorer.identity() for scorer in scorers],
                      "annotators": [annotator.identity() for annotator in annotators]}


def scoring_identity(configuration, scorers, annotators):
    return config_id(scoring_manifest(configuration, scorers, annotators))


def score_stage(stage, samples, scorers, checkpoint, configuration, annotators=(), chunk_size=64, evaluate=None):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    identity = register_configuration(checkpoint, scoring_manifest(configuration, scorers, annotators))
    done = load_jsonl(checkpoint, ("stage", "sample_id"))
    pending = [sample for sample in samples if not compatible_score(done.get((stage, sample.id)), sample, identity)]
    print(f"{stage}: {len(samples) - len(pending)}/{len(samples)} compatible scores; {len(pending)} pending", flush=True)
    evaluate = evaluate or ak.evaluate
    for start in range(0, len(pending), chunk_size):
        chunk = pending[start:start + chunk_size]
        result = evaluate(chunk, model="precomputed", scorers=scorers, annotators=list(annotators),
                          config=ak.RunConfig(max_retries=0, extra={"score_config_id": identity,
                                                "stage": stage, "attempt": uuid.uuid4().hex}))
        predictions = {prediction.sample_id: prediction for prediction in result.predictions}
        records = []
        required = {scorer.name for scorer in scorers}
        for sample in chunk:
            prediction = predictions.get(sample.id)
            scores = {score["name"]: {"value": score.get("value"), "metadata": score.get("metadata") or {}}
                      for score in (prediction.metadata.get("scores", []) if prediction else [])}
            context = {name: {"matched": value.get("matched")} for name, value in
                       ((prediction.context or {}) if prediction else {}).items() if isinstance(value, dict)}
            record = {"schema_version": 2, "stage": stage, "sample_id": sample.id,
                      "fingerprint": fingerprint(sample.input, sample.actual_output, sample.target),
                      "config_id": identity, "scoring_status": "completed" if required <= scores.keys() else "error",
                      "error": None if required <= scores.keys() else "Missing metrics: " + ", ".join(sorted(required - scores.keys())),
                      "scores": scores, "context": context, "timestamp": datetime.now(timezone.utc).isoformat()}
            records.append(record)
            done[(stage, sample.id)] = record
        append_records(checkpoint, records)
        print(f"{stage}: checkpointed {min(start + chunk_size, len(pending))}/{len(pending)} new scores", flush=True)
    return {sample.id: done[(stage, sample.id)] for sample in samples}


def score_answers(answers, variants, answer_scorers, pair_scorers, checkpoint, configuration,
                  annotators=(), chunk_size=64, evaluate=None):
    """Grade individual complete answers and embed pairs only when both answers completed."""
    rows = answers.to_dict("records")
    complete = [row for row in rows if generation_status(row) == "completed"]
    samples = [ak.Sample(id=case_id(row), input=row["prompt"], actual_output=row["answer"], target=row.get("reference"),
                         metadata={field: row[field] for field in ANSWER_KEY}) for row in complete]
    individual = score_stage("answer", samples, answer_scorers, checkpoint, configuration,
                             annotators, chunk_size, evaluate)
    by_key = {tuple(row[field] for field in ANSWER_KEY): row for row in rows}
    if len(by_key) != len(rows):
        raise ValueError("Duplicate answer case keys")
    pair_samples = []
    sources = sorted({key[:2] for key in by_key})
    for source_id, split in sources:
        for variant in variants:
            ja, en = (by_key.get((source_id, split, variant, language), {}) for language in ("ja", "en"))
            if generation_status(ja) != "completed" or generation_status(en) != "completed":
                continue
            pair_samples.append(ak.Sample(id="|".join((source_id, split, variant)), input=ja["prompt"],
                                          actual_output=ja["answer"], target=en["answer"]))
    paired = score_stage("pair", pair_samples, pair_scorers, checkpoint, configuration, (), chunk_size, evaluate)
    return individual, paired
