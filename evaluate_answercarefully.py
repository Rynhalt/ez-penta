"""Run a small AnswerCarefully experiment with sequential local inference."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path

import auditkit as ak

from answercarefully_dataset import load_development_data, select_samples
import chat
from guard_judge import GUARD_MODEL, make_guard
from local_server import local_server, require_free_port


def write_json(path, data):
    with path.open("x", encoding="utf-8") as output:
        json.dump(data, output, ensure_ascii=False, indent=2)
        output.write("\n")


def create_run_directory(path=None):
    if path is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S-%fZ")
        path = Path("runs") / f"answercarefully-{timestamp}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def generate_responses(samples, base_url, output_path, max_tokens):
    model_id = chat.loaded_model(base_url)
    if "tiny-aya-water" not in model_id.lower():
        raise RuntimeError(f"Expected Tiny Aya Water, found {model_id}")
    pending = iter(samples)
    with output_path.open("x", encoding="utf-8") as output:
        def answer_prompts(prompts):
            answers = []
            for prompt in prompts:
                sample = next(pending)
                if sample.input != prompt:
                    raise ValueError("AuditKIT changed the expected prompt order")
                # Each evaluation request starts with a fresh conversation.
                response = chat.complete(base_url, [{"role": "user", "content": prompt}], max_tokens)
                choice = response["choices"][0]
                answer = choice["message"]["content"]
                if not isinstance(answer, str) or not answer.strip():
                    raise ValueError(f"Empty/non-text answer for {sample.id}")
                sample.actual_output = answer
                record = {"sample": asdict(sample), "generation": {
                    "model": model_id, "temperature": 0.2, "max_tokens": max_tokens,
                    "finish_reason": choice.get("finish_reason"), "usage": response.get("usage"),
                    "timings": response.get("timings"),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }}
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                output.flush()
                print(f"\n{sample.id}\nPrompt: {prompt}\nAya: {answer}", flush=True)
                print(f"Finish: {choice.get('finish_reason')}", flush=True)
                answers.append(answer)
            return answers

        result = ak.evaluate(
            samples, model=answer_prompts, scorers=[],
            config=ak.RunConfig(
                temperature=0.2, max_tokens=max_tokens, concurrency=1, max_retries=0,
                extra={"model": model_id, "started_at": datetime.now(timezone.utc).isoformat()},
            ),
        )
    if result.errors:
        raise RuntimeError(f"Generation failed; completed answers remain in {output_path}: {result.errors}")
    print(f"Responses saved: {output_path}", flush=True)


def load_responses(path):
    samples = []
    excluded = []
    seen_ids = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            sample_data = dict(record["sample"])
            sample_id = sample_data["id"]
            if sample_id in seen_ids:
                raise ValueError(f"Duplicate sample ID: {sample_id}")
            seen_ids.add(sample_id)
            finish_reason = record["generation"]["finish_reason"]
            if finish_reason != "stop":
                excluded.append({"sample_id": sample_id, "finish_reason": finish_reason})
                continue
            # JSON stores enum values as strings; restore the AuditKIT type.
            sample_data["kind"] = ak.TaskKind(sample_data.get("kind", "generative"))
            sample = ak.Sample(**sample_data)
            if not isinstance(sample.actual_output, str) or not sample.actual_output.strip():
                raise ValueError("Empty/non-text answer")
            sample.metadata["generation"] = record["generation"]
            samples.append(sample)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid response record at {path}:{line_number}: {error}") from error
    if not samples:
        raise ValueError(f"No completed answers in {path}; increase --max-tokens and regenerate")
    return samples, excluded


def evaluate_responses(samples, excluded, base_url, output_dir):
    guard = make_guard(base_url)
    # Grade the saved transcript without generating new Tiny Aya answers.
    result = ak.evaluate(
        samples, model="precomputed", scorers=[guard],
        config=ak.RunConfig(max_retries=0, extra={
            "started_at": datetime.now(timezone.utc).isoformat(),
            "guard_selector": GUARD_MODEL, "guard_quantization": "Q8_0",
            "prefill_assistant": False,
        }),
    )
    sample_metadata = {sample.id: sample.metadata for sample in samples}
    unsafe_count = 0
    scored_count = 0
    for prediction in result.predictions:
        prediction.metadata.update(sample_metadata[prediction.sample_id])
        for score in prediction.metadata.get("scores", []):
            if score["name"] != guard.name:
                continue
            verdict = score.get("metadata", {})
            if not verdict.get("unknown"):
                scored_count += 1
                unsafe_count += int(score["value"] == 1.0)
            print(f"{prediction.sample_id}: {verdict.get('label', 'Unknown')}"
                  f" | refusal={verdict.get('refusal')}", flush=True)
    result.metadata.update({
        "auditkit_version": version("auditkit"), "excluded_generations": excluded,
    })
    summary = {
        "generated_count": len(samples) + len(excluded),
        "scored_count": scored_count,
        "safe_count": scored_count - unsafe_count,
        "unsafe_count": unsafe_count,
        "unsafe_rate": unsafe_count / scored_count if scored_count else None,
        "unscored_count": len(samples) - scored_count,
        "excluded_generation_count": len(excluded),
        "excluded_generations": excluded,
        "judging_error_count": len(result.errors),
        "guard_model": GUARD_MODEL,
    }
    write_json(output_dir / "auditkit-guard-results.json", result.to_dict())
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"Results: {output_dir}", flush=True)
    return result


def run_experiment(args):
    # Fail before downloading data or models if an existing service owns a port.
    require_free_port(args.aya_port)
    require_free_port(args.guard_port)
    print("Loading AnswerCarefully v2.2 development data...", flush=True)
    dataset, revision = load_development_data()
    samples = select_samples(dataset, revision, args.limit)
    output_dir = create_run_directory(args.output_dir)
    write_json(output_dir / "run-settings.json", {
        "dataset": "llm-jp/AnswerCarefully", "configuration": "v2.2", "split": "dev",
        "dataset_revision": revision, "sample_ids": [sample.id for sample in samples],
        "aya_model": chat.DEFAULT_MODEL, "temperature": 0.2, "max_tokens": args.max_tokens,
        "guard_model": GUARD_MODEL, "guard_temperature": 0.0,
        "context_size": 4096, "parallel_slots": 1, "seed": None,
        "auditkit_version": version("auditkit"),
    })
    responses_path = output_dir / "tiny-aya-responses.jsonl"
    with local_server(chat.DEFAULT_MODEL, args.aya_port, output_dir / "tiny-aya-server.log") as base_url:
        generate_responses(samples, base_url, responses_path, args.max_tokens)
    samples, excluded = load_responses(responses_path)
    with local_server(GUARD_MODEL, args.guard_port, output_dir / "qwen3guard-server.log", guard=True) as base_url:
        return evaluate_responses(samples, excluded, base_url, output_dir)


def score_saved_responses(args):
    require_free_port(args.aya_port)
    require_free_port(args.guard_port)
    samples, excluded = load_responses(args.responses)
    output_dir = create_run_directory(args.output_dir)
    with local_server(GUARD_MODEL, args.guard_port, output_dir / "qwen3guard-server.log", guard=True) as base_url:
        return evaluate_responses(samples, excluded, base_url, output_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Load data, generate answers, and evaluate them")
    run.add_argument("--limit", type=int, choices=range(1, 6), default=3)
    run.add_argument("--max-tokens", type=int, default=512)
    score = commands.add_parser("score", help="Evaluate an existing responses JSONL file")
    score.add_argument("--responses", type=Path, required=True)
    for command in (run, score):
        command.add_argument("--aya-port", type=int, default=8080)
        command.add_argument("--guard-port", type=int, default=8081)
        command.add_argument("--output-dir", type=Path, help="New output directory; defaults to a timestamp")
    args = parser.parse_args()
    if not all(1 <= port <= 65535 for port in (args.aya_port, args.guard_port)):
        parser.error("Ports must be between 1 and 65535")
    if args.aya_port == args.guard_port:
        parser.error("Tiny Aya and the guard must use different ports")
    if args.command == "run" and args.max_tokens <= 0:
        parser.error("--max-tokens must be positive")
    os.environ["XDG_CACHE_HOME"] = str(Path("runs/cache").resolve())
    try:
        result = run_experiment(args) if args.command == "run" else score_saved_responses(args)
    except KeyboardInterrupt:
        parser.exit(130, "\nInterrupted; completed records remain in the output directory.\n")
    except Exception as error:
        parser.exit(1, f"Error: {error}\n")
    if result.errors:
        parser.exit(1, "Some verdicts were unscored; inspect summary.json and auditkit-guard-results.json.\n")


if __name__ == "__main__":
    main()
