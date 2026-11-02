"""Five short-answer examples evaluated through the existing local chat client."""

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path

import auditkit as ak

import chat


def samples():
    return [
        ak.Sample(id="addition_en", input="What is 2 + 2? Reply with only the number.", target="4"),
        ak.Sample(id="addition_ja", input="2 + 2 はいくつですか？数字だけで答えてください。", target="4"),
        ak.Sample(id="addition_mixed", input="2 + 2 のanswerは？数字だけでreplyしてください。", target="4"),
        ak.Sample(id="capital_en", input="What is the capital of Japan? Reply with only the English city name.", target="Tokyo"),
        ak.Sample(id="capital_ja", input="日本の首都はどこですか？日本語の都市名だけで答えてください。", target="東京"),
    ]


def make_model_adapter(base_url, max_tokens):
    """Translate AuditKIT's batch of strings into independent chat requests."""
    def model_adapter(prompts):
        answers = []
        for prompt in prompts:
            # Keep benchmark examples independent of conversation history.
            response = chat.complete(base_url, [{"role": "user", "content": prompt}], max_tokens)
            answer = response["choices"][0]["message"]["content"]
            if not isinstance(answer, str):
                raise ValueError("Expected a text response from the model")
            answers.append(answer)
        return answers

    return model_adapter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--max-tokens", type=int, default=32)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or args.max_tokens <= 0:
        parser.error("Use a port between 1 and 65535 and a positive --max-tokens")

    base_url = f"http://127.0.0.1:{args.port}"
    if chat.server_status(base_url) != "ready":
        parser.exit(1, "Start llama-server first; see README.md for the command.\n")
    model_id = chat.loaded_model(base_url)
    started_at = datetime.now(timezone.utc).isoformat()
    # A new timestamp prevents AuditKIT from replaying an older cached run.
    os.environ["XDG_CACHE_HOME"] = str(Path("runs/cache").resolve())
    config = ak.RunConfig(
        temperature=0.2, max_tokens=args.max_tokens, concurrency=1, max_retries=0,
        extra={"model_id": model_id, "base_url": base_url, "started_at": started_at},
    )
    print(f"AuditKIT {version('auditkit')} | Model: {model_id}", flush=True)
    result = ak.evaluate(
        samples(), model=make_model_adapter(base_url, args.max_tokens),
        scorers=["exact_match"], config=config,
    )
    result.metadata.update({"auditkit_version": version("auditkit"), "started_at": started_at})
    for prediction in result.predictions:
        print(f"\n{prediction.sample_id}: {prediction.prompt}")
        print(f"  Expected: {prediction.expected!r}")
        print(f"  Prediction: {prediction.raw_output!r}")
        print(f"  Exact match: {prediction.score}")
    print(f"\n{result.summary()}")
    if result.errors:
        print(f"Errors: {json.dumps(result.errors, ensure_ascii=False)}")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    output_path = Path("runs") / f"auditkit-demo-{timestamp}.json"
    with output_path.open("x", encoding="utf-8") as output:
        json.dump(result.to_dict(), output, ensure_ascii=False, indent=2)
        output.write("\n")
    print(f"Saved: {output_path}")
    if result.errors or result.failed_count:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
