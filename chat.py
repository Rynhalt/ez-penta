"""A tiny command-line client for Tiny Aya, powered by a local llama.cpp server."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_MODEL = "CohereLabs/tiny-aya-water-GGUF:Q4_K_M"
MAX_EXCHANGES = 3  # Keep recent context modest for the 2,048-token server.


def read_json(url, data=None, timeout=10):
    """Make an HTTP request and decode the JSON response."""
    headers = {"Content-Type": "application/json"} if data is not None else {}
    body = json.dumps(data, ensure_ascii=False).encode("utf-8") if data is not None else None
    request = Request(url, data=body, headers=headers)
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


def server_status(base_url):
    """Return 'ready', 'loading', 'offline', or 'unknown'."""
    try:
        result = read_json(f"{base_url}/health", timeout=2)
        if not isinstance(result, dict):
            return "unknown"
        return {"ok": "ready", "loading": "loading"}.get(result.get("status"), "unknown")
    except HTTPError as error:
        return "loading" if error.code == 503 else "unknown"
    except (URLError, TimeoutError, OSError):
        return "offline"
    except (ValueError, TypeError):
        return "unknown"


def ensure_server(base_url, model, port, ctx_size):
    """Reuse an existing server, or start one owned by this program."""
    status = server_status(base_url)
    if status == "ready":
        print("Found an already-running llama.cpp server.")
        return None
    if status == "unknown":
        raise RuntimeError(f"A service on {base_url} did not behave like llama.cpp.")

    process = None
    if status == "offline":
        if shutil.which("llama-server") is None:
            raise RuntimeError("llama-server was not found. Install it with: brew install llama.cpp")
        Path("runs").mkdir(exist_ok=True)
        log_file = open("runs/llama-server.log", "a", encoding="utf-8")
        command = [
            "llama-server", "-hf", model,
            "--host", "127.0.0.1", "--port", str(port),
            "--ctx-size", str(ctx_size),
        ]
        print("Starting llama-server (model downloads automatically if needed).")
        print("Server output: runs/llama-server.log")
        try:
            process = subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT)
        finally:
            log_file.close()
    else:
        print("Server is already loading; checking for readiness.")

    try:
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            status = server_status(base_url)
            if status == "ready":
                print("Model is ready.")
                return process
            if status == "unknown":
                raise RuntimeError("The server returned an unexpected health response.")
            if process is not None and process.poll() is not None:
                raise RuntimeError("llama-server exited. Check runs/llama-server.log.")
            time.sleep(1)
        raise RuntimeError("Server did not become ready; inspect the server log.")
    except BaseException:
        stop_owned_server(process)  # Also handles Ctrl+C during first-time loading.
        raise


def loaded_model(base_url):
    models = read_json(f"{base_url}/v1/models")
    return models["data"][0]["id"]


def complete(base_url, messages, max_tokens):
    """Send conversation messages to the OpenAI-compatible local endpoint."""
    return read_json(
        f"{base_url}/v1/chat/completions",
        {"messages": messages, "temperature": 0.2, "max_tokens": max_tokens},
        timeout=180,
    )


def stop_owned_server(process):
    """Never stop a server that was already running before this script."""
    if process is not None and process.poll() is None:
        print("Stopping llama-server started by this script...")
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def main():
    parser = argparse.ArgumentParser(description="Chat locally with Tiny Aya via llama.cpp")
    parser.add_argument("--save", action="store_true", help="Save prompts and answers to runs/*.jsonl")
    parser.add_argument("--stats", action="store_true", help="Show inference token statistics")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face GGUF model selector")
    parser.add_argument("--port", type=int, default=8080, help="Local llama-server port")
    parser.add_argument("--ctx-size", type=int, default=2048, help="Context size when starting the server")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max generated tokens per response")
    args = parser.parse_args()

    base_url = f"http://127.0.0.1:{args.port}"
    process = None
    try:
        process = ensure_server(base_url, args.model, args.port, args.ctx_size)
        model_id = loaded_model(base_url)
        print(f"Model: {model_id}")
        if args.model.lower().split(":")[0] not in model_id.lower():
            print("Note: the existing server may be using a different model than --model.")
        print("Type a question in English or Japanese. /reset clears memory; /exit quits.")

        log_path = None
        if args.save:
            Path("runs").mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            log_path = Path("runs") / f"session-{timestamp}.jsonl"
            log_path.touch(exist_ok=False)
            print(f"Saving session data to {log_path} (local, git-ignored).")

        messages = []
        while True:
            try:
                prompt = input("\nYou> ").strip()
            except (KeyboardInterrupt, EOFError):
                print()
                break
            if prompt.lower() in ("/exit", "/quit"):
                break
            if prompt.lower() == "/reset":
                messages.clear()
                print("Conversation cleared.")
                continue
            if prompt.lower() == "/help":
                print("/reset clears conversation; /exit quits; --save enables JSONL logging.")
                continue
            if not prompt:
                continue

            messages.append({"role": "user", "content": prompt})
            try:
                result = complete(base_url, messages, args.max_tokens)
                answer = result["choices"][0]["message"]["content"] or ""
                if not isinstance(answer, str):
                    raise ValueError("Expected a text response from the model")
            except (HTTPError, URLError, OSError, KeyError, IndexError, ValueError, TypeError) as error:
                messages.pop()  # Do not retain an unsuccessful request.
                print(f"Request failed: {error}. Try /reset if the context is full.")
                continue

            print(f"\nAya> {answer}")
            if args.stats:
                usage = result.get("usage", {})
                speed = result.get("timings", {}).get("predicted_per_second")
                print(f"[{usage.get('prompt_tokens', '?')} prompt tokens, "
                      f"{usage.get('completion_tokens', '?')} response tokens" +
                      (f", {speed:.1f} tokens/sec" if isinstance(speed, (float, int)) else "") + "]")
            if log_path is not None:
                record = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "model": result.get("model", model_id),
                    "prompt": prompt,
                    "response": answer,
                    "messages": messages.copy(),
                    "usage": result.get("usage", {}),
                    "timings": result.get("timings", {}),
                    "finish_reason": result["choices"][0].get("finish_reason"),
                    "settings": {"temperature": 0.2, "max_tokens": args.max_tokens},
                }
                try:
                    with log_path.open("a", encoding="utf-8") as file:
                        file.write(json.dumps(record, ensure_ascii=False) + "\n")
                except OSError as error:
                    print(f"Could not save this response: {error}")
            messages.append({"role": "assistant", "content": answer})
            messages = messages[-2 * MAX_EXCHANGES:]
    except KeyboardInterrupt:
        print()
    except (RuntimeError, HTTPError, URLError, OSError, KeyError, IndexError, ValueError, TypeError) as error:
        print(f"Setup error: {error}")
        raise SystemExit(1) from error
    finally:
        stop_owned_server(process)


if __name__ == "__main__":
    main()
