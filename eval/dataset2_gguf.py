"""Dataset2's native llama.cpp GGUF backend, reusing the earlier Colab launcher/cache."""

from contextlib import ExitStack
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import urllib.request
import uuid

import chat
from colab_build_cache import (
    build_identity,
    cache_key,
    restore_build,
    save_build,
    validate_cuda_server,
)
from eval.checkpoints import config_id, file_sha256
from local_server import local_server, available_port, PortInUseError

LLAMA_COMMIT = "d1044bc82712fbf08996080bef6298b24a27e9c7"
MODEL_SPECS = {
    "aya": {
        "model_id": "CohereLabs/tiny-aya-water-GGUF",
        "revision": "bfe7fe10f8f206192097eeedcecdfbb1a11f9ca7",
        "filename": "tiny-aya-water-q4_k_m.gguf",
        "quantization": "Q4_K_M",
        "size": 2143977056,
        "sha256": "ecc6569cac41b36f4a06fdf9ccf5b315ec6f8efe44c3dc828f5e0a644dfbe8d9",
    },
    "guard": {
        "model_id": "QuantFactory/Qwen3Guard-Gen-0.6B-GGUF",
        "revision": "be4e562404e1bd6d6e01bed9c5fcf1ddb4a6c6aa",
        "filename": "Qwen3Guard-Gen-0.6B.Q8_0.gguf",
        "quantization": "Q8_0",
        "size": 804753408,
        "sha256": "a5108bc29824fbe805d4674cf8af438bb7221186c2fa940175a8dfdda4a8d2cc",
    },
}
GGUF_GENERATION = {
    "temperature": 0.0,
    "top_k": 0,
    "top_p": 1.0,
    "min_p": 0.0,
    "repeat_penalty": 1.0,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "seed": 42,
    "cache_prompt": False,
    "return_tokens": True,
    "stream": False,
    "stop": [],
}


def verify_model(path, spec):
    path = Path(path)
    return (
        path.is_file()
        and path.stat().st_size == spec["size"]
        and file_sha256(path) == spec["sha256"]
    )


def download_model(spec, destination, open_url=None):
    """Stream/resume the public immutable GGUF, then verify size and publisher LFS hash."""
    open_url = open_url or urllib.request.urlopen
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pending = destination.with_name(destination.name + ".partial")
    offset = pending.stat().st_size if pending.exists() else 0
    if offset > spec["size"]:
        raise ValueError(f"Oversized partial model: {pending}")
    if offset < spec["size"]:
        url = f"https://huggingface.co/{spec['model_id']}/resolve/{spec['revision']}/{spec['filename']}"
        request = urllib.request.Request(
            url, headers={"Range": f"bytes={offset}-"} if offset else {}
        )
        with open_url(request, timeout=120) as response:
            resumed = offset > 0 and response.status == 206
            if resumed and not response.headers.get("Content-Range", "").startswith(
                f"bytes {offset}-"
            ):
                raise ValueError("Model download returned the wrong byte range")
            written = offset if resumed else 0
            announced = written
            with pending.open("ab" if resumed else "wb") as output:
                while True:
                    block = response.read(8 * 1024 * 1024)
                    if not block:
                        break
                    output.write(block)
                    written += len(block)
                    if written > spec["size"]:
                        raise ValueError("Downloaded model exceeds pinned size")
                    if written - announced >= 128 * 1024 * 1024:
                        print(
                            f"{spec['filename']}: {written/1e9:.2f}/{spec['size']/1e9:.2f} GB",
                            flush=True,
                        )
                        announced = written
                output.flush()
                os.fsync(output.fileno())
    if not verify_model(pending, spec):
        raise ValueError(
            f"Model checksum/size mismatch: {pending}; preserve it for inspection before retrying"
        )
    os.replace(pending, destination)
    return destination


def prepare_model(
    role, cache_root, local_root="/content/models", local_path=None, download=None
):
    """Prefer local files, then persistent Drive models, then a public pinned download."""
    spec = MODEL_SPECS[role]
    destination = Path(local_root) / role / spec["filename"]
    if verify_model(destination, spec):
        if cache_root is not None:
            cached = Path(cache_root) / spec["sha256"][:16] / spec["filename"]
            if not cached.exists():
                cached.parent.mkdir(parents=True, exist_ok=True)
                pending = cached.with_name(cached.name + ".partial")
                shutil.copyfile(destination, pending)
                if not verify_model(pending, spec):
                    raise ValueError("Persistent model copy failed verification")
                os.replace(pending, cached)
        print("Reusing local GGUF:", destination, flush=True)
        return destination
    if destination.exists():
        raise ValueError(
            f"Existing local model has a different checksum: {destination}"
        )
    cached = (
        Path(cache_root) / spec["sha256"][:16] / spec["filename"]
        if cache_root is not None
        else None
    )
    source = (
        Path(local_path)
        if local_path is not None
        else (cached if cached is not None and cached.exists() else None)
    )
    if source is not None:
        if not verify_model(source, spec):
            raise ValueError(
                f"Local/cached GGUF does not match the pinned {role} model: {source}"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        pending = destination.with_name(destination.name + ".partial")
        shutil.copyfile(source, pending)
        if not verify_model(pending, spec):
            raise ValueError("Model copy failed checksum verification")
        os.replace(pending, destination)
        print("Copied cached GGUF to runtime:", destination, flush=True)
    else:
        print(
            "Downloading public GGUF:", spec["model_id"], spec["filename"], flush=True
        )
        # Retain partial downloads across runtime deletion, not only kernel restarts.
        download_target = cached if cached is not None else destination
        local_pending = destination.with_name(destination.name + ".partial")
        if cached is not None and local_pending.exists():
            cache_pending = cached.with_name(cached.name + ".partial")
            if not cache_pending.exists():
                cache_pending.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(local_pending, cache_pending)
        (download or download_model)(spec, download_target)
        if not verify_model(download_target, spec):
            raise ValueError("Downloaded GGUF does not match pinned identity")
        if download_target != destination:
            destination.parent.mkdir(parents=True, exist_ok=True)
            pending = destination.with_name(destination.name + ".partial")
            shutil.copyfile(download_target, pending)
            if not verify_model(pending, spec):
                raise ValueError("Model copy failed checksum verification")
            os.replace(pending, destination)
    if cached is not None and not cached.exists():
        cached.parent.mkdir(parents=True, exist_ok=True)
        pending = cached.with_name(cached.name + ".partial")
        shutil.copyfile(destination, pending)
        if not verify_model(pending, spec):
            raise ValueError("Persistent model copy failed verification")
        os.replace(pending, cached)
    return destination


def ensure_cuda_server(work="/content/llama.cpp", cache_root=None, force_rebuild=False):
    """Reuse the existing GPU-specific runtime/Drive build cache before compiling."""
    work = Path(work)
    options = [
        "-DGGML_CUDA=ON",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_CUDA_ARCHITECTURES=native",
        "-DLLAMA_BUILD_UI=OFF",
        "-DLLAMA_USE_PREBUILT_UI=OFF",
    ]
    try:
        identity = build_identity(LLAMA_COMMIT, options)
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError(
            "Select a Colab NVIDIA GPU runtime with nvcc before GGUF setup"
        ) from error
    key = cache_key(identity)
    binary = work / "build/bin"
    stamp = work / "ez-penta-build.json"
    archive = Path(cache_root) / f"{key}.zip" if cache_root is not None else None
    ready = False
    origin = "built"
    if not force_rebuild:
        try:
            if stamp.is_file() and json.loads(stamp.read_text()) == identity:
                devices, version = validate_cuda_server(binary)
                ready = True
                origin = "runtime"
            elif archive is not None and restore_build(archive, binary, identity):
                devices, version = validate_cuda_server(binary)
                ready = True
                origin = "drive"
        except Exception as error:
            print("Build cache needs rebuilding:", error, flush=True)
    if not ready:
        if shutil.which("cmake") is None:
            raise RuntimeError("cmake is missing from this Colab runtime")
        if not (work / ".git").exists():
            work.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "init", str(work)], check=True)
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(work),
                    "remote",
                    "add",
                    "origin",
                    "https://github.com/ggml-org/llama.cpp.git",
                ],
                check=True,
            )
        subprocess.run(
            ["git", "-C", str(work), "fetch", "--depth", "1", "origin", LLAMA_COMMIT],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(work), "checkout", "--detach", LLAMA_COMMIT], check=True
        )
        if (work / "build").exists():
            shutil.rmtree(work / "build")
        log = work / "dataset2-build.log"
        print("Building CUDA llama-server once; log:", log, flush=True)
        with log.open("w") as output:
            for command in [
                ["cmake", "-S", str(work), "-B", str(work / "build"), *options],
                [
                    "cmake",
                    "--build",
                    str(work / "build"),
                    "--target",
                    "llama-server",
                    "--parallel",
                    "2",
                ],
            ]:
                try:
                    subprocess.run(
                        command, stdout=output, stderr=subprocess.STDOUT, check=True
                    )
                except subprocess.CalledProcessError as error:
                    raise RuntimeError(
                        f"llama.cpp build failed; inspect {log}"
                    ) from error
        devices, version = validate_cuda_server(binary)
    if LLAMA_COMMIT[:7] not in version:
        raise RuntimeError(
            "llama-server version does not match the pinned source commit"
        )
    stamp.write_text(json.dumps(identity, indent=2) + "\n")
    if archive is not None and (origin == "built" or not archive.is_file()):
        save_build(binary, archive, identity)
    os.environ["PATH"] = str(binary) + os.pathsep + os.environ.get("PATH", "")
    os.environ["LD_LIBRARY_PATH"] = (
        str(binary) + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
    )
    result = {
        "llama_commit": LLAMA_COMMIT,
        "cmake_options": options,
        "identity": identity,
        "origin": origin,
        "version": version,
        "binary_sha256": file_sha256(binary / "llama-server"),
    }
    print("CUDA llama-server ready:", origin, version, flush=True)
    return result


class GGUFRunner:
    """One owned native model server, using the embedded template and exact token arrays."""

    def __init__(
        self,
        role,
        model_path,
        build,
        log_dir,
        port=None,
        context_size=8192,
        read_json=None,
        launcher=None,
    ):
        spec = MODEL_SPECS[role]
        if not verify_model(model_path, spec):
            raise ValueError(f"Invalid pinned {role} GGUF")
        self.role = role
        self.read = read_json or chat.read_json
        self.stack = ExitStack()
        self.build = build
        self.context_size = context_size
        if build["llama_commit"] != LLAMA_COMMIT:
            raise ValueError("Wrong llama.cpp commit")
        alias = spec["model_id"] + ":" + spec["quantization"]
        try:
            automatic_port = port is None
            for attempt in range(5):
                self.port = available_port() if automatic_port else port
                try:
                    self.base_url = self.stack.enter_context(
                        (launcher or local_server)(
                            alias,
                            self.port,
                            Path(log_dir) / (role + "-" + uuid.uuid4().hex + ".log"),
                            guard=role == "guard",
                            model_path=model_path,
                            gpu_layers="all",
                            context_size=context_size,
                        )
                    )
                    break
                except PortInUseError:
                    if not automatic_port or attempt == 4:
                        raise
                    print(
                        f"Port {self.port} became occupied; selecting another free port.",
                        flush=True,
                    )
            props = self.read(self.base_url + "/props")
            template = props.get("chat_template")
            if not isinstance(template, str) or not template:
                raise ValueError("Server did not expose its GGUF chat template")
            n_ctx = props.get("default_generation_settings", {}).get("n_ctx")
            if n_ctx != context_size:
                raise ValueError("Server context differs from requested context")
            self.configuration = {
                "backend": "llama.cpp-gguf",
                "model_id": spec["model_id"],
                "revision": spec["revision"],
                "filename": spec["filename"],
                "model_sha256": spec["sha256"],
                "quantization": spec["quantization"],
                "tokenizer_identity": spec["sha256"],
                "chat_template": template,
                "generation": dict(
                    GGUF_GENERATION, n_predict=128 if role == "guard" else 1024
                ),
                "context_size": context_size,
                "llama_commit": LLAMA_COMMIT,
                "build_options": build["cmake_options"],
                "prefill_assistant": role != "guard",
                "implementation": config_id(
                    [
                        file_sha256(__file__),
                        file_sha256(chat.__file__),
                        file_sha256(__import__("local_server").__file__),
                    ]
                ),
                "output_policy": "native-eos-only-complete-v1;array-input-no-extra-BOS",
                "batch_size": 1,
            }
        except BaseException:
            self.stack.close()
            raise

    def generate(self, messages, max_new_tokens=None):
        rendered = self.read(
            self.base_url + "/apply-template", {"messages": messages}, timeout=180
        )["prompt"]
        if not isinstance(rendered, str) or not rendered:
            raise ValueError("Invalid rendered chat prompt")
        ids = self.read(
            self.base_url + "/tokenize",
            {"content": rendered, "add_special": False, "parse_special": True},
            timeout=180,
        )["tokens"]
        if not isinstance(ids, list) or not ids or any(type(i) is not int for i in ids):
            raise ValueError("Invalid prompt token IDs")
        settings = dict(self.configuration["generation"])
        settings["n_predict"] = max_new_tokens or settings["n_predict"]
        if len(ids) + settings["n_predict"] > self.context_size:
            raise ValueError(
                "Prompt plus output exceeds context; no input truncation permitted"
            )
        reply = self.read(
            self.base_url + "/completion", {"prompt": ids, **settings}, timeout=600
        )
        text = reply.get("content")
        output = reply.get("tokens")
        stop = reply.get("stop_type")
        if reply.get("truncated"):
            raise ValueError("Server shifted/truncated context; result rejected")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Empty GGUF completion")
        if (
            not isinstance(output, list)
            or not output
            or any(type(i) is not int for i in output)
        ):
            raise ValueError("Server did not return exact completion token IDs")
        if stop not in ("eos", "limit"):
            raise RuntimeError(f"Unexpected GGUF stop state: {stop}")
        return {
            "answer": text.strip(),
            "rendered_input": rendered,
            "input_ids": ids,
            "output_ids": output,
            "input_tokens": len(ids),
            "output_tokens": len(output),
            "status": "completed" if stop == "eos" else "truncated",
            "finish_reason": stop,
            "truncated": stop != "eos",
            "native_generation_settings": reply.get("generation_settings"),
            "timings": reply.get("timings"),
            "runtime_provenance": self.build,
        }

    def answer(self, row):
        return self.generate([{"role": "user", "content": row["prompt"]}])

    def close(self):
        self.stack.close()
