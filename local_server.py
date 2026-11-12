"""Own a llama-server process and release it before loading another model."""

from contextlib import contextmanager
from pathlib import Path
import shutil
import socket
import errno
import subprocess
import time

import chat


class PortInUseError(RuntimeError):
    """An existing listener owns the requested port; never stop that process."""


def available_port():
    """Ask the OS for a loopback port; launcher still checks it before binding."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def require_free_port(port):
    if chat.server_status(f"http://127.0.0.1:{port}") != "offline":
        raise PortInUseError(f"Port {port} is occupied. Stop that server or choose another port.")
    # Health checks alone miss non-HTTP listeners. Probe actual bind availability.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        try:
            listener.bind(("127.0.0.1", port))
        except OSError as error:
            if error.errno == errno.EADDRINUSE:
                raise PortInUseError(f"Port {port} is occupied. Choose another port.") from error
            raise


@contextmanager
def local_server(model, port, log_path, guard=False, timeout=900, *, model_path=None, gpu_layers=None, context_size=4096):
    require_free_port(port)
    executable = shutil.which("llama-server")
    if executable is None:
        raise RuntimeError("llama-server is missing. Run bash run.sh --setup-only first.")
    # One inference slot limits memory use; reject overlong requests instead of shifting context.
    # Colab downloads an exact model revision first; macOS keeps its existing -hf path.
    if model_path is None:
        model_args = ["-hf", model]
    else:
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        model_args = ["-m", str(model_path), "--alias", model]
    command = [
        executable, *model_args, "--host", "127.0.0.1", "--port", str(port),
        "--ctx-size", str(context_size), "--parallel", "1", "--no-context-shift",
    ]
    if gpu_layers is not None:
        command.extend(["--gpu-layers", str(gpu_layers)])
    if guard:
        # The final assistant turn is evidence to classify, not text to continue.
        command.append("--no-prefill-assistant")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    base_url = f"http://127.0.0.1:{port}"
    process = None
    try:
        with log_path.open("x", encoding="utf-8") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        print(f"Loading {model}\nServer log: {log_path}", flush=True)
        deadline = time.monotonic() + timeout
        next_update = time.monotonic() + 15
        while True:
            if process.poll() is not None:
                # A listener may claim the port after our preflight bind probe.
                # Let automatic-port callers retry this specific startup failure.
                tail = log_path.read_text(errors="replace")[-16384:]
                if "address already in use" in tail.lower():
                    raise PortInUseError(f"Port {port} became occupied during startup.")
                raise RuntimeError(f"llama-server exited. Inspect {log_path}")
            status = chat.server_status(base_url)
            if status == "ready":
                break
            if status == "unknown" or time.monotonic() >= deadline:
                raise RuntimeError(f"Server did not become ready. Inspect {log_path}")
            if time.monotonic() >= next_update:
                print(f"Still downloading/loading; progress is in {log_path}", flush=True)
                next_update = time.monotonic() + 15
            time.sleep(1)
        # The caller can send HTTP requests only after the health check reports readiness.
        yield base_url
    finally:
        # Cleanup also runs when loading, generation, or judging is interrupted.
        chat.stop_owned_server(process)
