"""Cache llama-server and its shared libraries across Colab runtimes."""

import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import tempfile
import uuid
import zipfile


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_identity(commit, cmake_options):
    """Reuse builds only on a matching GPU, CUDA, CPU, and Linux environment."""
    def output(command):
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()

    # GGML/CUDA native builds can depend on both GPU and CPU instruction sets.
    cpu_lines = Path("/proc/cpuinfo").read_text().splitlines()
    cpu = {}
    for line in cpu_lines:
        key, separator, value = line.partition(":")
        if separator and key.strip() in ("model name", "flags"):
            cpu.setdefault(key.strip(), value.strip())
    return {
        "cache_format": 1, "llama_commit": commit, "cmake_options": list(cmake_options),
        "gpu": output(["nvidia-smi", "--query-gpu=name,compute_cap,driver_version",
                       "--format=csv,noheader"]),
        "cuda_compiler": output(["nvcc", "--version"]),
        "cxx_compiler": output(["c++", "--version"]),
        "machine": platform.machine(), "libc": list(platform.libc_ver()),
        "linux_release": Path("/etc/os-release").read_text(), "cpu": cpu,
    }


def cache_key(identity):
    encoded = json.dumps(identity, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24]


def validate_cuda_server(bin_dir):
    """Exercise the binary and its libraries before trusting a restored build."""
    executable = bin_dir / "llama-server"
    environment = os.environ.copy()
    environment["LD_LIBRARY_PATH"] = str(bin_dir) + os.pathsep + environment.get("LD_LIBRARY_PATH", "")
    devices = subprocess.check_output(
        [str(executable), "--list-devices"], text=True, stderr=subprocess.STDOUT,
        env=environment, timeout=60,
    )
    if not re.search(r"^\s*CUDA\d+:", devices, re.MULTILINE):
        raise RuntimeError(f"llama-server did not detect a CUDA GPU:\n{devices}")
    version = subprocess.check_output(
        [str(executable), "--version"], text=True, stderr=subprocess.STDOUT,
        env=environment, timeout=60,
    ).strip()
    return devices, version


def save_build(bin_dir, archive_path, identity):
    """Publish one complete archive; an interrupted copy cannot replace a good cache."""
    if not (bin_dir / "llama-server").is_file():
        raise FileNotFoundError(f"No compiled llama-server in {bin_dir}")
    # All runtime outputs share build/bin in this pinned llama.cpp version.
    # Reading symlinks as bytes also saves their library aliases as regular files.
    files = sorted(path for path in bin_dir.iterdir() if path.is_file())
    manifest = {"identity": identity, "files": {path.name: file_sha256(path) for path in files}}
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    pending = archive_path.with_name(f"{archive_path.name}.{uuid.uuid4().hex}.partial")
    try:
        with tempfile.TemporaryDirectory() as directory:
            local_archive = Path(directory) / "build.zip"
            with zipfile.ZipFile(local_archive, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("manifest.json", json.dumps(manifest, indent=2))
                for path in files:
                    archive.write(path, arcname=path.name)
            shutil.copyfile(local_archive, pending)
            pending.replace(archive_path)
    finally:
        pending.unlink(missing_ok=True)


def restore_build(archive_path, bin_dir, identity):
    """Check metadata and every file hash before replacing the local runtime files."""
    if not archive_path.is_file():
        return False
    bin_dir.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        if manifest["identity"] != identity:
            return False
        files = manifest["files"]
        if not files or "llama-server" not in files:
            raise ValueError("Build cache has no llama-server")
        # Cache entries are flat runtime files, never arbitrary extraction paths.
        if any(Path(name).name != name or name in (".", "..", "manifest.json") for name in files):
            raise ValueError("Invalid build cache filename")
        if sorted(archive.namelist()) != sorted(["manifest.json", *files]):
            raise ValueError("Build cache contents do not match its manifest")
        with tempfile.TemporaryDirectory(dir=bin_dir.parent) as directory:
            staged = Path(directory) / "bin"
            staged.mkdir()
            for name, expected_hash in files.items():
                destination = staged / name
                with archive.open(name) as source, destination.open("wb") as output:
                    shutil.copyfileobj(source, output)
                if file_sha256(destination) != expected_hash:
                    raise ValueError(f"Build cache checksum mismatch: {name}")
            (staged / "llama-server").chmod(0o755)
            if bin_dir.exists():
                shutil.rmtree(bin_dir)
            staged.replace(bin_dir)
    return True
