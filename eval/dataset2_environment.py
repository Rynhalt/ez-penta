"""Detect live-kernel package mixtures after Colab dependency installation."""

import importlib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import subprocess
import sys

MODULE_DISTRIBUTIONS = {
    "huggingface_hub": "huggingface_hub",
    "transformers": "transformers",
    "tokenizers": "tokenizers",
    "torch": "torch",
    "numpy": "numpy",
    "pandas": "pandas",
    "accelerate": "accelerate",
    "datasets": "datasets",
    "sentencepiece": "sentencepiece",
    "google.protobuf": "protobuf",
}


class RestartRequired(RuntimeError):
    """Installed files and already-loaded Python modules do not agree."""


def _installed(distribution):
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "not installed"


def _loaded_versions():
    return {
        module: str(
            getattr(sys.modules[module], "__version__", _installed(distribution))
        )
        for module, distribution in MODULE_DISTRIBUTIONS.items()
        if module in sys.modules and sys.modules[module] is not None
    }


def _restart(details):
    raise RestartRequired(
        "The running session has mixed dependency versions: "
        + "; ".join(details)
        + ". In Colab choose Runtime > Restart session, then run the notebook from the top. "
        "The installed packages and extracted files remain available. Drive checkpoints are preserved."
    )


def require_clean_environment():
    """Validate live modules, including the exact Hub import that failed in Colab."""
    differences = [
        f"{module}: memory={loaded}, installed={_installed(MODULE_DISTRIBUTIONS[module])}"
        for module, loaded in _loaded_versions().items()
        if loaded != _installed(MODULE_DISTRIBUTIONS[module])
    ]
    if differences:
        _restart(differences)
    try:
        hub_utils = importlib.import_module("huggingface_hub.utils")
        importlib.import_module("huggingface_hub.file_download")
        if not callable(getattr(hub_utils, "refresh_xet_connection_info", None)):
            raise ImportError("refresh_xet_connection_info is missing")
    except ImportError as error:
        _restart(
            [f"Hugging Face Hub imports are inconsistent ({type(error).__name__})"]
        )


def _pins(requirements):
    pins = {}
    for line in Path(requirements).read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        name, separator, expected = line.partition("==")
        if not separator or not name.strip() or not expected.strip():
            raise ValueError("Dataset2 requirements must use exact name==version pins")
        pins[name.strip()] = expected.strip()
    return pins


def _matches(actual, expected):
    # PEP 440: a public ==2.8.0 pin also accepts 2.8.0+cu128 wheels.
    return actual == expected or (
        "+" not in expected and actual.split("+", 1)[0] == expected
    )


def install_dependencies(requirements, installer=None):
    """Install only missing/different pins; request a restart only for actual mixtures."""
    pins = _pins(requirements)
    pending = {
        name: expected
        for name, expected in pins.items()
        if not _matches(_installed(name), expected)
    }
    if not pending:
        require_clean_environment()
        print(
            "Pinned dependencies already installed; skipped pip. Continue to authentication."
        )
        return False
    before = _loaded_versions()
    disk_before = {
        module: _installed(MODULE_DISTRIBUTIONS[module]) for module in before
    }
    print(
        "Installing required versions: "
        + ", ".join(f"{name}=={expected}" for name, expected in pending.items()),
        flush=True,
    )
    (installer or subprocess.check_call)(
        [sys.executable, "-m", "pip", "install", "-q", "-r", str(requirements)]
    )
    importlib.invalidate_caches()
    remaining = [
        name
        for name, expected in pins.items()
        if not _matches(_installed(name), expected)
    ]
    if remaining:
        raise RuntimeError(
            "Dependency installation did not establish required pins: "
            + ", ".join(remaining)
        )
    changed = [
        f"{module}: loaded={loaded}, installed={_installed(MODULE_DISTRIBUTIONS[module])}"
        for module, loaded in before.items()
        if loaded != _installed(MODULE_DISTRIBUTIONS[module])
        or disk_before[module] != _installed(MODULE_DISTRIBUTIONS[module])
    ]
    if changed:
        _restart(changed)
    require_clean_environment()
    print(
        "Dependencies and running imports are consistent; continue to authentication."
    )
    return True


def require_clean_gguf_environment():
    """GGUF inference does not use Hub/Transformers or inspect their live modules."""
    for module in ("numpy", "pandas", "auditkit"):
        loaded = sys.modules.get(module)
        if loaded is not None:
            actual = str(getattr(loaded, "__version__", _installed(module)))
            if actual != _installed(module):
                _restart([f"{module}: memory={actual}, installed={_installed(module)}"])


def install_gguf_dependencies(requirements, installer=None):
    """Retain compatible Colab array libraries; install only actual missing requirements."""
    from packaging.requirements import Requirement
    from packaging.version import InvalidVersion

    requirements = Path(requirements)
    specs = [
        Requirement(line.split("#", 1)[0].strip())
        for line in requirements.read_text().splitlines()
        if line.split("#", 1)[0].strip()
    ]

    def satisfied(spec):
        try:
            return spec.specifier.contains(_installed(spec.name), prereleases=True)
        except InvalidVersion:
            return False

    pending = [str(spec) for spec in specs if not satisfied(spec)]
    if pending:
        before = {
            module: _installed(module)
            for module in ("numpy", "pandas", "auditkit")
            if module in sys.modules
        }
        print("Installing GGUF dependencies: " + ", ".join(pending), flush=True)
        (installer or subprocess.check_call)(
            [sys.executable, "-m", "pip", "install", "-q", "-r", str(requirements)]
        )
        importlib.invalidate_caches()
        if any(not satisfied(spec) for spec in specs):
            raise RuntimeError(
                "GGUF dependency installation did not satisfy requirements"
            )
        changed = [
            module for module, old in before.items() if old != _installed(module)
        ]
        if changed:
            _restart(changed)
    require_clean_gguf_environment()
    print(
        "GGUF dependencies ready. Existing compatible libraries retained; no Hub/Transformers installation."
    )
    return bool(pending)
