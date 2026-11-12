"""Content-addressed identities for resumable private evaluation runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_tree_fingerprint(root: str | Path) -> dict[str, Any]:
    """Hash every regular file under a local model directory deterministically."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Local model checkpoint directory does not exist: {root}")
    files = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        files.append({"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": digest.hexdigest()})
    if not files:
        raise ValueError(f"Local model checkpoint directory is empty: {root}")
    return {"file_count": len(files), "files": files, "sha256": canonical_hash(files)}


def assert_local_arm_paths(arms: Mapping[str, str], base_model_id: str) -> None:
    missing = {name: ref for name, ref in arms.items() if ref != base_model_id and not Path(ref).is_dir()}
    if missing:
        details = ", ".join(f"{name}={path}" for name, path in sorted(missing.items()))
        raise FileNotFoundError(
            "Required local model arm checkpoint(s) are missing before inference: " + details
        )
    for name, ref in arms.items():
        if ref != base_model_id:
            validate_local_model_checkpoint(ref, arm=name)


def validate_local_model_checkpoint(root: str | Path, *, arm: str = "model") -> None:
    """Check a local full HF checkpoint's manifest and weight containers pre-inference."""
    root = Path(root)
    config_path = root / "config.json"
    if not config_path.is_file():
        raise ValueError(f"{arm} checkpoint lacks config.json: {root}")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{arm} checkpoint has unreadable config.json: {root}") from exc
    if not isinstance(config, dict):
        raise ValueError(f"{arm} checkpoint config.json must contain an object: {root}")

    weight_files = sorted(root.glob("model*.safetensors")) + sorted(root.glob("pytorch_model*.bin"))
    index_files = [root / "model.safetensors.index.json", root / "pytorch_model.bin.index.json"]
    existing_indexes = [path for path in index_files if path.is_file()]
    if existing_indexes:
        declared = set()
        for index_path in existing_indexes:
            try:
                index = json.loads(index_path.read_text(encoding="utf-8"))
                declared.update(index["weight_map"].values())
            except (OSError, json.JSONDecodeError, KeyError, AttributeError, TypeError) as exc:
                raise ValueError(f"{arm} checkpoint has invalid shard index: {index_path.name}") from exc
        for shard in declared:
            shard_path = (root / shard).resolve()
            if root.resolve() not in shard_path.parents or not shard_path.is_file():
                raise ValueError(f"{arm} checkpoint shard missing or outside checkpoint root: {shard}")
        weight_files = [root / name for name in sorted(declared)]
    if not weight_files:
        raise ValueError(f"{arm} checkpoint has no full-model safetensors or PyTorch weight files: {root}")
    for path in weight_files:
        if path.stat().st_size <= 0:
            raise ValueError(f"{arm} checkpoint contains an empty weight file: {path.name}")
        if path.suffix == ".safetensors":
            try:
                from safetensors import safe_open

                with safe_open(str(path), framework="pt", device="cpu") as handle:
                    if not handle.keys():
                        raise ValueError(f"{arm} checkpoint weight file has no tensors: {path.name}")
            except Exception as exc:
                raise ValueError(f"{arm} checkpoint has invalid safetensors metadata: {path.name}") from exc


def generation_key(
    *,
    arm: str,
    model_fingerprint: Mapping[str, Any],
    tokenizer_fingerprint: Mapping[str, Any],
    prompt_id: str,
    input_token_hash: str,
    generation_config: Mapping[str, Any],
) -> str:
    return canonical_hash({
        "stage": "generation-v1",
        "arm": arm,
        "model": model_fingerprint,
        "tokenizer": tokenizer_fingerprint,
        "prompt_id": prompt_id,
        "input_token_hash": input_token_hash,
        "generation_config": generation_config,
    })


def judgment_key(
    *,
    prompt_id: str,
    answer_sha256: str,
    guard_model: str,
    guard_revision: str,
    guard_input_token_hash: str,
    guard_config: Mapping[str, Any],
) -> str:
    return canonical_hash({
        "stage": "judgment-v1",
        "prompt_id": prompt_id,
        "answer_sha256": answer_sha256,
        "guard_model": guard_model,
        "guard_revision": guard_revision,
        "guard_input_token_hash": guard_input_token_hash,
        "guard_config": guard_config,
    })


def assert_resume_compatible(
    records: Iterable[Mapping[str, Any]],
    expected_keys: Mapping[str, str],
    *,
    key_field: str,
    stage: str,
) -> set[str]:
    """Reject duplicate, stale, or provenance-free rows; return valid IDs."""
    seen: set[str] = set()
    for record in records:
        row_id = record.get("prompt_id")
        if not row_id:
            raise ValueError(f"{stage} resume record is missing prompt_id")
        if row_id in seen:
            raise ValueError(f"duplicate {stage} resume record for prompt_id={row_id}")
        seen.add(row_id)
        if row_id not in expected_keys:
            raise ValueError(f"stale {stage} resume record for unexpected prompt_id={row_id}")
        if record.get(key_field) != expected_keys[row_id]:
            raise ValueError(
                f"incompatible {stage} resume record for prompt_id={row_id}; "
                "use a new output directory rather than mixing run configurations"
            )
    return seen
