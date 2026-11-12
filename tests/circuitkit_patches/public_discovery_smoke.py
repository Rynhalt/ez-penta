"""Synthetic CPU smoke of the public CircuitKIT discover_circuit API.

This exercises the exact pinned `attribute_node.py` EAP-IG path, custom TaskSpec
collator, position maps, score artifact serialization, and signed-edge capture.
It is not Tiny Aya or research evidence.
"""
from __future__ import annotations

import argparse
import importlib
from importlib import metadata
import json
import os
from pathlib import Path
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--circuitkit-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--source-commit", default="e3501bc72b349963db76df324f5dbd41ced2a02f")
    args = parser.parse_args()
    source = args.circuitkit_source.resolve()
    if not (source / "src/circuitkit").is_dir():
        raise FileNotFoundError(f"No CircuitKIT source package at {source}")
    sys.path.insert(0, str(source / "src"))
    repo_root = args.repo_root.resolve()
    sys.path.insert(0, str(repo_root))

    import torch
    import circuitkit as ck
    from transformers import GPT2TokenizerFast
    from transformer_lens import HookedTransformer, HookedTransformerConfig
    from circuitkit.tasks.registry import _TASKS, register_task
    from eval.track2_extension.circuitkit_data import RefusalComplianceTaskSpec

    torch.set_num_threads(2)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.chat_template = (
        "{% for message in messages %}"
        "{% if message['role'] == 'user' %}User: {{ message['content'] }}\\n"
        "{% elif message['role'] == 'assistant' %}Assistant: {{ message['content'] }}\\n"
        "{% endif %}{% endfor %}"
        "{% if add_generation_prompt %}Assistant:{% endif %}"
    )

    pair = {
        "data_mode": "synthetic_smoke",
        "seed_id": "synthetic-seed-0001",
        "clean_prompt": "A test request.",
        "corrupt_prompt": "A fake request.",
        "clean_answer": " I cannot help.",
        "corrupted_answer": " Here is help.",
    }
    task_name = "lexsi_public_api_synthetic_cpu_smoke"
    _TASKS.pop(task_name, None)
    register_task(RefusalComplianceTaskSpec(
        pairs=[pair], name=task_name, allow_synthetic_smoke=True,
    ))

    cfg = HookedTransformerConfig(
        n_layers=1, n_heads=1, d_model=16, d_head=16, d_mlp=32,
        d_vocab=tokenizer.vocab_size, n_ctx=256, act_fn="gelu",
        normalization_type="LN", model_name="gpt2", tokenizer_name="gpt2",
        use_attn_result=True, use_split_qkv_input=True, use_hook_mlp_in=True,
    )
    model = HookedTransformer(cfg, tokenizer=tokenizer).eval().to("cpu")
    # HookedTransformer may wrap tokenizer metadata; set the template on its own copy.
    model.tokenizer.chat_template = tokenizer.chat_template
    model.cfg.ungroup_grouped_query_attention = True

    output_path = args.output_dir / "synthetic_circuit.pt"
    signed_edges_path = args.output_dir / "raw_signed_edges.json"
    api = importlib.import_module("circuitkit.api")
    original_attribute_node = api.attribute_node

    def capture_signed_edges(*call_args, **call_kwargs):
        graph = call_args[1] if len(call_args) > 1 else call_kwargs["graph"]
        result = original_attribute_node(*call_args, **call_kwargs)
        rows = [
            {
                "edge": name,
                "signed_score": float(edge.score.detach().cpu()),
                "in_graph": bool(edge.in_graph),
            }
            for name, edge in graph.edges.items()
        ]
        signed_edges_path.write_text(json.dumps(rows, indent=2) + "\n")
        return result

    api.attribute_node = capture_signed_edges
    config = {
        "model": {"name": "gpt2", "precision": "float32", "device": "cpu"},
        "discovery": {
            "algorithm": "eap-ig", "task": task_name, "level": "node",
            "method": "EAP-IG-inputs", "batch_size": 2, "ig_steps": 2,
            "seed": 301, "max_seq_len": 256,
            "data_params": {"num_examples": 2, "seed": 301},
        },
        "pruning": {"target_sparsity": 0.30, "scope": "both", "intervention": "patching"},
        "output_path": str(output_path),
    }
    ck.discover_circuit(config, _model=model)
    expected = [
        output_path,
        output_path.with_name(output_path.stem + "_scores.pt"),
        output_path.with_name(output_path.stem + "_scores.json"),
        signed_edges_path,
    ]
    missing = [str(path) for path in expected if not path.is_file()]
    if missing:
        raise AssertionError(f"Public discovery failed to write expected artifacts: {missing}")
    edge_rows = json.loads(signed_edges_path.read_text())
    if not edge_rows or any(not isinstance(row["signed_score"], float) for row in edge_rows):
        raise AssertionError("Signed edge attribution capture is empty or malformed")
    versions = {
        name: metadata.version(name)
        for name in ("torch", "transformer-lens", "circuitkit", "transformers")
    }
    result = {
        "status": "synthetic_public_api_discovery_passed",
        "model": "random one-layer GPT2-shaped TransformerLens model",
        "data_mode": "synthetic_smoke",
        "algorithm": "eap-ig",
        "circuitkit_version": ck.__version__,
        "source_commit": args.source_commit,
        "versions": versions,
        "cuda_available": torch.cuda.is_available(),
        "n_signed_edge_scores": len(edge_rows),
        "artifacts": [str(path) for path in expected],
        "interpretation": "API integration only; no Tiny Aya, human-approved data, or mechanistic result.",
    }
    (args.output_dir / "smoke_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
