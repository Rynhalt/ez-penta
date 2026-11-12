import types
import torch
from torch.utils.data import DataLoader

from circuitkit.backends.eap.eap_utils import (
    _build_eap_ig_input_endpoints,
    collate_EAP_with_position_maps,
    _map_corrupted_positions,
    make_hooks_and_matrices,
)
from circuitkit.backends.eap import attribute as attribute_module
from circuitkit.backends.eap import attribute_node as attribute_node_module
from circuitkit.backends.eap.attribute import get_scores_eap_ig
from circuitkit.backends.eap.graph import Graph as TLGraph


def test_map_gathers_only_explicit_correspondences():
    acts = torch.arange(1 * 5 * 2, dtype=torch.float32).reshape(1, 5, 2)
    aligned, valid = _map_corrupted_positions(acts, [([0, 1, 4], [0, 2, 4])])
    assert valid.tolist() == [[True, True, False, False, True]]
    assert torch.equal(aligned[0, 0], acts[0, 0])
    assert torch.equal(aligned[0, 1], acts[0, 2])
    assert torch.equal(aligned[0, 2:4], torch.zeros(2, 2))
    assert torch.equal(aligned[0, 4], acts[0, 4])


def test_map_rejects_duplicate_and_out_of_range_indices():
    acts = torch.zeros(1, 5, 2)
    for m in [([0, 0], [1, 2]), ([0, 5], [1, 2]), ([0, 1], [1, 1])]:
        try:
            _map_corrupted_positions(acts, [m])
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid map accepted: {m}")


def test_eap_ig_endpoints_preserve_clean_unmapped_positions():
    clean = torch.full((1, 5, 2), 10.0)
    aligned_corrupt = torch.full((1, 5, 2), 20.0)
    valid = torch.tensor([[True, True, False, False, True]])
    clean_endpoint, corrupt_endpoint = _build_eap_ig_input_endpoints(clean, aligned_corrupt, valid)
    assert torch.equal(clean_endpoint, clean)
    assert torch.equal(corrupt_endpoint[0, [0, 1, 4]], torch.full((3, 2), 20.0))
    assert torch.equal(corrupt_endpoint[0, [2, 3]], torch.full((2, 2), 10.0))


def test_map_collator_preserves_variable_length_maps():
    class CharTokenizer:
        def __call__(self, text, add_special_tokens=False):
            assert add_special_tokens is False
            return {"input_ids": [ord(c) for c in text]}

    rows = [
        ("a--a", "a-b-a", 1, ((0, 3), (0, 4))),
        ("b-b", "b--b", -1, ((0, 2), (0, 3))),
    ]
    clean, corrupted, labels, maps = collate_EAP_with_position_maps(
        rows, tokenizer=CharTokenizer(), padding_side="left"
    )
    assert clean == ["a--a", "b-b"]
    assert corrupted == ["a-b-a", "b--b"]
    assert labels.tolist() == [1, -1]
    assert maps == [((1, 4), (0, 4)), ((2, 4), (1, 4))]


class Cfg:
    use_normalization_before_and_after = False
    d_model = 2
    d_mlp = 2
    device = "cpu"
    dtype = torch.float32


class Node:
    def __init__(self, name, out_hook=None, in_hook=None, qkv_inputs=None):
        self.name, self.out_hook, self.in_hook, self.qkv_inputs = name, out_hook, in_hook, qkv_inputs


class Graph:
    cfg = {"d_mlp": 2, "mlp_hook": "mlp_out", "n_layers": 1}
    n_forward = 4

    def __init__(self):
        self.nodes = {
            "input": Node("input", out_hook="input"),
            "a0.h0": Node("a0.h0", out_hook="attn", qkv_inputs=["q", "k", "v"]),
            "m0": Node("m0", out_hook="mlp", in_hook="mlp_in"),
            "logits": Node("logits", in_hook="logits_in"),
        }
        self.indices = {"input": 0, "a0.h0": 1, "m0": 2, "logits": 3}

    def forward_index(self, node): return self.indices[node.name]
    def backward_index(self, node, qkv=None): return {"q": 0, "k": 1, "v": 2}.get(qkv, 3)
    def prev_index(self, node): return self.indices[node.name]


class Model:
    cfg = Cfg()


def test_explicitly_rejects_unimplemented_separate_activation_mode():
    model = Model()
    model.cfg = types.SimpleNamespace(
        use_normalization_before_and_after=True,
        d_model=2,
        d_mlp=2,
        device="cpu",
        dtype=torch.float32,
    )
    try:
        make_hooks_and_matrices(model, Graph(), 1, 5, None, position_maps=[([0], [0])])
    except NotImplementedError as exc:
        assert "separate-activation" in str(exc)
    else:
        raise AssertionError("unsupported normalization branch must fail explicitly")


def test_hooks_gather_differences_and_mask_unmapped_gradients():
    graph = Graph()
    maps = [([0, 1, 4], [0, 2, 4])]
    state = {}
    scores = torch.zeros(4, 4)
    hooks, delta = make_hooks_and_matrices(Model(), graph, 1, 5, scores, position_maps=maps, alignment_state=state)
    corrupt = torch.arange(10, dtype=torch.float32).reshape(1, 5, 2)
    clean = corrupt + 100
    for _, fn in hooks[0]: fn(corrupt, types.SimpleNamespace(name="corrupt"))
    for _, fn in hooks[1]: fn(clean, types.SimpleNamespace(name="clean"))
    assert state["valid_positions"].tolist() == [[True, True, False, False, True]]
    expected = torch.zeros_like(clean)
    expected[0, 0] = corrupt[0, 0] - clean[0, 0]
    expected[0, 1] = corrupt[0, 2] - clean[0, 1]
    expected[0, 4] = corrupt[0, 4] - clean[0, 4]
    assert torch.equal(delta[0, :, 0], expected[0])

    # Drive a real backward hook on the MLP input and verify only mapped sites contribute.
    bwd_name, bwd_fn = hooks[2][-2]
    grads = torch.ones(1, 5, 2)
    bwd_fn(grads, types.SimpleNamespace(name=bwd_name))
    assert torch.isfinite(scores).all()
    expected_score = expected.sum()
    assert torch.allclose(scores[:2, 3], expected_score.expand(2))


def test_legacy_hook_path_keeps_same_position_subtraction():
    graph = Graph()
    scores = torch.zeros(4, 4)
    hooks, delta = make_hooks_and_matrices(Model(), graph, 1, 5, scores)
    corrupt = torch.arange(10, dtype=torch.float32).reshape(1, 5, 2)
    clean = corrupt + 100
    for _, fn in hooks[0]: fn(corrupt, types.SimpleNamespace(name="corrupt"))
    for _, fn in hooks[1]: fn(clean, types.SimpleNamespace(name="clean"))
    assert torch.equal(delta[0, :, 0], corrupt[0] - clean[0])


def test_full_eap_ig_smoke_with_explicit_maps(monkeypatch):
    """Run CircuitKIT's actual EAP-IG loop on a random 1-layer TL model."""
    from transformer_lens import HookedTransformer, HookedTransformerConfig

    cfg = HookedTransformerConfig(
        n_layers=1,
        n_heads=1,
        d_model=16,
        d_head=16,
        d_mlp=32,
        d_vocab=32,
        n_ctx=8,
        act_fn="gelu",
        normalization_type="LN",
    )
    model = HookedTransformer(cfg)
    model.cfg.use_attn_result = True
    model.cfg.use_split_qkv_input = True
    model.cfg.use_hook_mlp_in = True
    model.cfg.n_key_value_heads = None
    model.eval()
    graph = TLGraph.from_model(model)

    def fake_tokenize(_model, clean, corrupted, **_kwargs):
        c = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
        r = torch.tensor([[1, 7, 8, 4, 5]], dtype=torch.long)
        mask = torch.ones_like(c)
        return c, r, mask, mask, torch.tensor([5]), 5

    monkeypatch.setattr(attribute_module, "tokenize_batch_pair", fake_tokenize)
    rows = [("clean", "corrupt", 0, ((0, 1, 2, 3, 4), (0, 1, 2, 3, 4)))]
    mapped_dl = DataLoader(
        rows,
        batch_size=1,
        collate_fn=lambda xs: (["clean"], ["corrupt"], torch.tensor([0]), [xs[0][3]]),
    )
    plain_dl = [( ["clean"], ["corrupt"], torch.tensor([0]) )]

    def metric(logits, clean_logits, lengths, labels):
        del clean_logits, lengths, labels
        return logits[:, -1, 1].sum()

    scores = get_scores_eap_ig(model, graph, mapped_dl, metric, steps=2, quiet=True)
    assert torch.isfinite(scores).all()
    assert float(scores.abs().sum()) > 0.0
    baseline_scores = get_scores_eap_ig(model, graph, plain_dl, metric, steps=2, quiet=True)
    assert torch.allclose(scores, baseline_scores, atol=1e-6, rtol=1e-5)
    partial_dl = [(["clean"], ["corrupt"], torch.tensor([0]), [((0, 3, 4), (0, 3, 4))])]
    partial_scores = get_scores_eap_ig(model, graph, partial_dl, metric, steps=2, quiet=True)
    assert torch.isfinite(partial_scores).all()


def test_public_node_eap_ig_path_accepts_explicit_maps(monkeypatch):
    """Exercise the EAP-IG implementation called by discover_circuit."""
    from transformer_lens import HookedTransformer, HookedTransformerConfig

    cfg = HookedTransformerConfig(
        n_layers=1, n_heads=1, d_model=16, d_head=16, d_mlp=32,
        d_vocab=32, n_ctx=8, act_fn="gelu", normalization_type="LN",
    )
    model = HookedTransformer(cfg)
    model.cfg.use_attn_result = True
    model.cfg.use_split_qkv_input = True
    model.cfg.use_hook_mlp_in = True
    model.cfg.n_key_value_heads = None
    model.eval()
    graph = TLGraph.from_model(model, node_scores=True)

    def fake_tokenize(_model, clean, corrupted, **_kwargs):
        c = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
        r = torch.tensor([[1, 7, 8, 4, 5]], dtype=torch.long)
        mask = torch.ones_like(c)
        return c, r, mask, mask, torch.tensor([5]), 5

    monkeypatch.setattr(attribute_node_module, "tokenize_batch_pair", fake_tokenize)
    mapped_dl = [(
        ["clean"], ["corrupt"], torch.tensor([[1, 1, 3, 4, 5]]),
        [((0, 3, 4), (0, 3, 4))],
    )]

    def metric(logits, clean_logits, lengths, labels):
        del clean_logits, lengths, labels
        return logits[:, -1, 1].sum()

    scores = attribute_node_module.get_scores_eap_ig(
        model, graph, mapped_dl, metric, steps=2, quiet=True
    )
    assert torch.isfinite(scores).all()
    assert float(scores.abs().sum()) > 0.0
