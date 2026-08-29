"""Tests for the Dual-AR gradient checkpointing patch."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("loralib")

import loralib as lora  # noqa: E402

from fish_studio.training.lora_patch import (  # noqa: E402
    _gate_slow_linear,
    _gated_lora_linear_forward,
    _non_reentrant_checkpoint,
    _replace_embedding,
    _scale_vq_embeddings,
    _tied_lora_logits,
    apply_position_gate,
    freeze_semantic_id_adapter,
    lora_active_after_first_im_end,
    set_lora_position_mask,
    setup_lora,
)

R = 32
DIM = 256


def _linear_factor_norm() -> float:
    """The scale loralib gives the non-zero factor of a Linear adapter."""
    return float(lora.Linear(in_features=DIM, out_features=DIM, r=R).lora_A.norm())


class _Block(torch.nn.Module):
    """Stands in for a transformer layer holding trainable LoRA weights."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(4, 4, bias=False)

    def forward(self, x):
        return self.linear(x)


def _frozen_input() -> torch.Tensor:
    """Mimic the output of the frozen embeddings feeding the slow layers."""
    tensor = torch.ones(2, 4)
    assert not tensor.requires_grad
    return tensor


def test_reentrant_checkpointing_builds_no_graph_for_frozen_inputs() -> None:
    """Control: the upstream behaviour that silently zeroed the slow LoRA weights."""
    block = _Block()

    with pytest.warns(UserWarning, match="None of the inputs have requires_grad"):
        out = torch.utils.checkpoint.checkpoint(block, _frozen_input(), use_reentrant=True)

    assert out.grad_fn is None


def test_patched_checkpointing_keeps_gradients_for_frozen_inputs() -> None:
    block = _Block()

    out = _non_reentrant_checkpoint(block, _frozen_input())
    out.sum().backward()

    assert block.linear.weight.grad is not None
    assert block.linear.weight.grad.abs().sum().item() > 0


def test_patched_checkpointing_overrides_a_reentrant_request() -> None:
    block = _Block()

    out = _non_reentrant_checkpoint(block, _frozen_input(), use_reentrant=True)
    out.sum().backward()

    assert block.linear.weight.grad is not None


def test_stock_embedding_adapter_starts_far_out_of_scale() -> None:
    """Control: the upstream seeding this patch exists to correct."""
    stock = lora.Embedding(num_embeddings=1000, embedding_dim=DIM, r=R)

    # A is zero and B carries unit-variance noise with no fan-in term, so the
    # factor that scales the other one's gradient is orders of magnitude too big.
    assert float(stock.lora_A.norm()) == 0.0
    assert float(stock.lora_B.norm()) > 20 * _linear_factor_norm()


def test_embedding_adapter_matches_the_linear_update_scale() -> None:
    """One learning rate has to mean the same thing for both kinds of matrix."""
    embed = _replace_embedding(torch.nn.Embedding(1000, DIM), _config())

    ratio = float(embed.lora_B.norm()) / _linear_factor_norm()
    assert 0.8 < ratio < 1.25, f"embedding factor is {ratio:.2f}x the linear one"


def test_embedding_scale_does_not_drift_with_the_table_size() -> None:
    """Vocabulary size must not leak into the update scale, or every table differs.

    The audio decoder's codebook table is 40960 rows against 2560 for the text
    side; a fan-in read off the wrong axis makes those two move at different
    speeds under one learning rate.
    """
    small = _replace_embedding(torch.nn.Embedding(100, DIM), _config())
    large = _replace_embedding(torch.nn.Embedding(40960, DIM), _config())

    assert abs(float(small.lora_B.norm()) - float(large.lora_B.norm())) < 0.3


class _DualAR(torch.nn.Module):
    """The Dual-AR surface setup_lora touches."""

    def __init__(self) -> None:
        super().__init__()
        self.embeddings = torch.nn.Embedding(50, DIM)
        self.codebook_embeddings = torch.nn.Embedding(64, DIM)
        self.fast_embeddings = torch.nn.Embedding(16, DIM)
        self.layers = torch.nn.ModuleList()
        self.fast_layers = torch.nn.ModuleList()


def test_embeddings_target_arms_the_text_table_only() -> None:
    model = _DualAR()

    setup_lora(model, _config(targets=["embeddings"]))

    assert isinstance(model.embeddings, lora.Embedding)
    assert not isinstance(model.codebook_embeddings, lora.Embedding)
    assert not isinstance(model.fast_embeddings, lora.Embedding)


def test_semantic_id_rows_get_no_lora_gradient() -> None:
    embed = _replace_embedding(torch.nn.Embedding(20, DIM), _config())
    # Force a non-zero A so we can see the freeze clear it.
    embed.lora_A.data.normal_()
    freeze_semantic_id_adapter(embed, begin=12, end=15)
    # loralib Embedding: A is [r, vocab].
    assert float(embed.lora_A.data[:, 12:16].abs().sum()) == 0.0

    tokens = torch.tensor([1, 12, 14, 3])
    embed(tokens).sum().backward()
    assert embed.lora_A.grad is not None
    assert float(embed.lora_A.grad[:, 12:16].abs().sum()) == 0.0
    assert float(embed.lora_A.grad[:, 1].abs().sum()) > 0.0


def test_setup_lora_freezes_semantic_ids_from_model_config() -> None:
    model = _DualAR()
    model.config = type("Config", (), {"semantic_begin_id": 40, "semantic_end_id": 45})()

    setup_lora(model, _config(targets=["embeddings"]))
    model.embeddings.lora_A.data.fill_(1.0)
    model.embeddings(torch.tensor([1, 40, 45, 3])).sum().backward()

    assert float(model.embeddings.lora_A.grad[:, 40:46].abs().sum()) == 0.0
    assert float(model.embeddings.lora_A.grad[:, 1].abs().sum()) > 0.0


def test_codebook_embeddings_target_is_opt_in() -> None:
    model = _DualAR()

    setup_lora(model, _config(targets=["codebook_embeddings"]))

    assert not isinstance(model.embeddings, lora.Embedding)
    assert isinstance(model.codebook_embeddings, lora.Embedding)


def test_explicit_fast_embeddings_still_wraps() -> None:
    model = _DualAR()

    setup_lora(model, _config(targets=["fast_embeddings"]))

    assert not isinstance(model.embeddings, lora.Embedding)
    assert isinstance(model.fast_embeddings, lora.Embedding)


class _Attn(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.wqkv = torch.nn.Linear(DIM, DIM, bias=False)
        self.wo = torch.nn.Linear(DIM, DIM, bias=False)


class _FF(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w1 = torch.nn.Linear(DIM, DIM, bias=False)
        self.w2 = torch.nn.Linear(DIM, DIM, bias=False)
        self.w3 = torch.nn.Linear(DIM, DIM, bias=False)


class _Layer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attention = _Attn()
        self.feed_forward = _FF()


def test_mlp_w2_wraps_only_the_down_projection() -> None:
    model = _DualAR()
    model.layers.append(_Layer())

    setup_lora(model, _config(targets=["mlp_w2"]))

    ff = model.layers[0].feed_forward
    assert not isinstance(ff.w1, lora.Linear)
    assert isinstance(ff.w2, lora.Linear)
    assert not isinstance(ff.w3, lora.Linear)


def test_full_mlp_wins_over_mlp_w2() -> None:
    model = _DualAR()
    model.layers.append(_Layer())

    setup_lora(model, _config(targets=["mlp", "mlp_w2"]))

    ff = model.layers[0].feed_forward
    assert isinstance(ff.w1, lora.Linear)
    assert isinstance(ff.w2, lora.Linear)
    assert isinstance(ff.w3, lora.Linear)


def test_slow_targets_do_not_wrap_the_fast_stack() -> None:
    model = _DualAR()
    model.layers.append(_Layer())
    model.fast_layers.append(_Layer())

    setup_lora(model, _config(targets=["attention", "mlp"]))

    assert isinstance(model.layers[0].attention.wqkv, lora.Linear)
    assert isinstance(model.layers[0].feed_forward.w1, lora.Linear)
    assert not isinstance(model.fast_layers[0].attention.wqkv, lora.Linear)
    assert not isinstance(model.fast_layers[0].feed_forward.w1, lora.Linear)
    assert not isinstance(model.fast_embeddings, lora.Embedding)


def test_tied_logit_correction_matches_the_merged_weight() -> None:
    """Training must see the same logits the merged checkpoint will produce.

    s2-pro's tied head reads embeddings.weight directly, skipping loralib's
    forward, so during training the LoRA delta reaches the logits only through
    the _tied_lora_logits correction. That correction has to equal
    F.linear(hidden, delta) for the exact delta loralib merges on eval().
    """
    embed = _replace_embedding(torch.nn.Embedding(50, DIM), _config())
    torch.nn.init.normal_(embed.lora_A, std=0.1)  # a no-op delta would prove nothing
    hidden = torch.randn(3, 7, DIM)

    base_weight = embed.weight.data.clone()
    corrected = torch.nn.functional.linear(hidden, base_weight) + _tied_lora_logits(
        embed, hidden
    )

    embed.eval()  # loralib folds the delta into .weight here, as merge_lora does
    merged = torch.nn.functional.linear(hidden, embed.weight.data)

    # float32 matmuls reassociate differently per BLAS; logits are O(30), so
    # agreement to ~1e-3 absolute is the correct expectation, not 1e-5.
    assert torch.allclose(corrected, merged, atol=1e-3, rtol=1e-4)


def test_vq_positions_get_the_inference_scale() -> None:
    """embed() must feed the slow tower what forward_generate will feed it.

    s2-pro's inference divides VQ-position embeddings by sqrt(num_codebooks+1);
    upstream's training path forgets to, so a fine-tune optimises a model whose
    acoustic inputs are 3.3x too large and the served model breaks in
    proportion to how far the adapters moved.
    """
    config = type(
        "Config",
        (),
        {"semantic_begin_id": 100, "semantic_end_id": 200, "num_codebooks": 10},
    )()
    x = torch.randn(2, 4, 8)
    token_ids = torch.tensor([[5, 100, 200, 201], [99, 150, 3, 7]])

    scaled = _scale_vq_embeddings(x, token_ids, config)

    factor = (10 + 1) ** 0.5
    assert torch.allclose(scaled[0, 1], x[0, 1] / factor)
    assert torch.allclose(scaled[0, 2], x[0, 2] / factor)
    assert torch.allclose(scaled[1, 1], x[1, 1] / factor)
    untouched = [(0, 0), (0, 3), (1, 0), (1, 2), (1, 3)]
    for b, s in untouched:
        assert torch.equal(scaled[b, s], x[b, s])


def test_embedding_adapter_still_starts_as_a_no_op() -> None:
    """Rescaling must not disturb the base weights or add an update at step zero."""
    original = torch.nn.Embedding(1000, DIM)
    embed = _replace_embedding(original, _config())

    assert torch.equal(embed.weight.data, original.weight.data)
    assert float((embed.lora_B @ embed.lora_A).norm()) == 0.0


def _config(targets: list[str] | None = None):
    from fish_studio.training.lora_patch import LoraConfig

    if targets is None:
        return LoraConfig(r=R, lora_alpha=2.0 * R)
    return LoraConfig(r=R, lora_alpha=2.0 * R, target_modules=targets)


IM_END = 9


def test_lora_mask_opens_after_the_first_im_end() -> None:
    """System/ref prefix stays closed; user+assistant after the first <|im_end|> open."""
    tokens = torch.tensor([[1, 2, IM_END, 3, 4], [IM_END, 5, 6, 7, 8]])

    mask = lora_active_after_first_im_end(tokens, IM_END)

    assert mask.tolist() == [
        [False, False, False, True, True],
        [False, True, True, True, True],
    ]


def test_lora_mask_stays_open_when_im_end_is_missing() -> None:
    tokens = torch.tensor([[1, 2, 3], [4, 5, 6]])

    mask = lora_active_after_first_im_end(tokens, IM_END)

    assert mask.all()


def test_position_gate_zeros_the_system_prefix() -> None:
    delta = torch.ones(2, 4, 3)
    mask = torch.tensor([[False, False, True, True], [False, True, True, True]])

    gated = apply_position_gate(delta, mask)

    assert torch.equal(gated[0, :2], torch.zeros(2, 3))
    assert torch.equal(gated[0, 2:], torch.ones(2, 3))
    assert torch.equal(gated[1, 0], torch.zeros(3))
    assert torch.equal(gated[1, 1:], torch.ones(3, 3))


def test_gated_linear_keeps_the_system_prefix_stock() -> None:
    """LoRA delta must not reach positions before the first <|im_end|>."""
    linear = lora.Linear(in_features=8, out_features=8, bias=False, r=4, lora_alpha=8)
    torch.nn.init.ones_(linear.lora_A)
    torch.nn.init.ones_(linear.lora_B)
    _gate_slow_linear(linear)

    x = torch.ones(2, 5, 8)
    mask = torch.tensor(
        [[False, False, False, True, True], [False, False, True, True, True]]
    )
    set_lora_position_mask(mask)
    try:
        out = linear(x)
    finally:
        set_lora_position_mask(None)

    base = torch.nn.functional.linear(x, linear.weight)
    full_delta = (
        x @ linear.lora_A.transpose(0, 1) @ linear.lora_B.transpose(0, 1)
    ) * linear.scaling
    full = base + full_delta

    assert torch.allclose(out[0, :3], base[0, :3])
    assert torch.allclose(out[1, :2], base[1, :2])
    assert torch.allclose(out[0, 3:], full[0, 3:])
    assert torch.allclose(out[1, 2:], full[1, 2:])
    assert not torch.allclose(out[1, :2], full[1, :2])


def test_position_gate_survives_non_reentrant_checkpoint() -> None:
    """The mask is not a layer input; checkpoint must still see it on recompute."""
    linear = lora.Linear(in_features=8, out_features=8, bias=False, r=4, lora_alpha=8)
    torch.nn.init.ones_(linear.lora_A)
    torch.nn.init.ones_(linear.lora_B)
    _gate_slow_linear(linear)

    x = torch.ones(2, 5, 8)
    mask = torch.tensor(
        [[False, False, False, True, True], [False, False, True, True, True]]
    )
    set_lora_position_mask(mask)
    try:
        out = _non_reentrant_checkpoint(linear, x)
        out.sum().backward()
    finally:
        set_lora_position_mask(None)

    assert linear.lora_A.grad is not None
    assert float(linear.lora_A.grad.abs().sum()) > 0


def test_slow_attention_linears_are_position_gated() -> None:
    model = _DualAR()
    model.layers.append(_Layer())
    model.fast_layers.append(_Layer())

    setup_lora(model, _config(targets=["attention", "mlp"]))

    slow = model.layers[0].attention.wqkv
    assert slow.forward.__func__ is _gated_lora_linear_forward
    assert model.layers[0].feed_forward.w1.forward.__func__ is _gated_lora_linear_forward
    # Fast stack stays on loralib's own forward — we do not gate timbre.
    assert model.fast_layers[0].attention.wqkv.forward.__func__ is not _gated_lora_linear_forward
