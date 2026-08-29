"""Patch fish-speech LoRA setup for s2-pro Dual-AR checkpoints (v2.0.0-beta)."""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from types import MethodType

import loralib as lora
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as torch_checkpoint

# True on user+assistant positions. Thread-local so DataLoader workers stay isolated.
_position_gate = threading.local()
# s2-pro <|im_end|> measured on the served tokenizer; overridden if we can load it.
_DEFAULT_IM_END_ID = 151645


@dataclass
class LoraConfig:
    r: int
    lora_alpha: float
    lora_dropout: float = 0.0
    target_modules: list[str] = field(
        default_factory=lambda: ["attention", "mlp", "embeddings"]
    )


def _rescale_embedding_lora(new_embed) -> None:
    """Give an embedding adapter the same update scale as a linear one.

    ``loralib`` seeds the two wrappers as mirror images: ``Linear`` gets a
    Kaiming-scaled ``lora_A`` with ``lora_B`` at zero, while ``Embedding`` gets
    ``lora_A`` at zero and ``lora_B`` from ``normal_()`` — unit variance, no fan-in
    term. Both start at a zero update, so nothing looks wrong, but the non-zero
    factor is ~90x larger for the embedding (‖B‖ ≈ 287 against ‖A‖ ≈ 3.3 at
    r=32, d=2560). Since the gradient of one factor is proportional to the other,
    the embedding moves that much faster: measured against the released weights,
    500 steps put the codebook table 17.7% away from where it started while every
    transformer matrix sat under 2.5%, and the audio decoder lost the meaning of
    its own codes — synthesis came back 25-45 dB quiet.

    Matching the Kaiming scale makes one learning rate mean the same thing for
    every adapted matrix. The bound is written out rather than delegated to
    ``kaiming_uniform_``: the target is a total norm of ``sqrt(r/3)`` — what
    ``lora_A`` gets in a Linear adapter, whatever the shapes — and over the
    ``embedding_dim x r`` elements of ``lora_B`` that means a bound of
    ``1/sqrt(embedding_dim)``. ``kaiming_uniform_`` would read the fan-in of
    ``r`` off the trailing axis instead and land ``sqrt(embedding_dim/r)`` too
    high: about 9x on the 2560-wide tables at r=32.
    """
    bound = 1.0 / math.sqrt(new_embed.lora_B.size(0))
    nn.init.uniform_(new_embed.lora_B, -bound, bound)
    nn.init.zeros_(new_embed.lora_A)


def _replace_embedding(old_embed, lora_config: LoraConfig):
    new_embed = lora.Embedding(
        num_embeddings=old_embed.num_embeddings,
        embedding_dim=old_embed.embedding_dim,
        padding_idx=old_embed.padding_idx,
        r=lora_config.r,
        lora_alpha=lora_config.lora_alpha,
    )
    new_embed.weight.data.copy_(old_embed.weight.data)
    _rescale_embedding_lora(new_embed)
    return new_embed


def freeze_semantic_id_adapter(embed, begin: int, end: int) -> None:
    """Keep LoRA off the VQ/semantic token rows of the text embedding table.

    Those ids share ``embeddings`` with ordinary text. Left trainable they
    dominate the adapter and pull the clone path, not Ukrainian letters.
    Zero the matching slice of ``lora_A`` and drop its gradient so lookup
    and the tied logit head stay stock for every semantic id.
    """
    if begin < 0 or end < begin:
        raise ValueError(f"invalid semantic id range: {begin}..{end}")
    adapter = embed.lora_A
    # loralib Embedding stores A as [r, num_embeddings]. Do not guess from
    # which axis is longer: in tests r can exceed the toy vocab.
    n_vocab = embed.num_embeddings
    if adapter.size(1) == n_vocab:
        vocab_dim = 1
    elif adapter.size(0) == n_vocab:
        vocab_dim = 0
    else:
        raise ValueError(
            f"lora_A shape {tuple(adapter.shape)} does not match "
            f"num_embeddings={n_vocab}"
        )
    last = min(end + 1, adapter.size(vocab_dim))
    if begin >= last:
        return
    if vocab_dim == 0:
        adapter.data[begin:last].zero_()

        def hook(grad: torch.Tensor) -> torch.Tensor:
            grad = grad.clone()
            grad[begin:last] = 0
            return grad
    else:
        adapter.data[:, begin:last].zero_()

        def hook(grad: torch.Tensor) -> torch.Tensor:
            grad = grad.clone()
            grad[:, begin:last] = 0
            return grad

    adapter.register_hook(hook)


def _tied_lora_logits(embeddings, hidden):
    """The logit term the tied head misses while an embedding LoRA is unmerged.

    s2-pro computes token logits as ``F.linear(slow_out, embeddings.weight)``,
    reading the parameter directly and bypassing loralib's forward, so the
    adapter's delta never reaches the logits during training. This returns
    exactly that missing term, ``slow_out @ ΔW.T`` for ``ΔW = (B·A).T *
    scaling``, factored as ``((h @ B) @ A) * scaling`` so the full
    ``vocab x dim`` delta never materialises.
    """
    return (hidden @ embeddings.lora_B) @ embeddings.lora_A * embeddings.scaling


def lora_active_after_first_im_end(token_ids: torch.Tensor, im_end_id: int) -> torch.Tensor:
    """True after the first ``<|im_end|>`` — user + assistant, not the system/ref block."""
    if token_ids.dim() != 2:
        raise ValueError(f"expected [batch, time] token ids, got {tuple(token_ids.shape)}")
    is_end = token_ids == im_end_id
    has_end = is_end.any(dim=-1)
    first = is_end.int().argmax(dim=-1)
    steps = torch.arange(token_ids.size(1), device=token_ids.device)
    after = steps.unsqueeze(0) > first.unsqueeze(1)
    return torch.where(has_end.unsqueeze(1), after, torch.ones_like(after))


def apply_position_gate(delta: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Zero the LoRA delta on system/ref positions. ``mask`` is [batch, time] bool."""
    if delta.dim() == 3 and mask.shape == delta.shape[:2]:
        return delta * mask.unsqueeze(-1).to(dtype=delta.dtype)
    if delta.dim() == 2 and delta.size(0) == mask.numel():
        return delta * mask.reshape(-1).unsqueeze(-1).to(dtype=delta.dtype)
    return delta


def _current_position_mask() -> torch.Tensor | None:
    return getattr(_position_gate, "mask", None)


def set_lora_position_mask(mask: torch.Tensor | None) -> None:
    """Install the [batch, time] gate. Left set across backward for checkpoint recompute."""
    _position_gate.mask = mask


def _token_ids_from_inp(inp: torch.Tensor) -> torch.Tensor:
    """Text token row of a Dual-AR batch: ``inp`` is ``[B, n_codebooks+1, T]``."""
    if inp.dim() == 3:
        tokens = inp[:, 0]
    elif inp.dim() == 2:
        tokens = inp
    else:
        tokens = inp.view(1, -1)
    if tokens.dim() == 1:
        tokens = tokens.unsqueeze(0)
    return tokens


def _set_position_mask_from_inp(inp: torch.Tensor, im_end_id: int) -> None:
    tokens = _token_ids_from_inp(inp)
    set_lora_position_mask(lora_active_after_first_im_end(tokens, im_end_id))


def resolve_im_end_id(module) -> int:
    tokenizer = getattr(module, "tokenizer", None)
    if tokenizer is not None and hasattr(tokenizer, "get_token_id"):
        try:
            return int(tokenizer.get_token_id("<|im_end|>"))
        except Exception:  # noqa: BLE001 — fall back to the measured s2-pro id
            pass
    return _DEFAULT_IM_END_ID


def _gated_lora_linear_forward(self, x: torch.Tensor):
    """Stock matmul everywhere; LoRA delta only on user+assistant positions."""

    def transpose_weight(weight: torch.Tensor) -> torch.Tensor:
        return weight.transpose(0, 1) if self.fan_in_fan_out else weight

    base = F.linear(x, transpose_weight(self.weight), bias=self.bias)
    if self.r <= 0 or self.merged:
        return base
    delta = (
        self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)
    ) * self.scaling
    mask = _current_position_mask()
    if mask is not None:
        delta = apply_position_gate(delta, mask)
    return base + delta


def _gate_slow_linear(linear: nn.Module) -> None:
    linear.forward = MethodType(_gated_lora_linear_forward, linear)  # type: ignore[method-assign]


def _mlp_weight_names(targets: set[str]) -> tuple[str, ...]:
    """Which slow feed-forward matrices to wrap.

    ``mlp`` is the full SwiGLU block. ``mlp_w2`` is the down-projection only —
    the path that writes into the residual stream. ``mlp`` wins if both appear.
    """
    if "mlp" in targets:
        return ("w1", "w2", "w3")
    if "mlp_w2" in targets:
        return ("w2",)
    return ()


def setup_lora(model, lora_config: LoraConfig) -> None:
    """Wrap Dual-AR slow and/or fast modules listed in ``target_modules``.

    Slow names (``attention``, ``mlp``, ``mlp_w2``, ``embeddings``) adapt the
    text→semantic tower only. ``mlp`` wraps ``w1/w2/w3``; ``mlp_w2`` is the
    down-projection alone so gate/up stay stock after merge. ``embeddings`` is
    the text token table; ``codebook_embeddings`` is the acoustic codebook and
    must be listed separately. Fast / timbre adapters are opt-in via the
    matching ``fast_*`` names.
    """
    targets = set(lora_config.target_modules)
    # (module, attr, gate) — gate only the slow tower so the ref/VQ path can stay stock.
    linears: list[tuple[object, str, bool]] = []

    slow_attention = "attention" in targets
    slow_mlp = _mlp_weight_names(targets)
    slow_embeddings = "embeddings" in targets
    slow_codebook = "codebook_embeddings" in targets
    slow_output = "output" in targets
    fast_attention = "fast_attention" in targets
    fast_mlp = "fast_mlp" in targets
    fast_embeddings = "fast_embeddings" in targets
    fast_output = "fast_output" in targets

    if slow_embeddings:
        # On tied checkpoints the logit head reads embeddings.weight directly;
        # patch_tied_embedding_logits() keeps training consistent with the
        # merged model, so the text table is safe to adapt.
        model.embeddings = _replace_embedding(model.embeddings, lora_config)
        config = getattr(model, "config", None)
        begin = getattr(config, "semantic_begin_id", None)
        end = getattr(config, "semantic_end_id", None)
        if begin is not None and end is not None:
            freeze_semantic_id_adapter(model.embeddings, int(begin), int(end))
            print(
                f"[info] LoRA frozen on embeddings rows {int(begin)}..{int(end)} "
                "(semantic ids stay stock)"
            )

    if slow_codebook:
        model.codebook_embeddings = _replace_embedding(
            model.codebook_embeddings, lora_config
        )

    if slow_output:
        if hasattr(model, "output"):
            linears.append((model, "output", True))
        else:
            # s2-pro has no separate slow logit head, so this target silently adapts
            # nothing. Say so, or a run looks configured for something it never did.
            print("[warn] LoRA target 'output' has no matching module; nothing to adapt there")

    for layer in model.layers:
        if slow_attention:
            linears.extend(
                [(layer.attention, "wqkv", True), (layer.attention, "wo", True)]
            )
        for weight in slow_mlp:
            linears.append((layer.feed_forward, weight, True))

    if hasattr(model, "fast_layers"):
        if fast_embeddings:
            model.fast_embeddings = _replace_embedding(model.fast_embeddings, lora_config)
        if fast_output:
            linears.append((model, "fast_output", False))

        for layer in model.fast_layers:
            if fast_attention:
                linears.extend(
                    [(layer.attention, "wqkv", False), (layer.attention, "wo", False)]
                )
            if fast_mlp:
                linears.extend(
                    [
                        (layer.feed_forward, "w1", False),
                        (layer.feed_forward, "w2", False),
                        (layer.feed_forward, "w3", False),
                    ]
                )

    gated = 0
    for module, layer_name, gate in linears:
        old_linear = getattr(module, layer_name)
        updated_linear = lora.Linear(
            in_features=old_linear.in_features,
            out_features=old_linear.out_features,
            bias=old_linear.bias is not None,
            r=lora_config.r,
            lora_alpha=lora_config.lora_alpha,
            lora_dropout=lora_config.lora_dropout,
        )
        updated_linear.weight.data.copy_(old_linear.weight.data)
        if old_linear.bias is not None:
            updated_linear.bias.data.copy_(old_linear.bias.data)
        setattr(module, layer_name, updated_linear)
        if gate:
            _gate_slow_linear(updated_linear)
            gated += 1

    if gated:
        print(
            f"[info] LoRA position gate on {gated} slow linears "
            "(delta after first <|im_end|>; system/ref stays stock during train)"
        )

    lora.mark_only_lora_as_trainable(model, bias="none")


def _non_reentrant_checkpoint(function, *args, **kwargs):
    """Non-reentrant checkpoint that also carries the position gate.

    The mask lives outside each layer. Left as a thread-local read, non-reentrant
    checkpoint saves it on the original forward and not on recompute
    (76 vs 71 tensors — one extra mul per slow linear in the block). Passing
    it in as a checkpoint input makes both passes identical.
    """
    kwargs["use_reentrant"] = False
    mask = _current_position_mask()
    if mask is None:
        return torch_checkpoint.checkpoint(function, *args, **kwargs)

    def wrapped(mask_in, *inner_args):
        previous = _current_position_mask()
        set_lora_position_mask(mask_in)
        try:
            return function(*inner_args)
        finally:
            set_lora_position_mask(previous)

    return torch_checkpoint.checkpoint(wrapped, mask, *args, **kwargs)


def patch_gradient_checkpointing() -> None:
    """Force non-reentrant gradient checkpointing in the Dual-AR forward passes.

    Upstream calls ``checkpoint(..., use_reentrant=True)``. Reentrant checkpointing
    only builds a backward graph when an *input* of the wrapped block requires grad.
    The slow (text->semantic) layers receive the output of the frozen ``embeddings``,
    so with LoRA-only training their inputs never require grad and every LoRA weight
    inside them silently stays at its zero initialisation. The fast layers are fed by
    the LoRA-wrapped ``fast_embeddings`` and are therefore unaffected, which makes the
    failure look like a partially successful run.
    """
    import fish_speech.models.text2semantic.llama as fish_llama

    fish_llama.checkpoint = _non_reentrant_checkpoint


def patch_tied_embedding_logits() -> None:
    """Make the tied logit head see the embedding LoRA delta during training.

    ``BaseTransformer.forward`` on tied checkpoints computes
    ``F.linear(slow_out, self.embeddings.weight)`` — a direct parameter read
    that skips loralib's forward. Left alone, the delta trains against the
    input lookup only, and merging it into the shared weight afterwards shifts
    the logits of every token (semantic ids and im_end included) in a way no
    training step ever saw. This wrapper adds the same delta to the logits, so
    the function being trained and the merged checkpoint are identical.

    The correction reuses ``hidden_states``, which equals ``slow_out`` when
    ``norm_fastlayer_input`` is set (always true for s2-pro); anything else is
    refused loudly rather than trained subtly wrong.
    """
    import fish_speech.models.text2semantic.llama as fish_llama

    if getattr(fish_llama.BaseTransformer.forward, "_fish_studio_tied_lora_patch", False):
        return

    original_forward = fish_llama.BaseTransformer.forward

    def forward(self, inp, key_padding_mask=None):
        result = original_forward(self, inp, key_padding_mask=key_padding_mask)
        embeddings = self.embeddings
        if (
            getattr(self.config, "tie_word_embeddings", False)
            and isinstance(embeddings, lora.Embedding)
            and embeddings.r > 0
            and not embeddings.merged
        ):
            if not getattr(self.config, "norm_fastlayer_input", False):
                raise RuntimeError(
                    "tied embeddings with LoRA need norm_fastlayer_input=True to "
                    "rebuild slow_out from hidden_states; this checkpoint has it off"
                )
            result.logits = result.logits + _tied_lora_logits(embeddings, result.hidden_states)
        return result

    forward._fish_studio_tied_lora_patch = True  # type: ignore[attr-defined]
    fish_llama.BaseTransformer.forward = forward


def patch_lora_position_mask() -> None:
    """Mark user+assistant positions so gated slow LoRA leaves the ref block stock.

    Merge still folds ΔW into the base matrix, so serve applies the adapter
    everywhere. The mask is a train regularizer: gradients never see the
    system/ref+VQ prefix. Do not wrap layers in a new ``nn.Module`` — merge
    matches state-dict keys.
    """
    import fish_speech.models.text2semantic.llama as fish_llama

    if getattr(fish_llama.BaseTransformer.forward, "_fish_studio_pos_mask", False):
        return

    inner = fish_llama.BaseTransformer.forward

    def forward(self, inp, key_padding_mask=None):
        _set_position_mask_from_inp(inp, resolve_im_end_id(self))
        return inner(self, inp, key_padding_mask=key_padding_mask)

    forward._fish_studio_pos_mask = True  # type: ignore[attr-defined]
    fish_llama.BaseTransformer.forward = forward


def _scale_vq_embeddings(x, token_ids, config):
    """Divide VQ-position embeddings the way ``forward_generate`` does."""
    vq = (token_ids >= config.semantic_begin_id) & (token_ids <= config.semantic_end_id)
    scaled = x / math.sqrt(config.num_codebooks + 1)
    return torch.where(vq.unsqueeze(-1).expand_as(x), scaled, x)


def patch_scaled_codebook_embed() -> None:
    """Give training the same VQ embedding scale that inference applies.

    s2-pro loads with ``scale_codebook_embeddings=True``, and at inference
    ``forward_generate`` divides the summed embedding at every VQ position by
    ``sqrt(num_codebooks + 1)`` ≈ 3.3. The training path — ``forward`` via
    ``embed()`` — skips that division, so fine-tuning optimises a model whose
    acoustic inputs are 3.3x larger than anything the served model will ever
    see. The adapters learn to compensate for the wrong scale, and the served
    model degrades in proportion to how far they moved: a probe run stayed
    broken (32 dB level drop at 500 steps) even after the init-scale fix,
    while scaling its adapter down to x0.15 nearly recovered — both symptoms
    of a systematic train/inference mismatch, not of one bad matrix.
    """
    import fish_speech.models.text2semantic.llama as fish_llama

    if getattr(fish_llama.BaseTransformer.embed, "_fish_studio_vq_scale_patch", False):
        return

    original_embed = fish_llama.BaseTransformer.embed

    def embed(self, inp):
        x = original_embed(self, inp)
        if getattr(self.config, "scale_codebook_embeddings", False):
            x = _scale_vq_embeddings(x, inp[:, 0], self.config)
        return x

    embed._fish_studio_vq_scale_patch = True  # type: ignore[attr-defined]
    fish_llama.BaseTransformer.embed = embed


def patch_checkpoint_unpickling() -> None:
    """Allow Lightning resume of fish-speech ckpts under torch>=2.6.

    Training checkpoints pickle OmegaConf metadata that ``weights_only=True``
    rejects. Local runs only load our own ``step_*.ckpt`` files, so force the
    pre-2.6 behaviour for ``torch.load``.
    """
    import torch

    if getattr(torch.load, "_fish_studio_weights_only_patch", False):
        return

    original_load = torch.load

    def load_trusted(*args, **kwargs):
        kwargs["weights_only"] = False
        return original_load(*args, **kwargs)

    load_trusted._fish_studio_weights_only_patch = True  # type: ignore[attr-defined]
    torch.load = load_trusted  # type: ignore[assignment]


def patch_lora_checkpoint_strict() -> None:
    """LoRA ``step_*.ckpt`` files omit frozen base weights; resume with strict=False."""
    import lightning.pytorch.strategies.strategy as strat

    if getattr(strat.Strategy.load_model_state_dict, "_fish_studio_lora_strict_patch", False):
        return

    original = strat.Strategy.load_model_state_dict

    def load_model_state_dict(self, checkpoint, strict: bool = True):  # noqa: ARG001
        return original(self, checkpoint, strict=False)

    load_model_state_dict._fish_studio_lora_strict_patch = True  # type: ignore[attr-defined]
    strat.Strategy.load_model_state_dict = load_model_state_dict  # type: ignore[method-assign]


def apply_dual_ar_lora_patch() -> None:
    """Install Dual-AR LoRA + checkpoint patches before any fish-speech train/merge import."""
    import fish_speech.models.text2semantic.lora as fish_lora

    fish_lora.LoraConfig = LoraConfig
    fish_lora.setup_lora = setup_lora
    patch_gradient_checkpointing()
    patch_tied_embedding_logits()
    patch_lora_position_mask()
    patch_scaled_codebook_embed()
    patch_checkpoint_unpickling()
    patch_lora_checkpoint_strict()

