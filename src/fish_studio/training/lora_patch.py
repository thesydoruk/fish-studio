"""Patch fish-speech LoRA setup for s2-pro Dual-AR checkpoints (v2.0.0-beta)."""

from __future__ import annotations

from dataclasses import dataclass, field

import loralib as lora
import torch.utils.checkpoint as torch_checkpoint


@dataclass
class LoraConfig:
    r: int
    lora_alpha: float
    lora_dropout: float = 0.0
    target_modules: list[str] = field(
        default_factory=lambda: ["fast_attention", "fast_mlp", "fast_embeddings", "fast_output"]
    )


def _replace_embedding(old_embed, lora_config: LoraConfig):
    new_embed = lora.Embedding(
        num_embeddings=old_embed.num_embeddings,
        embedding_dim=old_embed.embedding_dim,
        padding_idx=old_embed.padding_idx,
        r=lora_config.r,
        lora_alpha=lora_config.lora_alpha,
    )
    new_embed.weight.data.copy_(old_embed.weight.data)
    return new_embed


def setup_lora(model, lora_config: LoraConfig) -> None:
    """Wrap Dual-AR slow and/or fast modules. Bare names also enable the matching fast_* targets."""
    targets = set(lora_config.target_modules)
    linears: list[tuple[object, str]] = []

    slow_attention = "attention" in targets
    slow_mlp = "mlp" in targets
    slow_embeddings = "embeddings" in targets
    slow_output = "output" in targets

    # Dual-AR: fast_* is the acoustic stack. A slow name also arms the matching fast module
    # so a config that says "attention" still trains both towers.
    fast_attention = slow_attention or "fast_attention" in targets
    fast_mlp = slow_mlp or "fast_mlp" in targets
    fast_embeddings = slow_embeddings or "fast_embeddings" in targets
    fast_output = slow_output or "fast_output" in targets

    if slow_embeddings:
        model.embeddings = _replace_embedding(model.embeddings, lora_config)
        model.codebook_embeddings = _replace_embedding(model.codebook_embeddings, lora_config)

    if slow_output and hasattr(model, "output"):
        linears.append((model, "output"))

    for layer in model.layers:
        if slow_attention:
            linears.extend([(layer.attention, "wqkv"), (layer.attention, "wo")])
        if slow_mlp:
            linears.extend(
                [
                    (layer.feed_forward, "w1"),
                    (layer.feed_forward, "w2"),
                    (layer.feed_forward, "w3"),
                ]
            )

    if hasattr(model, "fast_layers"):
        if fast_embeddings:
            model.fast_embeddings = _replace_embedding(model.fast_embeddings, lora_config)
        if fast_output:
            linears.append((model, "fast_output"))

        for layer in model.fast_layers:
            if fast_attention:
                linears.extend([(layer.attention, "wqkv"), (layer.attention, "wo")])
            if fast_mlp:
                linears.extend(
                    [
                        (layer.feed_forward, "w1"),
                        (layer.feed_forward, "w2"),
                        (layer.feed_forward, "w3"),
                    ]
                )

    for module, layer_name in linears:
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

    lora.mark_only_lora_as_trainable(model, bias="none")


def _non_reentrant_checkpoint(function, *args, **kwargs):
    kwargs["use_reentrant"] = False
    return torch_checkpoint.checkpoint(function, *args, **kwargs)


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
    patch_checkpoint_unpickling()
    patch_lora_checkpoint_strict()

