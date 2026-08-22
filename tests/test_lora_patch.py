"""Tests for the Dual-AR gradient checkpointing patch."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("loralib")

from fish_studio.training.lora_patch import _non_reentrant_checkpoint  # noqa: E402


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
