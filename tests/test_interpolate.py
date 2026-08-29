"""Blend helpers used by ``train merge`` (``TRAINING_MERGE_SCALE``)."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")

from fish_studio.training.interpolate import apply_merge_scale, interpolate  # noqa: E402


def _tiny_stock(tmp_path: Path) -> Path:
    stock = tmp_path / "stock"
    stock.mkdir()
    safetensors_torch.save_file(
        {
            "text_model.model.embeddings.weight": torch.zeros(2, 3),
            "text_model.model.layers.0.weight": torch.ones(2, 2),
        },
        str(stock / "model.safetensors"),
    )
    return stock


def _tiny_merged(tmp_path: Path) -> Path:
    path = tmp_path / "merged" / "model.pth"
    path.parent.mkdir()
    torch.save(
        {
            "embeddings.weight": torch.full((2, 3), 2.0),
            "layers.0.weight": torch.full((2, 2), 3.0),
        },
        path,
    )
    return path


def test_interpolate_halfway_between_stock_and_merged(tmp_path: Path) -> None:
    stock = _tiny_stock(tmp_path)
    merged = _tiny_merged(tmp_path)

    state = interpolate(stock, merged, 0.5)

    assert torch.equal(state["embeddings.weight"], torch.ones(2, 3))
    assert torch.equal(state["layers.0.weight"], torch.full((2, 2), 2.0))


def test_apply_merge_scale_is_noop_at_one(tmp_path: Path) -> None:
    merged = _tiny_merged(tmp_path)
    before = torch.load(merged, map_location="cpu", weights_only=True)

    apply_merge_scale(merged.parent, tmp_path / "missing-stock", 1.0)

    after = torch.load(merged, map_location="cpu", weights_only=True)
    assert torch.equal(after["embeddings.weight"], before["embeddings.weight"])


def test_apply_merge_scale_rewrites_model_pth(tmp_path: Path) -> None:
    stock = _tiny_stock(tmp_path)
    merged = _tiny_merged(tmp_path)

    apply_merge_scale(merged.parent, stock, 0.5)

    state = torch.load(merged, map_location="cpu", weights_only=True)
    assert torch.equal(state["embeddings.weight"], torch.ones(2, 3))
