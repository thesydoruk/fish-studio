"""Group merge used by ``train merge`` (``TRAINING_MERGE_SCALE_FOR``)."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")

from fish_studio.training.interpolate import (  # noqa: E402
    apply_merge_groups,
    interpolate,
    parse_scale_group,
    scale_for,
)


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


def test_groups_give_each_tensor_its_own_scale(tmp_path: Path) -> None:
    """Embeddings kept, layers not listed: the layers go back to stock."""
    stock = _tiny_stock(tmp_path)
    merged = _tiny_merged(tmp_path)

    state = interpolate(stock, merged, [parse_scale_group(r"^embeddings\.=1.0")])

    assert torch.equal(state["embeddings.weight"], torch.full((2, 3), 2.0))
    assert torch.equal(state["layers.0.weight"], torch.ones(2, 2))


def test_a_group_can_blend_halfway(tmp_path: Path) -> None:
    stock = _tiny_stock(tmp_path)
    merged = _tiny_merged(tmp_path)

    state = interpolate(stock, merged, [parse_scale_group(r"^layers\.=0.5")])

    assert torch.equal(state["layers.0.weight"], torch.full((2, 2), 2.0))
    assert torch.equal(state["embeddings.weight"], torch.zeros(2, 3))


def test_no_groups_is_refused_rather_than_silently_undoing_the_fold(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        interpolate(_tiny_stock(tmp_path), _tiny_merged(tmp_path), [])


def test_first_matching_group_wins_over_later_ones() -> None:
    groups = [parse_scale_group(r"^layers\.0\.=0.25"), parse_scale_group(r"^layers\.=1.0")]
    assert scale_for("layers.0.weight", groups) == 0.25
    assert scale_for("layers.7.weight", groups) == 1.0
    assert scale_for("embeddings.weight", groups) == 0.0


def test_parse_scale_group_rejects_bad_specs() -> None:
    with pytest.raises(ValueError):
        parse_scale_group("no-equals-sign")
    with pytest.raises(ValueError):
        parse_scale_group("^x=1.5")


def test_apply_merge_groups_rewrites_model_pth(tmp_path: Path) -> None:
    stock = _tiny_stock(tmp_path)
    merged = _tiny_merged(tmp_path)

    apply_merge_groups(merged.parent, stock, [parse_scale_group(r"^embeddings\.=1.0")])

    after = torch.load(merged, map_location="cpu", weights_only=True)
    assert torch.equal(after["embeddings.weight"], torch.full((2, 3), 2.0))
    assert torch.equal(after["layers.0.weight"], torch.ones(2, 2))
