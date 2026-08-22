"""Tests for LoRA target wiring between training and merging."""

from __future__ import annotations

from pathlib import Path

import pytest

from fish_studio.config import TrainingConfig
from fish_studio.training import merge_lora
from fish_studio.training.train_lora import _lora_targets


def test_lora_targets_accepts_slow_and_fast_modules() -> None:
    assert _lora_targets("attention, mlp ,fast_output") == "attention,mlp,fast_output"


def test_lora_targets_accepts_a_list() -> None:
    assert _lora_targets(["attention", "fast_mlp"]) == "attention,fast_mlp"


def test_lora_targets_rejects_unknown_module() -> None:
    with pytest.raises(ValueError, match="wqkv"):
        _lora_targets("attention,wqkv")


def test_lora_targets_rejects_empty() -> None:
    with pytest.raises(ValueError):
        _lora_targets(" , ")


def test_write_lora_config_mirrors_training_targets(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(merge_lora, "fish_site_packages", lambda: tmp_path)
    training = TrainingConfig(
        lora_config="r_8_alpha_16",
        lora_r=16,
        lora_alpha=32.0,
        lora_dropout=0.05,
        lora_target_modules=["attention", "mlp", "fast_embeddings"],
    )

    name = merge_lora.write_lora_config(training, training.lora_config)

    written = (tmp_path / "fish_speech" / "configs" / "lora" / f"{name}.yaml").read_text(
        encoding="utf-8"
    )
    assert name == "fish-studio-r_8_alpha_16"
    assert "_target_: fish_studio.training.lora_patch.LoraConfig" in written
    assert "r: 16" in written
    assert "lora_alpha: 32.0" in written
    assert "  - attention\n  - mlp\n  - fast_embeddings\n" in written
