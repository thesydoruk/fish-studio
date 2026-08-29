"""Tests for typed coercion of .env values into config dataclasses."""

from __future__ import annotations

import pytest

from fish_studio.config import (
    InferenceConfig,
    PipelineConfig,
    QualityConfig,
    TrainingConfig,
    _dataclass_from_env,
)


def test_int_fields_are_coerced(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INFERENCE_PORT", "9000")
    monkeypatch.setenv("INFERENCE_MAX_UPLOAD_BYTES", "15728640")

    config = _dataclass_from_env(InferenceConfig, "INFERENCE")

    assert config.port == 9000
    # A str would silently break the size comparison in the upload handler.
    assert isinstance(config.max_upload_bytes, int)
    assert config.max_upload_bytes == 15728640


def test_false_bool_is_not_truthy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PIPELINE_FORCE", "false")

    config = _dataclass_from_env(PipelineConfig, "PIPELINE")

    assert config.force is False


def test_missing_env_keeps_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("INFERENCE_PORT", raising=False)

    config = _dataclass_from_env(InferenceConfig, "INFERENCE")

    assert config.port == InferenceConfig().port


def test_training_merge_scale_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRAINING_MERGE_SCALE", "0.4")
    monkeypatch.setenv("TRAINING_LORA_TARGET_MODULES", "attention,mlp,embeddings")

    config = _dataclass_from_env(TrainingConfig, "TRAINING")

    assert config.merge_scale == 0.4
    assert config.lora_target_modules == ["attention", "mlp", "embeddings"]


def test_quality_audio_gates_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QUALITY_FILTER_AUDIO_QUALITY", "false")
    monkeypatch.setenv("QUALITY_MAX_NO_SPEECH_PROB", "0.4")

    config = _dataclass_from_env(QualityConfig, "QUALITY")

    assert config.filter_audio_quality is False
    assert config.max_no_speech_prob == 0.4
