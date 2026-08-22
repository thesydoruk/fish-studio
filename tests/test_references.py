"""Tests for multi-reference server helpers."""

from __future__ import annotations

import wave
from pathlib import Path

import pytest

from fish_studio.config import StressConfig
from fish_studio.server.references import (
    MAX_REFERENCES,
    format_reference_text,
    normalize_reference_texts,
    validate_reference_count,
)

pytest.importorskip("ukrainian_word_stress")


def _write_wav(path: Path, *, frames: int = 8000, rate: int = 44100) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"\x00\x01" * frames)


def test_validate_reference_count() -> None:
    validate_reference_count(1)
    validate_reference_count(MAX_REFERENCES)
    with pytest.raises(ValueError, match="at least one"):
        validate_reference_count(0)
    with pytest.raises(ValueError, match="at most"):
        validate_reference_count(MAX_REFERENCES + 1)


def test_format_single_reference_keeps_plain_text() -> None:
    assert format_reference_text(["референс"]) == "референс"


def test_format_multiple_references_adds_speaker_tags() -> None:
    joined = format_reference_text(["перший", "другий"])
    assert joined.splitlines() == ["<|speaker:0|>перший", "<|speaker:1|>другий"]


def test_normalize_reference_texts_fills_missing_with_default() -> None:
    texts = normalize_reference_texts(
        ["one"],
        count=2,
        default_text="fallback",
        stress=StressConfig(enabled=False),
    )
    assert texts == ["one", "fallback"]


def test_resolve_raw_speaker_texts_requires_one_per_clip() -> None:
    from fish_studio.server.app import _resolve_raw_speaker_texts

    with pytest.raises(ValueError, match="speaker_texts"):
        _resolve_raw_speaker_texts(count=2, speaker_text="only one", speaker_texts=None)
    stress = StressConfig(enabled=True)
    texts = normalize_reference_texts(
        ["ліхтарик"],
        count=1,
        default_text="",
        stress=stress,
    )
    assert "\u0301" in texts[0]


def test_concat_reference_audio_requires_ffmpeg(tmp_path: Path) -> None:
    import shutil

    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")
    from fish_studio.server.references import concat_reference_audio

    first = tmp_path / "a.wav"
    second = tmp_path / "b.wav"
    _write_wav(first, frames=4000)
    _write_wav(second, frames=6000)

    combined = concat_reference_audio([first, second])
    try:
        with wave.open(str(combined), "rb") as handle:
            assert handle.getnframes() == 4000 + 6000
    finally:
        combined.unlink(missing_ok=True)
