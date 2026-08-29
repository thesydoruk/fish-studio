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
    from fish_studio.server.references import _SEAM_GAP_SEC, concat_reference_audio

    first = tmp_path / "a.wav"
    second = tmp_path / "b.wav"
    _write_wav(first, frames=4000)
    _write_wav(second, frames=6000)

    combined = concat_reference_audio([first, second])
    try:
        with wave.open(str(combined), "rb") as handle:
            gap = int(44100 * _SEAM_GAP_SEC)
            assert handle.getnframes() == 4000 + gap + 6000
    finally:
        combined.unlink(missing_ok=True)


def test_concat_reference_audio_levels_later_clips_to_the_first(tmp_path: Path) -> None:
    import shutil

    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")
    np = pytest.importorskip("numpy")
    sf = pytest.importorskip("soundfile")

    from fish_studio.loudness import speech_lufs
    from fish_studio.server.references import _SEAM_GAP_SEC, concat_reference_audio

    rate = 44100
    t = np.arange(rate, dtype=np.float32) / rate
    tone = np.sin(2 * np.pi * 220.0 * t).astype(np.float32)
    quiet = tmp_path / "quiet.wav"
    loud = tmp_path / "loud.wav"
    sf.write(str(quiet), tone * 0.15, rate)
    sf.write(str(loud), tone * 0.75, rate)

    combined = concat_reference_audio([quiet, loud])
    try:
        audio, out_rate = sf.read(str(combined), dtype="float32", always_2d=False)
        gap = int(out_rate * _SEAM_GAP_SEC)
        head = audio[:out_rate]
        seam = audio[out_rate : out_rate + gap]
        tail = audio[out_rate + gap :]

        assert np.max(np.abs(seam)) == 0.0
        head_lufs = speech_lufs(head, out_rate)
        tail_lufs = speech_lufs(tail, out_rate)
        assert head_lufs is not None and tail_lufs is not None
        assert abs(tail_lufs - head_lufs) < 0.5
    finally:
        combined.unlink(missing_ok=True)
