"""Tests for synthesis waveform fade compensation."""

from __future__ import annotations

import numpy as np
import pytest

from fish_studio.waveform import compensate_fade, concat_audio_chunks

pytest.importorskip("numpy")

SAMPLE_RATE = 16_000


def _tone(duration_sec: float, amplitude: float, freq: float = 220.0) -> np.ndarray:
    n = int(SAMPLE_RATE * duration_sec)
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _rms(samples: np.ndarray, start: int, end: int) -> float:
    chunk = samples[start:end]
    return float(np.sqrt(np.mean(chunk * chunk)))


def test_compensate_fade_lifts_a_linear_decay() -> None:
    n = SAMPLE_RATE * 6
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    envelope = np.linspace(1.0, 0.25, n, dtype=np.float32)
    faded = (0.2 * envelope * np.sin(2 * np.pi * 220 * t)).astype(np.float32)

    fixed = compensate_fade(faded, SAMPLE_RATE)
    hop = SAMPLE_RATE
    early = _rms(fixed, hop, 2 * hop)
    late = _rms(fixed, n - 2 * hop, n - hop)
    faded_late = _rms(faded, n - 2 * hop, n - hop)

    assert late / early > 0.75
    assert late > faded_late * 1.5


def test_compensate_fade_leaves_short_clips_unchanged() -> None:
    short = _tone(0.4, 0.2)
    out = compensate_fade(short, SAMPLE_RATE)
    assert np.array_equal(out, short)


def test_compensate_fade_leaves_flat_speech_unchanged() -> None:
    flat = _tone(4.0, 0.2)
    out = compensate_fade(flat, SAMPLE_RATE)
    assert np.max(np.abs(out - flat)) < 1e-3


def test_concat_audio_chunks_crossfades() -> None:
    left = _tone(0.5, 0.2)
    right = _tone(0.5, 0.2, freq=440.0)
    joined = concat_audio_chunks([left, right], SAMPLE_RATE, crossfade_ms=40)
    overlap = int(SAMPLE_RATE * 0.04)
    assert joined.shape[0] == left.size + right.size - overlap
