"""Tests for waveform concat helpers."""

from __future__ import annotations

import numpy as np
import pytest

from fish_studio.waveform import concat_audio_chunks

pytest.importorskip("numpy")

SAMPLE_RATE = 16_000


def _tone(duration_sec: float, amplitude: float = 0.2) -> np.ndarray:
    n = int(SAMPLE_RATE * duration_sec)
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    return (amplitude * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def test_concat_crossfades_two_chunks() -> None:
    left = _tone(0.5)
    right = _tone(0.5, amplitude=0.1)
    out = concat_audio_chunks([left, right], SAMPLE_RATE, crossfade_ms=20.0)
    expected = left.size + right.size - int(SAMPLE_RATE * 0.02)
    assert out.size == expected
