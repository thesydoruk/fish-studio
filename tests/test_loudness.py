"""Speech-gated LUFS match is a linear gain only."""

from __future__ import annotations

import numpy as np
import pytest

from fish_studio.loudness import match_loudness_to_reference

pytest.importorskip("numpy")

SAMPLE_RATE = 16_000


def _tone(duration_sec: float, amplitude: float, freq: float = 220.0) -> np.ndarray:
    n = int(SAMPLE_RATE * duration_sec)
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_quiet_take_is_raised_to_the_reference() -> None:
    reference = _tone(1.0, 0.4)
    quiet = _tone(1.0, 0.1)
    fit = match_loudness_to_reference(quiet, SAMPLE_RATE, reference, SAMPLE_RATE)

    assert fit.skip_reason is None
    assert fit.gain > 1.0
    assert fit.lufs_after is not None and fit.lufs_reference is not None
    assert abs(fit.lufs_after - fit.lufs_reference) < 0.15


def test_loud_take_is_lowered_to_the_reference() -> None:
    reference = _tone(1.0, 0.1)
    loud = _tone(1.0, 0.4)
    fit = match_loudness_to_reference(loud, SAMPLE_RATE, reference, SAMPLE_RATE)

    assert fit.gain < 1.0
    assert fit.lufs_after is not None and fit.lufs_reference is not None
    assert abs(fit.lufs_after - fit.lufs_reference) < 0.15


def test_gain_does_not_clip() -> None:
    reference = _tone(1.0, 0.5)
    quiet = _tone(1.0, 0.05)
    quiet = quiet.copy()
    quiet[100] = 0.95
    fit = match_loudness_to_reference(quiet, SAMPLE_RATE, reference, SAMPLE_RATE)

    assert float(np.max(np.abs(fit.audio))) <= 10 ** (-0.1 / 20.0) + 1e-6
    assert fit.peak_limited is True


def test_silence_is_left_alone() -> None:
    silence = np.zeros(SAMPLE_RATE, dtype=np.float32)
    speech = _tone(1.0, 0.3)
    assert match_loudness_to_reference(silence, SAMPLE_RATE, speech, SAMPLE_RATE).skip_reason == (
        "no_speech_synth"
    )
    assert match_loudness_to_reference(speech, SAMPLE_RATE, silence, SAMPLE_RATE).skip_reason == (
        "no_speech_reference"
    )


def test_length_and_shape_are_unchanged() -> None:
    reference = _tone(0.8, 0.2)
    take = _tone(1.2, 0.05)
    fit = match_loudness_to_reference(take, SAMPLE_RATE, reference, SAMPLE_RATE)
    assert fit.audio.shape == take.shape
    assert fit.audio.dtype == np.float32
