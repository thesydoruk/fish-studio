"""Tests for matching synthesized speech level to the first reference clip."""

from __future__ import annotations

import math

import numpy as np
import pytest

from fish_studio.loudness import (
    finish_synthesis_audio,
    fit_level_to_reference,
    match_loudness_to_reference,
    match_peak_to_target,
    measure_peak,
    measure_speech_rms,
    scale_with_soft_ceiling,
)

pytest.importorskip("numpy")

SAMPLE_RATE = 22_050
I16 = 32_768.0


def _tone(duration_sec: float, amp_i16: float, freq: float = 220.0) -> np.ndarray:
    n = int(SAMPLE_RATE * duration_sec)
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    return np.float32(amp_i16 / I16) * np.sin(2 * np.pi * freq * t).astype(np.float32)


def _db_between(value: float, reference: float) -> float:
    return 20 * math.log10(value / reference)


def test_measure_speech_rms_ignores_pauses() -> None:
    speech = _tone(1, 8000)
    padded = np.concatenate([speech, np.zeros(SAMPLE_RATE, dtype=np.float32), speech])
    speech_rms = measure_speech_rms(speech, SAMPLE_RATE)
    assert abs(_db_between(measure_speech_rms(padded, SAMPLE_RATE), speech_rms)) < 0.5
    assert speech_rms == pytest.approx((8000 / I16) / math.sqrt(2), rel=0.05)


def test_measure_speech_rms_silence_is_zero() -> None:
    assert measure_speech_rms(np.zeros(SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE) == 0


def test_match_peak_to_target_scales_up_and_down() -> None:
    assert measure_peak(match_peak_to_target(_tone(0.5, 4000), 12_000 / I16)) == pytest.approx(
        12_000 / I16, rel=1e-4
    )
    assert measure_peak(match_peak_to_target(_tone(0.5, 20_000), 8_000 / I16)) == pytest.approx(
        8_000 / I16, rel=1e-4
    )


def test_soft_ceiling_keeps_peak_within_limit() -> None:
    loud = np.concatenate([_tone(0.2, 12_000), _tone(0.01, 30_000)])
    scaled = scale_with_soft_ceiling(loud, 3, 32_000 / I16)
    assert measure_peak(scaled) <= 32_000 / I16 + 1e-6


def test_soft_ceiling_scales_under_knee_linearly() -> None:
    quiet = _tone(0.2, 4000)
    scaled = scale_with_soft_ceiling(quiet, 2, 32_000 / I16)
    assert measure_peak(scaled) == pytest.approx(measure_peak(quiet) * 2, rel=1e-4)


def test_match_loudness_follows_speech_not_a_transient() -> None:
    english = _tone(2, 9000)
    tts = np.concatenate([_tone(2, 3000), _tone(0.02, 22_000)])
    matched = fit_level_to_reference(tts, SAMPLE_RATE, english, SAMPLE_RATE)
    target = measure_speech_rms(english, SAMPLE_RATE)
    assert abs(_db_between(measure_speech_rms(matched, SAMPLE_RATE), target)) < 1
    assert measure_peak(matched) <= 32_000 / I16 + 1e-5


def test_match_loudness_turns_a_loud_line_down() -> None:
    english = _tone(2, 4000)
    tts = _tone(2, 20_000)
    matched = fit_level_to_reference(tts, SAMPLE_RATE, english, SAMPLE_RATE)
    target = measure_speech_rms(english, SAMPLE_RATE)
    assert abs(_db_between(measure_speech_rms(matched, SAMPLE_RATE), target)) < 0.5


def test_quiet_tail_is_lifted_then_matched_to_the_first_clip() -> None:
    """Level-only scale keeps a fade; the loudness stage flattens it, then matches."""
    n = SAMPLE_RATE * 6
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    envelope = np.linspace(1.0, 0.25, n, dtype=np.float32)
    faded = (np.float32(9000 / I16) * envelope * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    reference = _tone(2, 9000)

    rms_only = fit_level_to_reference(faded, SAMPLE_RATE, reference, SAMPLE_RATE)
    hop = SAMPLE_RATE
    rms_only_ratio = _rms(rms_only, n - 2 * hop, n - hop) / _rms(rms_only, hop, 2 * hop)
    assert rms_only_ratio < 0.65

    fixed = match_loudness_to_reference(faded, SAMPLE_RATE, reference, SAMPLE_RATE)
    early = _rms(fixed, hop, 2 * hop)
    late = _rms(fixed, n - 2 * hop, n - hop)
    target = measure_speech_rms(reference, SAMPLE_RATE)
    assert late / early > 0.9
    assert abs(_db_between(measure_speech_rms(fixed, SAMPLE_RATE), target)) < 1.5


def test_loudness_stage_matches_source_level_after_remaster() -> None:
    reference = _tone(2.0, 9000)
    quiet = _tone(2.0, 2500) + _tone(2.0, 3200, freq=3200.0) * 0.08
    matched = match_loudness_to_reference(quiet, SAMPLE_RATE, reference, SAMPLE_RATE)
    target = measure_speech_rms(reference, SAMPLE_RATE)
    assert abs(_db_between(measure_speech_rms(matched, SAMPLE_RATE), target)) < 1.5
    assert matched.size == quiet.size


def test_finish_synthesis_flags_skip_loudness_and_timing() -> None:
    reference = _tone(2.0, 4000)
    loud = _tone(3.0, 16000)
    text = "tata tata tata tata"
    skipped = finish_synthesis_audio(
        loud,
        SAMPLE_RATE,
        reference,
        SAMPLE_RATE,
        synthesis_text=text,
        reference_text=text,
        match_loudness=False,
        match_timing=False,
    )
    assert skipped.size == loud.size
    assert abs(measure_speech_rms(skipped, SAMPLE_RATE) - measure_speech_rms(loud, SAMPLE_RATE)) < 0.01

    timed = finish_synthesis_audio(
        loud,
        SAMPLE_RATE,
        reference,
        SAMPLE_RATE,
        synthesis_text=text,
        reference_text=text,
        match_loudness=False,
        match_timing=True,
    )
    assert timed.size < loud.size


def _rms(samples: np.ndarray, start: int, end: int) -> float:
    chunk = samples[start:end]
    return float(np.sqrt(np.mean(chunk * chunk)))
