"""Broadcast remaster should lift presence and control peaks without stretching time."""

from __future__ import annotations

import math

import numpy as np
import pytest

from fish_studio.enhance import remaster_speech
from fish_studio.loudness import measure_peak, measure_speech_rms

pytest.importorskip("numpy")

SAMPLE_RATE = 22_050


def _tone(duration_sec: float, freq: float, amp: float) -> np.ndarray:
    n = int(SAMPLE_RATE * duration_sec)
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    return (np.float32(amp) * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _band_rms(samples: np.ndarray, low: float, high: float) -> float:
    spec = np.fft.rfft(samples)
    freqs = np.fft.rfftfreq(samples.size, 1.0 / SAMPLE_RATE)
    mask = (freqs >= low) & (freqs < high)
    if not np.any(mask):
        return 0.0
    band = np.zeros_like(spec)
    band[mask] = spec[mask]
    recovered = np.fft.irfft(band, n=samples.size)
    return float(np.sqrt(np.mean(recovered * recovered)))


def test_enhance_keeps_duration_and_limits_peak() -> None:
    dull = _tone(1.2, 400, 0.15) + _tone(1.2, 3200, 0.01)
    out = remaster_speech(dull, SAMPLE_RATE)
    assert out.size == dull.size
    assert measure_peak(out) <= 10 ** (-0.3 / 20.0) + 1e-3


def test_enhance_lifts_quiet_presence_relative_to_body() -> None:
    dull = _tone(1.5, 450, 0.18) + _tone(1.5, 3200, 0.012)
    out = remaster_speech(dull, SAMPLE_RATE)
    before = _band_rms(dull, 2500, 4500) / max(_band_rms(dull, 250, 800), 1e-8)
    after = _band_rms(out, 2500, 4500) / max(_band_rms(out, 250, 800), 1e-8)
    assert after > before * 1.4


def test_enhance_does_not_swell_pauses() -> None:
    speech = _tone(0.6, 400, 0.2)
    pause = np.zeros(int(SAMPLE_RATE * 0.4), dtype=np.float32)
    line = np.concatenate([speech, pause, speech])
    out = remaster_speech(line, SAMPLE_RATE)
    # FFT crossovers ring a few milliseconds at the edges; measure the pause center.
    edge = int(SAMPLE_RATE * 0.05)
    mid = out[speech.size + edge : speech.size + pause.size - edge]
    assert float(np.sqrt(np.mean(mid * mid))) < 0.02


def test_remaster_holds_a_faded_ending() -> None:
    n = int(SAMPLE_RATE * 2.0)
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    envelope = np.ones(n, dtype=np.float32)
    fade = int(SAMPLE_RATE * 0.35)
    envelope[-fade:] = np.linspace(1.0, 0.15, fade, dtype=np.float32)
    line = (0.18 * envelope * np.sin(2 * np.pi * 400 * t)).astype(np.float32)
    out = remaster_speech(line, SAMPLE_RATE)
    mid = float(np.sqrt(np.mean(out[SAMPLE_RATE : SAMPLE_RATE + SAMPLE_RATE // 4] ** 2)))
    late = float(np.sqrt(np.mean(out[-int(SAMPLE_RATE * 0.12) : -int(SAMPLE_RATE * 0.03)] ** 2)))
    assert late / mid > 0.7


def test_remaster_holds_a_whispered_last_word_after_a_gap() -> None:
    """Fish sometimes drops the last word below the flatten floor after a short dip."""
    speech = _tone(1.4, 400, 0.2)
    gap = _tone(0.06, 400, 0.002)
    whisper = _tone(0.35, 400, 0.012)
    line = np.concatenate([speech, gap, whisper])
    out = remaster_speech(line, SAMPLE_RATE)
    mid = float(np.sqrt(np.mean(out[int(SAMPLE_RATE * 0.4) : int(SAMPLE_RATE * 1.0)] ** 2)))
    late = float(np.sqrt(np.mean(out[-int(SAMPLE_RATE * 0.28) : -int(SAMPLE_RATE * 0.04)] ** 2)))
    assert late / mid > 0.7


def test_remaster_holds_an_unstressed_final_syllable() -> None:
    """Last syllable of the last word can sit below the 12% flatten floor with no gap."""
    stem = _tone(1.4, 400, 0.2)
    ending = _tone(0.22, 400, 0.018)
    line = np.concatenate([stem, ending])
    out = remaster_speech(line, SAMPLE_RATE)
    mid = float(np.sqrt(np.mean(out[int(SAMPLE_RATE * 0.4) : int(SAMPLE_RATE * 1.0)] ** 2)))
    late = float(np.sqrt(np.mean(out[-int(SAMPLE_RATE * 0.18) : -int(SAMPLE_RATE * 0.03)] ** 2)))
    assert late / mid > 0.7


def test_remaster_does_not_pump_a_silent_hole_inside_speech() -> None:
    """True silence inside a word cannot be invented; do not pump the hole."""
    left = _tone(0.8, 400, 0.2)
    hole = np.zeros(int(SAMPLE_RATE * 0.025), dtype=np.float32)
    right = _tone(0.35, 400, 0.2)
    line = np.concatenate([left, hole, right])
    out = remaster_speech(line, SAMPLE_RATE)
    edge = int(SAMPLE_RATE * 0.004)
    mid_hole = out[left.size + edge : left.size + hole.size - edge]
    around = out[left.size - int(SAMPLE_RATE * 0.08) : left.size]
    hole_rms = float(np.sqrt(np.mean(mid_hole * mid_hole)))
    around_rms = float(np.sqrt(np.mean(around * around)))
    assert hole_rms < around_rms * 0.35


def test_remaster_levels_a_quiet_dip_inside_speech() -> None:
    """A present but quiet stretch inside a word should rise toward the vowels."""
    left = _tone(0.8, 400, 0.2)
    dip = _tone(0.05, 400, 0.03)
    right = _tone(0.4, 400, 0.2)
    line = np.concatenate([left, dip, right])
    out = remaster_speech(line, SAMPLE_RATE)
    edge = int(SAMPLE_RATE * 0.008)
    mid_dip = out[left.size + edge : left.size + dip.size - edge]
    around = out[left.size - int(SAMPLE_RATE * 0.08) : left.size]
    dip_rms = float(np.sqrt(np.mean(mid_dip * mid_dip)))
    around_rms = float(np.sqrt(np.mean(around * around)))
    assert dip_rms / around_rms > 0.55


def test_remaster_does_not_chop_a_loud_ending() -> None:
    line = _tone(1.5, 400, 0.18)
    out = remaster_speech(line, SAMPLE_RATE)
    late = float(np.sqrt(np.mean(out[-int(SAMPLE_RATE * 0.04) : -int(SAMPLE_RATE * 0.01)] ** 2)))
    mid = float(np.sqrt(np.mean(out[int(SAMPLE_RATE * 0.5) : int(SAMPLE_RATE * 0.8)] ** 2)))
    assert late / mid > 0.7


def test_enhance_leaves_silence_alone() -> None:
    silent = np.zeros(SAMPLE_RATE, dtype=np.float32)
    out = remaster_speech(silent, SAMPLE_RATE)
    assert measure_speech_rms(out, SAMPLE_RATE) == 0
    assert math.isclose(float(np.max(np.abs(out))), 0.0, abs_tol=1e-7)
