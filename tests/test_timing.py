"""Tests for pause-then-tempo timing fit."""

from __future__ import annotations

import numpy as np
import pytest

from fish_studio.timing import (
    TimingStretchError,
    _tempo_rate,
    _trim_edge_silence,
    count_syllables,
    duration_in_slot,
    fit_timing_to_reference,
    match_timing_to_reference,
    measure_active_speech_sec,
    stretch_rate_bounds,
    strip_nonspeech,
    syllable_rate_bounds,
    tempo_rate_bounds,
    time_stretch,
)

pytest.importorskip("numpy")

SAMPLE_RATE = 16_000
TEXT_EN = "tata tata tata tata"
TEXT_UK = "тата тата тата тата"


def _tone(duration_sec: float, amplitude: float = 0.2, freq: float = 220.0) -> np.ndarray:
    n = int(SAMPLE_RATE * duration_sec)
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _duration(samples: np.ndarray) -> float:
    return samples.size / SAMPLE_RATE


def test_strip_nonspeech_and_syllables() -> None:
    assert strip_nonspeech("*sigh* Hello there") == "Hello there"
    assert strip_nonspeech("[Сарказм] Ну звісно") == "Ну звісно"
    assert count_syllables(TEXT_EN) >= 4
    assert count_syllables(TEXT_UK) >= 4
    assert duration_in_slot(4.0, 4.0)


def test_time_stretch_faster_is_shorter() -> None:
    pytest.importorskip("parselmouth")
    original = _tone(2.0)
    faster = time_stretch(original, 1.12, SAMPLE_RATE)
    assert _duration(faster) == pytest.approx(2.0 / 1.12, rel=0.15)


def test_time_stretch_requires_sample_rate() -> None:
    with pytest.raises(TimingStretchError, match="sample_rate"):
        time_stretch(_tone(1.0), 1.12)


def test_rate_bounds_allow_large_tempo_gap() -> None:
    low, high = stretch_rate_bounds()
    assert low <= 0.5
    assert high >= 2.0


def test_praat_stretch_when_available() -> None:
    pytest.importorskip("parselmouth")
    from fish_studio.timing import _praat_stretch

    original = _tone(1.5)
    out = _praat_stretch(original, 1.10, SAMPLE_RATE)
    assert out.size < original.size


def test_skips_short_reference() -> None:
    line = _tone(0.5)
    synth = _tone(2.0)
    out = match_timing_to_reference(
        synth, SAMPLE_RATE, line, SAMPLE_RATE, text_uk=TEXT_UK, text_en=TEXT_EN
    )
    assert out.size == synth.size


def test_skips_when_text_is_too_thin() -> None:
    line = _tone(3.0)
    synth = _tone(4.0)
    out = match_timing_to_reference(
        synth, SAMPLE_RATE, line, SAMPLE_RATE, text_uk="Так", text_en="Hi"
    )
    assert out.size == synth.size


def test_enormous_pause_is_capped() -> None:
    pytest.importorskip("parselmouth")
    speech = _tone(1.5)
    huge = np.zeros(int(SAMPLE_RATE * 1.8), dtype=np.float32)
    synth = np.concatenate([speech, huge, speech])
    # Ref has a modest clause pause only.
    line = np.concatenate(
        [speech, np.zeros(int(SAMPLE_RATE * 0.25), dtype=np.float32), speech]
    )
    out = match_timing_to_reference(
        synth, SAMPLE_RATE, line, SAMPLE_RATE, text_uk=TEXT_UK, text_en=TEXT_EN
    )
    assert _longest_internal_pause(out) < 0.55
    assert _duration(out) < _duration(synth)


def test_long_synth_shrinks_pause_not_speech_cuts() -> None:
    pytest.importorskip("parselmouth")
    speech = _tone(2.2)
    phrase_pause = np.zeros(int(SAMPLE_RATE * 0.80), dtype=np.float32)
    synth = np.concatenate([speech, phrase_pause, speech])
    line = np.concatenate(
        [speech, np.zeros(int(SAMPLE_RATE * 0.20), dtype=np.float32), speech]
    )
    out = match_timing_to_reference(
        synth, SAMPLE_RATE, line, SAMPLE_RATE, text_uk=TEXT_UK, text_en=TEXT_EN
    )
    assert _duration(out) < _duration(synth)
    # Still a real clause gap — not collapsed to a word dip.
    assert _longest_internal_pause(out) >= 0.15


def test_slow_active_speech_speeds_up_even_in_slot() -> None:
    """Same wall duration as the ref, but denser/slower speech → speed up."""
    pytest.importorskip("parselmouth")
    line = np.concatenate(
        [
            _tone(2.5),
            np.zeros(int(SAMPLE_RATE * 1.5), dtype=np.float32),
        ]
    )
    synth = _tone(4.0)
    assert abs(_duration(synth) - _duration(line)) < 0.05
    out = match_timing_to_reference(
        synth, SAMPLE_RATE, line, SAMPLE_RATE, text_uk=TEXT_UK, text_en=TEXT_EN
    )
    assert _duration(out) < 3.5


def test_fast_active_speech_is_not_slowed() -> None:
    pytest.importorskip("parselmouth")
    line = _tone(4.0)
    synth = _tone(2.5)
    out = match_timing_to_reference(
        synth, SAMPLE_RATE, line, SAMPLE_RATE, text_uk=TEXT_UK, text_en=TEXT_EN
    )
    assert _duration(out) == pytest.approx(_duration(synth), rel=0.08)
    assert _tempo_rate(
        syl_per_sec=6.0, syl_per_sec_ref=4.5, duration_sec=2.5, slot_sec=4.0
    ) == pytest.approx(1.0)


def test_tempo_rate_is_capped_by_the_syllable_band() -> None:
    high = syllable_rate_bounds()[1]
    rate = _tempo_rate(syl_per_sec=5.0, syl_per_sec_ref=5.0, duration_sec=10.0, slot_sec=2.0)
    assert rate == pytest.approx(high / 5.0)
    assert rate < tempo_rate_bounds()[1]


def test_tempo_rate_hits_the_speed_cap_when_the_band_still_has_room() -> None:
    rate = _tempo_rate(syl_per_sec=2.5, syl_per_sec_ref=5.0, duration_sec=8.0, slot_sec=2.0)
    assert rate == pytest.approx(tempo_rate_bounds()[1])
    assert rate == pytest.approx(1.3)
    assert 2.5 * rate <= syllable_rate_bounds()[1] + 1e-9


def test_tempo_rate_floor_yields_to_a_slow_reference() -> None:
    """A deliberately unhurried original must not be pushed to the band floor."""
    slow_ref = _tempo_rate(syl_per_sec=2.5, syl_per_sec_ref=2.4, duration_sec=3.0, slot_sec=3.0)
    brisk_ref = _tempo_rate(syl_per_sec=2.5, syl_per_sec_ref=5.5, duration_sec=3.0, slot_sec=3.0)
    assert slow_ref == pytest.approx(1.0, abs=0.02)
    assert brisk_ref > 1.1


def test_overlong_line_is_never_slowed_down() -> None:
    """A dense take above the band must not be stretched further past the slot."""
    rate = _tempo_rate(syl_per_sec=7.39, syl_per_sec_ref=6.19, duration_sec=3.48, slot_sec=3.20)
    assert rate >= 1.0


def test_brisk_reference_lifts_the_band_ceiling() -> None:
    in_slot = _tempo_rate(syl_per_sec=6.5, syl_per_sec_ref=6.2, duration_sec=3.0, slot_sec=3.0)
    assert in_slot == pytest.approx(1.0, abs=0.03)


def test_overlong_line_is_reported_not_over_compressed() -> None:
    pytest.importorskip("parselmouth")
    line = _tone(2.0)
    synth = _tone(5.0)
    fit = fit_timing_to_reference(
        synth, SAMPLE_RATE, line, SAMPLE_RATE, text_uk=TEXT_UK, text_en=TEXT_EN
    )
    assert fit.stretch_rate == pytest.approx(tempo_rate_bounds()[1], rel=0.01)
    assert fit.needs_shorter_line
    assert fit.overrun_sec > 0.3
    assert _duration(fit.audio) > fit.slot_sec


def test_fit_report_is_clean_when_the_line_fits() -> None:
    pytest.importorskip("parselmouth")
    speech = _tone(1.5)
    synth = np.concatenate([speech, np.zeros(int(SAMPLE_RATE * 1.8), dtype=np.float32), speech])
    line = np.concatenate([speech, np.zeros(int(SAMPLE_RATE * 0.25), dtype=np.float32), speech])
    fit = fit_timing_to_reference(
        synth, SAMPLE_RATE, line, SAMPLE_RATE, text_uk=TEXT_UK, text_en=TEXT_EN
    )
    assert fit.applied
    assert not fit.needs_shorter_line
    assert fit.overrun_sec == pytest.approx(0.0, abs=0.2)
    assert fit.metrics()["counts"]["syllables_uk"] >= 4


def test_word_gaps_are_not_ballooned() -> None:
    pytest.importorskip("parselmouth")
    speech = _tone(2.0)
    word_gap = np.zeros(int(SAMPLE_RATE * 0.08), dtype=np.float32)
    synth = np.concatenate([speech, word_gap, speech])
    line = np.concatenate([speech, np.zeros(int(SAMPLE_RATE * 1.0), dtype=np.float32), speech])
    out = match_timing_to_reference(
        synth, SAMPLE_RATE, line, SAMPLE_RATE, text_uk=TEXT_UK, text_en=TEXT_EN
    )
    # Tiny gaps must not become the only place to dump a 1s ref pause.
    assert _longest_internal_pause(out) < 0.55


def test_trim_keeps_a_quiet_ending() -> None:
    speech = _tone(2.0, amplitude=0.2)
    decay = _tone(0.18, amplitude=0.012)
    silence = np.zeros(int(SAMPLE_RATE * 0.30), dtype=np.float32)
    line = np.concatenate([speech, decay, silence])
    out = _trim_edge_silence(line, SAMPLE_RATE)
    assert out.size > speech.size + int(SAMPLE_RATE * 0.12)
    assert out.size < line.size - int(SAMPLE_RATE * 0.08)


def test_active_speech_ignores_silence() -> None:
    speech = _tone(1.0)
    padded = np.concatenate([speech, np.zeros(SAMPLE_RATE, dtype=np.float32), speech])
    assert measure_active_speech_sec(padded, SAMPLE_RATE) == pytest.approx(2.0, rel=0.15)


def _longest_internal_pause(samples: np.ndarray, floor_ratio: float = 0.05) -> float:
    abs_s = np.abs(samples)
    peak = float(np.max(abs_s)) or 1.0
    speech = abs_s >= peak * floor_ratio
    if not np.any(speech):
        return 0.0
    first = int(np.argmax(speech))
    last = int(speech.size - np.argmax(speech[::-1]))
    longest = 0
    run = 0
    for flag in speech[first:last]:
        if flag:
            run = 0
            continue
        run += 1
        longest = max(longest, run)
    return longest / SAMPLE_RATE
