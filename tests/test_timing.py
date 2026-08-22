"""Tests for slot matching after loudness."""

from __future__ import annotations

import numpy as np
import pytest

from fish_studio.timing import (
    _MIN_KEEP_PAUSE_SEC,
    _speech_pause_regions,
    _split_stretched,
    _trim_edge_silence,
    count_syllables,
    match_timing_to_reference,
    measure_active_speech_sec,
    strip_nonspeech,
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
    assert count_syllables(TEXT_EN) == count_syllables(TEXT_UK)


def test_time_stretch_faster_is_shorter() -> None:
    original = _tone(2.0)
    faster = time_stretch(original, 1.25, SAMPLE_RATE)
    assert _duration(faster) == pytest.approx(2.0 / 1.25, rel=0.08)


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


def test_long_synth_moves_toward_the_line_slot() -> None:
    line = _tone(4.0)
    synth = _tone(4.4)
    out = match_timing_to_reference(
        synth, SAMPLE_RATE, line, SAMPLE_RATE, text_uk=TEXT_UK, text_en=TEXT_EN
    )
    assert _duration(out) < _duration(synth)
    assert abs(_duration(out) - 4.0) < abs(_duration(synth) - 4.0)


def test_split_stretched_keeps_island_proportions() -> None:
    stretched = np.arange(100, dtype=np.float32)
    pieces = _split_stretched(stretched, [40, 10, 50])
    assert [piece.size for piece in pieces] == [40, 10, 50]
    assert np.concatenate(pieces).size == 100


def test_long_synth_stretches_speech_not_phrase_pauses() -> None:
    speech = _tone(2.2)
    phrase_pause = np.zeros(int(SAMPLE_RATE * 0.40), dtype=np.float32)
    synth = np.concatenate([speech, phrase_pause, speech])
    line = _tone(3.4)
    out = match_timing_to_reference(
        synth, SAMPLE_RATE, line, SAMPLE_RATE, text_uk=TEXT_UK, text_en=TEXT_EN
    )
    assert _longest_internal_pause(out) >= _longest_internal_pause(synth) - 0.02
    assert _duration(out) < _duration(synth)
    speech_out = _duration(out) - _longest_internal_pause(out)
    assert speech_out < 4.4


def test_short_synth_is_padded_toward_the_slot() -> None:
    line = _tone(4.0)
    synth = _tone(2.0)
    out = match_timing_to_reference(
        synth, SAMPLE_RATE, line, SAMPLE_RATE, text_uk=TEXT_UK, text_en=TEXT_EN
    )
    assert _duration(out) == pytest.approx(4.0, rel=0.08)


def test_pause_expansion_lengthens_phrase_gaps_only() -> None:
    speech = _tone(2.0)
    phrase_pause = np.zeros(int(SAMPLE_RATE * 0.22), dtype=np.float32)
    synth = np.concatenate([speech, phrase_pause, speech])
    line = np.concatenate([speech, np.zeros(int(SAMPLE_RATE * 1.0), dtype=np.float32), speech])
    out = match_timing_to_reference(
        synth, SAMPLE_RATE, line, SAMPLE_RATE, text_uk=TEXT_UK, text_en=TEXT_EN
    )
    assert _longest_internal_pause(out) > _longest_internal_pause(synth)
    assert _longest_internal_pause(out) <= 0.45


def test_keep_threshold_freezes_clause_gaps_not_word_gaps() -> None:
    """Stretch freezes ≥100 ms pauses; 60 ms word gaps still ride with speech."""
    speech = _tone(1.0)
    clause = np.zeros(int(SAMPLE_RATE * 0.12), dtype=np.float32)
    word = np.zeros(int(SAMPLE_RATE * 0.06), dtype=np.float32)
    line = np.concatenate([speech, clause, speech, word, speech])
    frozen = [
        (end - start) / SAMPLE_RATE
        for start, end, is_pause in _speech_pause_regions(line, SAMPLE_RATE)
        if is_pause
    ]
    assert any(pause >= _MIN_KEEP_PAUSE_SEC - 0.01 for pause in frozen)
    assert not any(0.04 <= pause <= 0.08 for pause in frozen)


def test_long_synth_does_not_shrink_clause_pauses() -> None:
    """A ~120 ms clause gap is below the expand floor but must not be sped up."""
    speech = _tone(2.2)
    clause_pause = np.zeros(int(SAMPLE_RATE * 0.12), dtype=np.float32)
    synth = np.concatenate([speech, clause_pause, speech])
    line = _tone(3.4)
    out = match_timing_to_reference(
        synth, SAMPLE_RATE, line, SAMPLE_RATE, text_uk=TEXT_UK, text_en=TEXT_EN
    )
    assert _longest_internal_pause(out) >= _MIN_KEEP_PAUSE_SEC - 0.02
    assert _duration(out) < _duration(synth)


def test_word_gaps_are_not_inflated() -> None:
    speech = _tone(2.0)
    word_gap = np.zeros(int(SAMPLE_RATE * 0.08), dtype=np.float32)
    synth = np.concatenate([speech, word_gap, speech])
    line = np.concatenate([speech, np.zeros(int(SAMPLE_RATE * 1.0), dtype=np.float32), speech])
    out = match_timing_to_reference(
        synth, SAMPLE_RATE, line, SAMPLE_RATE, text_uk=TEXT_UK, text_en=TEXT_EN
    )
    assert _longest_internal_pause(out) < 0.15
    assert _duration(out) > _duration(synth)


def _longest_internal_pause(samples: np.ndarray, floor_ratio: float = 0.1) -> float:
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
