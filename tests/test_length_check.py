"""Tests for the served-model early-termination probe."""

from __future__ import annotations

import array
import io
import math
import wave

from fish_studio.server.length_check import (
    BROKEN,
    OK,
    WEAK,
    LengthReport,
    format_report,
    verdict,
    wav_level_db,
    wav_seconds,
)


def make_report(ratios: list[float]) -> LengthReport:
    return LengthReport(label="step250", ratios=ratios, per_text=[(22, ratios)])


def tone(amplitude: float, seconds: float = 1.0, rate: int = 44100) -> bytes:
    samples = array.array(
        "h",
        (
            int(amplitude * 32767 * math.sin(2 * math.pi * 220 * i / rate))
            for i in range(int(rate * seconds))
        ),
    )
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(samples.tobytes())
    return buffer.getvalue()


def test_verdict_grades_on_the_worst_sample() -> None:
    assert verdict(0.79, fail_under=0.70) == OK
    assert verdict(0.70, fail_under=0.70) == OK
    assert verdict(0.65, fail_under=0.70) == WEAK
    assert verdict(0.37, fail_under=0.70) == BROKEN


def test_a_good_average_does_not_rescue_a_truncating_model() -> None:
    # Real signature of early termination: most samples fine, some collapse.
    report = make_report([0.95, 0.92, 0.90, 0.28])

    assert report.median > 0.85
    assert verdict(report.worst, fail_under=0.70) == BROKEN


def test_report_line_carries_verdict_and_per_length_detail() -> None:
    report = LengthReport(
        label="step250",
        ratios=[0.88, 0.79, 0.92],
        per_text=[(22, [0.88, 0.79]), (139, [0.92])],
    )

    line = format_report(report, fail_under=0.70)

    assert "step250" in line
    assert OK in line
    assert "22ch:88%/79%" in line
    assert "139ch:92%" in line


def test_a_quiet_checkpoint_fails_even_with_perfect_duration() -> None:
    """The failure that duration alone missed: right length, inaudible audio."""
    assert verdict(0.95, fail_under=0.70, level_drop=45.0) == BROKEN
    assert verdict(0.95, fail_under=0.70, level_drop=14.0) == WEAK
    assert verdict(0.95, fail_under=0.70, level_drop=3.0) == OK


def test_report_line_shows_the_level_gap() -> None:
    report = LengthReport(
        label="step12000",
        ratios=[0.95],
        per_text=[(22, [0.95])],
        level_drops=[12.0, 45.3, 30.0],
    )

    line = format_report(report, fail_under=0.70)

    assert "quiet= 45.3dB" in line
    assert BROKEN in line


def test_level_tracks_amplitude_in_decibels() -> None:
    loud = wav_level_db(tone(0.5))
    quiet = wav_level_db(tone(0.5 / 100))

    assert -9.5 < loud < -8.5  # a 0.5 sine is 0.354 rms, or -9.0 dBFS
    assert 39.0 < loud - quiet < 41.0  # a factor of 100 is 40 dB


def test_silence_does_not_blow_up_the_level_measurement() -> None:
    assert wav_level_db(tone(0.0)) <= -100.0


def test_wav_seconds_reads_the_header() -> None:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(44100)
        handle.writeframes(b"\x00\x00" * 44100 * 2)

    assert wav_seconds(buffer.getvalue()) == 2.0
