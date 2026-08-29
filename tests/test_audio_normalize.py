"""Tests for the ffmpeg filter chain applied to exported clip WAVs."""

from __future__ import annotations

from dataclasses import replace

import pytest

from fish_studio.config import SegmentationConfig, _dataclass_from_env
from fish_studio.dataset.audio_normalize import (
    build_silence_trim_filter,
    ffmpeg_output_args,
)


def test_trim_filter_reverses_to_reach_the_tail() -> None:
    chain = build_silence_trim_filter(SegmentationConfig())

    assert chain is not None
    # Two trims separated by areverse: the tail is only reachable reversed, and
    # trimming from the start each time leaves pauses inside speech alone.
    assert chain.count("silenceremove=") == 2
    assert chain.count("areverse") == 2
    assert chain.endswith("areverse")
    assert "start_threshold=-50.0dB" in chain
    assert "start_silence=0.05" in chain


def test_trim_filter_disabled() -> None:
    config = replace(SegmentationConfig(), trim_silence=False)

    assert build_silence_trim_filter(config) is None


def test_trim_runs_before_loudnorm() -> None:
    args = ffmpeg_output_args(SegmentationConfig())

    chain = args[args.index("-af") + 1]
    # loudnorm measures integrated loudness, so padding must be gone first.
    assert chain.index("silenceremove") < chain.index("loudnorm")


def test_seeked_cuts_never_get_the_reversing_trim() -> None:
    """areverse waits for an EOF a seeked span never delivers, and hangs ffmpeg forever."""
    args = ffmpeg_output_args(SegmentationConfig(), trim_silence=False)

    chain = args[args.index("-af") + 1]
    assert "areverse" not in chain
    assert "silenceremove" not in chain
    assert "loudnorm" in chain


def test_clip_cuts_ask_for_no_trim() -> None:
    """The segmenter reads `-ss X -to Y -i source.webm`, so it must opt out."""
    import inspect

    from fish_studio.dataset import segment

    source = inspect.getsource(segment.AudioSegmenter._export_one_clip)
    assert "trim_silence=False" in source


def test_output_args_without_any_filter() -> None:
    config = replace(SegmentationConfig(), trim_silence=False, normalize_loudness=False)

    args = ffmpeg_output_args(config)

    assert "-af" not in args
    assert args == ["-ac", "1", "-ar", str(config.sample_rate), "-c:a", "pcm_s16le"]


def test_trim_settings_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEGMENTATION_TRIM_SILENCE", "false")
    monkeypatch.setenv("SEGMENTATION_TRIM_SILENCE_DB", "-45")
    monkeypatch.setenv("SEGMENTATION_TRIM_SILENCE_KEEP_SEC", "0.1")

    config = _dataclass_from_env(SegmentationConfig, "SEGMENTATION")

    assert config.trim_silence is False
    assert config.trim_silence_db == -45.0
    assert config.trim_silence_keep_sec == 0.1
