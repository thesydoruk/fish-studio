"""ffmpeg dynaudnorm evens real ragged takes; linear gain cannot."""

from __future__ import annotations

from pathlib import Path
import shutil

import numpy as np
import pytest
import soundfile as sf

from fish_studio.speech_norm import normalize_speech, speech_norm_filter

pytest.importorskip("numpy")

_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "synth_raw_uneven.wav"


def _window_rms(samples: np.ndarray, sample_rate: int, hop_ms: float = 50.0) -> np.ndarray:
    hop = max(1, int(sample_rate * hop_ms / 1000.0))
    values = [
        float(np.sqrt(np.mean(np.square(chunk, dtype=np.float64))))
        for start in range(0, samples.size, hop)
        if (chunk := samples[start : start + hop]).size
    ]
    return np.asarray(values, dtype=np.float64)


def _speech_windows(rms: np.ndarray) -> np.ndarray:
    peak = float(rms.max()) if rms.size else 0.0
    if peak <= 1e-6:
        return np.zeros(0, dtype=np.float64)
    return rms[rms >= peak * 0.08]


def _raggedness(samples: np.ndarray, sample_rate: int) -> tuple[float, float]:
    speech = _speech_windows(_window_rms(samples, sample_rate))
    assert speech.size >= 4
    lo, hi = np.percentile(speech, [20, 95])
    cv = float(speech.std() / max(speech.mean(), 1e-9))
    return cv, float(hi / max(lo, 1e-9))


def test_filter_is_dynaudnorm_not_a_gain() -> None:
    graph = speech_norm_filter()
    assert graph.startswith("dynaudnorm=")
    assert "speechnorm" not in graph
    assert "loudnorm" not in graph
    assert "volume=" not in graph


def test_empty_audio_is_skipped() -> None:
    fit = normalize_speech(np.zeros(0, dtype=np.float32), 16_000)
    assert fit.applied is False
    assert fit.skip_reason == "empty_audio"


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")
def test_real_raw_take_is_less_ragged() -> None:
    assert _FIXTURE.is_file(), f"missing fixture {_FIXTURE}"
    raw, rate = sf.read(str(_FIXTURE), dtype="float32", always_2d=False)
    fit = normalize_speech(np.asarray(raw, dtype=np.float32), int(rate))
    assert fit.applied
    in_cv, in_ratio = _raggedness(np.asarray(raw, dtype=np.float32), int(rate))
    out_cv, out_ratio = _raggedness(fit.audio, fit.sample_rate)
    # Linear gain would keep both. On this FO4 raw take dynaudnorm closed
    # p95/p20 from ~5.5 to ~3.0; keep a floor so a milder ffmpeg still fails.
    assert out_ratio < in_ratio * 0.75
    assert out_cv < in_cv
