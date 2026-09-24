"""Conservative raw-synthesis silence / cutoff checks."""

from __future__ import annotations

import numpy as np
import pytest

from fish_studio.server.app import synthesis_response_headers
from fish_studio.server.settings import FishSpeechSettings
from fish_studio.server.synth_validate import (
    SYNTH_WARNING_HEADER,
    attach_voice,
    judge_raw_synth,
    line_voice_similarity,
    quality_warning,
)
from fish_studio.server.vllm_proxy import VllmFishProxy, _encode_wav
from fish_studio.server.voiceprint import VOICE_SIMILARITY_HEADER
from fish_studio.synthesis import SynthesisResult
from fish_studio.timing import count_syllables

pytest.importorskip("numpy")

SAMPLE_RATE = 16_000
# Long enough to score (syllables ≥ 4 and chars ≥ 12).
LINE = "Сьогодні гарна погода надворі"


def _tone(duration_sec: float, amplitude: float = 0.2) -> np.ndarray:
    n = int(SAMPLE_RATE * duration_sec)
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    return (amplitude * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def _silence(duration_sec: float = 1.0) -> np.ndarray:
    return np.zeros(int(SAMPLE_RATE * duration_sec), dtype=np.float32)


def test_silence_is_rejected() -> None:
    check = judge_raw_synth(_silence(2.0), SAMPLE_RATE, LINE)
    assert check.ok is False
    assert check.reason == "silence"
    assert check.active_speech_sec < 0.25


def test_empty_audio_is_silence() -> None:
    check = judge_raw_synth(np.zeros(0, dtype=np.float32), SAMPLE_RATE, LINE)
    assert check.ok is False
    assert check.reason == "silence"


def test_click_is_silence() -> None:
    n = int(SAMPLE_RATE * 0.08)
    click = np.zeros(n, dtype=np.float32)
    click[n // 2] = 0.9
    check = judge_raw_synth(click, SAMPLE_RATE, LINE)
    assert check.ok is False
    assert check.reason == "silence"


def test_cutoff_is_rejected() -> None:
    syllables = count_syllables(LINE)
    # Half a normal-rate take: implied syl/s == 2 × 5 == 10.
    audio = _tone(syllables / 10.0)
    check = judge_raw_synth(audio, SAMPLE_RATE, LINE)
    assert check.ok is False
    assert check.reason == "cutoff"
    assert check.implied_syl_per_sec >= 10.0


def test_valid_fast_speech_is_kept() -> None:
    syllables = count_syllables(LINE)
    # 8 syl/s is brisk but possible; below the cutoff ceiling.
    audio = _tone(syllables / 8.0)
    check = judge_raw_synth(audio, SAMPLE_RATE, LINE)
    assert check.ok is True
    assert check.reason == ""
    assert 7.0 < check.implied_syl_per_sec < 9.0


def test_trailing_pause_does_not_reject() -> None:
    syllables = count_syllables(LINE)
    speech = _tone(syllables / 5.0)
    audio = np.concatenate([speech, _silence(8.0)])
    check = judge_raw_synth(audio, SAMPLE_RATE, LINE)
    assert check.ok is True
    assert check.implied_syl_per_sec < 7.0


def test_short_line_is_not_scored() -> None:
    check = judge_raw_synth(_silence(0.4), SAMPLE_RATE, "Так.")
    assert check.ok is True
    assert check.reason == ""


def test_stage_direction_does_not_inflate_syllables() -> None:
    text = "[сміється] Так."
    check = judge_raw_synth(_silence(0.4), SAMPLE_RATE, text)
    assert check.ok is True


def _proxy() -> VllmFishProxy:
    return VllmFishProxy(
        FishSpeechSettings(
            base_url="http://127.0.0.1:9",
            voice="test",
            timeout_sec=5.0,
            max_new_tokens=64,
            chunk_length=200,
            max_concurrent_requests=1,
            default_reference_text="",
            default_language="uk",
            synth_log_enabled=False,
            synth_log_dir=None,
        )
    )


def test_quality_warning_empty_when_ok() -> None:
    assert quality_warning([{"ok": True, "reason": ""}]) == ""
    assert quality_warning([]) == ""


def test_quality_warning_describes_silence_and_cutoff() -> None:
    silence = quality_warning(
        [
            {
                "ok": False,
                "reason": "silence",
                "active_speech_sec": 0.0,
                "implied_syl_per_sec": 0.0,
            }
        ]
    )
    assert silence == "silence (0.00s active speech)"
    cutoff = quality_warning(
        [
            {
                "ok": False,
                "reason": "cutoff",
                "active_speech_sec": 0.41,
                "implied_syl_per_sec": 14.2,
            }
        ]
    )
    assert cutoff == "cutoff (14.2 syl/s, 0.41s active speech)"


def test_quality_warning_labels_chunks() -> None:
    text = quality_warning(
        [
            {"ok": True, "reason": ""},
            {
                "ok": False,
                "reason": "silence",
                "active_speech_sec": 0.0,
                "implied_syl_per_sec": 0.0,
            },
        ]
    )
    assert text == "chunk 2: silence (0.00s active speech)"


def test_response_headers_include_warning() -> None:
    headers = synthesis_response_headers(
        SynthesisResult(
            wav_bytes=b"RIFF",
            sample_rate=44100,
            language="uk",
            warning="silence (0.00s active speech)",
        )
    )
    assert headers["X-Sample-Rate"] == "44100"
    assert headers["X-Language"] == "uk"
    assert headers[SYNTH_WARNING_HEADER] == "silence (0.00s active speech)"


def test_response_headers_omit_warning_when_clean() -> None:
    headers = synthesis_response_headers(
        SynthesisResult(wav_bytes=b"RIFF", sample_rate=16000, language="uk")
    )
    assert SYNTH_WARNING_HEADER not in headers
    assert VOICE_SIMILARITY_HEADER not in headers


def test_response_headers_include_voice_similarity() -> None:
    headers = synthesis_response_headers(
        SynthesisResult(
            wav_bytes=b"RIFF",
            sample_rate=16000,
            language="uk",
            voice_similarity=0.312,
        )
    )
    assert headers[VOICE_SIMILARITY_HEADER] == "0.312"


def test_attach_voice_does_not_hide_silence() -> None:
    check = attach_voice(judge_raw_synth(_silence(2.0), SAMPLE_RATE, LINE), 0.05)
    assert check.ok is False
    assert check.reason == "silence"
    assert check.voice_similarity == 0.05


def test_line_voice_similarity_is_the_weakest_chunk() -> None:
    assert line_voice_similarity(
        [{"voice_similarity": 0.41}, {"voice_similarity": 0.29}]
    ) == pytest.approx(0.29)


def test_attach_voice_records_the_score_without_judging_it() -> None:
    """The server reports the cosine; a threshold is the client's business."""
    check = attach_voice(judge_raw_synth(_tone(3.0), SAMPLE_RATE, LINE), 0.05)
    assert check.ok is True
    assert check.voice_similarity == 0.05


def test_attach_voice_does_not_hide_silence_behind_a_score() -> None:
    check = attach_voice(judge_raw_synth(_silence(2.0), SAMPLE_RATE, LINE), 0.05)
    assert check.ok is False
    assert check.reason == "silence"


def test_one_take_per_chunk_even_when_it_is_silence(monkeypatch) -> None:
    proxy = _proxy()
    calls = {"n": 0}

    def _fake_speech(*_args, **_kwargs) -> bytes:
        calls["n"] += 1
        return _encode_wav(_silence(1.0), SAMPLE_RATE)

    monkeypatch.setattr(proxy, "_request_speech", _fake_speech)
    _, _, report = proxy._synthesize_chunk_judged(
        client=None,  # type: ignore[arg-type]
        chunk=LINE,
        mime="audio/wav",
        ref_b64="",
        reference_text="",
    )
    assert calls["n"] == 1
    assert report["ok"] is False
    assert report["reason"] == "silence"
