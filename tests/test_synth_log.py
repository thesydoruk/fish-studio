"""Synthesis request dump logger."""

from __future__ import annotations

import json

import numpy as np
import pytest
import soundfile as sf

from fish_studio.server.synth_log import SynthesisRequestLogger

pytest.importorskip("numpy")

SAMPLE_RATE = 16_000


def _tone(duration_sec: float = 0.2) -> np.ndarray:
    n = int(SAMPLE_RATE * duration_sec)
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    return (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def test_logger_writes_request_and_latest(tmp_path) -> None:
    ref = tmp_path / "ref.wav"
    sf.write(str(ref), _tone(0.3), SAMPLE_RATE)
    logger = SynthesisRequestLogger(tmp_path / "synthesis", keep=5, enabled=True)

    dest = logger.log(
        text_raw="Сира 2 грн",
        text_prepared="Сира дві гривні",
        chunks=["Сира дві гривні"],
        language="uk",
        match_timing=True,
        reference_paths=[ref],
        reference_texts=["реф"],
        raw_audio=_tone(0.5),
        final_audio=_tone(0.4),
        sample_rate=SAMPLE_RATE,
    )
    assert dest is not None
    assert (dest / "request.json").is_file()
    assert (dest / "synth_raw.wav").is_file()
    assert (dest / "synth_final.wav").is_file()
    assert (dest / "reference_0.wav").is_file()
    payload = json.loads((dest / "request.json").read_text(encoding="utf-8"))
    assert payload["text_raw"] == "Сира 2 грн"
    assert payload["text_prepared"] == "Сира дві гривні"
    assert payload["match_timing"] is True
    assert logger.latest_dir() == dest


def test_logger_prunes_old_requests(tmp_path) -> None:
    logger = SynthesisRequestLogger(tmp_path / "synthesis", keep=2, enabled=True)
    paths = []
    for index in range(4):
        paths.append(
            logger.log(
                text_raw=f"t{index}",
                text_prepared=f"t{index}",
                chunks=[f"t{index}"],
                language="uk",
                match_timing=False,
                reference_paths=[],
                reference_texts=[],
                raw_audio=_tone(0.1),
                final_audio=_tone(0.1),
                sample_rate=SAMPLE_RATE,
            )
        )
    surviving = sorted(p.name for p in (tmp_path / "synthesis").iterdir() if p.is_dir())
    assert len(surviving) == 2
    assert paths[-1] is not None
    assert paths[-1].name in surviving
    assert paths[0] is not None
    assert paths[0].name not in surviving


def test_logger_disabled_is_noop(tmp_path) -> None:
    logger = SynthesisRequestLogger(tmp_path / "synthesis", enabled=False)
    assert (
        logger.log(
            text_raw="x",
            text_prepared="x",
            chunks=["x"],
            language="uk",
            match_timing=True,
            reference_paths=[],
            reference_texts=[],
            raw_audio=_tone(0.1),
            final_audio=_tone(0.1),
            sample_rate=SAMPLE_RATE,
        )
        is None
    )
    assert not (tmp_path / "synthesis").exists()
