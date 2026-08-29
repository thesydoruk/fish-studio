"""Tests for acoustic stress fallback."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from fish_studio.config import StressConfig
from fish_studio.stress import COMBINING_ACUTE, stressify
from fish_studio.stress_acoustic import apply_acoustic_stress


def _write_two_syllable_wav(path: Path, *, louder: str = "second") -> None:
    sr = 16000
    t = np.linspace(0, 0.12, int(sr * 0.12), endpoint=False)
    soft = 0.15 * np.sin(2 * np.pi * 180 * t)
    loud = 0.85 * np.sin(2 * np.pi * 180 * t)
    audio = np.concatenate([loud, soft] if louder == "first" else [soft, loud])
    sf.write(str(path), audio.astype(np.float32), sr)


def test_acoustic_stress_picks_louder_syllable(tmp_path: Path) -> None:
    wav = tmp_path / "clip.wav"
    _write_two_syllable_wav(wav, louder="second")

    # Acute immediately after the second vowel: мама́
    assert apply_acoustic_stress("мама", wav) == "мама" + COMBINING_ACUTE


def test_acoustic_stress_picks_first_when_louder(tmp_path: Path) -> None:
    wav = tmp_path / "clip.wav"
    _write_two_syllable_wav(wav, louder="first")

    assert apply_acoustic_stress("мама", wav) == "ма" + COMBINING_ACUTE + "ма"


def test_acoustic_stress_skips_when_no_clear_peak(tmp_path: Path) -> None:
    sr = 16000
    t = np.linspace(0, 0.24, int(sr * 0.24), endpoint=False)
    audio = 0.4 * np.sin(2 * np.pi * 180 * t)
    wav = tmp_path / "flat.wav"
    sf.write(str(wav), audio.astype(np.float32), sr)

    assert apply_acoustic_stress("мама", wav) == "мама"


def test_stressify_acoustic_fills_oov(tmp_path: Path) -> None:
    wav = tmp_path / "clip.wav"
    _write_two_syllable_wav(wav, louder="first")
    config = StressConfig(
        enabled=True,
        lexicon_path="",
        disambiguation="dictionary",
        acoustic_fallback=True,
    )
    marked = stressify("бубуба", config, audio_path=wav)
    assert COMBINING_ACUTE in marked


def test_stressify_without_audio_skips_acoustic() -> None:
    config = StressConfig(
        enabled=True,
        lexicon_path="",
        disambiguation="dictionary",
        acoustic_fallback=True,
    )
    # No audio → OOV stays unmarked (dictionary miss + no acoustic).
    assert COMBINING_ACUTE not in stressify("бубуба", config)


def test_acoustic_skips_function_words(tmp_path: Path) -> None:
    wav = tmp_path / "clip.wav"
    _write_two_syllable_wav(wav, louder="second")
    # Pronoun would otherwise get a mark from energy; filter must leave it alone.
    assert apply_acoustic_stress("йому", wav) == "йому"
