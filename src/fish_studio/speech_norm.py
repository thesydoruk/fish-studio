"""Even ragged synth loudness with ffmpeg ``dynaudnorm`` before timing.

Linear gain cannot lift quiet syllables relative to loud ones: both sides
scale the same. On real Fish takes ``speechnorm`` mostly raises the peak;
``dynaudnorm`` with a short window actually closes the loud/quiet gap
*before* pause detection and tempo run.
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from fish_studio.ffmpeg import run_ffmpeg

logger = logging.getLogger(__name__)

# Measured on five raw FO4 lines (match_timing=false). Short frame + tiny
# Gaussian so a torn syllable is leveled; target RMS keeps speech, not noise.
# speechnorm on the same takes barely moved p95/p20.
_FILTER = "dynaudnorm=f=100:g=3:p=0.95:m=15:r=0.2"


@dataclass(frozen=True)
class SpeechNormFit:
    """Result of running ``dynaudnorm`` on one take."""

    audio: np.ndarray
    sample_rate: int
    filter: str = _FILTER
    applied: bool = False
    skip_reason: str | None = None

    def metrics(self) -> dict[str, str | bool | None]:
        return {
            "applied": self.applied,
            "filter": self.filter,
            "skip_reason": self.skip_reason,
        }


def speech_norm_filter() -> str:
    """ffmpeg ``-af`` graph used on every synthesized line (``dynaudnorm``)."""
    return _FILTER


def normalize_speech(samples: np.ndarray, sample_rate: int) -> SpeechNormFit:
    """Return ``samples`` with syllable-scale leveling, same rate."""
    audio = np.asarray(samples, dtype=np.float32)
    if audio.size == 0 or sample_rate <= 0:
        return SpeechNormFit(audio=audio, sample_rate=max(sample_rate, 0), skip_reason="empty_audio")

    with tempfile.TemporaryDirectory(prefix="fish-speech-norm-") as tmp:
        src = Path(tmp) / "in.wav"
        dest = Path(tmp) / "out.wav"
        sf.write(str(src), audio, sample_rate, format="WAV")
        result = run_ffmpeg(
            [
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(src),
                "-af",
                _FILTER,
                "-ac",
                "1",
                "-ar",
                str(sample_rate),
                "-c:a",
                "pcm_s16le",
                str(dest),
            ],
            text=True,
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "ffmpeg failed").strip()[-500:]
            raise RuntimeError(f"ffmpeg dynaudnorm failed: {detail}")
        out, rate = sf.read(str(dest), dtype="float32", always_2d=False)

    if not isinstance(out, np.ndarray) or out.size == 0:
        raise RuntimeError("ffmpeg dynaudnorm returned empty audio")
    if int(rate) != sample_rate:
        raise RuntimeError(f"ffmpeg dynaudnorm changed sample rate: {rate} != {sample_rate}")
    return SpeechNormFit(
        audio=np.asarray(out, dtype=np.float32),
        sample_rate=int(rate),
        applied=True,
    )
