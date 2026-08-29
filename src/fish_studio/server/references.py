"""Voice-cloning reference handling for the HTTP server."""

from __future__ import annotations

import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from fish_studio.config import StressConfig
from fish_studio.loudness import scale_to_speech_lufs, speech_lufs
from fish_studio.stress import stressify

MAX_REFERENCES = 5
# Fish Speech concatenates multi-clip prompts with these tags; skip if the caller already sent one.
_SPEAKER_TAG = re.compile(r"<\|speaker:\d+\|>")
# Clause-length gap between clips: enough to read as a pause, short enough not
# to teach the model that this voice trails off.
_SEAM_GAP_SEC = 0.25


@dataclass(frozen=True)
class ReferenceClip:
    """One clone sample plus its transcript (already stress-marked)."""

    audio_path: Path
    text: str


def validate_reference_count(count: int) -> None:
    if count < 1:
        raise ValueError("at least one reference clip is required")
    if count > MAX_REFERENCES:
        raise ValueError(f"at most {MAX_REFERENCES} reference clips are allowed")


def normalize_reference_texts(
    texts: list[str | None],
    *,
    count: int,
    default_text: str,
    stress: StressConfig,
) -> list[str]:
    """Return one transcript per reference, applying the project stress settings."""
    validate_reference_count(count)
    if len(texts) > count:
        raise ValueError(f"expected at most {count} speaker_text value(s), got {len(texts)}")

    resolved: list[str] = []
    for index in range(count):
        raw = texts[index] if index < len(texts) else default_text
        text = (raw or default_text).strip()
        if not text:
            raise ValueError(f"speaker_text for reference {index + 1} must not be empty")
        resolved.append(stressify(text, stress))
    return resolved


def format_reference_text(texts: list[str]) -> str:
    """Build the ref_text payload Fish Speech / vLLM expect."""
    validate_reference_count(len(texts))
    if len(texts) == 1:
        return texts[0]

    # Multi-clip prompts need speaker tags so the model can tell the concatenated WAVs apart.
    parts: list[str] = []
    for index, text in enumerate(texts):
        if _SPEAKER_TAG.search(text):
            parts.append(text)
        else:
            parts.append(f"<|speaker:{index}|>{text}")
    return "\n".join(parts)


def concat_reference_audio(paths: list[Path], *, sample_rate: int = 44100) -> Path:
    """Concatenate mono reference clips in order (native Fish multi-ref semantics).

    The model reads the whole prompt as one voice, so a clip that is louder than
    the others pulls the cloned timbre toward itself. Every clip is scaled to the
    first clip's speech loudness, and a short gap separates them so the seam
    reads as a phrase pause instead of one continuous utterance.
    """
    validate_reference_count(len(paths))
    if len(paths) == 1:
        return paths[0]

    temp_dir = Path(tempfile.mkdtemp(prefix="fish-refs-"))
    try:
        clips = [
            _decode_mono(path, temp_dir / f"{index:02d}.wav", sample_rate)
            for index, path in enumerate(paths)
        ]
        target_lufs = speech_lufs(clips[0], sample_rate)
        if target_lufs is not None:
            clips = [clips[0]] + [
                scale_to_speech_lufs(clip, sample_rate, target_lufs) for clip in clips[1:]
            ]

        gap = np.zeros(int(sample_rate * _SEAM_GAP_SEC), dtype=np.float32)
        joined: list[np.ndarray] = []
        for index, clip in enumerate(clips):
            if index:
                joined.append(gap)
            joined.append(clip)

        # Caller owns cleanup of the returned file.
        persistent = Path(tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name)
        sf.write(str(persistent), np.concatenate(joined), sample_rate, format="WAV")
        return persistent
    finally:
        for path in temp_dir.glob("*"):
            path.unlink(missing_ok=True)
        temp_dir.rmdir()


def _decode_mono(path: Path, dest: Path, sample_rate: int) -> np.ndarray:
    """Decode one clip to mono at ``sample_rate`` via ffmpeg."""
    if not path.is_file():
        raise FileNotFoundError(f"reference clip not found: {path}")
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(path),
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-c:a",
        "pcm_s16le",
        str(dest),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {path.name}: {result.stderr[-500:]}")
    audio, rate = sf.read(str(dest), dtype="float32", always_2d=False)
    if not isinstance(audio, np.ndarray) or audio.size == 0:
        raise ValueError(f"reference clip is empty: {path}")
    if int(rate) != sample_rate:
        raise RuntimeError(f"expected {sample_rate} Hz after decode, got {rate} for {path.name}")
    return np.asarray(audio, dtype=np.float32)
