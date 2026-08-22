"""Voice-cloning reference handling for the HTTP server."""

from __future__ import annotations

import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from fish_studio.config import StressConfig
from fish_studio.stress import stressify

MAX_REFERENCES = 5
# Fish Speech concatenates multi-clip prompts with these tags; skip if the caller already sent one.
_SPEAKER_TAG = re.compile(r"<\|speaker:\d+\|>")


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
    """Concatenate mono reference clips in order (native Fish multi-ref semantics)."""
    validate_reference_count(len(paths))
    if len(paths) == 1:
        return paths[0]

    normalized: list[Path] = []
    temp_dir = Path(tempfile.mkdtemp(prefix="fish-refs-"))
    try:
        for index, path in enumerate(paths):
            if not path.is_file():
                raise FileNotFoundError(f"reference clip not found: {path}")
            dest = temp_dir / f"{index:02d}.wav"
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
            normalized.append(dest)

        out = temp_dir / "combined.wav"
        list_file = temp_dir / "concat.txt"
        list_file.write_text(
            "\n".join(f"file '{path.resolve().as_posix()}'" for path in normalized) + "\n",
            encoding="utf-8",
        )
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_file),
            "-c:a",
            "pcm_s16le",
            str(out),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg concat failed: {result.stderr[-500:]}")

        # Caller owns cleanup of the returned path's parent directory.
        persistent = Path(tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name)
        persistent.write_bytes(out.read_bytes())
        return persistent
    finally:
        for path in temp_dir.glob("*"):
            path.unlink(missing_ok=True)
        temp_dir.rmdir()
