"""Transcript JSON schema shared by the dataset pipeline."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

from fish_studio.dataset.stats import TranscriptStats


@dataclass
class TranscriptWord:
    """Forced-alignment token used to cut clips on word boundaries."""

    word: st
    start: float
    end: float
    probability: float


@dataclass
class SoundEvent:
    """Non-speech timeline event from audio-intel (PANNs / AudioSet)."""

    label: st
    start: float
    end: float
    score: float
    prompt_relevant: bool = False


@dataclass
class SpeakerInfo:
    """Speaker roster entry from audio-intel diarization."""

    id: st
    speech_seconds: float = 0.0
    segment_count: int = 0
    # Optional pyannote embedding for cross-video clustering within a channel.
    embedding: list[float] | None = None


@dataclass
class TranscriptSegment:
    """One speech span from audio-intel, with Whisper quality metrics."""

    id: int
    start: float
    end: float
    text: st
    avg_logprob: float
    no_speech_prob: float
    compression_ratio: float
    words: list[TranscriptWord]
    language: str | None = None
    language_probability: float | None = None
    alignment_failed: bool = False
    speaker_id: str | None = None


@dataclass
class TranscriptResult:
    """Per-video transcript JSON written under ``work/<slug>/transcripts/``."""
    video_id: st
    source_audio: st
    language: st
    duration_sec: float
    segments: list[TranscriptSegment]
    transcribed_at: st
    aligned_at: str | None = None
    stats: TranscriptStats | None = None
    sound_events: list[SoundEvent] = field(default_factory=list)
    speakers: list[SpeakerInfo] = field(default_factory=list)


def probe_duration(audio_path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(audio_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        return 0.0
    try:
        return float(proc.stdout.strip())
    except ValueError:
        return 0.0


def save_transcript(
    result: TranscriptResult,
    output_dir: str | Path,
    *,
    out_path: Path | None = None,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = out_path or (output_dir / f"{result.video_id}.json")
    payload = asdict(result)
    if result.stats:
        payload["stats"] = result.stats.to_dict()

    fd, tmp_path = tempfile.mkstemp(suffix=".json.tmp", dir=output_dir, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        # Atomic replace so a crash mid-write cannot leave a truncated JSON for resume.
        os.replace(tmp_path, target)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    return target


def load_transcript(path: str | Path) -> TranscriptResult:
    with Path(path).open(encoding="utf-8") as handle:
        data = json.load(handle)

    segments: list[TranscriptSegment] = []
    for seg in data["segments"]:
        words = [TranscriptWord(**word) for word in seg.get("words", [])]
        segments.append(
            TranscriptSegment(
                id=seg["id"],
                start=seg["start"],
                end=seg["end"],
                text=seg["text"],
                avg_logprob=seg["avg_logprob"],
                no_speech_prob=seg["no_speech_prob"],
                compression_ratio=seg["compression_ratio"],
                words=words,
                language=seg.get("language"),
                language_probability=seg.get("language_probability"),
                alignment_failed=bool(seg.get("alignment_failed", False)),
                speaker_id=seg.get("speaker_id"),
            )
        )

    sound_events = [
        SoundEvent(
            label=str(event["label"]),
            start=float(event["start"]),
            end=float(event["end"]),
            score=float(event.get("score", 0.0)),
            prompt_relevant=bool(event.get("prompt_relevant", False)),
        )
        for event in data.get("sound_events") or []
    ]
    speakers = [
        SpeakerInfo(
            id=str(speaker["id"]),
            speech_seconds=float(speaker.get("speech_seconds", 0.0)),
            segment_count=int(speaker.get("segment_count", 0)),
            embedding=parse_embedding(speaker.get("embedding")),
        )
        for speaker in data.get("speakers") or []
    ]

    return TranscriptResult(
        video_id=data["video_id"],
        source_audio=data["source_audio"],
        language=data["language"],
        duration_sec=data["duration_sec"],
        segments=segments,
        transcribed_at=data["transcribed_at"],
        aligned_at=data.get("aligned_at"),
        stats=TranscriptStats.from_dict(data.get("stats")),
        sound_events=sound_events,
        speakers=speakers,
    )


def parse_embedding(raw: object) -> list[float] | None:
    """Parse a JSON embedding list; return None when missing or invalid."""
    if not isinstance(raw, list) or not raw:
        return None
    values: list[float] = []
    for item in raw:
        try:
            values.append(float(item))
        except (TypeError, ValueError):
            return None
    return values or None
