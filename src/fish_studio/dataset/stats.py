"""Counters and human-readable summaries for pipeline rejection stats."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class TranscriptStats:
    """Rejection counters written into each transcript JSON for pipeline summaries."""

    vad_regions: int = 0
    segments_kept: int = 0
    rejected_non_target_language: int = 0
    rejected_quality: int = 0
    rejected_empty: int = 0
    rejected_too_short: int = 0
    rejected_alignment_failed: int = 0
    marked_alignment_failed: int = 0
    detected_languages: dict[str, int] = field(default_factory=dict)
    kept_speech_duration_sec: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> TranscriptStats:
        if not data:
            return cls()
        langs = data.get("detected_languages") or {}
        return cls(
            vad_regions=int(data.get("vad_regions", 0)),
            segments_kept=int(data.get("segments_kept", 0)),
            rejected_non_target_language=int(data.get("rejected_non_target_language", 0)),
            rejected_quality=int(data.get("rejected_quality", 0)),
            rejected_empty=int(data.get("rejected_empty", 0)),
            rejected_too_short=int(data.get("rejected_too_short", 0)),
            rejected_alignment_failed=int(data.get("rejected_alignment_failed", 0)),
            marked_alignment_failed=int(data.get("marked_alignment_failed", 0)),
            detected_languages=dict(langs),
            kept_speech_duration_sec=float(data.get("kept_speech_duration_sec", 0.0)),
        )

    def merge(self, other: TranscriptStats) -> None:
        self.segments_kept += other.segments_kept
        self.rejected_non_target_language += other.rejected_non_target_language
        self.rejected_quality += other.rejected_quality
        self.rejected_empty += other.rejected_empty
        self.rejected_too_short += other.rejected_too_short
        self.rejected_alignment_failed += other.rejected_alignment_failed
        self.marked_alignment_failed += other.marked_alignment_failed
        self.kept_speech_duration_sec += other.kept_speech_duration_sec
        for lang, count in other.detected_languages.items():
            self.detected_languages[lang] = self.detected_languages.get(lang, 0) + count

    def summary(self, target_language: str) -> str:
        rejected_total = (
            self.rejected_non_target_language
            + self.rejected_quality
            + self.rejected_empty
            + self.rejected_too_short
            + self.rejected_alignment_failed
        )
        parts = [
            f"kept {self.segments_kept} segments",
            f"rejected {rejected_total}",
        ]
        non_target = {
            lang: count
            for lang, count in self.detected_languages.items()
            if lang != target_language
        }
        if non_target:
            lang_bits = ", ".join(f"{lang}:{count}" for lang, count in sorted(non_target.items()))
            parts.append(f"non-{target_language} regions kept ({lang_bits})")
        if self.rejected_quality:
            parts.append(f"quality {self.rejected_quality}")
        if self.rejected_empty:
            parts.append(f"empty {self.rejected_empty}")
        if self.rejected_too_short:
            parts.append(f"too short {self.rejected_too_short}")
        if self.marked_alignment_failed:
            parts.append(f"align marked {self.marked_alignment_failed}")
        parts.append(f"speech {self.kept_speech_duration_sec / 60:.1f} min")
        return " | ".join(parts)


@dataclass
class SegmentStats:
    """Per-video clip-building counters (quality, junk, duration, alignment)."""

    clips_kept: int = 0
    rejected_empty: int = 0
    rejected_quality: int = 0
    rejected_language: int = 0
    rejected_too_short: int = 0
    rejected_text_length: int = 0
    rejected_duration: int = 0
    rejected_word_quality: int = 0
    rejected_audio_quality: int = 0
    rejected_alignment_failed: int = 0
    rejected_junk: int = 0
    kept_duration_sec: float = 0.0

    def summary(self) -> str:
        rejected = (
            self.rejected_empty
            + self.rejected_quality
            + self.rejected_language
            + self.rejected_too_short
            + self.rejected_text_length
            + self.rejected_duration
            + self.rejected_word_quality
            + self.rejected_audio_quality
            + self.rejected_alignment_failed
            + self.rejected_junk
        )
        return (
            f"kept {self.clips_kept} clips | rejected {rejected} "
            f"(empty {self.rejected_empty}, quality {self.rejected_quality}, "
            f"language {self.rejected_language}, too short {self.rejected_too_short}, "
            f"text {self.rejected_text_length}, "
            f"duration {self.rejected_duration}, words {self.rejected_word_quality}, "
            f"audio {self.rejected_audio_quality}, "
            f"align failed {self.rejected_alignment_failed}, "
            f"junk {self.rejected_junk}) | "
            f"speech {self.kept_duration_sec / 60:.1f} min"
        )
