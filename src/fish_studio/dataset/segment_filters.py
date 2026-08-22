"""Transcript quality and alignment filters applied before clip building."""

from __future__ import annotations

from fish_studio.config import QualityConfig, SegmentationConfig
from fish_studio.dataset.audio_quality import passes_word_quality
from fish_studio.dataset.stats import SegmentStats
from fish_studio.dataset.transcript import TranscriptSegment


def passes_transcript_quality(segment: TranscriptSegment, config: QualityConfig) -> bool:
    """Reject hallucinations: low logprob, high no_speech_prob, or repetitive text."""
    if segment.avg_logprob < config.min_avg_logprob:
        return False
    if segment.no_speech_prob > config.max_no_speech_prob:
        return False
    return not (
        config.drop_compression_ratio_outliers
        and segment.compression_ratio > config.max_compression_ratio
    )


def filter_segments(
    segments: list[TranscriptSegment],
    *,
    quality: QualityConfig,
    segmentation: SegmentationConfig,
    target_language: str | None,
    aligned_at: str | None,
    stats: SegmentStats,
    filter_alignment_failed: bool = True,
) -> list[TranscriptSegment]:
    """Apply all transcript-level filters in one place before merge/split."""
    kept: list[TranscriptSegment] = []

    for seg in segments:
        if not seg.text.strip():
            stats.rejected_empty += 1
            continue

        if not passes_transcript_quality(seg, quality):
            stats.rejected_quality += 1
            continue

        if filter_alignment_failed and (
            seg.alignment_failed or (aligned_at is not None and seg.text.strip() and not seg.words)
        ):
            # Alignment ran but this span has no word timestamps — clip cuts would be guesses.
            stats.rejected_alignment_failed += 1
            continue

        if target_language and seg.language is not None and seg.language != target_language:
            stats.rejected_language += 1
            continue

        if seg.end - seg.start < segmentation.min_speech_duration_sec:
            stats.rejected_too_short += 1
            continue

        if not passes_word_quality(
            seg,
            quality,
            require_aligned_words=bool(aligned_at),
        ):
            stats.rejected_word_quality += 1
            continue

        kept.append(seg)

    return kept


def filter_allowed_languages(
    segments: list[TranscriptSegment],
    allowed: list[str],
    stats: SegmentStats,
) -> list[TranscriptSegment]:
    """Keep only ``uk`` / ``en`` (or whatever the source allowlist is)."""
    allowed_set = {item.strip().lower() for item in allowed if item.strip()}
    if not allowed_set:
        return segments

    kept: list[TranscriptSegment] = []
    for segment in segments:
        language = (segment.language or "").strip().lower()
        if language not in allowed_set:
            stats.rejected_language += 1
            continue
        kept.append(segment)
    return kept
