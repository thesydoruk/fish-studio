"""Volume and word-confidence checks for clip quality gating."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass

from fish_studio.config import QualityConfig
from fish_studio.dataset.transcript import TranscriptSegment, TranscriptWord


@dataclass
class SegmentAudioMetrics:
    mean_volume_db: float
    max_volume_db: float


@dataclass
class SegmentQualityMetrics:
    avg_word_score: float | None
    low_score_word_ratio: float | None
    mean_volume_db: float | None
    max_volume_db: float | None


def avg_word_score(words: list[TranscriptWord]) -> float | None:
    if not words:
        return None
    return sum(w.probability for w in words) / len(words)


def low_score_word_ratio(words: list[TranscriptWord], min_score: float) -> float | None:
    if not words:
        return None
    low = sum(1 for w in words if w.probability < min_score)
    return low / len(words)


def passes_word_quality(
    segment: TranscriptSegment,
    config: QualityConfig,
    *,
    require_aligned_words: bool = False,
) -> bool:
    if require_aligned_words and segment.text.strip() and not segment.words:
        return False
    if not segment.words:
        # Unaligned transcripts have no per-word scores; skip this gate rather than drop everything.
        return True

    avg = avg_word_score(segment.words)
    if avg is not None and avg < config.min_avg_word_score:
        return False

    ratio = low_score_word_ratio(segment.words, config.min_word_score)
    return not (ratio is not None and ratio > config.max_low_score_word_ratio)


def analyze_segment_volume(
    source_audio: str,
    start: float,
    end: float,
    padding_sec: float = 0.0,
) -> SegmentAudioMetrics | None:
    """Run ffmpeg ``volumedetect`` on the same padded span that will be exported."""
    start_at = max(0.0, start - padding_sec)
    end_at = end + padding_sec
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-nostats",
        "-ss",
        f"{start_at:.3f}",
        "-to",
        f"{end_at:.3f}",
        "-i",
        source_audio,
        "-af",
        "volumedetect",
        "-f",
        "null",
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    output = proc.stderr
    mean_match = re.search(r"mean_volume:\s(-?\d+(?:\.\d+)?)\s*dB", output)
    max_match = re.search(r"max_volume:\s(-?\d+(?:\.\d+)?)\s*dB", output)
    if not mean_match or not max_match:
        return None

    return SegmentAudioMetrics(
        mean_volume_db=float(mean_match.group(1)),
        max_volume_db=float(max_match.group(1)),
    )


def passes_audio_quality(metrics: SegmentAudioMetrics, config: QualityConfig) -> bool:
    if metrics.mean_volume_db < config.min_mean_volume_db:
        return False
    if metrics.mean_volume_db > config.max_mean_volume_db:
        return False
    return not metrics.max_volume_db > config.max_peak_volume_db


def evaluate_segment_quality(
    segment: TranscriptSegment,
    config: QualityConfig,
    source_audio: str,
    padding_sec: float,
    *,
    require_aligned_words: bool = False,
) -> tuple[bool, SegmentQualityMetrics]:
    metrics = SegmentQualityMetrics(
        avg_word_score=avg_word_score(segment.words),
        low_score_word_ratio=low_score_word_ratio(segment.words, config.min_word_score),
        mean_volume_db=None,
        max_volume_db=None,
    )

    if not passes_word_quality(segment, config, require_aligned_words=require_aligned_words):
        return False, metrics

    if not config.filter_audio_quality:
        return True, metrics

    audio = analyze_segment_volume(
        source_audio,
        segment.start,
        segment.end,
        padding_sec=padding_sec,
    )
    if audio is None:
        # Missing volumedetect output is treated as a fail so we do not keep unmeasured clips.
        return False, metrics

    metrics.mean_volume_db = audio.mean_volume_db
    metrics.max_volume_db = audio.max_volume_db
    return passes_audio_quality(audio, config), metrics
