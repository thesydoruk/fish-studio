"""Clip length has to cover what one synthesis request asks the model to generate."""

from __future__ import annotations

from fish_studio.config import QualityConfig, SegmentationConfig
from fish_studio.dataset.segment import AudioSegmenter
from fish_studio.dataset.transcript import (
    TranscriptResult,
    TranscriptSegment,
    TranscriptWord,
)

# Measured over the Ukrainian training corpus; used to turn a character budget
# into the seconds of speech a generation has to sustain.
CHARS_PER_SECOND = 14.2


def _segment(index: int, start: float, end: float, text: str) -> TranscriptSegment:
    step = (end - start) / max(1, len(text.split()))
    words = [
        TranscriptWord(word=word, start=start + i * step, end=start + (i + 1) * step,
                       probability=0.9)
        for i, word in enumerate(text.split())
    ]
    return TranscriptSegment(
        id=index,
        start=start,
        end=end,
        text=text,
        avg_logprob=-0.1,
        no_speech_prob=0.0,
        compression_ratio=1.0,
        words=words,
        language="uk",
        speaker_id="spk_0",
    )


def _transcript(segments: list[TranscriptSegment]) -> TranscriptResult:
    return TranscriptResult(
        video_id="v1",
        source_audio="/tmp/v1.wav",
        language="uk",
        duration_sec=segments[-1].end,
        segments=segments,
        transcribed_at="2026-01-01T00:00:00Z",
    )


def _segmenter(**overrides) -> AudioSegmenter:
    config = SegmentationConfig(min_speech_duration_sec=0.2, min_chars=3, **overrides)
    return AudioSegmenter(config, QualityConfig(filter_audio_quality=False))


def test_default_clip_cap_covers_a_full_synthesis_chunk() -> None:
    """A 200-character request is ~14 s of speech; clips must be allowed to run that long."""
    config = SegmentationConfig()
    assert config.max_duration_sec >= 200 / CHARS_PER_SECOND
    assert config.max_chars > 200


def test_merging_fills_a_clip_up_to_the_duration_cap() -> None:
    """Sentence-length pieces separated by ordinary pauses become one long clip."""
    segments = []
    start = 0.0
    for index in range(8):
        end = start + 2.0
        segments.append(_segment(index, start, end, "Це до́сить дов́ге ре́чення про мо́ву."))
        start = end + 0.4  # an ordinary sentence pause

    clips, _ = _segmenter().build_clips(_transcript(segments))

    assert clips, "expected at least one clip"
    longest = max(clip.duration for clip in clips)
    assert longest > 12.0, f"merging stopped at {longest:.1f}s, too short to teach long lines"
    assert longest <= SegmentationConfig().max_duration_sec


def test_a_tight_gap_budget_cannot_reach_long_clips() -> None:
    """The regression that shipped: pauses wider than the gap budget block every merge."""
    segments = []
    start = 0.0
    for index in range(8):
        end = start + 2.0
        segments.append(_segment(index, start, end, "Це до́сить дов́ге ре́чення про мо́ву."))
        start = end + 0.4

    clips, _ = _segmenter(merge_gap_sec=0.3).build_clips(_transcript(segments))

    assert max(clip.duration for clip in clips) < 3.0


def test_character_cap_does_not_bind_before_the_duration_cap() -> None:
    """max_chars must leave room, or clips stop short of the seconds we actually need."""
    config = SegmentationConfig()
    assert config.max_chars >= config.max_duration_sec * CHARS_PER_SECOND
