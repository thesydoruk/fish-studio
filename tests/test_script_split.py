"""Script-based EN/UK clip cuts and loudnorm skipping the source volume gate."""

from __future__ import annotations

from pathlib import Path

from fish_studio.config import QualityConfig, SegmentationConfig
from fish_studio.dataset.script_split import (
    assign_clip_language,
    split_segment_by_script,
    word_script,
)
from fish_studio.dataset.segment import AudioSegmenter
from fish_studio.dataset.transcript import (
    TranscriptResult,
    TranscriptSegment,
    TranscriptWord,
)


def _word(text: str, start: float, end: float) -> TranscriptWord:
    return TranscriptWord(word=text, start=start, end=end, probability=0.9)


def _segment(text: str, words: list[TranscriptWord]) -> TranscriptSegment:
    return TranscriptSegment(
        id=0,
        start=words[0].start,
        end=words[-1].end,
        text=text,
        avg_logprob=-0.1,
        no_speech_prob=0.0,
        compression_ratio=1.0,
        words=words,
        language="uk",
    )


def test_word_script_classifies_letters() -> None:
    assert word_script("injury") == "en"
    assert word_script("поранення") == "uk"
    assert word_script("—") is None
    assert word_script("PTSD") == "en"


def test_split_segment_keeps_monolingual() -> None:
    segment = _segment(
        "Це лише українська",
        [_word("Це", 0.0, 0.3), _word("лише", 0.3, 0.6), _word("українська", 0.6, 1.2)],
    )
    assert split_segment_by_script(segment) == [segment]


def test_split_segment_cuts_latin_from_cyrillic() -> None:
    segment = _segment(
        "An injury – поранення, травма.",
        [
            _word("An", 0.0, 0.2),
            _word("injury", 0.2, 0.7),
            _word("–", 0.7, 0.8),
            _word("поранення,", 0.8, 1.4),
            _word("травма.", 1.4, 2.0),
        ],
    )
    parts = split_segment_by_script(segment)
    assert [part.language for part in parts] == ["en", "uk"]
    assert parts[0].text == "An injury –"
    assert parts[1].text == "поранення, травма."
    assert parts[0].end == 0.8
    assert parts[1].start == 0.8


def test_assign_clip_language_keeps_en_from_ru_parent() -> None:
    assert assign_clip_language("one's shoulder", "en", "ru") == "en"


def test_assign_clip_language_marks_russian_letters() -> None:
    assert assign_clip_language("вывыхнутый плече", "uk", "uk") == "ru"


def test_assign_clip_language_uses_whisper_when_cyrillic_is_ambiguous() -> None:
    assert assign_clip_language("Армийско-английско", "uk", "ru") == "ru"
    assert assign_clip_language("поранення", "uk", "uk") == "uk"


def test_assign_clip_language_prefers_uk_letters_over_whisper_ru() -> None:
    assert assign_clip_language("синець, забій", "uk", "ru") == "uk"


def test_build_clips_drops_russian_keeps_en_and_uk(tmp_path: Path) -> None:
    transcript = TranscriptResult(
        video_id="v1",
        source_audio=str(tmp_path / "missing.wav"),
        language="uk",
        duration_sec=10.0,
        segments=[
            TranscriptSegment(
                id=0,
                start=0.0,
                end=2.0,
                text="An injury – вывыхнутый плече",
                avg_logprob=-0.1,
                no_speech_prob=0.0,
                compression_ratio=1.0,
                words=[
                    _word("An", 0.0, 0.3),
                    _word("injury", 0.3, 0.8),
                    _word("вывыхнутый", 0.9, 1.5),
                    _word("плече", 1.5, 2.0),
                ],
                language="ru",
            ),
            TranscriptSegment(
                id=1,
                start=3.0,
                end=5.0,
                text="синець, забій",
                avg_logprob=-0.1,
                no_speech_prob=0.0,
                compression_ratio=1.0,
                words=[_word("синець,", 3.0, 4.0), _word("забій", 4.0, 5.0)],
                language="uk",
            ),
        ],
        transcribed_at="2026-01-01T00:00:00+00:00",
        aligned_at="2026-01-01T00:00:01+00:00",
    )
    segmenter = AudioSegmenter(
        SegmentationConfig(
            min_duration_sec=0.5,
            min_speech_duration_sec=0.2,
            min_chars=3,
            normalize_loudness=True,
            split_by_script=True,
            allowed_languages=["uk", "en"],
        ),
        QualityConfig(filter_audio_quality=False),
    )
    clips, stats = segmenter.build_clips(transcript)
    texts = [clip.text for clip in clips]
    assert any("injury" in text for text in texts)
    assert any("забій" in text for text in texts)
    assert not any("вывыхнутый" in text for text in texts)
    assert stats.rejected_language >= 1


def test_merge_does_not_cross_script_language() -> None:
    segmenter = AudioSegmenter(
        SegmentationConfig(merge_gap_sec=0.5, max_duration_sec=12.0, max_chars=220),
        QualityConfig(filter_audio_quality=False),
    )
    merged = segmenter._merge_segments(
        [
            TranscriptSegment(
                id=0,
                start=0.0,
                end=1.0,
                text="An injury",
                avg_logprob=-0.1,
                no_speech_prob=0.0,
                compression_ratio=1.0,
                words=[],
                language="en",
                speaker_id="SPEAKER_00",
            ),
            TranscriptSegment(
                id=1,
                start=1.1,
                end=2.2,
                text="поранення",
                avg_logprob=-0.1,
                no_speech_prob=0.0,
                compression_ratio=1.0,
                words=[],
                language="uk",
                speaker_id="SPEAKER_00",
            ),
        ]
    )
    assert [part.language for part in merged] == ["en", "uk"]


def test_loudnorm_skips_source_volume_gate(tmp_path: Path) -> None:
    words = [_word("Привіт", 0.0, 0.4), _word("друже", 0.45, 1.6)]
    transcript = TranscriptResult(
        video_id="v1",
        source_audio=str(tmp_path / "missing.wav"),
        language="uk",
        duration_sec=4.0,
        segments=[
            TranscriptSegment(
                id=0,
                start=0.0,
                end=1.8,
                text="Привіт друже",
                avg_logprob=-0.1,
                no_speech_prob=0.0,
                compression_ratio=1.0,
                words=words,
                language="uk",
            )
        ],
        transcribed_at="2026-01-01T00:00:00+00:00",
        aligned_at="2026-01-01T00:00:01+00:00",
    )
    segmenter = AudioSegmenter(
        SegmentationConfig(
            min_duration_sec=0.5,
            min_speech_duration_sec=0.2,
            min_chars=3,
            normalize_loudness=True,
            split_by_script=True,
        ),
        QualityConfig(filter_audio_quality=True),
    )
    clips, stats = segmenter.build_clips(transcript)
    assert stats.rejected_audio_quality == 0
    assert len(clips) == 1
