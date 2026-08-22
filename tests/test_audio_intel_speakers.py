"""Tests for audio-intel diarization / sound-event parsing and junk filtering."""

from __future__ import annotations

from pathlib import Path

from fish_studio.config import AudioIntelConfig, QualityConfig, SegmentationConfig
from fish_studio.dataset.audio_intel import AudioIntelClient
from fish_studio.dataset.junk_filter import filter_junk_segments, overlaps_any_sound_event
from fish_studio.dataset.merge import list_export_ready_datasets, merge_datasets
from fish_studio.dataset.segment import AudioClip, AudioSegmenter
from fish_studio.dataset.speakers import resolve_speaker_name
from fish_studio.dataset.stats import SegmentStats
from fish_studio.dataset.transcript import (
    SoundEvent,
    TranscriptResult,
    TranscriptSegment,
    TranscriptWord,
    load_transcript,
    save_transcript,
)


def _speech(
    *,
    start: float,
    end: float,
    text: str,
    speaker_id: str | None = None,
    words: list[dict] | None = None,
) -> dict:
    payload = {
        "kind": "speech",
        "start": start,
        "end": end,
        "text": text,
        "avg_logprob": -0.2,
        "no_speech_prob": 0.05,
        "compression_ratio": 1.1,
        "language": "uk",
        "words": words
        or [
            {
                "word": part,
                "start": start + i * 0.2,
                "end": start + i * 0.2 + 0.15,
                "probability": 0.9,
            }
            for i, part in enumerate(text.split())
        ],
    }
    if speaker_id is not None:
        payload["speaker_id"] = speaker_id
    return payload


def test_from_api_payload_keeps_speakers_and_sound_events(tmp_path: Path) -> None:
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"RIFF")
    client = AudioIntelClient(AudioIntelConfig(diarize=True, sound_events=True, align=True))
    result = client._from_api_payload(
        "vid1",
        audio,
        {
            "language": "uk",
            "duration": 10.0,
            "alignment_applied": True,
            "speakers": [
                {
                    "id": "SPEAKER_00",
                    "speech_seconds": 4.0,
                    "segment_count": 1,
                    "embedding": [0.1, 0.2, 0.3],
                },
                {"id": "SPEAKER_01", "speech_seconds": 3.0, "segment_count": 1},
            ],
            "segments": [
                _speech(start=0.0, end=2.0, text="Привіт друже", speaker_id="SPEAKER_00"),
                {
                    "kind": "sound",
                    "label": "Music",
                    "start": 2.0,
                    "end": 5.0,
                    "score": 0.8,
                    "prompt_relevant": True,
                },
                _speech(start=5.0, end=7.0, text="Добрий день", speaker_id="SPEAKER_01"),
            ],
        },
    )

    assert [seg.speaker_id for seg in result.segments] == ["SPEAKER_00", "SPEAKER_01"]
    assert len(result.sound_events) == 1
    assert result.sound_events[0].label == "Music"
    assert [speaker.id for speaker in result.speakers] == ["SPEAKER_00", "SPEAKER_01"]
    assert result.speakers[0].embedding == [0.1, 0.2, 0.3]
    assert result.speakers[1].embedding is None

    path = save_transcript(result, tmp_path)
    loaded = load_transcript(path)
    assert loaded.segments[0].speaker_id == "SPEAKER_00"
    assert loaded.sound_events[0].label == "Music"
    assert loaded.speakers[1].id == "SPEAKER_01"
    assert loaded.speakers[0].embedding == [0.1, 0.2, 0.3]


def test_junk_overlap_drops_any_non_speech_event() -> None:
    contaminated = TranscriptSegment(
        id=0,
        start=1.0,
        end=3.0,
        text="текст під музику",
        avg_logprob=-0.2,
        no_speech_prob=0.1,
        compression_ratio=1.0,
        words=[],
    )
    clean = TranscriptSegment(
        id=1,
        start=4.0,
        end=6.0,
        text="чиста мова",
        avg_logprob=-0.2,
        no_speech_prob=0.1,
        compression_ratio=1.0,
        words=[],
    )
    events = [
        SoundEvent(label="Music", start=1.2, end=2.8, score=0.9),
        SoundEvent(label="Door", start=0.0, end=0.2, score=0.4),
    ]
    stats = SegmentStats()
    kept = filter_junk_segments([contaminated, clean], events, stats=stats)
    assert kept == [clean]
    assert stats.rejected_junk == 1
    assert overlaps_any_sound_event(contaminated, events) is True
    assert overlaps_any_sound_event(clean, events) is False


def test_merge_segments_does_not_cross_speakers() -> None:
    segmenter = AudioSegmenter(
        SegmentationConfig(merge_gap_sec=0.5, max_duration_sec=12.0, max_chars=220),
        QualityConfig(filter_audio_quality=False),
        speaker_name="host",
    )
    segments = [
        TranscriptSegment(
            id=0,
            start=0.0,
            end=1.0,
            text="один",
            avg_logprob=-0.1,
            no_speech_prob=0.0,
            compression_ratio=1.0,
            words=[],
            speaker_id="SPEAKER_00",
        ),
        TranscriptSegment(
            id=1,
            start=1.1,
            end=2.1,
            text="два",
            avg_logprob=-0.1,
            no_speech_prob=0.0,
            compression_ratio=1.0,
            words=[],
            speaker_id="SPEAKER_01",
        ),
    ]
    merged = segmenter._merge_segments(segments)
    assert len(merged) == 2
    assert merged[0].speaker_id == "SPEAKER_00"
    assert merged[1].speaker_id == "SPEAKER_01"


def test_resolve_speaker_name_suffix() -> None:
    assert (
        resolve_speaker_name("channel", "SPEAKER_00", video_id="vid_a")
        == "channel__vid_a__SPEAKER_00"
    )
    assert resolve_speaker_name("channel", None, video_id="vid_a") == "channel__vid_a"


def test_build_clips_assigns_per_speaker_names(tmp_path: Path) -> None:
    words_a = [
        TranscriptWord(word="Привіт", start=0.0, end=0.4, probability=0.95),
        TranscriptWord(word="друже", start=0.45, end=0.9, probability=0.95),
    ]
    words_b = [
        TranscriptWord(word="Добрий", start=3.0, end=3.4, probability=0.95),
        TranscriptWord(word="день", start=3.45, end=3.9, probability=0.95),
    ]
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
                text="Привіт друже",
                avg_logprob=-0.1,
                no_speech_prob=0.0,
                compression_ratio=1.0,
                words=words_a,
                language="uk",
                speaker_id="SPEAKER_00",
            ),
            TranscriptSegment(
                id=1,
                start=3.0,
                end=5.0,
                text="Добрий день",
                avg_logprob=-0.1,
                no_speech_prob=0.0,
                compression_ratio=1.0,
                words=words_b,
                language="uk",
                speaker_id="SPEAKER_01",
            ),
        ],
        transcribed_at="2026-01-01T00:00:00+00:00",
        aligned_at="2026-01-01T00:00:01+00:00",
        sound_events=[],
    )
    segmenter = AudioSegmenter(
        SegmentationConfig(
            min_duration_sec=0.5,
            max_duration_sec=12.0,
            min_speech_duration_sec=0.2,
            min_chars=3,
            max_chars=220,
        ),
        QualityConfig(filter_audio_quality=False),
        target_language="uk",
        speaker_name="host",
    )
    clips, stats = segmenter.build_clips(transcript)
    assert stats.clips_kept == 2
    assert {clip.speaker_name for clip in clips} == {
        "host__v1__SPEAKER_00",
        "host__v1__SPEAKER_01",
    }


def test_merge_datasets_preserves_speaker_names(tmp_path: Path) -> None:
    def _make_dataset(name: str, rows: list[tuple[str, str]]) -> Path:
        root = tmp_path / name
        wavs = root / "wavs"
        wavs.mkdir(parents=True)
        lines = ["audio_file|text|speaker_name"]
        for idx, (text, speaker) in enumerate(rows, start=1):
            rel = f"wavs/{idx:06d}.wav"
            (wavs / f"{idx:06d}.wav").write_bytes(b"RIFF")
            lines.append(f"{rel}|{text}|{speaker}")
        (root / "metadata_train.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return root

    src_a = _make_dataset("a", [("один", "host_SPEAKER_00"), ("два", "host_SPEAKER_01")])
    src_b = _make_dataset("b", [("три", "guest")])
    out = tmp_path / "combined"
    stats = merge_datasets([src_a, src_b], out, speaker_name="fallback")
    assert stats.total_clips == 3
    train = (out / "metadata_train.csv").read_text(encoding="utf-8")
    eval_text = (out / "metadata_eval.csv").read_text(encoding="utf-8")
    combined = train + eval_text
    assert "host_SPEAKER_00" in combined
    assert "host_SPEAKER_01" in combined
    assert "guest" in combined
    assert "fallback" not in combined.splitlines()[1:]


def test_list_export_ready_datasets_skips_output_and_incomplete(tmp_path: Path) -> None:
    ready = tmp_path / "uk-mix"
    (ready / "wavs").mkdir(parents=True)
    (ready / "metadata_train.csv").write_text(
        "audio_file|text|speaker_name\nwavs/1.wav|привіт|filatov\n",
        encoding="utf-8",
    )
    (ready / "wavs" / "1.wav").write_bytes(b"RIFF")

    incomplete = tmp_path / "half"
    incomplete.mkdir()
    (incomplete / "metadata_train.csv").write_text(
        "audio_file|text|speaker_name\n", encoding="utf-8"
    )

    combined = tmp_path / "combined"
    (combined / "wavs").mkdir(parents=True)
    (combined / "metadata_train.csv").write_text(
        "audio_file|text|speaker_name\nwavs/1.wav|x|y\n", encoding="utf-8"
    )

    found = list_export_ready_datasets(tmp_path, exclude_names={"combined"})
    assert [path.name for path in found] == ["uk-mix"]


def test_export_writes_per_clip_speakers(tmp_path: Path, monkeypatch) -> None:
    from fish_studio.config import ExportConfig
    from fish_studio.dataset.export import DatasetExporter

    segments = tmp_path / "segments" / "v1"
    segments.mkdir(parents=True)
    (segments / "v1_0000.wav").write_bytes(b"RIFF")
    (segments / "v1_0001.wav").write_bytes(b"RIFF")

    def fake_ffmpeg(cmd, capture_output=True, text=True, check=False):  # noqa: ANN001
        Path(cmd[-1]).write_bytes(b"RIFF")

        class _Proc:
            returncode = 0

        return _Proc()

    monkeypatch.setattr("fish_studio.dataset.export.subprocess.run", fake_ffmpeg)

    clips = [
        AudioClip(
            clip_id="v1_0000",
            video_id="v1",
            start=0.0,
            end=2.0,
            text="Привіт",
            duration=2.0,
            source_audio="x.wav",
            speaker_name="host_SPEAKER_00",
            avg_word_score=0.9,
        ),
        AudioClip(
            clip_id="v1_0001",
            video_id="v1",
            start=3.0,
            end=5.0,
            text="День",
            duration=2.0,
            source_audio="x.wav",
            speaker_name="host_SPEAKER_01",
            avg_word_score=0.8,
        ),
    ]
    exporter = DatasetExporter(
        ExportConfig(output_dir=str(tmp_path / "dataset"), speaker_name="host", seed=1)
    )
    stats = exporter.export(clips, tmp_path / "segments")
    assert stats.total_clips == 2
    meta = (tmp_path / "dataset" / "metadata_train.csv").read_text(encoding="utf-8")
    eval_meta = (tmp_path / "dataset" / "metadata_eval.csv").read_text(encoding="utf-8")
    combined = meta + eval_meta
    assert "host_SPEAKER_00" in combined
    assert "host_SPEAKER_01" in combined
    assert (tmp_path / "dataset" / "reference.wav").is_file()
    assert (tmp_path / "dataset" / "references" / "host_SPEAKER_00.wav").is_file()
    assert (tmp_path / "dataset" / "references" / "host_SPEAKER_01.wav").is_file()
