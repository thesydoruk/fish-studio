"""Tests for cross-video speaker clustering."""

from __future__ import annotations

from pathlib import Path

from fish_studio.dataset.speaker_cluster import (
    SpeakerNode,
    apply_speaker_map_to_name,
    cluster_local_speakers,
    cluster_source_transcripts,
    cosine_similarity,
    remap_dataset_speakers,
)
from fish_studio.dataset.transcript import (
    SpeakerInfo,
    TranscriptResult,
    save_transcript,
)


def test_cosine_similarity_identical() -> None:
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_cluster_merges_similar_embeddings() -> None:
    nodes = [
        SpeakerNode(
            local_name="host__v1__spk_0",
            video_id="v1",
            speaker_id="spk_0",
            speech_seconds=10.0,
            embedding=[1.0, 0.0, 0.0],
        ),
        SpeakerNode(
            local_name="host__v2__spk_0",
            video_id="v2",
            speaker_id="spk_0",
            speech_seconds=8.0,
            embedding=[0.99, 0.01, 0.0],
        ),
        SpeakerNode(
            local_name="host__v2__spk_1",
            video_id="v2",
            speaker_id="spk_1",
            speech_seconds=3.0,
            embedding=[0.0, 1.0, 0.0],
        ),
    ]
    mapping = cluster_local_speakers(nodes, base_speaker="host", threshold=0.9)
    assert mapping["host__v1__spk_0"] == mapping["host__v2__spk_0"] == "host_s0"
    assert mapping["host__v2__spk_1"] == "host_s1"


def test_cluster_source_writes_map(tmp_path: Path) -> None:
    work = tmp_path / "work"
    transcripts = work / "transcripts"
    transcripts.mkdir(parents=True)
    for video_id, emb in (("v1", [1.0, 0.0]), ("v2", [1.0, 0.0])):
        save_transcript(
            TranscriptResult(
                video_id=video_id,
                source_audio=str(tmp_path / f"{video_id}.wav"),
                language="uk",
                duration_sec=1.0,
                segments=[],
                transcribed_at="2026-01-01T00:00:00+00:00",
                speakers=[
                    SpeakerInfo(
                        id="spk_0",
                        speech_seconds=5.0,
                        segment_count=1,
                        embedding=emb,
                    )
                ],
            ),
            transcripts,
        )
    result = cluster_source_transcripts(
        work,
        transcripts,
        base_speaker="olena",
        threshold=0.9,
    )
    assert Path(result.map_path).is_file()
    assert len(set(result.mapping.values())) == 1
    assert apply_speaker_map_to_name("olena__v1__spk_0", result.mapping) == "olena_s0"


def test_prune_tiny_speakers() -> None:
    from dataclasses import dataclass

    from fish_studio.dataset.speaker_cluster import prune_tiny_speakers

    @dataclass
    class _Clip:
        speaker_name: str
        duration: float

    clips = [
        _Clip("big", 10.0),
        *[_Clip("big", 5.0) for _ in range(99)],
        _Clip("tiny", 2.0),
        _Clip("tiny", 2.0),
        *[_Clip("mid", 4.0) for _ in range(50)],  # 50 clips, 200s — fails 300s floor
    ]
    kept, dropped = prune_tiny_speakers(clips, min_clips=100, min_speech_sec=300.0)
    assert all(clip.speaker_name == "big" for clip in kept)
    assert set(dropped) == {"tiny", "mid"}
    assert dropped["tiny"] == (2, 4.0)


def test_fallback_by_diarization_id() -> None:
    from fish_studio.dataset.speaker_cluster import fallback_map_by_diarization_id

    mapping = fallback_map_by_diarization_id(
        [
            "host__v1__spk_0",
            "host__v2__spk_0",
            "host__v2__spk_1",
        ],
        base_speaker="host",
    )
    assert mapping["host__v1__spk_0"] == mapping["host__v2__spk_0"] == "host_s0"
    assert mapping["host__v2__spk_1"] == "host_s1"


def test_remap_dataset_speakers(tmp_path: Path) -> None:
    ds = tmp_path / "combined"
    (ds / "wavs").mkdir(parents=True)
    (ds / "metadata_train.csv").write_text(
        "audio_file|text|speaker_name\nwavs/1.wav|a|olena_s0\nwavs/2.wav|b|rostyslav_s0\n",
        encoding="utf-8",
    )
    (ds / "metadata_eval.csv").write_text(
        "audio_file|text|speaker_name\nwavs/3.wav|c|olena_s0\n",
        encoding="utf-8",
    )
    counts = remap_dataset_speakers(
        ds,
        {"olena_s0": "teacher", "rostyslav_s0": "teacher"},
    )
    assert counts["metadata_train.csv"] == 2
    assert counts["metadata_eval.csv"] == 1
    train = (ds / "metadata_train.csv").read_text(encoding="utf-8")
    assert "teacher" in train
    assert "olena_s0" not in train
