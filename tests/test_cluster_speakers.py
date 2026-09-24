"""Speaker groups become voices by embedding; tiny voices are dropped."""

from __future__ import annotations

import csv
import wave
from pathlib import Path

import numpy as np

from fish_studio.dataset.cluster_speakers import (
    cluster_dataset_speakers,
    cluster_embeddings,
    load_embedding_cache,
)


def test_cluster_embeddings_splits_two_directions_and_keeps_noise_together() -> None:
    rng = np.random.default_rng(0)
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    vectors = np.stack(
        [a + rng.normal(0, 0.05, 3) for _ in range(6)]
        + [b + rng.normal(0, 0.05, 3) for _ in range(4)]
    )
    labels = cluster_embeddings(vectors, threshold=0.35)
    assert len(set(labels[:6])) == 1
    assert len(set(labels[6:])) == 1
    assert labels[0] != labels[6]


def test_cluster_embeddings_handles_empty_and_single() -> None:
    assert cluster_embeddings(np.zeros((0, 3)), 0.5) == []
    assert cluster_embeddings(np.ones((1, 3)), 0.5) == [0]


def _write_wav(path: Path, seconds: float = 4.0) -> None:
    with wave.open(str(path), "w") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(b"\x00\x01" * int(seconds * 16_000))


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="|"))


def test_cluster_dataset_speakers_rewrites_groups_into_voices_and_prunes(tmp_path: Path) -> None:
    """One episode with two people and a stray clip: two voices kept, the stray dropped."""
    wavs = tmp_path / "wavs"
    wavs.mkdir()
    lines = ["audio_file|text|speaker_name"]
    direction = {}
    for i in range(7):
        _write_wav(wavs / f"pods-ep1-{i:03d}.wav")
        lines.append(f"wavs/pods-ep1-{i:03d}.wav|text {i}|pods-ep1")
        direction[f"wavs/pods-ep1-{i:03d}.wav"] = [1.0, 0.0, 0.0] if i < 4 else [0.0, 1.0, 0.0]
    _write_wav(wavs / "pods-ep1-stray.wav")
    lines.append("wavs/pods-ep1-stray.wav|stray|pods-ep1")
    direction["wavs/pods-ep1-stray.wav"] = [0.0, 0.0, 1.0]
    _write_wav(wavs / "other-000.wav")
    lines.append("wavs/other-000.wav|untouched|other")
    (tmp_path / "metadata_train.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def embed(path: Path) -> np.ndarray:
        return np.asarray(direction[f"wavs/{path.name}"], dtype=np.float32)

    stats = cluster_dataset_speakers(
        tmp_path,
        embed=embed,
        threshold=0.35,
        min_clips=2,
        min_seconds=4.0,
        only_groups={"pods-ep1"},
    )

    assert stats.groups == 1
    assert stats.voices_kept == 2
    assert stats.voices_dropped == 1
    rows = _rows(tmp_path / "metadata_train.csv")
    voices = {r["audio_file"]: r["speaker_name"] for r in rows}
    assert voices["wavs/other-000.wav"] == "other"
    assert "wavs/pods-ep1-stray.wav" not in voices
    assert voices["wavs/pods-ep1-000.wav"] == "pods-ep1_s0"  # the larger voice ranks first
    assert voices["wavs/pods-ep1-005.wav"] == "pods-ep1_s1"
    assert set(load_embedding_cache(tmp_path)) == set(direction)
    assert (tmp_path / "speakers.json").is_file()


def test_dry_run_reports_but_writes_no_metadata(tmp_path: Path) -> None:
    wavs = tmp_path / "wavs"
    wavs.mkdir()
    _write_wav(wavs / "g-000.wav")
    _write_wav(wavs / "g-001.wav")
    text = "audio_file|text|speaker_name\nwavs/g-000.wav|a|g\nwavs/g-001.wav|b|g\n"
    (tmp_path / "metadata_train.csv").write_text(text, encoding="utf-8")

    stats = cluster_dataset_speakers(
        tmp_path,
        embed=lambda p: np.ones(3, dtype=np.float32),
        min_clips=1,
        min_seconds=1,
        dry_run=True,
    )

    assert stats.per_group["g"][0][1] == 2
    assert (tmp_path / "metadata_train.csv").read_text(encoding="utf-8") == text
    assert not (tmp_path / "speakers.json").exists()
