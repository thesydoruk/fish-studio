"""Tests for Fish Speech dataset export."""

from __future__ import annotations

import wave
from pathlib import Path

from fish_studio.training.export_dataset import export_labeled_dataset


def _write_wav(path: Path, *, duration_sec: float = 0.1, sample_rate: int = 22050) -> None:
    frames = int(duration_sec * sample_rate)
    with wave.open(str(path), "w") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\x00\x00" * frames)


def test_export_labeled_dataset_writes_wav_and_lab(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "combined"
    wavs_dir = dataset_dir / "wavs"
    wavs_dir.mkdir(parents=True)
    _write_wav(wavs_dir / "000001.wav")
    (dataset_dir / "metadata_train.csv").write_text(
        "audio_file|text|speaker_name\nwavs/000001.wav|Привіт, як справи?|speaker\n",
        encoding="utf-8",
    )

    output_dir = tmp_path / "fish-raw"
    stats = export_labeled_dataset(
        dataset_dir,
        output_dir,
        speaker_name="speaker",
    )

    speaker_dir = output_dir / "speaker"
    assert stats.clips == 1
    assert (speaker_dir / "000001.wav").is_file()
    assert (speaker_dir / "000001.lab").read_text(encoding="utf-8") == "Привіт, як справи?\n"


def test_export_links_audio_instead_of_copying_it(tmp_path: Path) -> None:
    """A 76 GB corpus copied next to itself buys nothing; nothing downstream rewrites a wav."""
    dataset_dir = tmp_path / "combined"
    wavs_dir = dataset_dir / "wavs"
    wavs_dir.mkdir(parents=True)
    _write_wav(wavs_dir / "000001.wav")
    (dataset_dir / "metadata_train.csv").write_text(
        "audio_file|text|speaker_name\nwavs/000001.wav|Привіт|speaker\n", encoding="utf-8"
    )

    export_labeled_dataset(dataset_dir, tmp_path / "fish-raw", speaker_name="speaker")

    exported = tmp_path / "fish-raw" / "speaker" / "000001.wav"
    assert exported.read_bytes() == (wavs_dir / "000001.wav").read_bytes()
    assert exported.stat().st_nlink == 2 or exported.stat().st_ino != 0


def test_export_with_a_worker_pool_matches_the_serial_export(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "combined"
    wavs_dir = dataset_dir / "wavs"
    wavs_dir.mkdir(parents=True)
    lines = ["audio_file|text|speaker_name"]
    for index in range(6):
        _write_wav(wavs_dir / f"{index:06d}.wav")
        lines.append(f"wavs/{index:06d}.wav|Речення номер {index}|spk{index % 2}")
    (dataset_dir / "metadata_train.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    stats = export_labeled_dataset(
        dataset_dir, tmp_path / "fish-raw", speaker_name="speaker", workers=2, progress_every=0
    )

    assert stats.clips == 6
    assert stats.speakers == {"spk0": 3, "spk1": 3}
    assert (tmp_path / "fish-raw" / "spk1" / "000005.lab").read_text(encoding="utf-8") == (
        "Речення номер 5\n"
    )


def test_export_labeled_dataset_can_include_eval_split(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "combined"
    wavs_dir = dataset_dir / "wavs"
    wavs_dir.mkdir(parents=True)
    _write_wav(wavs_dir / "000001.wav")
    _write_wav(wavs_dir / "000002.wav")
    (dataset_dir / "metadata_train.csv").write_text(
        "audio_file|text|speaker_name\nwavs/000001.wav|Перший|speaker\n",
        encoding="utf-8",
    )
    (dataset_dir / "metadata_eval.csv").write_text(
        "audio_file|text|speaker_name\nwavs/000002.wav|Другий|speaker\n",
        encoding="utf-8",
    )

    stats = export_labeled_dataset(
        dataset_dir,
        tmp_path / "fish-raw",
        speaker_name="speaker",
        include_eval=True,
    )
    speaker_dir = tmp_path / "fish-raw" / "speaker"
    assert stats.clips == 2
    assert (speaker_dir / "000002.lab").read_text(encoding="utf-8") == "Другий\n"


def test_export_labeled_dataset_splits_speakers_into_folders(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "hf"
    wavs_dir = dataset_dir / "wavs"
    wavs_dir.mkdir(parents=True)
    _write_wav(wavs_dir / "000001.wav")
    _write_wav(wavs_dir / "000002.wav")
    (dataset_dir / "metadata_train.csv").write_text(
        "audio_file|text|speaker_name\n"
        "wavs/000001.wav|Перший|mykyta\n"
        "wavs/000002.wav|Другий|lada\n",
        encoding="utf-8",
    )

    output_dir = tmp_path / "fish-raw"
    stats = export_labeled_dataset(dataset_dir, output_dir, speaker_name="speaker")

    assert stats.speakers == {"mykyta": 1, "lada": 1}
    assert (output_dir / "mykyta" / "000001.lab").read_text(encoding="utf-8") == "Перший\n"
    assert (output_dir / "lada" / "000002.lab").read_text(encoding="utf-8") == "Другий\n"
    assert not (output_dir / "speaker").exists()
