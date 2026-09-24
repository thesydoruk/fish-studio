"""Tests for Hugging Face dataset import helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from fish_studio.dataset.hf_import import (
    SourceStats,
    _Record,
    _read_manifest,
    _write_manifest,
    clean_text,
    parse_source,
    wav_duration,
)


def test_parse_source_reads_repo_and_speaker() -> None:
    source = parse_source("speech-uk/opentts-mykyta=Mykyta")

    assert source.repo_id == "speech-uk/opentts-mykyta"
    assert source.speaker == "mykyta"
    assert source.config_name is None
    assert source.split == "train"


def test_parse_source_reads_config_and_split() -> None:
    source = parse_source("Yehor/opentts-uk:lada@test=lada")

    assert source.repo_id == "Yehor/opentts-uk"
    assert source.config_name == "lada"
    assert source.split == "test"
    assert source.speaker == "lada"


def test_wav_duration_reads_header(tmp_path: Path) -> None:
    import wave

    path = tmp_path / "clip.wav"
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(44100)
        handle.writeframes(b"\x00\x00" * 66150)

    assert wav_duration(path) == pytest.approx(1.5)


def test_wav_duration_returns_none_for_non_wav(tmp_path: Path) -> None:
    path = tmp_path / "clip.wav"
    path.write_bytes(b"not audio")

    assert wav_duration(path) is None


def test_manifest_round_trip_marks_records_as_resumed(tmp_path: Path) -> None:
    stats = SourceStats(speaker="lada", repo_id="speech-uk/opentts-lada", kept=2)
    stats.sample_rates = {48000: 2}
    records = [
        _Record(file_id="lada-000001", text="Привіт", speaker="lada", duration=1.5),
        _Record(file_id="lada-000002", text="Світ", speaker="lada", duration=2.0),
    ]
    path = tmp_path / "lada.json"
    _write_manifest(path, stats, records)

    restored_stats, restored_records = _read_manifest(path)

    assert restored_stats.resumed is True
    assert restored_stats.sample_rates == {48000: 2}
    assert restored_records == records


def test_read_manifest_returns_none_for_corrupt_file(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")

    assert _read_manifest(path) is None


@pytest.mark.parametrize(
    "spec",
    [
        "speech-uk/opentts-mykyta",
        "=mykyta",
        "repo=",
    ],
)
def test_parse_source_rejects_malformed_specs(spec: str) -> None:
    with pytest.raises(ValueError):
        parse_source(spec)


def test_clean_text_strips_pipes_that_would_break_metadata() -> None:
    assert clean_text("Привіт | як  справи?") == "Привіт як справи?"


def test_clean_text_keeps_ukrainian_punctuation() -> None:
    assert clean_text("  Ґанок, «дім» — це  добре!  ") == "Ґанок, «дім» — це добре!"


def test_row_speaker_uses_the_first_group_of_the_regex() -> None:
    from dataclasses import replace

    source = replace(
        parse_source("taras-sereda/uk-pods=pods"),
        group_regex=r"clips/([0-9a-f-]{36})_",
    )
    row = {"audio_filepath": "clips/c5ecdab7-3a47-479b-886f-add595d90209_0042.wav"}
    assert source.row_speaker(row) == "pods-c5ecdab7-3a4"
    assert source.row_speaker({"audio_filepath": "elsewhere.wav"}) == "pods"
    assert parse_source("x/y=plain").row_speaker(row) == "plain"


def test_archive_rows_extract_once_and_point_audio_into_the_tree(
    tmp_path: Path, monkeypatch
) -> None:
    import io
    import json
    import tarfile
    from dataclasses import replace

    huggingface_hub = pytest.importorskip("huggingface_hub")
    import fish_studio.dataset.hf_import as module

    archive = tmp_path / "clips.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        data = b"RIFF"
        info = tarfile.TarInfo("clips/a_0001.wav")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    manifest = tmp_path / "pods_train.json"
    manifest.write_text(
        json.dumps([{"audio_filepath": "clips/a_0001.wav", "text": "hello", "duration": 2.0}]),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        huggingface_hub,
        "hf_hub_download",
        lambda repo_id, filename, repo_type: str(tmp_path / filename),
    )
    source = replace(
        parse_source("taras-sereda/uk-pods=pods"),
        archive="clips.tar.gz",
        manifest="pods_train.json",
    )
    cache = tmp_path / "cache"

    rows = list(module._load_rows(source, cache_dir=cache))

    assert rows[0]["text"] == "hello"
    assert Path(rows[0]["audio"]["path"]).read_bytes() == b"RIFF"
    assert module._audio_bytes(rows[0]) == b"RIFF"
    assert list(module._load_rows(source, cache_dir=cache))  # second pass reuses the extraction
