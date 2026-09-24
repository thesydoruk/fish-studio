"""The VQ wrapper checks its result: upstream exits 0 even when its workers die."""

from __future__ import annotations

from pathlib import Path

from fish_studio.training.extract_vq import clips_without_tokens


def test_clips_without_tokens_lists_wavs_that_got_no_npy(tmp_path: Path) -> None:
    speaker = tmp_path / "spk"
    speaker.mkdir()
    (speaker / "a.wav").write_bytes(b"RIFF")
    (speaker / "a.npy").write_bytes(b"\x93NUMPY")
    (speaker / "b.wav").write_bytes(b"RIFF")

    assert clips_without_tokens(tmp_path) == [speaker / "b.wav"]
