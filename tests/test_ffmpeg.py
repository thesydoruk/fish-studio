"""Concurrent ffmpeg instances must not share stdin or a thread pool."""

from __future__ import annotations

import subprocess

from fish_studio.ffmpeg import ffmpeg_cmd, ffprobe_cmd, run_ffmpeg


def test_ffmpeg_cmd_disables_stdin_and_limits_threads() -> None:
    cmd = ffmpeg_cmd("-y", "-i", "in.wav", "out.wav")
    assert cmd[:8] == [
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-threads",
        "1",
        "-filter_threads",
        "1",
        "-y",
    ]


def test_ffprobe_cmd_is_non_interactive() -> None:
    cmd = ffprobe_cmd("-v", "error", "clip.wav")
    assert cmd[0] == "ffprobe"
    assert "-hide_banner" in cmd


def test_run_ffmpeg_closes_stdin_when_not_piping(monkeypatch) -> None:
    seen: dict[str, object] = {}

    def fake_run(**kwargs):  # noqa: ANN003
        seen.update(kwargs)

        class _Proc:
            returncode = 0
            stdout = b""
            stderr = b""

        return _Proc()

    monkeypatch.setattr("fish_studio.ffmpeg.subprocess.run", fake_run)
    run_ffmpeg(["-y", "-i", "in.wav", "out.wav"])
    assert seen["stdin"] is subprocess.DEVNULL
    argv = seen["args"]
    assert isinstance(argv, list)
    assert argv[0] == "ffmpeg"
    assert "-nostdin" in argv
