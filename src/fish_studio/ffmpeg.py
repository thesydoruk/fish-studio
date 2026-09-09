"""ffmpeg / ffprobe argv that can run next to other instances.

Without ``-nostdin``, ffmpeg reads inherited stdin for interactive ``q``.
Concurrent synths or dataset workers then steal each other's stdin and one
hangs or dies. ``-threads 1`` keeps a short clip from spawning a thread
pool that fights sibling processes.
"""

from __future__ import annotations

import subprocess
from collections.abc import Sequence


def ffmpeg_cmd(*args: str) -> list[str]:
    return [
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-threads",
        "1",
        "-filter_threads",
        "1",
        *args,
    ]


def ffprobe_cmd(*args: str) -> list[str]:
    return ["ffprobe", "-hide_banner", *args]


def run_ffmpeg(
    args: Sequence[str],
    *,
    input: bytes | str | None = None,
    text: bool = False,
    check: bool = False,
) -> subprocess.CompletedProcess[str] | subprocess.CompletedProcess[bytes]:
    return _run(ffmpeg_cmd(*args), input=input, text=text, check=check)


def run_ffprobe(
    args: Sequence[str],
    *,
    input: bytes | str | None = None,
    text: bool = False,
    check: bool = False,
) -> subprocess.CompletedProcess[str] | subprocess.CompletedProcess[bytes]:
    return _run(ffprobe_cmd(*args), input=input, text=text, check=check)


def _run(
    cmd: list[str],
    *,
    input: bytes | str | None,
    text: bool,
    check: bool,
) -> subprocess.CompletedProcess[str] | subprocess.CompletedProcess[bytes]:
    kwargs: dict[str, object] = {
        "args": cmd,
        "capture_output": True,
        "text": text,
        "check": check,
    }
    if input is None:
        # Close stdin even with -nostdin: some builds still peek at it.
        kwargs["stdin"] = subprocess.DEVNULL
    else:
        kwargs["input"] = input
    return subprocess.run(**kwargs)
