"""Download YouTube audio or register local files; manifest tracks completed items."""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fish_studio.config import YouTubeConfig


@dataclass
class DownloadedVideo:
    video_id: str
    title: str
    url: str
    duration_sec: float
    audio_path: str
    downloaded_at: str


AUDIO_EXTENSIONS = {".wav", ".m4a", ".webm", ".opus", ".ogg", ".mp3", ".aac"}


class YouTubeDownloader:
    """yt-dlp wrapper with JSONL manifest for resume-safe downloads."""

    def __init__(self, config: YouTubeConfig, downloads_dir: str | Path) -> None:
        self.config = config
        self.downloads_dir = Path(downloads_dir)
        self.downloads_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.downloads_dir / "manifest.jsonl"

    def list_pending_videos(self) -> list[dict[str, Any]]:
        entries = self._extract_entries()
        completed = self._completed_ids()
        pending = [e for e in entries if e["id"] not in completed]
        return pending

    def download_all(self, force: bool = False) -> list[DownloadedVideo]:
        """Download pending videos. ``force`` re-fetches even if the manifest already has them."""
        entries = self._extract_entries()
        completed = set() if force else self._completed_ids()
        manifest = {item.video_id: item for item in self.load_manifest()}
        results: list[DownloadedVideo] = []

        for entry in entries:
            video_id = entry["id"]
            if video_id in completed and not force:
                continue

            duration = float(entry.get("duration") or 0)
            if duration < self.config.min_video_duration_sec:
                # Short clips are usually intros/outros; skip before spending yt-dlp time.
                continue
            if (
                self.config.max_video_duration_sec is not None
                and duration > self.config.max_video_duration_sec
            ):
                continue

            try:
                audio_path = self._download_video(entry)
            except (RuntimeError, FileNotFoundError) as exc:
                title = entry.get("title") or video_id
                print(f"  Skipping {video_id} ({title[:50]}): {exc}")
                continue

            record = DownloadedVideo(
                video_id=video_id,
                title=entry.get("title") or video_id,
                url=entry.get("webpage_url") or entry.get("url") or "",
                duration_sec=duration,
                audio_path=str(audio_path),
                downloaded_at=datetime.now(timezone.utc).isoformat(),
            )
            manifest[video_id] = record
            results.append(record)

        if results:
            self._write_manifest(manifest.values())
        return results

    def sync_local(self, force: bool = False) -> list[DownloadedVideo]:
        """Register audio files placed manually in downloads_dir (games, etc.)."""
        manifest = {item.video_id: item for item in self.load_manifest()}
        results: list[DownloadedVideo] = []

        for path in sorted(self.downloads_dir.iterdir()):
            if not path.is_file() or path.suffix.lower() not in AUDIO_EXTENSIONS:
                continue
            video_id = path.stem
            if video_id in manifest and not force:
                continue

            record = DownloadedVideo(
                video_id=video_id,
                title=video_id,
                url="",
                duration_sec=0.0,
                audio_path=str(path),
                downloaded_at=datetime.now(timezone.utc).isoformat(),
            )
            manifest[video_id] = record
            results.append(record)

        if results or force:
            self._write_manifest(manifest.values())
        return results

    def load_manifest(self) -> list[DownloadedVideo]:
        """Load the JSONL manifest, dropping duplicate ids (keep the newest existing file)."""
        if not self.manifest_path.exists():
            return []
        items: list[DownloadedVideo] = []
        with self.manifest_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                items.append(DownloadedVideo(**data))
        deduped = self._dedupe_manifest(items)
        if len(deduped) != len(items):
            self._write_manifest(deduped)
        return deduped

    @staticmethod
    def _dedupe_manifest(items: list[DownloadedVideo]) -> list[DownloadedVideo]:
        by_id: dict[str, DownloadedVideo] = {}
        for item in items:
            prev = by_id.get(item.video_id)
            if prev is None:
                by_id[item.video_id] = item
                continue
            prev_exists = Path(prev.audio_path).exists()
            cur_exists = Path(item.audio_path).exists()
            if (
                cur_exists
                and not prev_exists
                or cur_exists == prev_exists
                and item.downloaded_at >= prev.downloaded_at
            ):
                by_id[item.video_id] = item
        return list(by_id.values())

    def _completed_ids(self) -> set[str]:
        return {item.video_id for item in self.load_manifest() if Path(item.audio_path).exists()}

    def _write_manifest(self, records: Iterable[DownloadedVideo]) -> None:
        lines = [json.dumps(asdict(record), ensure_ascii=False) for record in records]
        self.manifest_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    def _subprocess_env(self) -> dict[str, str]:
        # yt-dlp needs a JS runtime (deno) for some YouTube extractors.
        env = os.environ.copy()
        deno_bin = Path.home() / ".deno" / "bin"
        if deno_bin.is_dir():
            env["PATH"] = f"{deno_bin}{os.pathsep}{env.get('PATH', '')}"
        return env

    def _yt_dlp_base_cmd(self) -> list[str]:
        cmd = ["yt-dlp"]
        deno = Path.home() / ".deno" / "bin" / "deno"
        if deno.is_file():
            cmd.extend(["--js-runtimes", f"deno:{deno}"])
        return cmd

    def _extract_entries(self) -> list[dict[str, Any]]:
        cmd = self._yt_dlp_base_cmd() + [
            "--flat-playlist",
            "--dump-json",
            self.config.url,
        ]
        if self.config.max_videos:
            cmd[1:1] = ["--playlist-end", str(self.config.max_videos)]

        proc = subprocess.run(
            cmd, capture_output=True, text=True, check=False, env=self._subprocess_env()
        )
        if proc.returncode != 0:
            raise RuntimeError(f"yt-dlp failed:\n{proc.stderr}")

        entries: list[dict[str, Any]] = []
        for line in proc.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            if data.get("_type") == "url" and data.get("url"):
                entries.append(
                    {
                        "id": data["id"],
                        "title": data.get("title"),
                        "url": data["url"],
                        "webpage_url": data.get("webpage_url"),
                        "duration": data.get("duration"),
                    }
                )
            elif data.get("id"):
                entries.append(data)

        if not entries:
            # Single video URL (not a playlist) — fetch metadata directly.
            single = self._extract_single(self.config.url)
            if single:
                entries = [single]

        return entries

    def _extract_single(self, url: str) -> dict[str, Any] | None:
        cmd = self._yt_dlp_base_cmd() + ["--dump-json", url]
        proc = subprocess.run(
            cmd, capture_output=True, text=True, check=False, env=self._subprocess_env()
        )
        if proc.returncode != 0:
            return None
        return json.loads(proc.stdout)

    def _download_video(self, entry: dict[str, Any]) -> Path:
        video_id = entry["id"]
        out_template = str(self.downloads_dir / f"{video_id}.%(ext)s")
        url = entry.get("webpage_url") or entry.get("url") or self.config.url

        if self.config.convert_on_download:
            # Decode to mono WAV at download time so later ffmpeg cuts skip a transcode.
            cmd = self._yt_dlp_base_cmd() + [
                "-x",
                "--audio-format",
                "wav",
                "--audio-quality",
                "0",
                "--postprocessor-args",
                "ffmpeg:-ar 48000 -ac 1",
                "-o",
                out_template,
                "--no-overwrites",
                url,
            ]
        else:
            cmd = self._yt_dlp_base_cmd() + [
                "-f",
                self.config.audio_format,
                "-o",
                out_template,
                "--no-overwrites",
                url,
            ]

        proc = subprocess.run(
            cmd, capture_output=True, text=True, check=False, env=self._subprocess_env()
        )
        if proc.returncode != 0:
            raise RuntimeError(f"Failed to download {video_id}:\n{proc.stderr}")

        candidates = list(self.downloads_dir.glob(f"{video_id}.*"))
        audio_files = [p for p in candidates if p.suffix.lower() in AUDIO_EXTENSIONS]
        if not audio_files:
            raise FileNotFoundError(f"No audio file found for {video_id} in {self.downloads_dir}")

        return audio_files[0]
