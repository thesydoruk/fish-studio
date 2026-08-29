"""End-to-end dataset pipeline: download → transcribe → segment → cluster → export."""

from __future__ import annotations

import json
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

from rich.console import Console
from rich.table import Table

from fish_studio.config import AppConfig, resolve_steps
from fish_studio.dataset.audio_intel import AudioIntelClient
from fish_studio.dataset.download import YouTubeDownloader
from fish_studio.dataset.export import DatasetExporter, clear_dataset_dir
from fish_studio.dataset.segment import AudioClip, AudioSegmenter
from fish_studio.dataset.speaker_cluster import (
    apply_speaker_map_to_name,
    cluster_source_transcripts,
    load_speaker_map,
    prune_tiny_speakers,
)
from fish_studio.dataset.transcript import load_transcript

STEP_LABELS = {
    "download": "Syncing source audio",
    "transcribe": "Transcribing via audio-intel",
    "segment": "Segmenting audio",
    "cluster": "Clustering speakers across videos",
    "export": "Exporting pipe-delimited dataset",
}


class Pipeline:
    """Orchestrates pipeline steps with resume support (skip existing outputs unless --force)."""

    @staticmethod
    def _format_progress(completed: int, total: int) -> str:
        remaining = max(total - completed, 0)
        return f"[{completed}/{total} · {remaining} left]"

    def __init__(
        self,
        config: AppConfig,
        console: Console | None = None,
        config_path: str = ".env",
    ) -> None:
        self.config = config
        self.config_path = config_path
        self.console = console or Console()
        self.steps = resolve_steps(config.pipeline.steps)

    def run(self) -> None:
        self.config.ensure_dirs()
        source = self.config.source
        self.console.print("[bold]Fish Speech Dataset Builder[/bold] — Ukrainian pipeline\n")
        self.console.print(f"[dim]Source:[/dim] {source.id} ({source.kind})")
        self.console.print(f"[dim]Work:[/dim] {self.config.paths.work_dir}")
        self.console.print(f"[dim]Dataset:[/dim] {self.config.export.output_dir}\n")

        total = len(self.steps)
        for index, step in enumerate(self.steps, start=1):
            self.console.print(f"[cyan]Step {index}/{total}[/cyan] {STEP_LABELS[step]}...")
            runner = getattr(self, f"_run_{step}")
            runner()

        self.console.print("\n[green]Done.[/green]")

    def _run_download(self) -> None:
        downloader = YouTubeDownloader(self.config.youtube, self.config.paths.downloads_dir)
        if self.config.source.kind == "local":
            results = downloader.sync_local(force=self.config.pipeline.force)
            self.console.print(f"  Registered {len(results)} local audio file(s)")
            return

        results = downloader.download_all(force=self.config.pipeline.force)
        self.console.print(f"  Downloaded {len(results)} new video(s)")

    def _run_transcribe(self) -> None:
        downloader = YouTubeDownloader(self.config.youtube, self.config.paths.downloads_dir)
        videos = downloader.load_manifest()
        if not videos:
            self.console.print("[yellow]  No audio files found. Run download first.[/yellow]")
            return

        client = AudioIntelClient(self.config.audio_intel)
        try:
            health = client.health()
        except Exception as exc:
            raise RuntimeError(
                f"audio-intel is not reachable at {self.config.audio_intel.base_url}: {exc}"
            ) from exc

        self.console.print(f"  [green]{client.describe_runtime()}[/green]")
        self.console.print(
            f"  Service model: {health.get('model', '?')} | "
            f"alignment: {health.get('alignment_enabled', '?')} | "
            f"speakers: {health.get('speakers_enabled', health.get('diarization_enabled', '?'))} | "
            f"sound_events: {health.get('sound_events_enabled', '?')}"
        )

        transcripts_dir = Path(self.config.paths.transcripts_dir)
        total = len(videos)
        done = 0

        for video in videos:
            out_path = transcripts_dir / f"{video.video_id}.json"
            if out_path.exists() and not self.config.pipeline.force:
                if not self.config.audio_intel.align:
                    done += 1
                    continue
                existing = load_transcript(out_path)
                if existing.aligned_at:
                    done += 1
                    continue
                # JSON exists but alignment never ran — re-transcribe instead of skipping.

            progress = self._format_progress(done + 1, total)
            self.console.print(f"  {progress} Transcribing: {video.title[:60]}...")
            result = client.transcribe_file(video.video_id, video.audio_path)
            client.save_transcript(result, transcripts_dir)
            done += 1
            if result.stats:
                self.console.print(
                    f"    → {result.stats.summary(self.config.audio_intel.language)}"
                )

        self.console.print(f"  Processed {done} transcript(s)")

    def _run_segment(self) -> None:
        target_language = (
            self.config.audio_intel.language
            if self.config.segmentation.filter_non_target_language
            else None
        )
        segmenter = AudioSegmenter(
            self.config.segmentation,
            self.config.quality,
            target_language,
            filter_alignment_failed=self.config.audio_intel.drop_failed_segments,
            speaker_name=self.config.export.speaker_name,
        )
        workers = max(1, self.config.pipeline.num_workers)

        transcripts_dir = Path(self.config.paths.transcripts_dir)
        segments_dir = Path(self.config.paths.segments_dir)
        all_clips: list[AudioClip] = []

        jobs = []
        skipped = 0
        for transcript_path in sorted(transcripts_dir.glob("*.json")):
            video_id = transcript_path.stem
            clip_manifest = segments_dir / video_id / "clips.jsonl"
            # Resume: clips.jsonl means this video already survived filter/cut.
            if clip_manifest.exists() and not self.config.pipeline.force:
                all_clips.extend(self._load_clips(clip_manifest))
                skipped += 1
                continue
            jobs.append(transcript_path)

        total = skipped + len(jobs)
        attempted = 0

        if jobs:
            self.console.print(
                f"  {self._format_progress(skipped, total)} starting "
                f"({len(jobs)} to segment, {skipped} already done)"
            )

        def process_video(transcript_path: Path) -> tuple[str, list[AudioClip], str]:
            video_id = transcript_path.stem
            transcript = load_transcript(transcript_path)
            clips, seg_stats = segmenter.build_clips(transcript)
            video_segments_dir = segments_dir / video_id
            if self.config.pipeline.force and video_segments_dir.exists():
                # Wipe old WAVs so a tighter filter does not leave orphan clips on disk.
                shutil.rmtree(video_segments_dir)
            segmenter.export_clips(clips, video_segments_dir, num_workers=workers)
            segmenter.save_clip_manifest(clips, video_segments_dir)
            return video_id, clips, seg_stats.summary()

        def report_segment(video_id: str, summary: str, *, completed: int) -> None:
            progress = self._format_progress(completed, total)
            self.console.print(f"  {progress} {video_id}: {summary}")

        if workers == 1:
            for transcript_path in jobs:
                video_id, clips, summary = process_video(transcript_path)
                all_clips.extend(clips)
                attempted += 1
                report_segment(video_id, summary, completed=skipped + attempted)
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(process_video, path): path for path in jobs}
                for future in as_completed(futures):
                    video_id, clips, summary = future.result()
                    all_clips.extend(clips)
                    attempted += 1
                    report_segment(video_id, summary, completed=skipped + attempted)

        total_duration = sum(c.duration for c in all_clips)
        self.console.print(f"  Total: {len(all_clips)} clips, {total_duration / 3600:.2f} hours")

    def _run_cluster(self) -> None:
        transcripts_dir = Path(self.config.paths.transcripts_dir)
        if not any(transcripts_dir.glob("*.json")):
            self.console.print("[yellow]  No transcripts to cluster.[/yellow]")
            return
        result = cluster_source_transcripts(
            self.config.paths.work_dir,
            transcripts_dir,
            base_speaker=self.config.export.speaker_name,
            threshold=self.config.speaker_cluster.threshold,
            segments_dir=self.config.paths.segments_dir,
        )
        globals_n = len(set(result.mapping.values()))
        self.console.print(
            f"  Mapped {len(result.mapping)} local speakers → {globals_n} global "
            f"(merged locals={result.clustered}, singletons={result.singletons}, "
            f"missing embeddings={result.missing_embeddings})"
        )
        if result.missing_embeddings and not result.mapping:
            self.console.print(
                "  [yellow]No speaker_map written — re-transcribe with diarize "
                "to store roster embeddings, or ensure segments exist for "
                "label-suffix fallback.[/yellow]"
            )
        elif result.missing_embeddings:
            self.console.print(
                "  [yellow]Embeddings missing on older transcripts — used "
                "diarization-id fallback. Re-transcribe to cluster by voice.[/yellow]"
            )
        self.console.print(f"  Wrote {result.map_path}")

    def _run_export(self) -> None:
        segments_dir = Path(self.config.paths.segments_dir)
        all_clips: list[AudioClip] = []

        for manifest in sorted(segments_dir.glob("*/clips.jsonl")):
            all_clips.extend(self._load_clips(manifest))

        if not all_clips:
            self.console.print("[yellow]  No clips to export.[/yellow]")
            return

        speaker_map = load_speaker_map(self.config.paths.work_dir)
        if speaker_map:
            all_clips = [
                replace(
                    clip,
                    speaker_name=apply_speaker_map_to_name(clip.speaker_name, speaker_map),
                )
                for clip in all_clips
            ]
            self.console.print(f"  Applied speaker_map.json ({len(speaker_map)} local ids)")
        else:
            self.console.print(
                "  [dim]No speaker_map.json — exporting video-scoped local names[/dim]"
            )

        before = len(all_clips)
        all_clips, dropped = prune_tiny_speakers(
            all_clips,
            min_clips=self.config.speaker_cluster.min_clips,
            min_speech_sec=self.config.speaker_cluster.min_speech_sec,
        )
        if dropped:
            dropped_clips = before - len(all_clips)
            self.console.print(
                f"  Pruned {len(dropped)} tiny speaker(s) "
                f"({dropped_clips} clips; need ≥{self.config.speaker_cluster.min_clips} clips "
                f"and ≥{self.config.speaker_cluster.min_speech_sec:.0f}s)"
            )
            for speaker, (count, speech) in sorted(
                dropped.items(), key=lambda item: item[1][1]
            )[:12]:
                self.console.print(
                    f"    [dim]- {speaker}: {count} clips, {speech:.1f}s[/dim]"
                )
            if len(dropped) > 12:
                self.console.print(f"    [dim]… +{len(dropped) - 12} more[/dim]")

        if not all_clips:
            self.console.print("[yellow]  No clips left after speaker prune.[/yellow]")
            return

        if self.config.pipeline.force:
            clear_dataset_dir(self.config.export.output_dir)

        exporter = DatasetExporter(self.config.export, stress=self.config.stress)
        stats = exporter.export(all_clips, segments_dir)

        table = Table(title="Dataset Summary")
        table.add_column("Metric")
        table.add_column("Value")
        table.add_row("Clips", str(stats.total_clips))
        if stats.eval_metadata_path:
            table.add_row("Train / Eval", f"{stats.train_clips} / {stats.eval_clips}")
        table.add_row("Duration", f"{stats.total_duration_sec / 3600:.2f} h")
        table.add_row("Metadata", stats.metadata_path)
        if stats.eval_metadata_path:
            table.add_row("Eval metadata", stats.eval_metadata_path)
        table.add_row("WAVs", stats.wavs_dir)
        table.add_row("Reference", stats.reference_path or "—")
        self.console.print(table)

    @staticmethod
    def _load_clips(manifest_path: Path) -> list[AudioClip]:
        clips: list[AudioClip] = []
        with manifest_path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                clips.append(AudioClip(**json.loads(line)))
        return clips
