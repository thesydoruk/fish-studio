"""CLI entry point: dataset pipeline, merge, Hugging Face import, init."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from fish_studio.config import ProjectConfig, load_config, _slugify_source_id
from fish_studio.dataset.export import clear_dataset_dir
from fish_studio.dataset.hf_import import import_sources, parse_source, probe_sources
from fish_studio.dataset.merge import (
    is_export_ready,
    list_export_ready_datasets,
    merge_datasets,
)
from fish_studio.dataset.pipeline import Pipeline


def _run_sources(
    project: ProjectConfig,
    *,
    source_ids: tuple[str, ...],
    steps: tuple[str, ...],
    force: bool,
    config_path: str,
) -> None:
    """Run the pipeline for each selected source sequentially."""
    console = Console()
    try:
        sources = project.resolve_sources(source_ids or None)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    if not sources:
        raise click.ClickException("No enabled sources in .env")

    project.workspace().ensure_layout()

    for index, source in enumerate(sources, start=1):
        if len(sources) > 1:
            console.print(f"\n[bold]Source {index}/{len(sources)}:[/bold] {source.id}\n")
        app_config = project.app_config(source)
        if steps:
            app_config.pipeline.steps = list(steps)
        if force:
            app_config.pipeline.force = True
        Pipeline(app_config, console, config_path=config_path).run()


@click.group()
@click.version_option()
def main() -> None:
    """Fish Speech dataset builder and shared project configuration."""


@main.command("run")
@click.option(
    "-c",
    "--config",
    "config_path",
    default=".env",
    show_default=True,
    type=click.Path(exists=False),
    help="Path to .env",
)
@click.option(
    "--source",
    "source_ids",
    multiple=True,
    help="Run only these source id(s). Default: all enabled sources.",
)
@click.option(
    "--step",
    "steps",
    multiple=True,
    type=click.Choice(["download", "transcribe", "segment", "cluster", "export", "all"]),
    help="Run specific pipeline step(s). Default: from config.",
)
@click.option("--force", is_flag=True, help="Re-process already completed items")
def run_cmd(
    config_path: str,
    source_ids: tuple[str, ...],
    steps: tuple[str, ...],
    force: bool,
) -> None:
    """Run the dataset building pipeline."""
    project = load_config(config_path)
    if steps:
        project = replace(project, pipeline=replace(project.pipeline, steps=list(steps)))
    if force:
        project = replace(project, pipeline=replace(project.pipeline, force=True))
    _run_sources(project, source_ids=source_ids, steps=steps, force=force, config_path=config_path)


@main.command("sources")
@click.option("-c", "--config", "config_path", default=".env")
def sources_cmd(config_path: str) -> None:
    """List configured SOURCES and every export-ready dataset on disk."""
    project = load_config(config_path)
    ws = project.workspace()
    console = Console()

    table = Table(title="Configured SOURCES (.env)")
    table.add_column("ID")
    table.add_column("Kind")
    table.add_column("Enabled")
    table.add_column("Work")
    table.add_column("Dataset")
    table.add_column("Clips")

    for source in project.sources:
        app = project.app_config(source)
        dataset_dir = Path(app.export.output_dir)
        clip_count = "—"
        wavs = dataset_dir / "wavs"
        if wavs.is_dir():
            clip_count = str(len(list(wavs.glob("*.wav"))))
        table.add_row(
            source.id,
            source.kind,
            "yes" if source.enabled else "no",
            app.paths.work_dir,
            app.export.output_dir,
            clip_count,
        )
    if not project.sources:
        table.add_row("—", "—", "—", "—", "(SOURCES is empty)", "—")
    table.add_row("", "", "", "", f"[dim]data_root: {ws.data_root}[/dim]", "")
    console.print(table)
    console.print()
    _print_on_disk_datasets(ws.datasets_root)


@main.command("datasets")
@click.option("-c", "--config", "config_path", default=".env")
def datasets_cmd(config_path: str) -> None:
    """List export-ready folders under ``data/datasets/`` (YouTube, HF, merged)."""
    project = load_config(config_path)
    _print_on_disk_datasets(project.workspace().datasets_root)


def _print_on_disk_datasets(datasets_root: Path) -> None:
    table = Table(title="Export-ready datasets on disk")
    table.add_column("ID")
    table.add_column("Train clips")
    table.add_column("Eval clips")
    table.add_column("WAVs")
    table.add_column("Hours (stats)")
    table.add_column("Path")

    ready = list_export_ready_datasets(datasets_root)
    if not ready:
        table.add_row("—", "—", "—", "—", "—", str(datasets_root))
        Console().print(table)
        return

    for path in ready:
        train_n = _count_metadata_rows(path / "metadata_train.csv")
        eval_n = _count_metadata_rows(path / "metadata_eval.csv")
        wav_n = len(list((path / "wavs").glob("*.wav"))) if (path / "wavs").is_dir() else 0
        hours = "—"
        stats_path = path / "stats.json"
        if stats_path.is_file():
            try:
                payload = json.loads(stats_path.read_text(encoding="utf-8"))
                dur = payload.get("total_duration_sec")
                if dur is not None:
                    hours = f"{float(dur) / 3600.0:.2f}"
            except (OSError, ValueError, TypeError):
                pass
        table.add_row(path.name, str(train_n), str(eval_n), str(wav_n), hours, str(path))
    Console().print(table)


def _count_metadata_rows(path: Path) -> int:
    if not path.is_file():
        return 0
    # Header + one row per clip; ignore blank lines.
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return max(0, len(lines) - 1)


@main.command("merge")
@click.option("-c", "--config", "config_path", default=".env")
@click.option(
    "-o",
    "--output",
    "output_id",
    default="combined",
    show_default=True,
    help="Output dataset slug under {data_root}/datasets/",
)
@click.option(
    "--source",
    "source_ids",
    multiple=True,
    help=(
        "Dataset id(s) to merge (folder under datasets/, or a SOURCES id). "
        "Default: every export-ready dataset on disk except the output slug."
    ),
)
@click.option(
    "--from-sources",
    is_flag=True,
    help="Only merge enabled SOURCES entries (ignore HF imports and other on-disk datasets).",
)
def merge_cmd(
    config_path: str,
    output_id: str,
    source_ids: tuple[str, ...],
    from_sources: bool,
) -> None:
    """Merge exported datasets into one pipe-delimited dataset for training.

    By default merges *all* export-ready folders under ``datasets/`` (YouTube
    exports, HF imports, …), excluding ``-o``. ``--from-sources`` limits the
    merge to enabled SOURCES entries; ``--source`` picks specific dataset ids.
    """
    project = load_config(config_path)
    ws = project.workspace()
    ws.ensure_layout()
    output_dir = ws.dataset_dir(_slugify_source_id(output_id))

    if from_sources and source_ids:
        raise click.ClickException("Use either --from-sources or --source, not both")

    try:
        if from_sources:
            sources = project.resolve_sources(None)
            source_dirs = [Path(project.app_config(source).export.output_dir) for source in sources]
        elif source_ids:
            source_dirs = [_resolve_merge_dataset(project, dataset_id) for dataset_id in source_ids]
        else:
            source_dirs = list_export_ready_datasets(
                ws.datasets_root,
                exclude_names={output_dir.name},
            )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    if not source_dirs:
        raise click.ClickException(
            "No export-ready datasets to merge. "
            "Run dataset export / hf-import first, or pass --source / --from-sources."
        )

    missing = [path for path in source_dirs if not is_export_ready(path)]
    if missing:
        raise click.ClickException(
            "Not export-ready (need metadata_train.csv + wavs/): "
            + ", ".join(str(path) for path in missing)
        )

    # Drop a previous combined tree so leftover WAVs cannot pollute the new merge.
    clear_dataset_dir(output_dir)

    stats = merge_datasets(
        source_dirs,
        output_dir,
        eval_split_size=project.export.eval_split_size,
        seed=project.export.seed,
        speaker_name=project.export.speaker_name,
    )

    table = Table(title="Merged Dataset")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Output", stats.output_dir)
    table.add_row("Inputs", ", ".join(path.name for path in source_dirs))
    table.add_row("Total clips", str(stats.total_clips))
    table.add_row("Train / Eval", f"{stats.train_clips} / {stats.eval_clips}")
    for source_id, count in stats.source_clips.items():
        table.add_row(f"From {source_id}", str(count))
    Console().print(table)


def _resolve_merge_dataset(project: ProjectConfig, dataset_id: str) -> Path:
    """Resolve ``--source`` to an on-disk dataset dir (folder name or SOURCES id)."""
    slug = _slugify_source_id(dataset_id)
    candidate = project.workspace().dataset_dir(slug)
    if is_export_ready(candidate):
        return candidate

    # Fall back to configured SOURCES so --source <SOURCES.id> still resolves
    # when the on-disk folder name differs from the slug.
    try:
        sources = project.resolve_sources((dataset_id,))
    except ValueError as exc:
        raise ValueError(
            f"Dataset '{dataset_id}' not found under {project.workspace().datasets_root} "
            f"and not in SOURCES ({exc})"
        ) from exc
    path = Path(project.app_config(sources[0]).export.output_dir)
    if not is_export_ready(path):
        raise ValueError(f"Dataset '{dataset_id}' is not export-ready: {path}")
    return path


def _single_step_cmd(config_path: str, step: str, source_ids: tuple[str, ...], force: bool) -> None:
    project = load_config(config_path)
    project = replace(project, pipeline=replace(project.pipeline, steps=[step], force=force))
    _run_sources(
        project, source_ids=source_ids, steps=(step,), force=force, config_path=config_path
    )


@main.command("download")
@click.option("-c", "--config", "config_path", default=".env")
@click.option("--source", "source_ids", multiple=True)
@click.option("--force", is_flag=True)
def download_cmd(config_path: str, source_ids: tuple[str, ...], force: bool) -> None:
    """Download or register source audio."""
    _single_step_cmd(config_path, "download", source_ids, force)


@main.command("transcribe")
@click.option("-c", "--config", "config_path", default=".env")
@click.option("--source", "source_ids", multiple=True)
@click.option("--force", is_flag=True)
def transcribe_cmd(config_path: str, source_ids: tuple[str, ...], force: bool) -> None:
    """Transcribe downloaded audio via audio-intel."""
    _single_step_cmd(config_path, "transcribe", source_ids, force)


@main.command("segment")
@click.option("-c", "--config", "config_path", default=".env")
@click.option("--source", "source_ids", multiple=True)
@click.option("--force", is_flag=True)
def segment_cmd(config_path: str, source_ids: tuple[str, ...], force: bool) -> None:
    """Segment audio into training clips."""
    _single_step_cmd(config_path, "segment", source_ids, force)


@main.command("cluster")
@click.option("-c", "--config", "config_path", default=".env")
@click.option("--source", "source_ids", multiple=True)
@click.option("--force", is_flag=True, help="Ignored (cluster always rewrites speaker_map.json)")
def cluster_cmd(config_path: str, source_ids: tuple[str, ...], force: bool) -> None:
    """Cluster video-local speakers within each source via embeddings."""
    _single_step_cmd(config_path, "cluster", source_ids, force)


@main.command("speakers-cluster")
@click.option("-c", "--config", "config_path", default=".env")
@click.option("--source", "source_ids", multiple=True)
def speakers_cluster_cmd(config_path: str, source_ids: tuple[str, ...]) -> None:
    """Alias for ``cluster``: write work/<source>/speaker_map.json."""
    _single_step_cmd(config_path, "cluster", source_ids, force=False)


@main.command("speakers-remap")
@click.option("-c", "--config", "config_path", default=".env")
@click.option(
    "-i",
    "--map",
    "map_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="JSON object mapping old speaker_name → new speaker_name",
)
@click.option(
    "--dataset",
    "dataset_id",
    required=True,
    help="Dataset slug under data/datasets/ (e.g. combined)",
)
def speakers_remap_cmd(config_path: str, map_path: str, dataset_id: str) -> None:
    """Manually remap speaker_name values in an exported dataset's metadata CSVs."""
    from fish_studio.dataset.speaker_cluster import remap_dataset_speakers

    project = load_config(config_path)
    dataset_dir = project.workspace().dataset_dir(_slugify_source_id(dataset_id))
    if not is_export_ready(dataset_dir):
        raise click.ClickException(f"Dataset not export-ready: {dataset_dir}")
    raw = json.loads(Path(map_path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not raw:
        raise click.ClickException("Map file must be a non-empty JSON object")
    mapping = {str(key): str(value) for key, value in raw.items() if key and value}
    counts = remap_dataset_speakers(dataset_dir, mapping)
    console = Console()
    console.print(f"Remapped speakers in {dataset_dir}")
    for name, changed in counts.items():
        console.print(f"  {name}: {changed} row(s) changed")


@main.command("export")
@click.option("-c", "--config", "config_path", default=".env")
@click.option("--source", "source_ids", multiple=True)
@click.option("--force", is_flag=True, help="Remove existing dataset output before export")
def export_cmd(config_path: str, source_ids: tuple[str, ...], force: bool) -> None:
    """Export clips to pipe-delimited dataset format for Fish Speech training."""
    _single_step_cmd(config_path, "export", source_ids, force)


@main.command("hf-import")
@click.option("-c", "--config", "config_path", default=".env")
@click.option(
    "--source",
    "source_specs",
    multiple=True,
    required=True,
    help="Hugging Face source as repo_id[:config][@split]=speaker (repeatable)",
)
@click.option(
    "-o",
    "--output",
    "output_id",
    default="hf",
    show_default=True,
    help="Output dataset slug under {data_root}/datasets/",
)
@click.option(
    "--max-wer",
    type=float,
    help="Drop rows whose dataset-provided wer exceeds this value",
)
@click.option("--workers", type=int, default=8, show_default=True, help="Parallel ffmpeg workers")
@click.option(
    "--streaming",
    is_flag=True,
    help="Stream rows instead of downloading the whole repo (for very large datasets)",
)
@click.option("--force", is_flag=True, help="Re-import sources that already have a manifest")
@click.option("--probe", is_flag=True, help="Report source audio specs without importing")
def hf_import_cmd(
    config_path: str,
    source_specs: tuple[str, ...],
    output_id: str,
    max_wer: float | None,
    workers: int,
    streaming: bool,
    force: bool,
    probe: bool,
) -> None:
    """Import Hugging Face speech datasets into a pipe-delimited dataset."""
    project = load_config(config_path)
    console = Console()

    try:
        sources = [parse_source(spec) for spec in source_specs]
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    if probe:
        report = probe_sources(sources, streaming=streaming)
        table = Table(title="Hugging Face Source Probe")
        table.add_column("Source")
        table.add_column("Sample rates")
        table.add_column("Mean dur")
        table.add_column("Fields")
        table.add_column("Sample text")
        for label, info in report.items():
            rates = ", ".join(f"{rate} Hz x{count}" for rate, count in info["sample_rates"].items())
            table.add_row(
                label,
                rates or "—",
                f"{info['mean_duration_sec']} s" if info["mean_duration_sec"] else "—",
                ", ".join(info["fields"]),
                "\n".join(info["texts"]) or "—",
            )
        console.print(table)
        return

    output_dir = project.workspace().dataset_dir(output_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    def on_source(stats) -> None:  # noqa: ANN001 - rich progress callback
        suffix = " (resumed)" if stats.resumed else ""
        console.print(
            f"[green]{stats.speaker}[/green]: {stats.kept} clips, "
            f"{stats.duration_sec / 3600:.2f} h from {stats.repo_id}{suffix}"
        )

    stats = import_sources(
        sources,
        output_dir,
        segmentation=project.segmentation,
        export=project.export,
        max_wer=max_wer,
        num_workers=workers,
        streaming=streaming,
        force=force,
        progress=on_source,
    )

    table = Table(title="Imported Dataset")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Output", stats.output_dir)
    table.add_row("Total clips", str(stats.total_clips))
    table.add_row("Train / Eval", f"{stats.train_clips} / {stats.eval_clips}")
    table.add_row("Duration", f"{stats.duration_sec / 3600:.2f} h")
    for source_stats in stats.sources:
        rates = ", ".join(f"{r} Hz" for r in sorted(source_stats.sample_rates))
        table.add_row(
            f"Speaker {source_stats.speaker}",
            f"{source_stats.kept} clips"
            f"{f' @ {rates}' if rates else ''}"
            f" (skipped: {source_stats.skipped_text} text,"
            f" {source_stats.skipped_duration} duration,"
            f" {source_stats.skipped_wer} wer,"
            f" {source_stats.skipped_audio} audio)",
        )
    console.print(table)


@main.command("init")
@click.option("-c", "--config", "config_path", default=".env")
def init_cmd(config_path: str) -> None:
    """Create .env from example template."""
    dest = Path(config_path)
    if dest.exists():
        raise click.ClickException(f"{dest} already exists")

    example = Path(".env.example")
    if not example.exists():
        raise click.ClickException(".env.example not found")

    dest.write_text(example.read_text(encoding="utf-8"), encoding="utf-8")
    click.echo(f"Created {dest}. Edit sources and run: fish-dataset run")


if __name__ == "__main__":
    main()
