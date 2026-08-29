"""Fish Speech fine-tuning path helpers."""

from __future__ import annotations

from pathlib import Path

from fish_studio.config import ProjectConfig, TrainingConfig
from fish_studio.paths import WorkspacePaths, resolve_under_data_root


def training_layout(ws: WorkspacePaths) -> dict[str, Path]:
    """Standard tree under ``data/training/``: raw → protos → runs → merged → vllm."""
    base = ws.training_dir
    return {
        "base": base,
        "raw": base / "raw",  # pipe-delimited dataset copied for VQ extraction
        "protos": base / "protos",  # packed protobuf shards consumed by the trainer
        "runs": base / "runs",  # Lightning logs + step_*.ckpt
        "merged": base / "merged",  # LoRA folded into a standalone LLAMA checkpoint
        "vllm": base / "vllm",  # HF key layout for vLLM-Omni
    }


def project_run_dir(ws: WorkspacePaths, project_name: str) -> Path:
    return training_layout(ws)["runs"] / project_name


def checkpoint_dir(ws: WorkspacePaths, project_name: str) -> Path:
    return project_run_dir(ws, project_name) / "checkpoints"


def resolve_base_checkpoint(project: ProjectConfig, ws: WorkspacePaths) -> Path:
    training: TrainingConfig = project.training
    if training.base_checkpoint:
        return resolve_under_data_root(ws.data_root, training.base_checkpoint)
    return resolve_under_data_root(ws.data_root, project.fish_speech.llama_checkpoint)


def resolve_merged_checkpoint(project: ProjectConfig, ws: WorkspacePaths) -> Path:
    training = project.training
    if training.merged_checkpoint:
        return resolve_under_data_root(ws.data_root, training.merged_checkpoint)
    return training_layout(ws)["merged"]


def latest_lora_checkpoint(checkpoint_dir_path: Path) -> Path | None:
    """Newest ``step_*.ckpt`` by mtime (not step number — interrupted runs can skip)."""
    if not checkpoint_dir_path.is_dir():
        return None
    candidates = sorted(
        checkpoint_dir_path.glob("step_*.ckpt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def ensure_training_dirs(ws: WorkspacePaths) -> dict[str, Path]:
    layout = training_layout(ws)
    for path in layout.values():
        path.mkdir(parents=True, exist_ok=True)
    return layout
