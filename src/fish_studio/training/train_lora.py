#!/usr/bin/env python3
"""LoRA fine-tune Fish Speech s2-pro text-to-semantic model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from fish_studio.project_context import try_load_project
from fish_studio.training.layout import (
    ensure_training_dirs,
    latest_lora_checkpoint,
    project_run_dir,
    resolve_base_checkpoint,
)
from fish_studio.training.upstream import run_fish_train


# The upstream dataset packs a fish-speech 1.5 prompt that s2-pro never sees at
# generation time; training on it drives the slow layers to silence.
DATASET_TARGET = "fish_studio.training.reference_dataset.ReferenceConditionedIterableDataset"

LORA_TARGETS = {
    "attention",
    "mlp",
    "embeddings",
    "output",
    "fast_attention",
    "fast_mlp",
    "fast_embeddings",
    "fast_output",
}


def _quote_hydra_path(path: Path) -> str:
    """Quote for Hydra: unquoted paths split on ``=``; POSIX form keeps Windows drives intact."""
    return f"'{path.as_posix()}'"


def _trainer_strategy() -> str:
    """Pick a Lightning strategy for LoRA fine-tuning.

    A lone GPU stays off DDP: it only adds gradient-sync overhead and its
    unused-parameter bookkeeping is easy to trip with adapters on a subset of layers.
    """
    try:
        import torch

        devices = torch.cuda.device_count()
    except Exception:  # noqa: BLE001 - strategy choice must never block training
        devices = 1
    return "auto" if devices <= 1 else "ddp_find_unused_parameters_true"


def _lora_targets(raw: str | list[str]) -> str:
    values = raw.split(",") if isinstance(raw, str) else list(raw)
    targets = [value.strip() for value in values if value.strip()]
    unknown = sorted(set(targets) - LORA_TARGETS)
    if unknown:
        raise ValueError(
            f"unknown LoRA target(s): {', '.join(unknown)}. "
            f"Valid: {', '.join(sorted(LORA_TARGETS))}"
        )
    if not targets:
        raise ValueError("at least one LoRA target module is required")
    return ",".join(targets)


def parse_args() -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("-c", "--config", default=".env")
    pre_args, _ = pre.parse_known_args()
    project = try_load_project(pre_args.config)

    defaults: dict[str, object] = {
        "project_name": "fish-uk",
        "max_steps": 10000,
        "batch_size": 4,
        "grad_accum": 1,
        "lr": 1e-4,
        "val_interval": 100,
        "save_top_k": -1,
        "lora_config": "r_8_alpha_16",
        "lora_r": 8,
        "lora_alpha": 16.0,
        "lora_dropout": 0.01,
        "lora_target_modules": ["fast_attention", "fast_mlp", "fast_embeddings", "fast_output"],
        "protos_dir": None,
        "base_checkpoint": None,
        "run_dir": None,
        "resume": "",
    }
    if project is not None:
        ws = project.workspace()
        ft = project.training
        defaults.update(
            {
                "project_name": ft.project_name,
                "max_steps": ft.max_steps,
                "batch_size": ft.batch_size,
                "grad_accum": ft.grad_accum,
                "lr": ft.lr,
                "val_interval": ft.val_check_interval,
                "save_top_k": ft.save_top_k,
                "lora_config": ft.lora_config,
                "lora_r": ft.lora_r,
                "lora_alpha": ft.lora_alpha,
                "lora_dropout": ft.lora_dropout,
                "lora_target_modules": ft.lora_target_modules,
                "protos_dir": ensure_training_dirs(ws)["protos"],
                "base_checkpoint": resolve_base_checkpoint(project, ws),
                "run_dir": project_run_dir(ws, ft.project_name),
                "resume": ft.continue_path,
            }
        )

    parser = argparse.ArgumentParser(description=__doc__, parents=[pre])
    parser.add_argument("--project-name", default=defaults["project_name"])
    parser.add_argument("--protos-dir", type=Path, default=defaults["protos_dir"])
    parser.add_argument("--base-checkpoint", type=Path, default=defaults["base_checkpoint"])
    parser.add_argument("--run-dir", type=Path, default=defaults["run_dir"])
    parser.add_argument("--max-steps", type=int, default=defaults["max_steps"])
    parser.add_argument("--batch-size", type=int, default=defaults["batch_size"])
    parser.add_argument("--grad-accum", type=int, default=defaults["grad_accum"])
    parser.add_argument("--lr", type=float, default=defaults["lr"])
    parser.add_argument("--val-interval", type=int, default=defaults["val_interval"])
    parser.add_argument(
        "--save-top-k",
        type=int,
        default=defaults["save_top_k"],
        help="Checkpoints to retain; -1 keeps every one so the best step stays available",
    )
    parser.add_argument("--lora-config", default=defaults["lora_config"])
    parser.add_argument("--lora-r", type=int, default=defaults["lora_r"])
    parser.add_argument("--lora-alpha", type=float, default=defaults["lora_alpha"])
    parser.add_argument("--lora-dropout", type=float, default=defaults["lora_dropout"])
    parser.add_argument(
        "--lora-target-modules",
        default=",".join(defaults["lora_target_modules"]),
        help="Comma-separated LoRA targets: attention, mlp, embeddings, output "
        "(slow text->semantic) and their fast_* acoustic counterparts",
    )
    parser.add_argument(
        "--resume",
        default=defaults["resume"],
        help='LoRA checkpoint path, or "auto" for latest step_*.ckpt',
    )
    return parser.parse_args()


def resolve_resume_checkpoint(args: argparse.Namespace, checkpoint_dir: Path) -> str | None:
    resume = (args.resume or "").strip()
    if not resume:
        return None
    if resume == "auto":
        latest = latest_lora_checkpoint(checkpoint_dir)
        if latest is None:
            return None
        print(f"[info] auto-resume from {latest}")
        return str(latest)
    path = Path(resume)
    if not path.is_file():
        raise FileNotFoundError(f"resume checkpoint not found: {path}")
    return str(path)


def main() -> None:
    args = parse_args()
    if args.protos_dir is None or not args.protos_dir.is_dir():
        print("[error] protos dir not found; run protos step first", file=sys.stderr)
        sys.exit(1)
    if args.base_checkpoint is None or not args.base_checkpoint.is_dir():
        print(f"[error] base checkpoint not found: {args.base_checkpoint}", file=sys.stderr)
        sys.exit(1)
    if args.run_dir is None:
        print("[error] run dir is required", file=sys.stderr)
        sys.exit(1)

    try:
        lora_targets = _lora_targets(args.lora_target_modules)
    except ValueError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        sys.exit(1)
    print(f"[info] LoRA r={args.lora_r} alpha={args.lora_alpha} targets=[{lora_targets}]")
    print("[info] dataset: reference-conditioned (s2-pro inference template)")

    args.run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.run_dir / "checkpoints"

    protos = _quote_hydra_path(args.protos_dir.resolve())
    base = _quote_hydra_path(args.base_checkpoint.resolve())
    run_dir = _quote_hydra_path(args.run_dir.resolve())
    ckpt_dir = _quote_hydra_path(checkpoint_dir.resolve())

    overrides = [
        "--config-name",
        "text2semantic_finetune",
        f"project={args.project_name}",
        f"pretrained_ckpt_path={base}",
        f"tokenizer.model_path={base}",
        f"paths.run_dir={run_dir}",
        f"paths.ckpt_dir={ckpt_dir}",
        f"train_dataset.proto_files=[{protos}]",
        f"val_dataset.proto_files=[{protos}]",
        f"train_dataset._target_={DATASET_TARGET}",
        f"val_dataset._target_={DATASET_TARGET}",
        f"trainer.max_steps={args.max_steps}",
        f"trainer.strategy={_trainer_strategy()}",
        f"trainer.accumulate_grad_batches={args.grad_accum}",
        f"trainer.val_check_interval={args.val_interval}",
        f"callbacks.model_checkpoint.every_n_train_steps={args.val_interval}",
        f"callbacks.model_checkpoint.save_top_k={args.save_top_k}",
        f"data.batch_size={args.batch_size}",
        f"data.num_workers={max(1, args.batch_size // 2)}",
        f"model.optimizer.lr={args.lr}",
        (
            "+model.model.lora_config={_target_:fish_studio.training.lora_patch.LoraConfig,"
            f"r:{args.lora_r},lora_alpha:{args.lora_alpha},lora_dropout:{args.lora_dropout},"
            f"target_modules:[{lora_targets}]}}"
        ),
    ]

    resume_ckpt = resolve_resume_checkpoint(args, checkpoint_dir)
    if resume_ckpt:
        # Not in the base hydra struct — must add with +.
        overrides.append(f"+ckpt_path={_quote_hydra_path(Path(resume_ckpt).resolve())}")

    run_fish_train(overrides)
    latest = latest_lora_checkpoint(checkpoint_dir)
    if latest:
        print(f"[done] latest LoRA checkpoint: {latest}")
    else:
        print("[done] training finished (no step_*.ckpt found yet)")


if __name__ == "__main__":
    main()
