#!/usr/bin/env bash
# Unified project control: install, TTS server, dataset pipeline, Fish LoRA training.
#
# Checkpoints are downloaded by install (not by train). Runtime data lives under
# DATA_ROOT in .env (default ./data).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# shellcheck disable=SC1091
source "${ROOT}/scripts/env.sh"

usage() {
  cat <<'EOF'
Usage: ./run.sh <command> [args...]

Commands:
  install [target]     Create venv, install deps, download checkpoints.
                       Targets: server (default) | dataset | training | all | dev
  server [-c config]   Start Fish Speech TTS HTTP server (proxy to vLLM)
  vllm <cmd>           vLLM-Omni TTS server: install, start, stop, restart, status
  dataset <cmd>        Dataset builder (fish-dataset): run, merge, all, sources,
                       datasets, hf-import, …
  dataset-build        Run all sources + merge (see scripts/dataset-build.sh)
  train <step>         Fish Speech s2-pro LoRA fine-tuning
  bg-train [fish]      Start Fish training in background with logging
  tensorboard <cmd>    TensorBoard for training runs: start, stop, status
  server <start|…>     Start/stop/restart TTS server, or length-check it
  status               Dataset, training, server, GPU overview
  logs <name>          Tail server / training / pipeline log
  analyze <kind> PATH  Quality helpers: transcripts | clips
  synthesize           Quick HTTP synthesis test
  init [-c path]       Create .env from .env.example

Examples:
  ./run.sh install all
  ./run.sh server
  ./run.sh server -c .env
  ./run.sh dataset run --source game-vo-1
  ./run.sh dataset datasets
  ./run.sh dataset all
  ./run.sh dataset-build
  ./run.sh bg-train
  ./run.sh server start
  ./run.sh vllm install
  ./run.sh vllm start
  ./run.sh status
  ./run.sh analyze transcripts data/work/channel-a/transcripts
  ./run.sh synthesize -t "Привіт" -w ref.wav
  ./run.sh train all
  ./run.sh train infer --text "Привіт" --speaker-wav ref.wav --out out.wav
  ./run.sh server length-check --ref ref.wav --ref-text "текст референсу"
  ./run.sh init

Environment:
  PROJECT_VENV         Override venv path (default: ./.venv)

Ukrainian LoRA end-to-end: docs/ukrainian-lora-playbook.md
See also: ./run.sh install --help
EOF
}

train_usage() {
  cat <<'EOF'
Usage: ./run.sh train <export|vq|protos|train|merge|export-vllm|infer|all> [args...]

Steps:
  export       pipe-delimited dataset → Fish .wav + .lab under {data_root}/training/raw/
               (stress marks applied here from STRESS_* + lexicon + acoustic fallback)
  vq           Extract semantic tokens (.npy) with the stock s2-pro codec
  protos       Pack tokens into protobuf shards for training
  train        LoRA fine-tune the s2-pro LLAMA weights
  merge        Merge LoRA into a standalone checkpoint → training/merged/
  export-vllm  Convert merged checkpoint to HF layout for vLLM-Omni → training/vllm/
  infer        CLI test synthesis with the merged checkpoint
  all          export → vq → protos → train → merge (extra args go to train only)

Requires stock Fish Speech s2-pro from: ./run.sh install all
Requires exported dataset (e.g. ./run.sh dataset run && ./run.sh dataset merge)

To serve the fine-tuned model:
  FISH_SPEECH_USE_FINETUNED=true
EOF
}

cmd="${1:-}"
if [[ $# -gt 0 ]]; then
  shift
fi

case "${cmd}" in
  install)
    exec "${ROOT}/scripts/install.sh" "$@"
    ;;
  dataset)
    activate_venv "${ROOT}"
    if [[ "${1:-}" == "all" ]]; then
      shift
      exec "${ROOT}/scripts/dataset-build.sh" "$@"
    fi
    exec fish-dataset "$@"
    ;;
  dataset-build)
    exec "${ROOT}/scripts/dataset-build.sh" "$@"
    ;;
  status)
    exec "${ROOT}/scripts/status.sh" "$@"
    ;;
  logs)
    exec "${ROOT}/scripts/logs.sh" "$@"
    ;;
  analyze)
    kind="${1:-}"
    if [[ $# -gt 0 ]]; then
      shift
    fi
    activate_venv "${ROOT}"
    case "${kind}" in
      transcripts)
        exec python "${ROOT}/scripts/analyze_transcripts.py" "$@"
        ;;
      clips)
        exec python "${ROOT}/scripts/analyze_clips.py" "$@"
        ;;
      -h | --help | help | "")
        cat <<'EOF'
Usage: ./run.sh analyze <transcripts|clips> [PATH]

  transcripts  ASR JSON folder (default inside script if PATH omitted)
  clips        segments/ folder with */clips.jsonl
EOF
        exit 0
        ;;
      *)
        echo "error: unknown analyze kind: ${kind} (use transcripts|clips)" >&2
        exit 1
        ;;
    esac
    ;;
  synthesize)
    exec "${ROOT}/scripts/synthesize.sh" "$@"
    ;;
  vllm)
    exec "${ROOT}/scripts/vllm.sh" "$@"
    ;;
  bg-train)
    exec "${ROOT}/scripts/bg-train.sh" "$@"
    ;;
  tensorboard | tb)
    exec "${ROOT}/scripts/tensorboard.sh" "$@"
    ;;
  server)
    sub="${1:-}"
    case "${sub}" in
      start | stop | restart | status)
        shift
        exec "${ROOT}/scripts/server.sh" "${sub}" "$@"
        ;;
      length-check)
        shift
        activate_venv "${ROOT}"
        exec python -m fish_studio.server.length_check "$@"
        ;;
      *)
        activate_venv "${ROOT}"
        exec python -m fish_studio.server.serve "$@"
        ;;
    esac
    ;;
  init)
    activate_venv "${ROOT}"
    exec fish-dataset init "$@"
    ;;
  train)
    step="${1:-all}"
    if [[ $# -gt 0 ]]; then
      shift
    fi
    activate_venv "${ROOT}"
    export PYTHONUNBUFFERED=1
    case "${step}" in
      export)
        exec python -m fish_studio.training.export_dataset "$@"
        ;;
      vq)
        exec python -m fish_studio.training.extract_vq "$@"
        ;;
      protos)
        exec python -m fish_studio.training.build_protos "$@"
        ;;
      train)
        exec python -m fish_studio.training.train_lora "$@"
        ;;
      merge)
        exec python -m fish_studio.training.merge_lora "$@"
        ;;
      export-vllm)
        exec python -m fish_studio.training.export_vllm "$@"
        ;;
      infer)
        exec python -m fish_studio.training.infer "$@"
        ;;
      all)
        python -m fish_studio.training.export_dataset
        python -m fish_studio.training.extract_vq
        python -m fish_studio.training.build_protos
        python -m fish_studio.training.train_lora "$@"
        exec python -m fish_studio.training.merge_lora
        ;;
      -h | --help | help)
        train_usage
        exit 0
        ;;
      *)
        echo "error: unknown train step: ${step}" >&2
        train_usage >&2
        exit 1
        ;;
    esac
    ;;
  "" | -h | --help | help)
    usage
    exit 0
    ;;
  *)
    echo "error: unknown command: ${cmd}" >&2
    usage >&2
    exit 1
    ;;
esac
