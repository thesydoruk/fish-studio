#!/usr/bin/env bash
# Create .venv (if needed), install Python dependencies, and fetch model checkpoints.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./install.sh [target]

Targets:
  server   Install HTTP TTS server (default)
  dataset  Install dataset builder CLI
  training Install Fish Speech LoRA fine-tuning stack
  all      Install server, dataset builder, and training
  dev      Install all extras plus dev tools (pytest, ruff, pre-commit)

Environment:
  INSTALL_VENV    Path to venv directory (default: ./.venv)
  INSTALL_DATA_ROOT  Checkpoint download directory (default: DATA_ROOT from .env, else ./data)
  PYTHON          Python interpreter for venv creation (default: python3)
  SKIP_DOWNLOAD   Set to 1 to skip checkpoint download

Examples:
  ./install.sh
  ./install.sh all
  ./install.sh dataset
  ./install.sh training
  SKIP_DOWNLOAD=1 ./install.sh all
  INSTALL_VENV=/opt/venvs/fish ./install.sh all
EOF
}

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

resolve_data_root() {
  if [[ -n "${INSTALL_DATA_ROOT:-}" ]]; then
    echo "${INSTALL_DATA_ROOT}"
    return
  fi

  if [[ -f "${ROOT}/.env" ]] && command -v python3 >/dev/null 2>&1; then
  python3 - <<'PY'
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv:
    load_dotenv(".env")

data_root = Path(os.environ.get("DATA_ROOT", "./data"))
if data_root.is_absolute():
    print(data_root)
else:
    print((Path.cwd() / data_root).resolve())
PY
    return
  fi

  echo "${ROOT}/data"
}

TARGET="${1:-server}"
PYTHON="${PYTHON:-python3}"
VENV="${INSTALL_VENV:-${ROOT}/.venv}"
DATA_ROOT="$(resolve_data_root)"

case "$TARGET" in
  server | dataset | training | all | dev) ;;
  -h | --help | help)
    usage
    exit 0
    ;;
  *)
    echo "error: unknown target: $TARGET" >&2
    usage >&2
    exit 1
    ;;
esac

ensure_hf_hub() {
  python -m pip install -q huggingface_hub
}

download_fish_checkpoints() {
  local dir="${DATA_ROOT}/checkpoints/fish-speech/s2-pro"
  if [[ -f "${dir}/codec.pth" ]]; then
    echo "Fish Speech checkpoints already present at ${dir}"
    return 0
  fi

  echo "Downloading Fish Speech s2-pro checkpoints to ${dir}..."
  mkdir -p "$dir"
  ensure_hf_hub
  FISH_CKPT_DIR="$dir" python - <<'PY'
import os

from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="fishaudio/s2-pro",
    local_dir=os.environ["FISH_CKPT_DIR"],
)
PY
}

download_checkpoints() {
  case "$TARGET" in
    server | training | all | dev)
      download_fish_checkpoints
      ;;
    dataset) ;;
  esac
}

if [[ ! -d "$VENV" ]]; then
  echo "Creating venv at $VENV"
  "$PYTHON" -m venv "$VENV"
fi

if [[ ! -f "${VENV}/bin/activate" ]]; then
  echo "error: venv activate script not found: ${VENV}/bin/activate" >&2
  exit 1
fi

# shellcheck disable=SC1091
source "${VENV}/bin/activate"

python -m pip install -U pip wheel setuptools

case "$TARGET" in
  server)
    pip install -e .
    ;;
  dataset)
    pip install -e ".[dataset]"
    ;;
  training)
    pip install -e ".[training,dataset]"
    ;;
  all)
    pip install -e ".[dataset,training]"
    ;;
  dev)
    pip install -e ".[dataset,training,dev]"
    pre-commit install
    ;;
esac

if [[ "${SKIP_DOWNLOAD:-0}" != "1" ]]; then
  download_checkpoints
else
  echo "Skipping checkpoint download (SKIP_DOWNLOAD=1)"
fi

# fish-speech dataset packing needs protobuf 4.x (conflicts with descript-audiotools pin)
case "$TARGET" in
  training | all | dev)
    pip install -q 'protobuf>=4.25.3,<6' || true
    ;;
esac

echo "Done. Activate with: source ${VENV}/bin/activate"
echo "Quick start:  ./run.sh server"
echo "Dataset:      ./run.sh dataset run"
echo "Training:     ./run.sh train all"
