#!/usr/bin/env bash
# Tail project log files.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/scripts/common.sh"

usage() {
  cat <<'EOF'
Usage: ./scripts/logs.sh <name> [tail options...]

Names:
  server          TTS HTTP server
  training        LoRA training pipeline
  pipeline        Dataset builder (fish-dataset run)
  hf-import       Hugging Face dataset import
  tensorboard     TensorBoard server

Examples:
  ./scripts/logs.sh training
  ./scripts/logs.sh server -f
EOF
}

NAME="${1:-}"
shift || true

if [[ -z "$NAME" || "$NAME" == "-h" || "$NAME" == "--help" ]]; then
  usage
  exit 0
fi

case "$NAME" in
  server) FILE="$(common_log_file server)" ;;
  training | train) FILE="$(common_log_file training)" ;;
  pipeline | dataset) FILE="$(common_log_file pipeline)" ;;
  hf-import | import) FILE="$(common_log_file hf-import)" ;;
  tensorboard | tb) FILE="$(common_log_file tensorboard)" ;;
  *)
    echo "error: unknown log name: $NAME" >&2
    usage >&2
    exit 1
    ;;
esac

if [[ ! -f "$FILE" ]]; then
  echo "log not found: $FILE" >&2
  exit 1
fi

if [[ $# -eq 0 ]]; then
  set -- -f
fi
exec tail "$@" "$FILE"
