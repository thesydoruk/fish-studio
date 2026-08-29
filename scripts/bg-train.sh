#!/usr/bin/env bash
# Run Fish Speech fine-tuning in the background with logging.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# shellcheck disable=SC1091
source "${ROOT}/scripts/env.sh"
# shellcheck disable=SC1091
source "${ROOT}/scripts/common.sh"

usage() {
  cat <<'EOF'
Usage: ./scripts/bg-train.sh [step] [train args...]

Run Fish LoRA training in background. Logs are appended to data/logs/.

Sub-steps: export | vq | protos | train | merge | all (default: all)

Examples:
  ./scripts/bg-train.sh --batch-size 1 --grad-accum 2
  ./scripts/bg-train.sh train
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

common_require_config
activate_venv "${ROOT}"
common_ensure_logs

export PYTHONUNBUFFERED=1

LOG="$(common_log_file training)"
STEP="${1:-all}"
if [[ $# -gt 0 ]]; then shift; fi
echo "[bg-train] fish ${STEP} → ${LOG}"
nohup bash "${ROOT}/run.sh" train "$STEP" "$@" >>"$LOG" 2>&1 &

PID=$!
echo "[bg-train] pid=${PID}"
echo "[bg-train] tail -f ${LOG}"
