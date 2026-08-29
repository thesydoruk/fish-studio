#!/usr/bin/env bash
# Serve TensorBoard for Fish Speech LoRA training runs.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# shellcheck disable=SC1091
source "${ROOT}/scripts/env.sh"
# shellcheck disable=SC1091
source "${ROOT}/scripts/common.sh"

usage() {
  cat <<'EOF'
Usage: ./scripts/tensorboard.sh [start|stop|status] [--port N] [--host H]

Serves {data_root}/training/runs so every project run shows up as a series.

Examples:
  ./scripts/tensorboard.sh start
  ./scripts/tensorboard.sh start --port 6007
  ./scripts/tensorboard.sh stop
EOF
}

CMD="${1:-start}"
if [[ $# -gt 0 ]]; then shift; fi

PORT="${TENSORBOARD_PORT:-6006}"
HOST="${TENSORBOARD_HOST:-0.0.0.0}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --port) PORT="$2"; shift 2 ;;
    --host) HOST="$2"; shift 2 ;;
    -h | --help) usage; exit 0 ;;
    *) echo "error: unknown option: $1" >&2; usage >&2; exit 1 ;;
  esac
done

DATA_ROOT="$(common_data_root)"
LOGDIR="${DATA_ROOT}/training/runs"
PATTERN="tensorboard.main --logdir ${LOGDIR}"

case "$CMD" in
  start)
    common_require_config
    activate_venv "${ROOT}"
    if pgrep -f "$PATTERN" >/dev/null 2>&1; then
      echo "[tensorboard] already running on port ${PORT}"
      exit 0
    fi
    mkdir -p "$LOGDIR"
    LOG="$(common_log_file tensorboard)"
    nohup python -m tensorboard.main --logdir "$LOGDIR" --host "$HOST" --port "$PORT" \
      >>"$LOG" 2>&1 &
    echo "[tensorboard] pid=$! logdir=${LOGDIR}"
    echo "[tensorboard] http://${HOST}:${PORT}  (log: ${LOG})"
    ;;
  stop)
    if pkill -f "$PATTERN"; then
      echo "[tensorboard] stopped"
    else
      echo "[tensorboard] not running"
    fi
    ;;
  status)
    if pgrep -af "$PATTERN"; then
      exit 0
    fi
    echo "[tensorboard] not running"
    ;;
  -h | --help | help)
    usage
    ;;
  *)
    echo "error: unknown command: $CMD" >&2
    usage >&2
    exit 1
    ;;
esac
