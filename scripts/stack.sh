#!/usr/bin/env bash
# Start or stop the whole TTS stack (vLLM-Omni + HTTP proxy) in the background.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

usage() {
  cat <<'EOF'
Usage: ./scripts/stack.sh [start|stop|restart|status] [extra vllm args...]

Both processes run in the background (nohup). start waits for vLLM /health
before bringing the HTTP proxy up.

Commands:
  start     Start vLLM, then the HTTP proxy (default)
  stop      Stop the HTTP proxy, then vLLM
  restart   stop + start
  status    Show both processes and /health

Examples:
  ./run.sh stack
  ./run.sh stack start
  ./run.sh stack stop
EOF
}

ACTION="${1:-start}"
if [[ $# -gt 0 ]]; then
  shift
fi

case "$ACTION" in
  start)
    "${ROOT}/scripts/vllm.sh" start "$@"
    "${ROOT}/scripts/server.sh" start
    ;;
  stop)
    "${ROOT}/scripts/server.sh" stop
    "${ROOT}/scripts/vllm.sh" stop
    ;;
  restart)
    "${ROOT}/scripts/server.sh" stop
    "${ROOT}/scripts/vllm.sh" stop
    "${ROOT}/scripts/vllm.sh" start "$@"
    "${ROOT}/scripts/server.sh" start
    ;;
  status)
    "${ROOT}/scripts/vllm.sh" status
    "${ROOT}/scripts/server.sh" status
    ;;
  -h | --help | help)
    usage
    ;;
  *)
    echo "error: unknown action: $ACTION" >&2
    usage >&2
    exit 1
    ;;
esac
