#!/usr/bin/env bash
# Start, stop, or check the unified TTS HTTP server.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# shellcheck disable=SC1091
source "${ROOT}/scripts/env.sh"
# shellcheck disable=SC1091
source "${ROOT}/scripts/common.sh"

usage() {
  cat <<'EOF'
Usage: ./scripts/server.sh <start|stop|restart|status> [options]

Commands:
  start     Start server in background (logs → data/logs/server.log)
  stop      Stop running server
  restart   stop + start
  status    Show process and /health

Options (start/restart):
  -c, --config PATH   Config file (default: .env)
  -f, --foreground    Run in foreground (no nohup)

Environment:
  SERVER_HOST / SERVER_PORT — override inference bind (optional)
EOF
}

ACTION="${1:-status}"
shift || true

CONFIG=".env"
FOREGROUND=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    -c | --config)
      CONFIG="$2"
      shift 2
      ;;
    -f | --foreground)
      FOREGROUND=1
      shift
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

LOG_FILE="$(common_log_file server)"
PID_PATTERN='python -m fish_studio.server.serve'

server_running() {
  pgrep -f "$PID_PATTERN" >/dev/null 2>&1
}

server_stop() {
  if server_running; then
    pkill -f "$PID_PATTERN" || true
    sleep 2
    if server_running; then
      pkill -9 -f "$PID_PATTERN" || true
      sleep 1
    fi
    echo "[server] stopped"
  else
    echo "[server] not running"
  fi
}

server_start() {
  common_require_config
  activate_venv "${ROOT}"

  if server_running; then
    echo "[server] already running: $(pgrep -af "$PID_PATTERN" | head -1)"
    return 0
  fi

  common_ensure_logs
  local cmd=(python -m fish_studio.server.serve -c "$CONFIG")
  if ((FOREGROUND)); then
    echo "[server] foreground: ${cmd[*]}"
    exec "${cmd[@]}"
  fi

  nohup "${cmd[@]}" >>"$LOG_FILE" 2>&1 &
  local pid=$!
  echo "[server] started pid=${pid} log=${LOG_FILE}"

  for _ in $(seq 1 30); do
  sleep 1
    if curl -sf http://127.0.0.1:8080/health >/dev/null 2>&1; then
      curl -s http://127.0.0.1:8080/health | sed 's/^/[server] health: /'
      return 0
    fi
  done
  echo "[server] warning: /health not ready yet (model loads on first request)" >&2
  tail -5 "$LOG_FILE" 2>/dev/null || true
}

server_status() {
  if server_running; then
    pgrep -af "$PID_PATTERN" | sed 's/^/[server] /'
    curl -sf http://127.0.0.1:8080/health 2>/dev/null | sed 's/^/[server] health: /' || echo "[server] health: unreachable"
  else
    echo "[server] not running"
  fi
}

case "$ACTION" in
  start)
    server_start
    ;;
  stop)
    server_stop
    ;;
  restart)
    server_stop
    server_start
    ;;
  status)
    server_status
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
