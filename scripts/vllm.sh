#!/usr/bin/env bash
# Start, stop, or check the vLLM-Omni Fish Speech TTS server.
#
# Runs in its own venv (.venv-vllm) because vllm pins torch/transformers
# versions that conflict with the main project venv.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# shellcheck disable=SC1091
source "${ROOT}/scripts/common.sh"

VENV="${VLLM_VENV:-${ROOT}/.venv-vllm}"

project_python() {
  if [[ -x "${ROOT}/.venv/bin/python3" ]]; then
    printf '%s\n' "${ROOT}/.venv/bin/python3"
  else
    printf '%s\n' python3
  fi
}

usage() {
  cat <<'EOF'
Usage: ./scripts/vllm.sh <install|start|stop|restart|status> [extra vllm args...]

Commands:
  install   Create .venv-vllm and install vllm-omni + fish-speech (DAC codec)
  start     Start vLLM-Omni server in background (logs → data/logs/vllm.log)
  stop      Stop running vLLM-Omni server
  restart   stop + start
  status    Show process and /health

Configuration (.env):
  FISH_SPEECH_MODEL          checkpoint dir under DATA_ROOT, or HF repo id
  FISH_SPEECH_BASE_URL       port is taken from here (default 8091)
  FISH_SPEECH_GPU_MEMORY_UTILIZATION   passed to vllm serve
  FISH_SPEECH_MAX_CONCURRENT_REQUESTS  AR batch size + HTTP proxy pool (default 6)
  PYTORCH_CUDA_ALLOC_CONF              default expandable_segments:True (less VRAM fragmentation)

Environment:
  VLLM_VENV   Override venv path (default: ./.venv-vllm)
EOF
}

vllm_config() {
  # Prints: model_path<TAB>port<TAB>gpu_util
  "$(project_python)" - <<'PY' "${ROOT}/$(common_config_path)"
import os
import sys
from pathlib import Path
from urllib.parse import urlparse

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

env_path = Path(sys.argv[1])
if load_dotenv and env_path.is_file():
    load_dotenv(env_path)

data_root = Path(os.environ.get("DATA_ROOT", "./data"))
if not data_root.is_absolute():
    anchor = env_path.resolve().parent if env_path.is_file() else Path.cwd()
    data_root = (anchor / data_root).resolve()

model = os.environ.get("FISH_SPEECH_MODEL", "checkpoints/fish-speech/s2-pro").strip()
local = data_root / model
if local.is_dir():
    model = str(local)

base_url = os.environ.get("FISH_SPEECH_BASE_URL", "http://127.0.0.1:8091")
port = urlparse(base_url).port or 8091
gpu_util = os.environ.get("FISH_SPEECH_GPU_MEMORY_UTILIZATION", "0.72")
print(f"{model}\t{port}\t{gpu_util}")
PY
}

PID_PATTERN='vllm serve .* --omni'
PID_FILE="${ROOT}/data/logs/vllm.pid"

vllm_running() {
  pgrep -f "$PID_PATTERN" >/dev/null 2>&1
}

# Kill the whole process group: vllm-omni spawns stage engine subprocesses
# that survive a plain pkill of the API server and keep holding VRAM.
vllm_kill_group() {
  local sig="$1"
  if [[ -f "$PID_FILE" ]]; then
    local pid
    pid="$(cat "$PID_FILE")"
    if [[ -n "$pid" ]]; then
      kill "-${sig}" -- "-${pid}" 2>/dev/null || true
    fi
  fi
  pkill "-${sig}" -f "$PID_PATTERN" 2>/dev/null || true
}

vllm_install() {
  if [[ ! -f "${VENV}/bin/activate" ]]; then
    echo "[vllm] creating venv: ${VENV}"
    python3 -m venv "$VENV"
  fi
  # shellcheck disable=SC1091
  source "${VENV}/bin/activate"
  pip install --upgrade pip
  pip install vllm-omni
  # fish-speech (PyPI) is required for the DAC codec, per the vllm-omni recipe.
  pip install fish-speech
  echo "[vllm] install done"
}

vllm_stop() {
  if vllm_running || [[ -f "$PID_FILE" ]]; then
    vllm_kill_group TERM
    sleep 3
    if vllm_running; then
      vllm_kill_group KILL
      sleep 1
    fi
    rm -f "$PID_FILE"
    echo "[vllm] stopped"
  else
    echo "[vllm] not running"
  fi
}

vllm_start() {
  if [[ ! -f "${VENV}/bin/activate" ]]; then
    echo "error: ${VENV} not found (run: ./scripts/vllm.sh install)" >&2
    exit 1
  fi
  if vllm_running; then
    echo "[vllm] already running: $(pgrep -af "$PID_PATTERN" | head -1)"
    return 0
  fi

  local model port gpu_util
  IFS=$'\t' read -r model port gpu_util < <(vllm_config)

  local deploy_config
  deploy_config="$("$(project_python)" "${ROOT}/scripts/generate_fish_deploy.py" "$(common_config_path)")"

  # shellcheck disable=SC1091
  source "${VENV}/bin/activate"
  common_ensure_logs
  local log_file
  log_file="$(common_log_file vllm)"

  # FlashInfer JIT needs nvcc; without a CUDA toolkit (e.g. WSL with only the
  # Windows driver) fall back to the torch sampler.
  if ! command -v nvcc >/dev/null 2>&1 && [[ ! -d "${CUDA_HOME:-/usr/local/cuda}" ]]; then
    export VLLM_USE_FLASHINFER_SAMPLER=0
    echo "[vllm] no CUDA toolkit found, setting VLLM_USE_FLASHINFER_SAMPLER=0"
  fi
  # Let the CUDA allocator grow/reuse segments so DAC can take leftover VRAM
  # after the AR stage has already reserved most of the GPU.
  if [[ -z "${PYTORCH_CUDA_ALLOC_CONF:-}" ]]; then
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  fi
  echo "[vllm] PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"

  local cmd=(vllm serve "$model" --omni --port "$port" --gpu-memory-utilization "$gpu_util")
  if [[ -f "$deploy_config" ]]; then
    cmd+=(--deploy-config "$deploy_config")
  elif [[ -f "${ROOT}/configs/fish_speech_deploy.yaml" ]]; then
    cmd+=(--deploy-config "${ROOT}/configs/fish_speech_deploy.yaml")
  fi
  cmd+=("$@")
  echo "[vllm] starting: ${cmd[*]}"
  nohup setsid "${cmd[@]}" >>"$log_file" 2>&1 &
  local pid=$!
  echo "$pid" > "$PID_FILE"
  echo "[vllm] started pid=${pid} log=${log_file}"

  echo "[vllm] waiting for /health on port ${port} (model load takes a few minutes)..."
  for _ in $(seq 1 120); do
    sleep 5
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "[vllm] error: process exited during startup" >&2
      tail -30 "$log_file" >&2 || true
      exit 1
    fi
    if curl -sf "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
      echo "[vllm] ready"
      return 0
    fi
  done
  echo "[vllm] warning: /health not ready after 10 min, check ${log_file}" >&2
}

vllm_status() {
  local model port gpu_util
  IFS=$'\t' read -r model port gpu_util < <(vllm_config)
  if vllm_running; then
    pgrep -af "$PID_PATTERN" | sed 's/^/[vllm] /'
    curl -sf "http://127.0.0.1:${port}/health" >/dev/null 2>&1 \
      && echo "[vllm] health: ok (port ${port})" \
      || echo "[vllm] health: unreachable (port ${port})"
  else
    echo "[vllm] not running"
  fi
}

ACTION="${1:-status}"
shift || true

case "$ACTION" in
  install)
    vllm_install
    ;;
  start)
    vllm_start "$@"
    ;;
  stop)
    vllm_stop
    ;;
  restart)
    vllm_stop
    vllm_start "$@"
    ;;
  status)
    vllm_status
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
