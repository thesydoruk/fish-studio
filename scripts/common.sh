#!/usr/bin/env bash
# Shared helpers for project scripts (sourced, not executed directly).

common_root() {
  if [[ -n "${PROJECT_ROOT:-}" ]]; then
    printf '%s\n' "$PROJECT_ROOT"
    return
  fi
  local src
  src="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  PROJECT_ROOT="$src"
  printf '%s\n' "$PROJECT_ROOT"
}

common_config_path() {
  printf '%s\n' "${CONFIG_PATH:-.env}"
}

common_data_root() {
  local root
  root="$(common_root)"
  if command -v python3 >/dev/null 2>&1; then
    python3 - <<'PY' "${root}/$(common_config_path)"
import os
import sys
from pathlib import Path

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
print(data_root)
PY
    return
  fi
  printf '%s\n' "${root}/data"
}

common_logs_dir() {
  printf '%s\n' "$(common_data_root)/logs"
}

common_ensure_logs() {
  mkdir -p "$(common_logs_dir)"
}

common_log_file() {
  local name="$1"
  common_ensure_logs
  printf '%s\n' "$(common_logs_dir)/${name}.log"
}

common_require_config() {
  local root config
  root="$(common_root)"
  config="$(common_config_path)"
  if [[ -f "${root}/${config}" ]]; then
    return 0
  fi
  if [[ -n "${DATA_ROOT:-}" || -n "${FISH_SPEECH_BASE_URL:-}" ]]; then
    return 0
  fi
  echo "error: ${config} not found (run: ./run.sh init)" >&2
  exit 1
}
