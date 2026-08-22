#!/usr/bin/env bash
# Shared venv activation for project scripts.

activate_venv() {
  local root="${1:-.}"
  local venv="${PROJECT_VENV:-${TTS_SERVER_VENV:-}}"

  if [[ -n "${venv}" && -f "${venv}/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${venv}/bin/activate"
  elif [[ -f "${root}/.venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${root}/.venv/bin/activate"
  else
    echo "error: no venv found (run ./run.sh install or set PROJECT_VENV)" >&2
    exit 1
  fi
}
