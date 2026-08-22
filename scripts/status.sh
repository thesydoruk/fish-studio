#!/usr/bin/env bash
# Show dataset, training, server, and GPU status.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# shellcheck disable=SC1091
source "${ROOT}/scripts/env.sh"
# shellcheck disable=SC1091
source "${ROOT}/scripts/common.sh"

CONFIG="$(common_config_path)"
DATA_ROOT="$(common_data_root)"

echo "=== Project status ==="
echo "config:    ${ROOT}/${CONFIG}"
echo "data_root: ${DATA_ROOT}"
if [[ -f "${ROOT}/${CONFIG}" ]]; then
  # shellcheck disable=SC1090
  set -a
  # Prefer grep over sourcing whole .env (may contain spaces/JSON).
  training_ds="$(grep -E '^TRAINING_DATASET_ID=' "${ROOT}/${CONFIG}" | tail -1 | cut -d= -f2- || true)"
  training_proj="$(grep -E '^TRAINING_PROJECT_NAME=' "${ROOT}/${CONFIG}" | tail -1 | cut -d= -f2- || true)"
  set +a
  [[ -n "${training_ds}" ]] && echo "train ds:  ${training_ds}"
  [[ -n "${training_proj}" ]] && echo "train run: ${training_proj}"
fi
echo

if [[ -f "${ROOT}/${CONFIG}" ]]; then
  activate_venv "${ROOT}" 2>/dev/null || true
  if command -v fish-dataset >/dev/null 2>&1; then
    echo "=== Dataset sources + on-disk exports ==="
    fish-dataset -c "$CONFIG" sources 2>/dev/null || true
    echo
  fi
fi

echo "=== Exported / merged datasets ==="
if [[ -d "${DATA_ROOT}/datasets" ]]; then
  for dir in "${DATA_ROOT}/datasets"/*; do
    [[ -d "$dir" ]] || continue
    name="$(basename "$dir")"
    train="${dir}/metadata_train.csv"
    wavs="${dir}/wavs"
    clips="—"
    if [[ -d "$wavs" ]]; then
      clips="$(find "$wavs" -maxdepth 1 -name '*.wav' 2>/dev/null | wc -l | tr -d ' ')"
    fi
    ref="no"
    [[ -f "${dir}/reference.wav" ]] && ref="yes"
    hours="—"
    if [[ -f "${dir}/stats.json" ]] && command -v python3 >/dev/null 2>&1; then
      hours="$(
        python3 -c "import json,sys; d=json.load(open(sys.argv[1],encoding='utf-8')); print(f\"{d.get('total_duration_sec',0)/3600:.2f}h\")" \
          "${dir}/stats.json" 2>/dev/null || echo "—"
      )"
    fi
    printf '  %-24s clips=%-8s hours=%-8s reference=%s\n' "$name" "$clips" "$hours" "$ref"
  done
else
  echo "  (no datasets yet)"
fi
echo

echo "=== Fish Speech training ==="
raw_wavs=0
if [[ -d "${DATA_ROOT}/training/raw" ]]; then
  raw_wavs="$(find "${DATA_ROOT}/training/raw" -name '*.wav' 2>/dev/null | wc -l | tr -d ' ')"
fi
echo "  training/raw wavs: ${raw_wavs}"
fish_runs="${DATA_ROOT}/training/runs"
fish_merged="${DATA_ROOT}/training/merged"
if compgen -G "${fish_runs}/*/checkpoints/step_*.ckpt" >/dev/null 2>&1; then
  ls -1t "${fish_runs}"/*/checkpoints/step_*.ckpt 2>/dev/null | head -3 | sed 's/^/  /'
else
  echo "  (no LoRA checkpoints)"
fi
if [[ -f "${fish_merged}/model.pth" ]]; then
  echo "  merged: ${fish_merged}/model.pth"
else
  echo "  merged: (not built yet)"
fi
if pgrep -f 'python -m fish_studio.training' >/dev/null 2>&1 \
  || pgrep -f 'run.sh train' >/dev/null 2>&1; then
  echo "  active:"
  pgrep -af 'fish_studio.training|run.sh train' 2>/dev/null | grep -v 'pgrep' | head -5 | sed 's/^/    /' || true
fi
echo

echo "=== TTS server ==="
if pgrep -f 'python -m fish_studio.server.serve' >/dev/null 2>&1; then
  echo "  running: $(pgrep -af 'python -m fish_studio.server.serve' | head -1)"
  if command -v curl >/dev/null 2>&1; then
    curl -sf http://127.0.0.1:8080/health 2>/dev/null | sed 's/^/  health: /' || echo "  health: unreachable"
  fi
else
  echo "  not running"
fi
echo

echo "=== audio-intel (dataset ASR) ==="
if pgrep -f 'python -m audio_intel.server' >/dev/null 2>&1; then
  echo "  running (holds GPU VRAM — stop it before ./run.sh bg-train / vllm)"
  pgrep -af 'python -m audio_intel.server' | head -1 | sed 's/^/  /'
  if command -v curl >/dev/null 2>&1; then
    curl -sf -m 3 http://127.0.0.1:8081/health 2>/dev/null | sed 's/^/  health: /' || echo "  health: unreachable"
  fi
else
  echo "  not running"
fi
echo

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "=== GPU ==="
  nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader | sed 's/^/  /'
fi
