#!/usr/bin/env bash
set -euo pipefail

export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

export PATH="/usr/local/bin:${PATH}"

deploy_config="$(python3 /app/scripts/generate_fish_deploy.py)"

IFS=$'\t' read -r model port gpu_util < <(python3 - <<'PY'
import os
from pathlib import Path
from urllib.parse import urlparse

data_root = Path(os.environ.get("DATA_ROOT", "/data"))
if not data_root.is_absolute():
    data_root = (Path.cwd() / data_root).resolve()

model = os.environ.get("FISH_SPEECH_MODEL", "checkpoints/fish-speech/s2-pro").strip()
local = data_root / model
if local.is_dir():
    model = str(local)

base_url = os.environ.get("FISH_SPEECH_BASE_URL", "http://127.0.0.1:8091")
port = urlparse(base_url).port or int(os.environ.get("VLLM_PORT", "8091"))
gpu_util = os.environ.get("FISH_SPEECH_GPU_MEMORY_UTILIZATION", "0.72")
print(f"{model}\t{port}\t{gpu_util}")
PY
)

cmd=(vllm serve "$model" --omni --port "$port" --gpu-memory-utilization "$gpu_util")
if [[ -f "$deploy_config" ]]; then
  cmd+=(--deploy-config "$deploy_config")
fi

echo "[vllm] starting: ${cmd[*]}"
exec "${cmd[@]}"
