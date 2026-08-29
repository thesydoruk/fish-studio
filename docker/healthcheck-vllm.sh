#!/usr/bin/env sh
set -eu
if [ -f /app/src/fish_studio/runtime/vllm_health.py ]; then
  exec python3 /app/src/fish_studio/runtime/vllm_health.py --container
fi
exec python3 -m fish_studio.runtime.vllm_health --container
