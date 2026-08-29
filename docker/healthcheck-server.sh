#!/usr/bin/env sh
set -eu
port="${INFERENCE_PORT:-8080}"
curl -sf "http://127.0.0.1:${port}/health"
