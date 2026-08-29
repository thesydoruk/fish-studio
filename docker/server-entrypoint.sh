#!/usr/bin/env bash
set -euo pipefail

exec python -m fish_studio.server.serve "$@"
