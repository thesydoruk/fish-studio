#!/usr/bin/env bash
# Build dataset from all enabled sources and merge for training.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# shellcheck disable=SC1091
source "${ROOT}/scripts/env.sh"
# shellcheck disable=SC1091
source "${ROOT}/scripts/common.sh"

usage() {
  cat <<'EOF'
Usage: ./scripts/dataset-build.sh [options] [-- fish-dataset run args...]

Run the full dataset pipeline for enabled sources, then merge into one training set.

Options:
  -c, --config PATH     Config file (default: .env)
  -o, --output ID       Merge output dataset id (default: combined)
  --source ID           Run only this source (repeatable). Merge still
                        includes every export-ready dataset on disk.
  --force               Re-process completed pipeline steps
  --skip-merge          Only run per-source pipeline, do not merge
  --merge-from-sources  Merge only enabled SOURCES (skip HF / other folders)
  -h, --help            Show this help

Examples:
  ./scripts/dataset-build.sh
  ./scripts/dataset-build.sh --source game-vo-1 --force
  ./scripts/dataset-build.sh -o farid-govoryt --source farid-govoryt --skip-merge
EOF
}

CONFIG=".env"
OUTPUT_ID="combined"
SKIP_MERGE=0
FORCE=0
MERGE_FROM_SOURCES=0
SOURCES=()
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -c | --config)
      CONFIG="$2"
      shift 2
      ;;
    -o | --output)
      OUTPUT_ID="$2"
      shift 2
      ;;
    --source)
      SOURCES+=("$2")
      shift 2
      ;;
    --force)
      FORCE=1
      shift
      ;;
    --skip-merge)
      SKIP_MERGE=1
      shift
      ;;
    --merge-from-sources)
      MERGE_FROM_SOURCES=1
      shift
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

common_require_config
activate_venv "${ROOT}"

RUN_ARGS=(-c "$CONFIG")
if ((${#SOURCES[@]})); then
  for id in "${SOURCES[@]}"; do
    RUN_ARGS+=(--source "$id")
  done
fi
if ((FORCE)); then
  RUN_ARGS+=(--force)
fi
if ((${#EXTRA_ARGS[@]})); then
  RUN_ARGS+=("${EXTRA_ARGS[@]}")
fi

echo "[dataset-build] running pipeline: fish-dataset run ${RUN_ARGS[*]}"
fish-dataset run "${RUN_ARGS[@]}"

if ((SKIP_MERGE)); then
  echo "[dataset-build] skip-merge set; done"
  exit 0
fi

MERGE_ARGS=(-c "$CONFIG" -o "$OUTPUT_ID")
if ((MERGE_FROM_SOURCES)); then
  MERGE_ARGS+=(--from-sources)
fi

echo "[dataset-build] merging → datasets/${OUTPUT_ID}"
fish-dataset merge "${MERGE_ARGS[@]}"
echo "[dataset-build] done: $(common_data_root)/datasets/${OUTPUT_ID}"
