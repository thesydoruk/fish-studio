#!/usr/bin/env bash
# Quick CLI test synthesis via the HTTP TTS server.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# shellcheck disable=SC1091
source "${ROOT}/scripts/common.sh"

usage() {
  cat <<'EOF'
Usage: ./scripts/synthesize.sh [options]

Synthesize speech through the running TTS server (must be started first).

Options:
  -t, --text TEXT         Text to synthesize (required)
  -w, --speaker-wav PATH  Reference WAV (~6 s)
  -s, --speaker-text TXT  Reference transcript (recommended for Fish)
  -o, --out PATH          Output WAV (default: synthesized.wav)
  -u, --url URL           Server base URL (default: http://127.0.0.1:8080)
  -h, --help

Examples:
  ./scripts/synthesize.sh -t "Привіт!" -w data/datasets/combined/reference.wav \
    -s "Привіт, як справи?"
EOF
}

TEXT=""
SPEAKER_WAV=""
SPEAKER_TEXT=""
OUT="synthesized.wav"
URL="http://127.0.0.1:8080"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -t | --text)
      TEXT="$2"
      shift 2
      ;;
    -w | --speaker-wav)
      SPEAKER_WAV="$2"
      shift 2
      ;;
    -s | --speaker-text)
      SPEAKER_TEXT="$2"
      shift 2
      ;;
    -o | --out)
      OUT="$2"
      shift 2
      ;;
    -u | --url)
      URL="$2"
      shift 2
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

if [[ -z "$TEXT" ]]; then
  echo "error: --text is required" >&2
  exit 1
fi

if [[ -z "$SPEAKER_WAV" ]]; then
  DATA="$(common_data_root)"
  for candidate in \
    "${DATA}/datasets/combined/reference.wav" \
    "${DATA}/datasets/farid-govoryt/reference.wav"; do
    if [[ -f "$candidate" ]]; then
      SPEAKER_WAV="$candidate"
      break
    fi
  done
fi

if [[ -z "$SPEAKER_WAV" || ! -f "$SPEAKER_WAV" ]]; then
  echo "error: --speaker-wav not found (pass explicitly or build a dataset with reference.wav)" >&2
  exit 1
fi

ARGS=(-s -w "HTTP %{http_code} size:%{size_download}\n" -o "$OUT")
ARGS+=(-F "text=${TEXT}" -F "speaker_wav=@${SPEAKER_WAV}")
if [[ -n "$SPEAKER_TEXT" ]]; then
  ARGS+=(-F "speaker_text=${SPEAKER_TEXT}")
fi

echo "[synthesize] ${URL}/v1/synthesize → ${OUT}"
curl -X POST "${URL}/v1/synthesize" "${ARGS[@]}"
echo "[synthesize] done: ${OUT}"
