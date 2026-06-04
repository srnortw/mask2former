#!/usr/bin/env bash
# Send one image to the API and save an overlay.
# Usage: ./scripts/predict_image.sh INPUT.jpg OUTPUT.jpg [conf_threshold]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

INPUT="${1:-}"
OUTPUT="${2:-}"
CONF="${3:-}"

if [[ -z "${INPUT}" || -z "${OUTPUT}" ]]; then
  echo "Usage: $0 INPUT.jpg OUTPUT.jpg [conf_threshold]"
  echo ""
  echo "Examples:"
  echo "  $0 data/raw/valid/frame.jpg reports/my_overlay.jpg"
  echo "  $0 frame.jpg reports/strict.jpg 0.65"
  echo ""
  echo "API must be running: docker compose up -d"
  exit 1
fi

[[ -d .venv ]] || { echo "Missing .venv — run: python3 -m venv .venv"; exit 1; }

if ! curl -sf http://localhost:8000/health >/dev/null 2>&1; then
  echo "API not reachable on :8000. Start it:"
  echo "  docker compose up -d"
  exit 1
fi

if [[ -n "${CONF}" ]]; then
  exec .venv/bin/python scripts/visualize_predict.py "${INPUT}" "${OUTPUT}" \
    --conf-threshold "${CONF}"
else
  exec .venv/bin/python scripts/visualize_predict.py "${INPUT}" "${OUTPUT}"
fi
