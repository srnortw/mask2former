#!/usr/bin/env bash
# POST validation images to local /predict (logs to MongoDB if MONGO_URI is set).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
URL="${PREDICT_URL:-http://localhost:8000/predict}"
DIR="${1:-$ROOT/data/raw/valid}"
count=0
for img in "$DIR"/*.jpg; do
  [[ -f "$img" ]] || continue
  curl -s -X POST "$URL" -F "file=@$img" > /dev/null
  echo "logged: $(basename "$img")"
  count=$((count + 1))
done
echo "Done. $count images sent to $URL"
