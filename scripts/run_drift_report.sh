#!/usr/bin/env bash
# Run Evidently drift report using project .venv (not system Python).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
[[ -d .venv ]] || { echo "Create venv first: python3 -m venv .venv"; exit 1; }
if [[ -f .env ]]; then set -a; source .env; set +a; fi
exec .venv/bin/python -m src.monitoring.drift_report "$@"
