#!/usr/bin/env bash
# Provision MongoDB Atlas for Mask2Former monitoring (Phase 07).
# Prerequisites: atlas CLI logged in (atlas auth login)
#
# Usage:
#   atlas auth login
#   ./scripts/setup_atlas_mongo.sh
#
# Atlas project: mask2former-mlops (6a21b995cbf3f23e5981be8f)
# Atlas cluster: mask2former-cluster (M0, EU_CENTRAL_1)
# Writes MONGO_URI + ATLAS_PROJECT_ID + ATLAS_CLUSTER to .env
#
# Optional env:
#   ATLAS_PROJECT_ID   — override project
#   ATLAS_CLUSTER      — override cluster name
#   ATLAS_DB_USER      — default: mask2former_api
#   ATLAS_REGION       — default: EU_CENTRAL_1
#   ALLOW_ANY_IP=1     — also allow 0.0.0.0/0 (dev only)

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_FILE="${ROOT}/.env"
DEFAULT_PROJECT_ID="6a21b995cbf3f23e5981be8f"
CLUSTER="${ATLAS_CLUSTER:-mask2former-cluster}"
DB_USER="${ATLAS_DB_USER:-mask2former_api}"
DB_NAME="${MONGO_DB_NAME:-mask2former}"
REGION="${ATLAS_REGION:-EU_CENTRAL_1}"
PROVIDER="${ATLAS_PROVIDER:-AWS}"

log() { echo "[setup-atlas] $*"; }
die() { echo "[setup-atlas] ERROR: $*" >&2; exit 1; }

command -v atlas >/dev/null || die "Install Atlas CLI: https://www.mongodb.com/docs/atlas/cli/"
command -v jq >/dev/null || die "Install jq for JSON parsing"

if ! atlas projects list -o json >/dev/null 2>&1; then
  die "Atlas session expired. Run: atlas auth login"
fi

PROJECT_ID="${ATLAS_PROJECT_ID:-}"
if [[ -z "${PROJECT_ID}" ]] && [[ -f "${ENV_FILE}" ]]; then
  PROJECT_ID="$(grep -E '^ATLAS_PROJECT_ID=' "${ENV_FILE}" 2>/dev/null | cut -d= -f2- || true)"
fi
if [[ -z "${PROJECT_ID}" ]]; then
  PROJECT_ID="$(atlas config get project_id 2>/dev/null || true)"
fi
PROJECT_ID="${PROJECT_ID:-${DEFAULT_PROJECT_ID}}"

log "Project: ${PROJECT_ID}"
log "Cluster: ${CLUSTER}"

cluster_exists() {
  atlas clusters list --projectId "${PROJECT_ID}" -o json 2>/dev/null \
    | jq -e --arg n "${CLUSTER}" '.results[]? | select(.name == $n)' >/dev/null
}

if cluster_exists; then
  log "Cluster '${CLUSTER}' already exists — skipping create"
else
  log "Creating M0 cluster '${CLUSTER}' (${PROVIDER} / ${REGION})..."
  atlas clusters create "${CLUSTER}" \
    --projectId "${PROJECT_ID}" \
    --provider "${PROVIDER}" \
    --region "${REGION}" \
    --tier M0 \
    --mdbVersion 7.0
fi

log "Waiting for cluster to become available (this can take 3–8 min)..."
for _ in $(seq 1 60); do
  state="$(atlas clusters describe "${CLUSTER}" --projectId "${PROJECT_ID}" -o json 2>/dev/null \
    | jq -r '.stateName // .state // empty' || true)"
  if [[ "${state}" == "IDLE" ]]; then
    log "Cluster is IDLE"
    break
  fi
  printf "  state=%s — waiting 15s...\n" "${state:-UNKNOWN}"
  sleep 15
done

state="$(atlas clusters describe "${CLUSTER}" --projectId "${PROJECT_ID}" -o json | jq -r '.stateName // .state')"
[[ "${state}" == "IDLE" ]] || die "Cluster not ready (state=${state}). Check Atlas UI."

log "Allowing your current IP..."
atlas accessLists create --currentIp --projectId "${PROJECT_ID}" 2>/dev/null || true

if [[ "${ALLOW_ANY_IP:-0}" == "1" ]]; then
  log "ALLOW_ANY_IP=1 — adding 0.0.0.0/0 (dev only)"
  atlas accessLists create 0.0.0.0/0 --type cidrBlock --projectId "${PROJECT_ID}" \
    --comment "mask2former dev" 2>/dev/null || true
fi

if [[ -n "${ATLAS_DB_PASSWORD:-}" ]]; then
  DB_PASS="${ATLAS_DB_PASSWORD}"
else
  DB_PASS="$(openssl rand -base64 24 | tr -d '/+=' | head -c 24)"
  log "Generated DB password (save it): ${DB_PASS}"
fi

if atlas dbusers list --projectId "${PROJECT_ID}" -o json 2>/dev/null \
  | jq -e --arg u "${DB_USER}" '.results[]? | select(.username == $u)' >/dev/null; then
  log "DB user '${DB_USER}' already exists — skipping create"
  if [[ -z "${ATLAS_DB_PASSWORD:-}" ]]; then
    die "Set ATLAS_DB_PASSWORD to the existing user's password, or delete the user in Atlas UI and re-run"
  fi
  DB_PASS="${ATLAS_DB_PASSWORD}"
else
  log "Creating DB user '${DB_USER}' with readWrite on ${DB_NAME}..."
  atlas dbusers create \
    --username "${DB_USER}" \
    --password "${DB_PASS}" \
    --role "readWrite@${DB_NAME}" \
    --projectId "${PROJECT_ID}"
fi

log "Fetching connection string..."
cs_json="$(atlas clusters connectionStrings describe "${CLUSTER}" --projectId "${PROJECT_ID}" -o json)"
SRV="$(echo "${cs_json}" | jq -r '.standardSrv // .connectionStrings.standardSrv // empty' | head -1)"
[[ -n "${SRV}" && "${SRV}" != "null" ]] || die "Could not parse SRV connection string"

HOST_PATH="${SRV#mongodb+srv://}"
ENC_USER="$(python3 -c "import urllib.parse; print(urllib.parse.quote('${DB_USER}', safe=''))")"
ENC_PASS="$(python3 -c "import urllib.parse; print(urllib.parse.quote('${DB_PASS}', safe=''))")"
MONGO_URI="mongodb+srv://${ENC_USER}:${ENC_PASS}@${HOST_PATH%/}/${DB_NAME}?retryWrites=true&w=majority"

log "Testing connection..."
python3 - <<PY
import os
os.environ["MONGO_URI"] = """${MONGO_URI}"""
from pymongo import MongoClient
c = MongoClient(os.environ["MONGO_URI"], serverSelectionTimeoutMS=10000)
c.admin.command("ping")
db = c["${DB_NAME}"]
db["predictions"].create_index("timestamp")
db["predictions"].create_index("image_id")
db["predictions"].create_index("instances.category_id")
db["drift_reports"].create_index("timestamp")
print("ping OK — indexes ensured on predictions + drift_reports")
PY

if [[ -f "${ENV_FILE}" ]]; then
  for kv in "MONGO_URI=${MONGO_URI}" \
            "MONGO_DB_NAME=${DB_NAME}" \
            "MONGO_COLLECTION_PREDICTIONS=predictions" \
            "MONGO_COLLECTION_DRIFT=drift_reports" \
            "ATLAS_PROJECT_ID=${PROJECT_ID}" \
            "ATLAS_CLUSTER=${CLUSTER}"; do
    key="${kv%%=*}"
    if grep -q "^${key}=" "${ENV_FILE}"; then
      sed -i "s|^${key}=.*|${kv}|" "${ENV_FILE}"
    else
      echo "${kv}" >> "${ENV_FILE}"
    fi
  done
  log "Updated ${ENV_FILE}"
else
  log "No .env found — add manually:"
  echo "MONGO_URI=${MONGO_URI}"
fi

log "Done."
log "  Project: mask2former-mlops (${PROJECT_ID})"
log "  Cluster: ${CLUSTER}"
log "  Next: source .env && docker compose up --build -d && curl http://localhost:8000/health"
