# 07 — Monitoring

## Overview

```
FastAPI /predict endpoint (Phase 06)
  ↓
MongoDB Atlas (project mask2former-mlops)  ← log prediction metadata
  ↓
Evidently AI 0.7  ← drift report: val set vs live predictions
  ↓
HTML in reports/  ← open in browser; summary also stored in drift_reports
  ↓
If drift detected → review and consider retraining
```

---

## What We Actually Built

**Commits:** `ff85c3c` — `feat(monitoring): Phase 07 MongoDB Atlas + Evidently drift reports` · `a221e3c` — package init + dotenv in requirements-monitoring

| File | Purpose |
|------|---------|
| `src/mongo_logger.py` | `PredictionLogger` + `try_create_logger()` — optional if no `MONGO_URI` |
| `src/monitoring/drift_report.py` | Evidently **0.7** drift report (reference COCO val vs MongoDB) |
| `src/monitoring/__init__.py` | Package marker |
| `api/main.py` | Calls `mongo_logger.log_prediction()` after each `/predict` |
| `api/Dockerfile` | Copies `src/mongo_logger.py` into API image |
| `api/requirements-api.txt` | Added `pymongo` |
| `docker-compose.yml` | Passes `MONGO_*` from `.env` |
| `config.yaml` | `services.mongodb.project_id`, `cluster`, collections |
| `requirements-monitoring.txt` | evidently 0.7, pandas, pycocotools, pymongo, python-dotenv |
| `.env.example` | Template for Atlas + Mongo vars |
| `scripts/setup_atlas_mongo.sh` | Atlas CLI: cluster, user, indexes, update `.env` |
| `scripts/seed_predictions.sh` | Batch `POST /predict` on val images |
| `scripts/run_drift_report.sh` | Drift report via **`.venv/bin/python`** (not base Python) |
| `tests/test_mongo_logger.py` | Unit tests (no live MongoDB) |

### Design choices

- **MongoDB is optional** — API starts and serves without `MONGO_URI`; `/health` reports `"mongodb": false`.
- **Metadata only** — documents store bbox, score, category, latency; **no mask pixels** (keeps Atlas free tier small).
- **Dedicated Atlas project** — `mask2former-mlops`, not the old `Computer_vision` / **Cluster0** project (that cluster was stuck `UPDATING`).
- **Same name as Drive** — `mask2former-mlops` on Google Drive and Atlas for one mental model.
- **Drift in project `.venv`** — Evidently 0.7 API (`Report` → `Snapshot.save_html`); install with `.venv/bin/pip`, never system `pip`.
- **Reference distribution** — COCO val annotations (`num_instances`, synthetic `mean_score=1.0`, `inference_ms=90`); **current** = recent MongoDB prediction stats.

### MongoDB document shape (`predictions`)

| Field | Description |
|-------|-------------|
| `image_id` | Upload filename |
| `timestamp` | UTC |
| `model_version` | e.g. `mask2former_int8.onnx` |
| `source` | `"api"` (or `"ros2"` later) |
| `inference_ms` | ONNX runtime only |
| `num_instances` | Count after threshold |
| `mean_score` | Average instance confidence |
| `instances[]` | `category_id`, `category_name`, `score`, `bbox` (no `mask_b64`) |

Indexes: `timestamp`, `image_id`, `instances.category_id`.

### Atlas setup (mask2former-mlops)

| Item | Value |
|------|--------|
| Atlas project | **`mask2former-mlops`** (`6a21b995cbf3f23e5981be8f`) |
| Cluster | **`mask2former-cluster`** (M0, **EU_CENTRAL_1**) |
| SRV host | `mask2former-cluster.ceg04zs.mongodb.net` |
| DB user | `mask2former_api` → `readWrite@mask2former` |
| Database | `mask2former` |
| Collections | `predictions`, `drift_reports` |
| Network | Current IP + `0.0.0.0/0` (dev) |

**Not used:** `Computer_vision` project / **Cluster0** (`cluster0.cuzjsc9.mongodb.net`) — left as-is or delete in Atlas UI later.

### Build log — issues and fixes

| Issue | Fix |
|-------|-----|
| Placeholder `MONGO_URI` in `.env` | Created real project via Atlas CLI; updated `.env` |
| **Cluster0** stuck `UPDATING` / no primary | New project **mask2former-mlops** + new cluster (reached **IDLE** in ~30s) |
| Renamed CV project to Robotic_Perception by mistake | Reverted to **Computer_vision**; MLOps uses separate **mask2former-mlops** project |
| `ImportError: ColumnDriftMetric` (Evidently 0.4 API) | Migrated to Evidently **0.7**: `DataDriftPreset`, `snapshot.save_html()` |
| `source .env` + shell script failed to pass `MONGO_URI` | `load_dotenv()` in `drift_report.py` + `run_drift_report.sh` uses `.venv/bin/python` |
| ROS pytest plugins break `pytest` | `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` for local test runs |

### Verification (2026-06-04) — Phase 07 complete

| Check | Result |
|-------|--------|
| `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/test_mongo_logger.py tests/test_inference.py` | **5/5 passed** |
| `curl http://localhost:8000/health` | `"mongodb": true`, `"loaded": true` |
| MongoDB ping (`mask2former-mlops`) | OK |
| `predictions` collection | **102** documents (live `/predict` incremented count by 1) |
| `drift_reports` collection | Reports metadata stored after each drift run |
| `POST /predict` → Mongo | 19 instances, ~2.7s CPU; new doc logged |
| `./scripts/run_drift_report.sh` | `reports/drift_report_20260604_1813.html` (~3.8 MB) |
| GitHub `main` | Pushed `ff85c3c`, `a221e3c` |

### Environment variables

| Variable | Example / default | Description |
|----------|-------------------|-------------|
| `MONGO_URI` | `mongodb+srv://mask2former_api:...@mask2former-cluster....mongodb.net/mask2former?...` | Full connection string |
| `MONGO_DB_NAME` | `mask2former` | Database name |
| `MONGO_COLLECTION_PREDICTIONS` | `predictions` | Prediction logs |
| `MONGO_COLLECTION_DRIFT` | `drift_reports` | Drift run metadata |
| `ATLAS_PROJECT_ID` | `6a21b995cbf3f23e5981be8f` | Atlas project mask2former-mlops |
| `ATLAS_CLUSTER` | `mask2former-cluster` | Cluster name |

Copy from `.env.example`; never commit `.env`.

---

## 1. MongoDB Atlas Setup (Atlas CLI)

Atlas CLI: `atlascli 1.55+`. Default project:

```bash
atlas auth login
atlas config set project_id 6a21b995cbf3f23e5981be8f
```

### One-command setup

```bash
cd ~/Desktop/mask2former
./scripts/setup_atlas_mongo.sh
```

Creates cluster (if missing), DB user, IP allowlist, indexes, updates `.env`.

**Docker / changing IP:**

```bash
ALLOW_ANY_IP=1 ./scripts/setup_atlas_mongo.sh
```

### Manual CLI (same project)

```bash
export ATLAS_PROJECT_ID=6a21b995cbf3f23e5981be8f

atlas clusters create mask2former-cluster \
  --projectId "$ATLAS_PROJECT_ID" \
  --provider AWS --region EU_CENTRAL_1 --tier M0

atlas accessLists create --currentIp --projectId "$ATLAS_PROJECT_ID"

atlas dbusers create \
  --username mask2former_api \
  --password 'YOUR_SECURE_PASSWORD' \
  --role readWrite@mask2former \
  --projectId "$ATLAS_PROJECT_ID"

atlas clusters connectionStrings describe mask2former-cluster \
  --projectId "$ATLAS_PROJECT_ID" -o json
```

---

## 2. Run monitoring locally

### A) API + MongoDB logging

```bash
cd ~/Desktop/mask2former
source .env

docker compose up --build -d
curl -s http://localhost:8000/health | python3 -m json.tool
# → "mongodb": true
```

### B) Seed predictions

```bash
./scripts/seed_predictions.sh
# or: for img in data/raw/valid/*.jpg; do curl -s -X POST http://localhost:8000/predict -F "file=@$img" > /dev/null; done
```

### C) Drift report (use `.venv`)

```bash
.venv/bin/pip install -r requirements-monitoring.txt
./scripts/run_drift_report.sh --val-ann data/raw/valid/_annotations.coco.json
```

Open `reports/drift_report_*.html` in a browser.

### D) Re-run verification

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/test_mongo_logger.py -q
curl -s http://localhost:8000/health
./scripts/run_drift_report.sh --min-samples 5
```

---

## 3. Evidently drift detection

Compares **reference** (COCO val stats per image) vs **current** (MongoDB predictions in last N hours).

| Column | Reference | Current (live) |
|--------|-----------|----------------|
| `num_instances` | Count from COCO anns | From API response |
| `mean_score` | `1.0` (GT proxy) | Mean confidence per image |
| `inference_ms` | `90.0` (baseline) | Measured ONNX time |

Implementation (Evidently 0.7):

```python
from evidently import Report
from evidently.presets import DataDriftPreset

snapshot = Report([DataDriftPreset()]).run(
    current_data=current_df,
    reference_data=reference_df,
)
snapshot.save_html("reports/drift_report_YYYYMMDD_HHMM.html")
```

Requires **≥ 10** current samples (`--min-samples`); otherwise script exits with a message.

### Schedule (cron)

```bash
0 */6 * * * cd ~/Desktop/mask2former && ./scripts/run_drift_report.sh >> /tmp/drift.log 2>&1
```

---

## 4. MongoDB queries (debugging)

```python
from pathlib import Path
from dotenv import load_dotenv
import os
from pymongo import MongoClient

load_dotenv(Path(".env"))
client = MongoClient(os.environ["MONGO_URI"])
db = client["mask2former"]

# Last 100 predictions — avg stats
list(db.predictions.aggregate([
    {"$sort": {"timestamp": -1}},
    {"$limit": 100},
    {"$group": {
        "_id": None,
        "avg_score": {"$avg": "$mean_score"},
        "avg_latency": {"$avg": "$inference_ms"},
        "avg_instances": {"$avg": "$num_instances"},
    }},
]))

# Category counts
list(db.predictions.aggregate([
    {"$unwind": "$instances"},
    {"$group": {"_id": "$instances.category_name", "count": {"$sum": 1}}},
    {"$sort": {"count": -1}},
]))
```

Use `.venv/bin/python` or ensure `pymongo` is in the active venv.

---

## 5. Optional — FiftyOne review (not implemented)

Future: pull low-confidence rows from `predictions` into a FiftyOne dataset for visual QA. Schema would need `image_path` on each doc if added later.

---

## Summary

| Tool | Role in Phase 07 |
|------|------------------|
| MongoDB Atlas **mask2former-mlops** | Store live prediction metadata |
| Atlas CLI + `setup_atlas_mongo.sh` | Provision cluster and `.env` |
| `mongo_logger.py` | Insert from FastAPI (and ROS2 later) |
| Evidently 0.7 | Drift HTML reports in `reports/` |
| `.venv` | All monitoring Python deps (not base Python) |
| Docker | `pymongo` in API container via `MONGO_URI` |

**Status:** Phase 07 **complete** (2026-06-04).

**Next:** [08 — CI/CD](08_cicd.md)
