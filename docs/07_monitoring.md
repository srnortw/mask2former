# 07 — Monitoring

## Overview

```
FastAPI /predict endpoint
  ↓
MongoDB Atlas  ← log every prediction (score, category, latency, image_id)
  ↓
Evidently AI   ← compute drift reports (score distribution shift, class distribution)
  ↓
HTML reports in reports/  ← open in browser
  ↓
Alert: if drift detected → consider retraining
```

---

## What We Actually Built

| File | Purpose |
|------|---------|
| `src/mongo_logger.py` | `PredictionLogger` + `try_create_logger()` (optional if no `MONGO_URI`) |
| `src/monitoring/drift_report.py` | Evidently 0.7 drift report: val set vs MongoDB predictions |
| `scripts/run_drift_report.sh` | Runs drift report via `.venv/bin/python` |
| `scripts/seed_predictions.sh` | Batch `/predict` on val images for MongoDB seeding |
| `api/main.py` | Logs each `/predict` to MongoDB (metadata only, no masks) |
| `requirements-monitoring.txt` | evidently, pandas, pycocotools, pymongo |
| `docker-compose.yml` | Passes `MONGO_*` env vars to API container |

### Behaviour

- API **works without MongoDB** — if `MONGO_URI` is unset, serving is unchanged
- `/health` includes `"mongodb": true/false`
- Stored fields: `image_id`, `timestamp`, `instances` (bbox/score/name), `mean_score`, `inference_ms`

### Atlas setup (2026-06-04)

Dedicated MLOps project (separate from old **Computer_vision** project where **Cluster0** was stuck `UPDATING`):

| Item | Value |
|------|--------|
| Project | **`mask2former-mlops`** (`6a21b995cbf3f23e5981be8f`) |
| Cluster | **`mask2former-cluster`** (M0, `EU_CENTRAL_1`) — **IDLE** |
| SRV host | `mask2former-cluster.ceg04zs.mongodb.net` |
| DB user | `mask2former_api` with `readWrite@mask2former` |
| Database | `mask2former` |
| Collections | `predictions`, `drift_reports` (indexes created) |
| `.env` | `MONGO_URI` points at new cluster |

Leave **Cluster0** in the old project alone (or delete later in Atlas UI); this pipeline does not use it.

### Phase 07 complete (2026-06-04)

| Check | Result |
|-------|--------|
| Atlas **mask2former-mlops** / **mask2former-cluster** | IDLE, `mongodb: true` on `/health` |
| Predictions logged | 100+ docs in `predictions` |
| Drift report | `scripts/run_drift_report.sh` → HTML in `reports/` (uses **`.venv`**) |
| Tests | `tests/test_mongo_logger.py`, `tests/test_inference.py` |

**Use project venv** (not system Python):

```bash
.venv/bin/pip install -r requirements-monitoring.txt
./scripts/run_drift_report.sh
```

### Local test

- Docker build OK with `mongo_logger.py` in image
- `source .env && docker compose up --build -d` → `/health` → `"mongodb": true`

---

## 1. MongoDB Atlas Setup (Atlas CLI)

Atlas CLI is installed (`atlascli 1.55+`). Default Atlas project: **`mask2former-mlops`** (`6a21b995cbf3f23e5981be8f`).

```bash
atlas config set project_id 6a21b995cbf3f23e5981be8f
```

### One-command setup (recommended)

```bash
cd ~/Desktop/mask2former

# Refresh login if session expired
atlas auth login

# Creates M0 cluster, DB user, IP allowlist, indexes, updates .env
./scripts/setup_atlas_mongo.sh
```

The script:

| Step | Atlas CLI / action |
|------|---------------------|
| Project | `mask2former-mlops` (create once in UI or `atlas projects create mask2former-mlops`) |
| Auth check | `atlas projects list` |
| Create cluster | `atlas clusters create mask2former-cluster` in that project (skipped if exists) |
| Wait until ready | polls until `stateName == IDLE` |
| Network access | `atlas accessLists create --currentIp` |
| DB user | `mask2former_api` with `readWrite@mask2former` |
| Connection URI | `atlas clusters connectionStrings describe` → writes `MONGO_URI` to `.env` |
| Collections | creates indexes on `predictions` and `drift_reports` via pymongo |

**Docker from a changing IP?** Re-run with dev-wide access (not for production):

```bash
ALLOW_ANY_IP=1 ./scripts/setup_atlas_mongo.sh
```

### Manual Atlas CLI (project `mask2former-mlops`)

```bash
atlas auth login

export ATLAS_PROJECT_ID=6a21b995cbf3f23e5981be8f
atlas config set project_id "$ATLAS_PROJECT_ID"

atlas clusters create mask2former-cluster \
  --projectId "$ATLAS_PROJECT_ID" \
  --provider AWS \
  --region EU_CENTRAL_1 \
  --tier M0

atlas accessLists create --currentIp --projectId "$ATLAS_PROJECT_ID"

atlas dbusers create \
  --username mask2former_api \
  --password 'YOUR_SECURE_PASSWORD' \
  --role readWrite@mask2former \
  --projectId "$ATLAS_PROJECT_ID"

atlas clusters connectionStrings describe mask2former-cluster \
  --projectId "$ATLAS_PROJECT_ID" -o json
# e.g. mongodb+srv://mask2former-cluster.ceg04zs.mongodb.net
```

`.env` (copy from `.env.example`; never commit `.env`):

```bash
MONGO_URI=mongodb+srv://mask2former_api:PASSWORD@mask2former-cluster.ceg04zs.mongodb.net/mask2former?retryWrites=true&w=majority
MONGO_DB_NAME=mask2former
MONGO_COLLECTION_PREDICTIONS=predictions
MONGO_COLLECTION_DRIFT=drift_reports
ATLAS_PROJECT_ID=6a21b995cbf3f23e5981be8f
ATLAS_CLUSTER=mask2former-cluster
```

---

## 2. Test monitoring locally

### A) Start API with MongoDB logging

```bash
cd ~/Desktop/mask2former
source .env    # MONGO_URI, HF_TOKEN, ...

docker compose up --build -d
curl http://localhost:8000/health
# → "mongodb": true
```

### B) Send a few predictions

```bash
for img in data/raw/valid/*.jpg; do
  curl -s -X POST http://localhost:8000/predict -F "file=@$img" > /dev/null
  echo "logged: $img"
done
```

Or use `scripts/visualize_predict.py` (also hits `/predict`).

### C) Run drift report

```bash
# Always use project .venv (not base/system Python)
.venv/bin/pip install -r requirements-monitoring.txt

./scripts/seed_predictions.sh   # optional if MongoDB has < 10 samples

./scripts/run_drift_report.sh \
  --val-ann data/raw/valid/_annotations.coco.json \
  --hours 24 \
  --output-dir reports
```

Equivalent without helper script:

```bash
source .env
.venv/bin/python -m src.monitoring.drift_report --val-ann data/raw/valid/_annotations.coco.json
```

Open `reports/drift_report_*.html` in a browser.

---

## 3. Evidently AI — Drift Detection

Evidently compares **reference data** (training distribution) vs **current data** (live predictions).

### What we monitor

| Metric | What drift looks like |
|---|---|
| Score distribution | Mean confidence drops → model uncertain on new data |
| Class distribution | Unexpected class prevalence shift |
| Inference latency | Performance degradation |
| Instances per image | Scene complexity changed |

```python
# src/monitoring/drift_report.py
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.metrics import ColumnDriftMetric, DatasetDriftMetric
from mongo_logger import PredictionLogger
import os


def build_reference_data(val_json_path: str) -> pd.DataFrame:
    """Build reference DataFrame from validation set statistics."""
    from pycocotools.coco import COCO
    coco = COCO(val_json_path)

    rows = []
    for img_id, anns in coco.imgToAnns.items():
        rows.append({
            "num_instances": len(anns),
            "mean_score": 1.0,   # ground truth = perfect score
            "inference_ms": 90.0,
        })
    return pd.DataFrame(rows)


def build_current_data(logger: PredictionLogger, hours: int = 24) -> pd.DataFrame:
    """Build current DataFrame from recent MongoDB predictions."""
    preds = logger.get_recent_predictions(hours=hours)
    rows = []
    for p in preds:
        rows.append({
            "num_instances": p["num_instances"],
            "mean_score": p["mean_score"],
            "inference_ms": p["inference_ms"],
        })
    return pd.DataFrame(rows)


def run_drift_report(
    val_json_path: str = "data/processed/valid/_annotations.coco.json",
    output_dir: str = "reports",
    hours: int = 24,
):
    os.makedirs(output_dir, exist_ok=True)

    logger = PredictionLogger()
    reference = build_reference_data(val_json_path)
    current = build_current_data(logger, hours=hours)

    if len(current) < 10:
        print(f"Not enough data for drift analysis ({len(current)} samples). Need at least 10.")
        return

    report = Report(metrics=[
        DatasetDriftMetric(),
        ColumnDriftMetric(column_name="mean_score"),
        ColumnDriftMetric(column_name="num_instances"),
        ColumnDriftMetric(column_name="inference_ms"),
    ])

    report.run(reference_data=reference, current_data=current)

    # Save HTML report
    report_path = f"{output_dir}/drift_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.html"
    report.save_html(report_path)
    print(f"Report saved: {report_path}")

    # Check if drift detected
    drift_result = report.as_dict()
    dataset_drift = drift_result["metrics"][0]["result"]["dataset_drift"]

    if dataset_drift:
        print("DRIFT DETECTED — consider retraining!")
        # Store drift alert in MongoDB
        logger.drift_reports.insert_one({
            "timestamp": pd.Timestamp.now().to_pydatetime(),
            "drift_detected": True,
            "report_path": report_path,
            "samples_analyzed": len(current),
        })

    return dataset_drift
```

### Schedule drift report (cron)

```bash
# Run drift check every 6 hours
# Add to crontab: crontab -e
0 */6 * * * cd ~/Desktop/mask2former && source .env && ./scripts/run_drift_report.sh
```

---

## 4. MongoDB Queries for Debugging

```python
from pymongo import MongoClient
import os

client = MongoClient(os.environ["MONGO_URI"])
db = client["mask2former"]

# Average confidence over last 100 predictions
pipeline = [
    {"$sort": {"timestamp": -1}},
    {"$limit": 100},
    {"$group": {
        "_id": None,
        "avg_score": {"$avg": "$mean_score"},
        "avg_latency": {"$avg": "$inference_ms"},
        "avg_instances": {"$avg": "$num_instances"},
    }}
]
result = list(db.predictions.aggregate(pipeline))
print(result)

# Most common categories predicted
pipeline = [
    {"$unwind": "$instances"},
    {"$group": {"_id": "$instances.category_name", "count": {"$sum": 1}}},
    {"$sort": {"count": -1}},
]
categories = list(db.predictions.aggregate(pipeline))
print(categories)
```

---

## 5. FiftyOne — Production Data Review

Pull recent predictions from MongoDB and review in FiftyOne:

```python
# src/monitoring/fiftyone_review.py
import fiftyone as fo
from mongo_logger import PredictionLogger

def review_low_confidence_predictions(threshold: float = 0.4):
    logger = PredictionLogger()
    preds = logger.get_recent_predictions(hours=48)

    # Filter low confidence predictions
    low_conf = [p for p in preds if p["mean_score"] < threshold]
    print(f"Low confidence predictions: {len(low_conf)}")

    dataset = fo.Dataset("low_confidence_review")
    for pred in low_conf[:100]:
        # Requires image path stored in prediction doc
        if "image_path" in pred:
            sample = fo.Sample(filepath=pred["image_path"])
            sample["mean_score"] = pred["mean_score"]
            sample["num_instances"] = pred["num_instances"]
            dataset.add_sample(sample)

    session = fo.launch_app(dataset)
    return session
```

---

## Summary

| Tool | Role |
|---|---|
| MongoDB Atlas | Project **mask2former-mlops** — store every prediction with metadata |
| Atlas CLI | Manage **mask2former-cluster** from terminal |
| Evidently AI | Statistical drift detection (score, class, latency) |
| FiftyOne | Visual review of low confidence predictions |
| Cron job | Scheduled drift reports every 6 hours |

**Next:** [08 — CI/CD](08_cicd.md)
