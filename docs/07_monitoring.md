# 07 — Monitoring

## Overview

```
FastAPI /predict endpoint
  ↓
MongoDB Atlas  ← log every prediction (score, category, latency, image_id)
  ↓
Evidently AI   ← compute drift reports (score distribution shift, class distribution)
  ↓
Grafana / HTML reports  ← visualize drift over time
  ↓
Alert: if drift detected → trigger retraining pipeline
```

---

## 1. MongoDB Atlas Setup

Atlas CLI is already installed (`atlascli 1.55.0`).

```bash
# Login to Atlas
atlas auth login

# Create a free cluster (M0, 512MB)
atlas clusters create mask2former-cluster \
  --provider AWS \
  --region US_EAST_1 \
  --tier M0 \
  --projectId your-project-id

# Get connection string
atlas clusters connectionStrings describe mask2former-cluster
# → mongodb+srv://user:pass@mask2former-cluster.xxxxx.mongodb.net/
```

Set connection string in `.env`:
```bash
MONGO_URI=mongodb+srv://user:pass@mask2former-cluster.xxxxx.mongodb.net/mask2former
```

---

## 2. Prediction Logger

```python
# src/mongo_logger.py
import os
import datetime
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
import logging

logger = logging.getLogger(__name__)


class PredictionLogger:
    def __init__(self, mongo_uri: str = None):
        uri = mongo_uri or os.environ["MONGO_URI"]
        self.client = MongoClient(uri, serverSelectionTimeoutMS=3000)
        self.db = self.client["mask2former"]
        self.predictions = self.db["predictions"]
        self.drift_reports = self.db["drift_reports"]
        self._ensure_indexes()

    def _ensure_indexes(self):
        self.predictions.create_index("timestamp")
        self.predictions.create_index("image_id")
        self.predictions.create_index("category_id")

    def log_prediction(
        self,
        image_id: str,
        instances: list,
        inference_ms: float,
        model_version: str,
        source: str = "api",   # "api" or "ros2"
    ):
        doc = {
            "image_id": image_id,
            "timestamp": datetime.datetime.utcnow(),
            "model_version": model_version,
            "source": source,
            "inference_ms": inference_ms,
            "num_instances": len(instances),
            "instances": [
                {
                    "category_id": inst["category_id"],
                    "category_name": inst["category_name"],
                    "score": inst["score"],
                    "bbox": inst["bbox"],
                }
                for inst in instances
            ],
            # Average confidence score across all instances
            "mean_score": sum(i["score"] for i in instances) / len(instances) if instances else 0.0,
        }

        try:
            self.predictions.insert_one(doc)
        except Exception as e:
            logger.warning(f"Failed to log prediction: {e}")

    def get_recent_predictions(self, hours: int = 24, limit: int = 1000):
        cutoff = datetime.datetime.utcnow() - datetime.timedelta(hours=hours)
        return list(
            self.predictions.find(
                {"timestamp": {"$gte": cutoff}},
                {"_id": 0}
            ).limit(limit)
        )
```

### Integrate logger into FastAPI

```python
# api/main.py — add to lifespan and predict endpoint

from mongo_logger import PredictionLogger

mongo_logger: PredictionLogger = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global session, mongo_logger
    # ... model loading ...
    mongo_logger = PredictionLogger()
    yield


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    # ... inference ...
    result = PredictResponse(...)

    # Log to MongoDB (non-blocking, fire-and-forget)
    mongo_logger.log_prediction(
        image_id=file.filename,
        instances=[i.dict() for i in result.instances],
        inference_ms=result.inference_ms,
        model_version=MODEL_FILE,
    )

    return result
```

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
0 */6 * * * cd ~/Desktop/mask2former && source .venv/bin/activate && python src/monitoring/drift_report.py
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
| MongoDB Atlas | Store every prediction with metadata |
| Atlas CLI | Cluster management from terminal |
| Evidently AI | Statistical drift detection (score, class, latency) |
| FiftyOne | Visual review of low confidence predictions |
| Cron job | Scheduled drift reports every 6 hours |

**Next:** [08 — CI/CD](08_cicd.md)
