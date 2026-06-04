"""
Log FastAPI / ROS2 predictions to MongoDB Atlas for drift monitoring.

Atlas project: mask2former-mlops
Cluster: mask2former-cluster
Database: mask2former (collections: predictions, drift_reports)
"""

from __future__ import annotations

import datetime
import logging
import os
from typing import Any

from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

logger = logging.getLogger(__name__)


class PredictionLogger:
    def __init__(
        self,
        mongo_uri: str | None = None,
        db_name: str | None = None,
        predictions_collection: str | None = None,
        drift_collection: str | None = None,
    ):
        uri = mongo_uri or os.environ.get("MONGO_URI")
        if not uri:
            raise ValueError("MONGO_URI is not set")

        self.db_name = db_name or os.environ.get("MONGO_DB_NAME", "mask2former")
        pred_coll = predictions_collection or os.environ.get(
            "MONGO_COLLECTION_PREDICTIONS", "predictions"
        )
        drift_coll = drift_collection or os.environ.get(
            "MONGO_COLLECTION_DRIFT", "drift_reports"
        )

        self.client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        self.client.admin.command("ping")
        self.db = self.client[self.db_name]
        self.predictions = self.db[pred_coll]
        self.drift_reports = self.db[drift_coll]
        self._ensure_indexes()
        logger.info("MongoDB connected: %s.%s", self.db_name, pred_coll)

    def _ensure_indexes(self):
        self.predictions.create_index("timestamp")
        self.predictions.create_index("image_id")
        self.predictions.create_index("instances.category_id")

    def log_prediction(
        self,
        image_id: str,
        instances: list[dict[str, Any]],
        inference_ms: float,
        model_version: str,
        source: str = "api",
    ):
        """Store prediction metadata (no mask pixels — keeps documents small)."""
        slim_instances = [
            {
                "category_id": inst["category_id"],
                "category_name": inst["category_name"],
                "score": inst["score"],
                "bbox": inst["bbox"],
            }
            for inst in instances
        ]
        scores = [i["score"] for i in slim_instances]

        doc = {
            "image_id": image_id,
            "timestamp": datetime.datetime.now(datetime.timezone.utc),
            "model_version": model_version,
            "source": source,
            "inference_ms": inference_ms,
            "num_instances": len(slim_instances),
            "instances": slim_instances,
            "mean_score": sum(scores) / len(scores) if scores else 0.0,
        }

        try:
            self.predictions.insert_one(doc)
        except Exception as e:
            logger.warning("Failed to log prediction: %s", e)

    def get_recent_predictions(self, hours: int = 24, limit: int = 1000) -> list[dict]:
        cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(
            hours=hours
        )
        return list(
            self.predictions.find(
                {"timestamp": {"$gte": cutoff}},
                {"_id": 0},
            )
            .sort("timestamp", -1)
            .limit(limit)
        )

    def save_drift_report(
        self,
        drift_detected: bool,
        report_path: str,
        samples_analyzed: int,
        details: dict | None = None,
    ):
        doc = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc),
            "drift_detected": drift_detected,
            "report_path": report_path,
            "samples_analyzed": samples_analyzed,
            "details": details or {},
        }
        self.drift_reports.insert_one(doc)


def try_create_logger() -> PredictionLogger | None:
    """Return logger if MONGO_URI is configured; otherwise None (API still works)."""
    if not os.environ.get("MONGO_URI"):
        return None
    try:
        return PredictionLogger()
    except (ConnectionFailure, Exception) as e:
        logger.warning("MongoDB unavailable, predictions will not be logged: %s", e)
        return None
