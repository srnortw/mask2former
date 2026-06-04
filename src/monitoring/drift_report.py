"""
Evidently drift report: reference (validation set stats) vs live MongoDB predictions.
Uses Evidently 0.7+ API (Report → Snapshot.save_html).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.mongo_logger import try_create_logger  # noqa: E402


def build_reference_data(val_ann_path: str) -> pd.DataFrame:
    from pycocotools.coco import COCO

    coco = COCO(val_ann_path)
    rows = []
    for img_id in coco.getImgIds():
        anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
        rows.append({
            "num_instances": len(anns),
            "mean_score": 1.0,
            "inference_ms": 90.0,
        })
    return pd.DataFrame(rows)


def build_current_data(logger, hours: int = 24) -> pd.DataFrame:
    preds = logger.get_recent_predictions(hours=hours)
    rows = []
    for p in preds:
        rows.append({
            "num_instances": p["num_instances"],
            "mean_score": p["mean_score"],
            "inference_ms": p["inference_ms"],
        })
    return pd.DataFrame(rows)


def _drift_detected_from_snapshot(snapshot) -> bool:
    """Best-effort parse of Evidently 0.7 snapshot for dataset drift."""
    try:
        raw = snapshot.dict() if hasattr(snapshot, "dict") else snapshot.json()
        if isinstance(raw, str):
            raw = json.loads(raw)
        text = json.dumps(raw).lower()
        if "dataset_drift" in text and "true" in text:
            return True
        if "share_of_drifted_columns" in text:
            # heuristic: any drifted column share > 0
            for block in raw.get("metrics", []) if isinstance(raw, dict) else []:
                payload = json.dumps(block).lower()
                if "drift" in payload and "detected" in payload:
                    return True
    except Exception:
        pass
    return False


def run_drift_report(
    val_ann_path: str,
    output_dir: str = "reports",
    hours: int = 24,
    min_samples: int = 10,
) -> bool | None:
    from evidently import Report
    from evidently.presets import DataDriftPreset

    logger = try_create_logger()
    if logger is None:
        print("MONGO_URI not set or MongoDB unreachable — skip drift report.")
        return None

    reference = build_reference_data(val_ann_path)
    current = build_current_data(logger, hours=hours)

    if len(current) < min_samples:
        print(
            f"Not enough predictions for drift analysis ({len(current)} < {min_samples}). "
            "Run the API and send more /predict requests first."
        )
        return None

    snapshot = Report([DataDriftPreset()]).run(
        current_data=current,
        reference_data=reference,
    )

    os.makedirs(output_dir, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    report_path = os.path.join(output_dir, f"drift_report_{stamp}.html")
    snapshot.save_html(report_path)
    print(f"Report saved: {report_path}")

    drift_detected = _drift_detected_from_snapshot(snapshot)

    if drift_detected:
        print("DRIFT DETECTED — review report and consider retraining.")
    else:
        print("No clear dataset drift flag (see HTML report for column-level drift).")

    logger.save_drift_report(
        drift_detected=drift_detected,
        report_path=report_path,
        samples_analyzed=len(current),
        details={"hours": hours, "reference_rows": len(reference)},
    )
    return drift_detected


def main():
    from dotenv import load_dotenv

    load_dotenv(_ROOT / ".env")

    parser = argparse.ArgumentParser(description="Run Evidently drift report")
    parser.add_argument(
        "--val-ann",
        default="data/raw/valid/_annotations.coco.json",
        help="COCO annotations for reference distribution",
    )
    parser.add_argument("--hours", type=int, default=24)
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--min-samples", type=int, default=10)
    args = parser.parse_args()

    run_drift_report(
        val_ann_path=args.val_ann,
        output_dir=args.output_dir,
        hours=args.hours,
        min_samples=args.min_samples,
    )


if __name__ == "__main__":
    main()
