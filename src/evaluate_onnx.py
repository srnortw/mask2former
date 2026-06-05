"""
Fast ONNX evaluation for Phase 4 (Colab + local CLI).

Uses pycocotools mask mAP — same metric as evaluate.py — on configurable splits
(default: valid, test). No FiftyOne dependency.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from typing import Any

import cv2
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from src.inference import create_session, postprocess_to_coco_results, preprocess_image_rgb

DEFAULT_SPLITS = ("valid", "test")
ANN_FILENAME = "_annotations.coco.json"


def resolve_split_dir(raw_dir: str, split: str) -> str:
    """Return image directory for a split (Roboflow flat or split/images/)."""
    images_subdir = os.path.join(raw_dir, split, "images")
    if os.path.isdir(images_subdir):
        return images_subdir
    return os.path.join(raw_dir, split)


def resolve_ann_path(raw_dir: str, split: str, ann_filename: str = ANN_FILENAME) -> str:
    return os.path.join(raw_dir, split, ann_filename)


def assert_onnx_artifacts(*paths: str) -> None:
    missing = [p for p in paths if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(
            "Missing ONNX file(s):\n  " + "\n  ".join(missing)
        )


def _is_colab() -> bool:
    return "google.colab" in sys.modules


def run_coco_segm_eval(gt_file: str, results: list[dict[str, Any]]) -> tuple[float, COCOeval]:
    """Return mask AP @ IoU 0.50:0.95 and the evaluator."""
    if not results:
        coco_gt = COCO(gt_file)
        coco_dt = coco_gt.loadRes([])
        evaluator = COCOeval(coco_gt, coco_dt, iouType="segm")
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
        return 0.0, evaluator

    coco_gt = COCO(gt_file)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(results, f)
        pred_file = f.name

    try:
        coco_dt = coco_gt.loadRes(pred_file)
        evaluator = COCOeval(coco_gt, coco_dt, iouType="segm")
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
        return float(evaluator.stats[0]), evaluator
    finally:
        os.unlink(pred_file)


def collect_split_predictions(
    session,
    raw_dir: str,
    split: str,
    img_size: int = 512,
    score_threshold: float = 0.5,
    max_samples: int | None = None,
    ann_filename: str = ANN_FILENAME,
) -> tuple[list[dict[str, Any]], str, int]:
    """
    Run ONNX on one split. Returns (coco_results, ann_path, n_images).
    """
    ann_path = resolve_ann_path(raw_dir, split, ann_filename)
    if not os.path.isfile(ann_path):
        raise FileNotFoundError(f"Missing annotations: {ann_path}")

    img_dir = resolve_split_dir(raw_dir, split)
    coco = COCO(ann_path)
    img_ids = sorted(coco.imgs.keys())
    if max_samples is not None:
        img_ids = img_ids[:max_samples]

    results: list[dict[str, Any]] = []
    for img_id in img_ids:
        info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(img_dir, info["file_name"])
        img = cv2.imread(img_path)
        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor, orig_h, orig_w = preprocess_image_rgb(img_rgb, img_size)
        masks_logits, class_logits = session.run(None, {"pixel_values": tensor})
        results.extend(
            postprocess_to_coco_results(
                masks_logits,
                class_logits,
                orig_h,
                orig_w,
                image_id=img_id,
                score_threshold=score_threshold,
            )
        )

    return results, ann_path, len(img_ids)


def evaluate_onnx_model(
    onnx_path: str,
    raw_dir: str,
    splits: list[str] | None = None,
    img_size: int = 512,
    score_threshold: float = 0.5,
    max_samples: int | None = None,
) -> dict[str, dict[str, Any]]:
    """Evaluate one ONNX model on multiple splits. Returns per-split mAP."""
    splits = list(splits or DEFAULT_SPLITS)
    session = create_session(onnx_path)
    summary: dict[str, dict[str, Any]] = {}

    for split in splits:
        print(f"\n--- {split} ---")
        results, ann_path, n_images = collect_split_predictions(
            session,
            raw_dir,
            split,
            img_size=img_size,
            score_threshold=score_threshold,
            max_samples=max_samples,
        )
        mAP, _ = run_coco_segm_eval(ann_path, results)
        summary[split] = {
            "mAP": mAP,
            "n_images": n_images,
            "n_predictions": len(results),
            "ann_path": ann_path,
        }
        print(f"  images={n_images}  preds={len(results)}  mAP={mAP:.4f}")

    return summary


def run_phase4_evaluation(
    fp32_onnx_path: str,
    int8_onnx_path: str,
    raw_dir: str = "data/raw",
    splits: list[str] | None = None,
    img_size: int = 512,
    score_threshold: float = 0.5,
    max_samples: int | None = None,
) -> dict[str, dict[str, dict[str, Any]]]:
    """
    Compare fp32 vs INT8 ONNX on valid + test (or custom splits).

    Returns: {split: {fp32: {...}, int8: {...}}}
    """
    splits = list(splits or DEFAULT_SPLITS)
    assert_onnx_artifacts(fp32_onnx_path, int8_onnx_path)

    for split in splits:
        ann = resolve_ann_path(raw_dir, split)
        if not os.path.isfile(ann):
            raise FileNotFoundError(f"Split '{split}' not found: {ann}")

    print(f"Phase 4 ONNX eval — splits={splits}  raw_dir={raw_dir}")
    if max_samples is not None:
        print(f"  (capped at {max_samples} images per split)")

    summary: dict[str, dict[str, dict[str, Any]]] = {}

    for tag, path in (("fp32", fp32_onnx_path), ("int8", int8_onnx_path)):
        print(f"\n========== {tag.upper()} ONNX ==========")
        per_split = evaluate_onnx_model(
            path,
            raw_dir,
            splits=splits,
            img_size=img_size,
            score_threshold=score_threshold,
            max_samples=max_samples,
        )
        for split, info in per_split.items():
            summary.setdefault(split, {})[tag] = info

    print("\n=== mAP summary ===")
    for split in splits:
        fp32_map = summary[split]["fp32"]["mAP"]
        int8_map = summary[split]["int8"]["mAP"]
        delta = int8_map - fp32_map
        print(f"  {split}: fp32={fp32_map:.4f}  int8={int8_map:.4f}  delta={delta:+.4f}")

    if _is_colab():
        print("\n(Colab: visual QA with FiftyOne is local-only — see src/fiftyone_onnx_review.py)")

    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fast ONNX mAP eval (Phase 4)")
    parser.add_argument("--fp32", required=True, help="Path to mask2former_fp32.onnx")
    parser.add_argument("--int8", required=True, help="Path to mask2former_int8.onnx")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        help="Splits under raw_dir (default: valid test)",
    )
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--img-size", type=int, default=512)
    args = parser.parse_args()

    run_phase4_evaluation(
        fp32_onnx_path=args.fp32,
        int8_onnx_path=args.int8,
        raw_dir=args.raw_dir,
        splits=args.splits,
        img_size=args.img_size,
        score_threshold=args.threshold,
        max_samples=args.max_samples,
    )
