"""
FiftyOne evaluation for exported ONNX models (Phase 4 — Colab or local).

Compare fp32 vs INT8 on valid/test splits with COCO-style mask mAP + FiftyOne App (PR curves).
"""

from __future__ import annotations

import base64
import os
from typing import Any

import cv2
import numpy as np

from src.inference import (
    DEFAULT_CATEGORIES,
    create_session,
    load_categories,
    postprocess_instances,
    preprocess_image_bytes,
)


def _require_fiftyone():
    import fiftyone as fo

    return fo


def load_fiftyone_split(
    split: str,
    raw_dir: str = "data/raw",
    ann_filename: str = "_annotations.coco.json",
    persistent: bool = True,
):
    """
    Load Roboflow COCO split into FiftyOne.
    split: 'valid' or 'test' (folder names under data/raw).
    """
    fo = _require_fiftyone()
    name = f"mask2former_eval_{split}"

    if persistent and fo.dataset_exists(name):
        return fo.load_dataset(name)

    if fo.dataset_exists(name):
        fo.delete_dataset(name)

    split_dir = os.path.join(raw_dir, split)
    images_dir = os.path.join(split_dir, "images")
    data_path = images_dir if os.path.isdir(images_dir) else split_dir

    dataset = fo.Dataset.from_dir(
        dataset_type=fo.types.COCODetectionDataset,
        data_path=data_path,
        labels_path=os.path.join(split_dir, ann_filename),
        name=name,
        label_types=["segmentations"],
        overwrite=True,
    )
    print(f"FiftyOne dataset '{name}': {len(dataset)} samples")
    return dataset


def instances_to_detections(
    instances: list[dict[str, Any]],
    width: int,
    height: int,
):
    """Convert postprocess_instances() output to FiftyOne Detections (with masks)."""
    fo = _require_fiftyone()
    dets = []
    for inst in instances:
        mask_h, mask_w = inst["mask_shape"]
        raw = base64.b64decode(inst["mask_b64"])
        mask = np.frombuffer(raw, dtype=np.uint8).reshape(mask_h, mask_w)
        if mask.shape != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

        x, y, bw, bh = inst["bbox"]
        dets.append(
            fo.Detection(
                label=inst["category_name"],
                confidence=float(inst["score"]),
                bounding_box=[
                    x / width,
                    y / height,
                    bw / width,
                    bh / height,
                ],
                mask=mask.astype(bool),
            )
        )
    return fo.Detections(detections=dets)


def apply_onnx_predictions(
    dataset,
    session,
    pred_field: str,
    img_size: int = 512,
    score_threshold: float = 0.5,
    max_samples: int | None = None,
    categories: dict[int, str] | None = None,
) -> None:
    """Run ONNX on each sample and store predictions in pred_field."""
    categories = categories or load_categories()
    view = dataset
    if max_samples is not None:
        view = dataset.take(max_samples)

    for sample in view.iter_samples(progress=True):
        with open(sample.filepath, "rb") as f:
            image_bytes = f.read()

        tensor, orig_h, orig_w = preprocess_image_bytes(image_bytes, img_size)
        masks_logits, class_logits = session.run(None, {"pixel_values": tensor})
        instances = postprocess_instances(
            masks_logits,
            class_logits,
            orig_h,
            orig_w,
            score_threshold=score_threshold,
            categories=categories,
        )
        sample[pred_field] = instances_to_detections(
            instances,
            sample.metadata.width,
            sample.metadata.height,
        )
        sample.save()


def evaluate_predictions(
    dataset,
    pred_field: str,
    eval_key: str,
    gt_field: str = "ground_truth",
):
    """COCO-style instance segmentation mAP (use_masks=True)."""
    results = dataset.evaluate_detections(
        pred_field,
        gt_field=gt_field,
        eval_key=eval_key,
        method="coco",
        use_masks=True,
        compute_mAP=True,
    )
    print(f"\n=== {eval_key} ({pred_field}) ===")
    results.print_report()
    return results


def run_phase4_evaluation(
    fp32_onnx_path: str,
    int8_onnx_path: str,
    splits: list[str] | None = None,
    raw_dir: str = "data/raw",
    img_size: int = 512,
    score_threshold: float = 0.5,
    max_samples: int | None = None,
    launch_app: bool = True,
    gt_field: str = "ground_truth",
) -> dict[str, Any]:
    """
    Phase 4 sign-off: evaluate fp32 and INT8 ONNX on valid + test in FiftyOne.

    Recommended order in Colab (after Cells 16–17):
      1) fp32 ONNX  — validates export vs training (compare to Phase 3 evaluate.py mAP)
      2) int8 ONNX — validates deployed artifact (what HF + Docker serve)

    Returns dict: {split: {model: mAP, eval_results}}
    """
    splits = splits or ["valid", "test"]
    summary: dict[str, Any] = {}

    fp32_session = create_session(fp32_onnx_path)
    int8_session = create_session(int8_onnx_path)

    for split in splits:
        dataset = load_fiftyone_split(split, raw_dir=raw_dir)
        summary[split] = {}

        for tag, session in (("fp32", fp32_session), ("int8", int8_session)):
            pred_field = f"predictions_{tag}"
            eval_key = f"eval_{tag}"

            if pred_field in dataset.get_field_schema():
                dataset.delete_sample_field(pred_field)
            if eval_key in dataset.list_evaluations():
                dataset.delete_evaluation(eval_key)

            print(f"\n--- {split} / {tag} ONNX ---")
            apply_onnx_predictions(
                dataset,
                session,
                pred_field=pred_field,
                img_size=img_size,
                score_threshold=score_threshold,
                max_samples=max_samples,
            )
            results = evaluate_predictions(
                dataset,
                pred_field=pred_field,
                eval_key=eval_key,
                gt_field=gt_field,
            )
            mAP = float(results.mAP()) if hasattr(results, "mAP") else None
            summary[split][tag] = {"mAP": mAP, "eval_key": eval_key, "pred_field": pred_field}

        if launch_app:
            fo = _require_fiftyone()
            print(f"\nLaunching FiftyOne App for '{dataset.name}' …")
            print("  Compare fields: predictions_fp32 vs predictions_int8 vs ground_truth")
            print("  Evaluations tab: PR curves / per-class AP")
            fo.launch_app(dataset)

    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FiftyOne ONNX eval (Phase 4)")
    parser.add_argument("--fp32", required=True, help="Path to mask2former_fp32.onnx")
    parser.add_argument("--int8", required=True, help="Path to mask2former_int8.onnx")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["valid", "test"],
        help="Data splits under data/raw (default: valid test)",
    )
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--no-app", action="store_true")
    args = parser.parse_args()

    run_phase4_evaluation(
        fp32_onnx_path=args.fp32,
        int8_onnx_path=args.int8,
        splits=args.splits,
        raw_dir=args.raw_dir,
        score_threshold=args.threshold,
        max_samples=args.max_samples,
        launch_app=not args.no_app,
    )
