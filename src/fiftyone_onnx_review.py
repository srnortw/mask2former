"""
FiftyOne visual review for ONNX models (local only).

Loads a small subset per split (default 25) for side-by-side fp32 vs INT8 vs GT.
Full mAP sign-off uses src/evaluate_onnx.py (fast pycocotools, no FiftyOne).
"""

from __future__ import annotations

import os
import sys
from typing import Any

import cv2
import numpy as np

from src.evaluate_onnx import (
    ANN_FILENAME,
    DEFAULT_SPLITS,
    assert_onnx_artifacts,
    resolve_ann_path,
    resolve_split_dir,
)
from src.inference import create_session, postprocess_instances, preprocess_image_rgb

DEFAULT_MAX_SAMPLES = 25


def _require_fiftyone():
    import fiftyone as fo

    return fo


def _load_category_names(ann_path: str) -> dict[int, str]:
    """COCO category id (1-indexed) → name from annotations."""
    from pycocotools.coco import COCO

    coco = COCO(ann_path)
    return {c["id"]: c["name"] for c in coco.loadCats(coco.getCatIds())}


def _model_label_names(cat_names: dict[int, str]) -> dict[int, str]:
    """Map model output index (0-based) → COCO category name."""
    return {i: cat_names.get(i + 1, str(i)) for i in range(max(len(cat_names), 1))}


def instances_to_detections(
    instances: list[dict[str, Any]],
    width: int,
    height: int,
    label_names: dict[int, str],
):
    fo = _require_fiftyone()
    dets = []
    for inst in instances:
        import base64

        mask_h, mask_w = inst["mask_shape"]
        raw = base64.b64decode(inst["mask_b64"])
        mask = np.frombuffer(raw, dtype=np.uint8).reshape(mask_h, mask_w)
        if mask.shape != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

        x, y, bw, bh = inst["bbox"]
        cat_id = inst["category_id"]
        dets.append(
            fo.Detection(
                label=label_names.get(cat_id, inst.get("category_name", str(cat_id))),
                confidence=float(inst["score"]),
                bounding_box=[x / width, y / height, bw / width, bh / height],
                mask=mask.astype(bool),
            )
        )
    return fo.Detections(detections=dets)


def load_review_dataset(
    split: str,
    raw_dir: str,
    max_samples: int = DEFAULT_MAX_SAMPLES,
    ann_filename: str = ANN_FILENAME,
    persistent: bool = True,
):
    fo = _require_fiftyone()
    name = f"mask2former_review_{split}"

    if persistent and fo.dataset_exists(name):
        ds = fo.load_dataset(name)
        if len(ds) <= max_samples:
            return ds
        fo.delete_dataset(name)
    elif fo.dataset_exists(name):
        fo.delete_dataset(name)

    split_dir = os.path.join(raw_dir, split)
    data_path = resolve_split_dir(raw_dir, split)

    dataset = fo.Dataset.from_dir(
        dataset_type=fo.types.COCODetectionDataset,
        data_path=data_path,
        labels_path=os.path.join(split_dir, ann_filename),
        name=name,
        label_types=["segmentations"],
        max_samples=max_samples,
        overwrite=True,
    )
    print(f"FiftyOne review '{name}': {len(dataset)} samples (cap={max_samples})")
    return dataset


def attach_onnx_predictions(
    dataset,
    fp32_session,
    int8_session,
    img_size: int = 512,
    score_threshold: float = 0.5,
    label_names: dict[int, str] | None = None,
) -> None:
    """Run fp32 + int8 on each sample in one pass (single image read)."""
    label_names = label_names or {}

    for field in ("predictions_fp32", "predictions_int8"):
        if field in dataset.get_field_schema():
            dataset.delete_sample_field(field)

    for sample in dataset.iter_samples(progress=True):
        img = cv2.imread(sample.filepath)
        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor, orig_h, orig_w = preprocess_image_rgb(img_rgb, img_size)
        w, h = sample.metadata.width, sample.metadata.height

        for tag, session in (("predictions_fp32", fp32_session), ("predictions_int8", int8_session)):
            masks_logits, class_logits = session.run(None, {"pixel_values": tensor})
            instances = postprocess_instances(
                masks_logits,
                class_logits,
                orig_h,
                orig_w,
                score_threshold=score_threshold,
                categories=label_names,
            )
            sample[tag] = instances_to_detections(instances, w, h, label_names)
        sample.save()


def run_visual_review(
    fp32_onnx_path: str,
    int8_onnx_path: str,
    raw_dir: str = "data/raw",
    splits: list[str] | None = None,
    max_samples_per_split: int = DEFAULT_MAX_SAMPLES,
    img_size: int = 512,
    score_threshold: float = 0.5,
    launch_app: bool | None = None,
) -> dict[str, Any]:
    """
    Local FiftyOne visual QA on a subset per split.

    launch_app defaults to True on desktop, False in Colab.
    """
    splits = list(splits or DEFAULT_SPLITS)
    assert_onnx_artifacts(fp32_onnx_path, int8_onnx_path)

    if launch_app is None:
        launch_app = "google.colab" not in sys.modules

    fp32_session = create_session(fp32_onnx_path)
    int8_session = create_session(int8_onnx_path)
    fo = _require_fiftyone()
    datasets = []

    for split in splits:
        ann_path = resolve_ann_path(raw_dir, split)
        cat_names = _load_category_names(ann_path)
        label_names = _model_label_names(cat_names)

        dataset = load_review_dataset(
            split,
            raw_dir,
            max_samples=max_samples_per_split,
        )
        print(f"\n--- {split}: running fp32 + int8 on {len(dataset)} samples ---")
        attach_onnx_predictions(
            dataset,
            fp32_session,
            int8_session,
            img_size=img_size,
            score_threshold=score_threshold,
            label_names=label_names,
        )
        datasets.append(dataset)

    if launch_app and datasets:
        combined = datasets[0]
        for ds in datasets[1:]:
            combined.merge(ds)
        print("\nFiftyOne App — compare predictions_fp32 / predictions_int8 / ground_truth")
        print("For full mAP on all images: python -m src.evaluate_onnx --fp32 ... --int8 ...")
        fo.launch_app(combined)

    return {"splits": splits, "datasets": [d.name for d in datasets]}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FiftyOne visual ONNX review (local)")
    parser.add_argument("--fp32", required=True)
    parser.add_argument("--int8", required=True)
    parser.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS))
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--max-samples", type=int, default=DEFAULT_MAX_SAMPLES)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--no-app", action="store_true")
    args = parser.parse_args()

    run_visual_review(
        fp32_onnx_path=args.fp32,
        int8_onnx_path=args.int8,
        raw_dir=args.raw_dir,
        splits=args.splits,
        max_samples_per_split=args.max_samples,
        score_threshold=args.threshold,
        launch_app=not args.no_app,
    )
