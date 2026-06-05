"""
FiftyOne visual review for ONNX models (Colab + local).

Loads a capped subset per split for side-by-side:
  ground_truth vs predictions_fp32 vs predictions_int8

Full mAP on all images: src/evaluate_onnx.py (fast pycocotools).
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


def _is_colab() -> bool:
    return "google.colab" in sys.modules


def _colab_app_hint(session) -> None:
    """Print fallback URL when Colab iframe embedding fails (Chrome 403)."""
    if not _is_colab() or session is None:
        return
    url = getattr(session, "url", None)
    if url:
        print(f"\nColab: if the App is blank, open in a new tab:\n  {url}?polling=true")


def _load_category_names(ann_path: str) -> dict[int, str]:
    """COCO category id (1-indexed) → name from annotations."""
    from pycocotools.coco import COCO

    coco = COCO(ann_path)
    return {c["id"]: c["name"] for c in coco.loadCats(coco.getCatIds())}


def _model_label_names(cat_names: dict[int, str]) -> dict[int, str]:
    """Map model output index (0-based) → COCO category name."""
    max_id = max(cat_names.keys()) if cat_names else 0
    return {i: cat_names.get(i + 1, str(i)) for i in range(max_id)}


def _bbox_from_mask(mask: np.ndarray) -> tuple[float, float, float, float] | None:
    """
    Pixel bbox (x, y, w, h). FiftyOne IoU crashes on zero-width/height boxes,
    so enforce at least 1px extent (thin lane masks can be 1px wide).
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    x0, y0 = float(xs.min()), float(ys.min())
    x1, y1 = float(xs.max()), float(ys.max())
    bw = max(1.0, x1 - x0 + 1.0)
    bh = max(1.0, y1 - y0 + 1.0)
    return x0, y0, bw, bh


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

        mask_bool = mask.astype(bool)
        box = _bbox_from_mask(mask_bool)
        if box is None or width <= 0 or height <= 0:
            continue

        x, y, bw, bh = box
        cat_id = inst["category_id"]
        dets.append(
            fo.Detection(
                label=label_names.get(cat_id, inst.get("category_name", str(cat_id))),
                confidence=float(inst["score"]),
                bounding_box=[x / width, y / height, bw / width, bh / height],
                mask=mask_bool,
            )
        )
    return fo.Detections(detections=dets)


def load_review_dataset(
    split: str,
    raw_dir: str,
    max_samples: int = DEFAULT_MAX_SAMPLES,
    ann_filename: str = ANN_FILENAME,
    persistent: bool = False,
):
    fo = _require_fiftyone()
    name = f"mask2former_review_{split}"

    if fo.dataset_exists(name):
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
    print(f"FiftyOne '{name}': {len(dataset)} samples (cap={max_samples})")
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


def _tag_split(dataset, split: str) -> None:
    """Tag samples so valid/test can be filtered in the FiftyOne App."""
    for sample in dataset:
        tags = list(sample.tags or [])
        if split not in tags:
            tags.append(split)
            sample.tags = tags
            sample.save()


def _combine_datasets(datasets: list, splits: list[str]):
    """Combine per-split datasets (FiftyOne has no Dataset.merge())."""
    if not datasets:
        return None
    if len(datasets) == 1:
        return datasets[0]

    combined = datasets[0]
    for ds in datasets[1:]:
        combined.add_samples(ds)
    print(f"Combined dataset '{combined.name}': {len(combined)} samples ({', '.join(splits)})")
    return combined


def evaluate_subset(
    dataset,
    pred_field: str,
    eval_key: str,
    gt_field: str = "ground_truth",
):
    """COCO mask mAP on the loaded subset — enables TP/FP/FN filters in the App."""
    fo = _require_fiftyone()
    for key in dataset.list_evaluations():
        if key == eval_key:
            dataset.delete_evaluation(eval_key)

    results = dataset.evaluate_detections(
        pred_field,
        gt_field=gt_field,
        eval_key=eval_key,
        method="coco",
        use_masks=True,
        compute_mAP=True,
    )
    print(f"\n=== {eval_key} ({pred_field} vs {gt_field}) ===")
    results.print_report()
    return results


def run_visual_review(
    fp32_onnx_path: str,
    int8_onnx_path: str,
    raw_dir: str = "data/raw",
    splits: list[str] | None = None,
    max_samples_per_split: int = DEFAULT_MAX_SAMPLES,
    img_size: int = 512,
    score_threshold: float = 0.5,
    launch_app: bool = True,
    run_eval: bool = True,
    gt_field: str = "ground_truth",
) -> dict[str, Any]:
    """
    FiftyOne visual QA: ground_truth vs fp32/int8 predictions.

    Works in Colab (embedded App) and locally. Keep returned `session` alive in
    the notebook so the App stays connected.
    """
    splits = list(splits or DEFAULT_SPLITS)
    assert_onnx_artifacts(fp32_onnx_path, int8_onnx_path)

    for split in splits:
        ann = resolve_ann_path(raw_dir, split)
        if not os.path.isfile(ann):
            raise FileNotFoundError(f"Split '{split}' not found: {ann}")

    fp32_session = create_session(fp32_onnx_path)
    int8_session = create_session(int8_onnx_path)
    fo = _require_fiftyone()
    datasets = []

    for split in splits:
        ann_path = resolve_ann_path(raw_dir, split)
        label_names = _model_label_names(_load_category_names(ann_path))

        dataset = load_review_dataset(split, raw_dir, max_samples=max_samples_per_split)
        print(f"\n--- {split}: fp32 + int8 inference on {len(dataset)} images ---")
        attach_onnx_predictions(
            dataset,
            fp32_session,
            int8_session,
            img_size=img_size,
            score_threshold=score_threshold,
            label_names=label_names,
        )

        if run_eval:
            evaluate_subset(dataset, "predictions_fp32", f"eval_fp32_{split}", gt_field)
            evaluate_subset(dataset, "predictions_int8", f"eval_int8_{split}", gt_field)

        datasets.append(dataset)

    session = None
    combined = None
    if launch_app and datasets:
        combined = datasets[0]
        for ds in datasets[1:]:
            combined.merge(ds)

        print("\nFiftyOne App fields:")
        print(f"  • {gt_field}  (Roboflow COCO labels)")
        print("  • predictions_fp32 / predictions_int8")
        print("  Evaluations tab → filter TP / FP / FN per model")
        session = fo.launch_app(combined)
        _colab_app_hint(session)

    return {
        "splits": splits,
        "datasets": [d.name for d in datasets],
        "dataset": combined,
        "session": session,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FiftyOne ONNX visual review")
    parser.add_argument("--fp32", required=True)
    parser.add_argument("--int8", required=True)
    parser.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS))
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--max-samples", type=int, default=DEFAULT_MAX_SAMPLES)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--no-app", action="store_true")
    parser.add_argument("--no-eval", action="store_true")
    args = parser.parse_args()

    run_visual_review(
        fp32_onnx_path=args.fp32,
        int8_onnx_path=args.int8,
        raw_dir=args.raw_dir,
        splits=args.splits,
        max_samples_per_split=args.max_samples,
        score_threshold=args.threshold,
        launch_app=not args.no_app,
        run_eval=not args.no_eval,
    )
