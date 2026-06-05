"""Unit tests for ONNX COCO evaluation helpers (no model file)."""

import numpy as np

from src.evaluate_onnx import resolve_ann_path, resolve_split_dir
from src.inference import postprocess_to_coco_results


def test_resolve_split_paths():
    assert resolve_split_dir("/data/raw", "valid").endswith("valid")
    assert resolve_ann_path("/data/raw", "test").endswith("test/_annotations.coco.json")


def test_postprocess_to_coco_results_category_id_one_indexed():
    q, h, w = 2, 32, 32
    num_classes = 5

    masks_logits = np.zeros((1, q, h, w), dtype=np.float32)
    masks_logits[0, 0] = 10.0

    class_logits = np.full((1, q, num_classes + 1), -10.0, dtype=np.float32)
    class_logits[0, 0, 2] = 10.0  # model class 2 → COCO category_id 3

    results = postprocess_to_coco_results(
        masks_logits,
        class_logits,
        orig_h=64,
        orig_w=64,
        image_id=42,
        score_threshold=0.5,
    )

    assert len(results) == 1
    assert results[0]["image_id"] == 42
    assert results[0]["category_id"] == 3
    assert "segmentation" in results[0]
    assert results[0]["score"] > 0.5


def test_postprocess_to_coco_results_empty_below_threshold():
    masks_logits = np.zeros((1, 1, 16, 16), dtype=np.float32)
    class_logits = np.zeros((1, 1, 6), dtype=np.float32)

    results = postprocess_to_coco_results(
        masks_logits,
        class_logits,
        orig_h=32,
        orig_w=32,
        image_id=1,
        score_threshold=0.99,
    )
    assert results == []
