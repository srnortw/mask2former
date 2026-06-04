"""Unit tests for ONNX postprocessing (no model file required)."""

import numpy as np

from src.inference import DEFAULT_CATEGORIES, postprocess_instances


def test_postprocess_instances_filters_low_score():
    q, h, w = 4, 64, 64
    num_classes = 5

    masks_logits = np.zeros((1, q, h, w), dtype=np.float32)
    masks_logits[0, 0] = 5.0  # high sigmoid for query 0

    class_logits = np.full((1, q, num_classes + 1), -10.0, dtype=np.float32)
    class_logits[0, 0, 0] = 10.0  # class 0, high score
    class_logits[0, 1, 1] = 10.0  # query 1 also passes

    instances = postprocess_instances(
        masks_logits,
        class_logits,
        orig_h=128,
        orig_w=128,
        score_threshold=0.5,
        categories=DEFAULT_CATEGORIES,
    )

    assert len(instances) >= 1
    assert instances[0]["category_id"] == 0
    assert "mask_b64" in instances[0]
    assert instances[0]["mask_shape"] == [128, 128]


def test_postprocess_returns_empty_when_all_below_threshold():
    masks_logits = np.zeros((1, 2, 32, 32), dtype=np.float32)
    class_logits = np.zeros((1, 2, 6), dtype=np.float32)

    instances = postprocess_instances(
        masks_logits,
        class_logits,
        orig_h=64,
        orig_w=64,
        score_threshold=0.99,
    )
    assert instances == []
