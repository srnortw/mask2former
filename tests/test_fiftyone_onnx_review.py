"""Unit tests for FiftyOne review helpers (no FiftyOne import required)."""

import numpy as np

from src.fiftyone_onnx_review import _bbox_from_mask, _model_label_names


def test_bbox_from_mask_single_pixel():
    mask = np.zeros((10, 10), dtype=bool)
    mask[5, 5] = True
    box = _bbox_from_mask(mask)
    assert box is not None
    x, y, bw, bh = box
    assert bw >= 1.0
    assert bh >= 1.0


def test_bbox_from_mask_thin_line():
    """Vertical 1px-wide lane line — caused ZeroDivisionError before fix."""
    mask = np.zeros((20, 20), dtype=bool)
    mask[2:18, 10] = True
    box = _bbox_from_mask(mask)
    assert box is not None
    _, _, bw, bh = box
    assert bw >= 1.0
    assert bh >= 1.0


def test_bbox_from_mask_empty():
    assert _bbox_from_mask(np.zeros((5, 5), dtype=bool)) is None


def test_model_label_names_matches_coco_gt():
    cat_names = {
        1: "Lane-Markings",
        2: "Left Boundary -Dashed-",
        3: "Left Boundary -Solid-",
        4: "Right Boundary -Dashed-",
        5: "Right Boundary -Solid-",
    }
    labels = _model_label_names(cat_names)
    assert labels[0] == "Lane-Markings"
    assert labels[4] == "Right Boundary -Solid-"
    assert "4" not in labels.values()
