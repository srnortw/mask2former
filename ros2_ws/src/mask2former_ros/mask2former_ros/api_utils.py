"""Decode /predict JSON and build mask / visualization images."""

from __future__ import annotations

import base64
from typing import Any

import cv2
import numpy as np

COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 128, 255),
    (255, 128, 0),
    (255, 0, 255),
    (0, 255, 255),
]


def decode_instance_masks(
    instances: list[dict[str, Any]],
    height: int,
    width: int,
) -> list[dict[str, Any]]:
    """Expand base64 masks from API response to numpy arrays."""
    decoded = []
    for inst in instances:
        mask_h, mask_w = inst["mask_shape"]
        raw = base64.b64decode(inst["mask_b64"])
        mask = np.frombuffer(raw, dtype=np.uint8).reshape(mask_h, mask_w)
        if mask.shape != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        decoded.append({**inst, "mask": mask})
    return decoded


def build_combined_mask(
    instances: list[dict[str, Any]],
    height: int,
    width: int,
) -> np.ndarray:
    """Single-channel mask: pixel value = instance_id + 1 (0 = background)."""
    combined = np.zeros((height, width), dtype=np.uint8)
    for inst in instances:
        mask = inst["mask"]
        combined[mask > 0] = int(inst["instance_id"]) + 1
    return combined


def build_visualization(
    bgr_image: np.ndarray,
    instances: list[dict[str, Any]],
) -> np.ndarray:
    """Color overlay + bbox labels (BGR in/out, same as visualize_predict.py)."""
    overlay = bgr_image.copy()
    h, w = overlay.shape[:2]

    for inst in instances:
        mask = inst["mask"]
        if mask.shape != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        color = COLORS[int(inst["category_id"]) % len(COLORS)]
        colored = np.zeros_like(overlay)
        colored[mask > 0] = color
        overlay = cv2.addWeighted(overlay, 1.0, colored, 0.45, 0)

        x, y, bw, bh = [int(v) for v in inst["bbox"]]
        label = f"{inst['category_name']} {inst['score']:.2f}"
        cv2.rectangle(overlay, (x, y), (x + bw, y + bh), color, 2)
        cv2.putText(
            overlay,
            label,
            (x, max(y - 6, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    return overlay
