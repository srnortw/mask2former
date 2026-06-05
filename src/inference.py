"""
ONNX inference helpers for Mask2Former (serving + ROS2).
Matches postprocessing in evaluate.py, using numpy/onnxruntime only.
"""

from __future__ import annotations

import base64
import os
from typing import Any

import cv2
import numpy as np

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Roboflow lane dataset class order (0-indexed, matches training labels)
DEFAULT_CATEGORIES = {
    0: "Lane-Markings",
    1: "Left-Boundary-Dashed",
    2: "Left-Boundary-Solid",
    3: "Right-Boundary-Dashed",
    4: "Right-Boundary-Solid",
}


def load_categories() -> dict[int, str]:
    """Optional JSON env CATEGORIES='{"0":"Lane-Markings",...}'"""
    raw = os.environ.get("CATEGORIES")
    if not raw:
        return DEFAULT_CATEGORIES.copy()
    import json

    parsed = json.loads(raw)
    return {int(k): v for k, v in parsed.items()}


def create_session(model_path: str):
    import onnxruntime as ort

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    available = ort.get_available_providers()
    providers = [p for p in providers if p in available]
    if not providers:
        providers = ["CPUExecutionProvider"]
    return ort.InferenceSession(model_path, providers=providers)


def resolve_model_path(
    hf_repo_id: str,
    model_file: str,
    hf_token: str | None = None,
    local_path: str | None = None,
) -> str:
    if local_path and os.path.isfile(local_path):
        return local_path
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id=hf_repo_id,
        filename=model_file,
        token=hf_token or None,
    )


def preprocess_image_rgb(
    img_rgb: np.ndarray,
    img_size: int = 512,
) -> tuple[np.ndarray, int, int]:
    """Resize + ImageNet normalize an RGB image for ONNX input."""
    orig_h, orig_w = img_rgb.shape[:2]
    resized = cv2.resize(img_rgb, (img_size, img_size))
    normalized = resized.astype(np.float32) / 255.0
    normalized = (normalized - IMAGENET_MEAN) / IMAGENET_STD
    tensor = normalized.transpose(2, 0, 1)[np.newaxis].astype(np.float32)
    return tensor, orig_h, orig_w


def preprocess_image_bytes(
    image_bytes: bytes,
    img_size: int = 512,
) -> tuple[np.ndarray, int, int]:
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image bytes")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return preprocess_image_rgb(img, img_size)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x.astype(np.float64), -50.0, 50.0)
    return (1.0 / (1.0 + np.exp(-x))).astype(np.float32)


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x.astype(np.float64)
    x = x - np.max(x, axis=axis, keepdims=True)
    x = np.clip(x, -50.0, 50.0)
    e = np.exp(x)
    return (e / np.sum(e, axis=axis, keepdims=True)).astype(np.float32)


def postprocess_instances(
    masks_queries_logits: np.ndarray,
    class_queries_logits: np.ndarray,
    orig_h: int,
    orig_w: int,
    score_threshold: float = 0.5,
    categories: dict[int, str] | None = None,
) -> list[dict[str, Any]]:
    """
    Convert ONNX outputs to instance dicts (JSON-serializable).
    Input shapes: [1, Q, H, W] and [1, Q, num_classes+1]
    """
    categories = categories or DEFAULT_CATEGORIES

    masks = _sigmoid(masks_queries_logits)[0]          # [Q, H, W]
    logits = class_queries_logits[0]                   # [Q, C+1]
    probs = _softmax(logits[:, :-1], axis=-1)         # drop no-object class
    scores = probs.max(axis=-1)
    labels = probs.argmax(axis=-1)

    instances = []
    instance_id = 0

    for q in range(masks.shape[0]):
        score = float(scores[q])
        if not np.isfinite(score) or score < score_threshold:
            continue

        mask_resized = cv2.resize(
            masks[q].astype(np.float32),
            (orig_w, orig_h),
            interpolation=cv2.INTER_LINEAR,
        )
        binary = (mask_resized > 0.5).astype(np.uint8)

        ys, xs = np.where(binary > 0)
        if len(xs) == 0:
            continue

        x0, y0 = int(xs.min()), int(ys.min())
        x1, y1 = int(xs.max()), int(ys.max())
        # min 1px — avoids zero-area boxes in FiftyOne IoU eval (thin lane masks)
        bbox = [float(x0), float(y0), max(1.0, float(x1 - x0 + 1)), max(1.0, float(y1 - y0 + 1))]

        label = int(labels[q])
        instances.append({
            "instance_id": instance_id,
            "category_id": label,
            "category_name": categories.get(label, str(label)),
            "score": score,
            "mask_b64": base64.b64encode(binary.tobytes()).decode("utf-8"),
            "mask_shape": [orig_h, orig_w],
            "bbox": bbox,
        })
        instance_id += 1

    return instances


def postprocess_to_coco_results(
    masks_queries_logits: np.ndarray,
    class_queries_logits: np.ndarray,
    orig_h: int,
    orig_w: int,
    image_id: int,
    score_threshold: float = 0.5,
) -> list[dict[str, Any]]:
    """
    Convert ONNX outputs to COCO detection results (segmentation RLE).
    category_id is 1-indexed to match Roboflow COCO annotations.
    """
    from pycocotools import mask as coco_mask

    masks = _sigmoid(masks_queries_logits)[0]
    logits = class_queries_logits[0]
    probs = _softmax(logits[:, :-1], axis=-1)
    scores = probs.max(axis=-1)
    labels = probs.argmax(axis=-1)

    results = []
    for q in range(masks.shape[0]):
        score = float(scores[q])
        if not np.isfinite(score) or score < score_threshold:
            continue

        mask_resized = cv2.resize(
            masks[q].astype(np.float32),
            (orig_w, orig_h),
            interpolation=cv2.INTER_LINEAR,
        )
        binary = (mask_resized > 0.5).astype(np.uint8)
        if not binary.any():
            continue

        rle = coco_mask.encode(np.asfortranarray(binary))
        rle["counts"] = rle["counts"].decode("utf-8")

        results.append({
            "image_id": image_id,
            "category_id": int(labels[q]) + 1,
            "segmentation": rle,
            "score": score,
        })

    return results


def run_inference(
    session,
    image_bytes: bytes,
    img_size: int = 512,
    score_threshold: float = 0.5,
    categories: dict[int, str] | None = None,
) -> tuple[list[dict[str, Any]], float]:
    import time

    tensor, orig_h, orig_w = preprocess_image_bytes(image_bytes, img_size)

    t0 = time.perf_counter()
    masks_logits, class_logits = session.run(
        None,
        {"pixel_values": tensor},
    )
    inference_ms = (time.perf_counter() - t0) * 1000.0

    instances = postprocess_instances(
        masks_logits,
        class_logits,
        orig_h,
        orig_w,
        score_threshold=score_threshold,
        categories=categories,
    )
    return instances, inference_ms
