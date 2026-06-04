#!/usr/bin/env python3
"""Call local FastAPI /predict and save an overlay visualization."""

import argparse
import base64
import sys
from pathlib import Path

import cv2
import numpy as np
import requests

COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 128, 255),
    (255, 128, 0),
    (255, 0, 255),
    (0, 255, 255),
]


def main():
    parser = argparse.ArgumentParser(description="Visualize /predict API output")
    parser.add_argument("image", type=Path, help="Input image path")
    parser.add_argument(
        "--url",
        default="http://localhost:8000/predict",
        help="Predict endpoint URL",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("reports/predict_overlay.jpg"),
        help="Output image path",
    )
    args = parser.parse_args()

    if not args.image.is_file():
        print(f"Image not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    with open(args.image, "rb") as f:
        resp = requests.post(args.url, files={"file": f}, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    img = cv2.imread(str(args.image))
    if img is None:
        print(f"Could not read image: {args.image}", file=sys.stderr)
        sys.exit(1)

    overlay = img.copy()
    h, w = img.shape[:2]

    for i, inst in enumerate(data["instances"]):
        mask_h, mask_w = inst["mask_shape"]
        raw = base64.b64decode(inst["mask_b64"])
        mask = np.frombuffer(raw, dtype=np.uint8).reshape(mask_h, mask_w)
        if mask.shape != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        color = COLORS[inst["category_id"] % len(COLORS)]
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

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), overlay)

    print(f"Instances: {len(data['instances'])}")
    print(f"Inference:  {data['inference_ms']:.1f} ms")
    print(f"Saved:       {args.output.resolve()}")


if __name__ == "__main__":
    main()
