#!/usr/bin/env python3
"""Send an image to FastAPI /predict and save an overlay with masks + boxes."""

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
    parser = argparse.ArgumentParser(
        description="POST image to /predict and save visualization.",
        epilog=(
            "Examples:\n"
            "  .venv/bin/python scripts/visualize_predict.py in.jpg out.jpg\n"
            "  .venv/bin/python scripts/visualize_predict.py in.jpg reports/my_result.jpg "
            "--conf-threshold 0.65\n"
            "  ./scripts/predict_image.sh in.jpg out.jpg"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input_image",
        type=Path,
        help="Input image file (sent to the API)",
    )
    parser.add_argument(
        "output_image",
        nargs="?",
        type=Path,
        default=Path("reports/predict_overlay.jpg"),
        metavar="OUTPUT",
        help="Output overlay path (default: reports/predict_overlay.jpg)",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output_opt",
        type=Path,
        help="Same as OUTPUT positional (optional alternative)",
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8000/predict",
        help="Predict endpoint URL",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=None,
        help="Override score threshold for this request (server default is 0.5)",
    )
    args = parser.parse_args()

    output = args.output_opt or args.output_image
    if not args.input_image.is_file():
        print(f"Input not found: {args.input_image}", file=sys.stderr)
        sys.exit(1)

    params = {}
    if args.conf_threshold is not None:
        params["conf_threshold"] = args.conf_threshold

    with open(args.input_image, "rb") as f:
        resp = requests.post(args.url, files={"file": f}, params=params, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    img = cv2.imread(str(args.input_image))
    if img is None:
        print(f"Could not read image: {args.input_image}", file=sys.stderr)
        sys.exit(1)

    overlay = img.copy()
    h, w = img.shape[:2]

    for inst in data["instances"]:
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

    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), overlay)

    print(f"Input:      {args.input_image.resolve()}")
    print(f"Instances:  {len(data['instances'])}")
    print(f"Inference:  {data['inference_ms']:.1f} ms")
    print(f"Saved:      {output.resolve()}")


if __name__ == "__main__":
    main()
