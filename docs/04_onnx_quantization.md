# 04 — ONNX Export + INT8 Quantization

## Overview

```
best_model.pth (PyTorch fp32)
  ↓
[Step 1] Export to ONNX (fp32)
  ↓
[Step 2] Verify ONNX model (check outputs match PyTorch)
  ↓
[Step 3] Static INT8 Quantization (calibration with ~200 images)
  ↓
[Step 4] FiftyOne — visual comparison: fp32 vs INT8 masks
  ↓
mask2former_int8.onnx  →  FastAPI serving  →  ROS2 node
```

### Why INT8 for ROS2?

| Model | Size | Inference (CPU) | mAP |
|---|---|---|---|
| PyTorch fp32 | ~200-400 MB | slow | baseline |
| ONNX fp32 | ~200-400 MB | ~2x faster | same |
| ONNX INT8 | ~50-100 MB | ~3-4x faster | -1 to 3% |

INT8 fits on embedded hardware (Jetson, NUC) used in ROS2 robots.

---

## 1. Export to ONNX (fp32) — run in Colab after training

```python
# src/export_onnx.py
import torch
import torch.onnx
from models.mask2former import build_model_from_checkpoint


def export_to_onnx(
    checkpoint_path: str,
    output_path: str = "checkpoints/mask2former_fp32.onnx",
    img_size: int = 512,
    opset_version: int = 17,
):
    model = build_model_from_checkpoint(checkpoint_path)
    model.eval()
    model.cuda()

    # Dummy input — same shape as training images
    dummy_input = torch.randn(1, 3, img_size, img_size).cuda()

    print("Exporting to ONNX...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=opset_version,
        input_names=["image"],
        output_names=["pred_masks", "pred_logits"],
        dynamic_axes={
            "image":       {0: "batch_size"},
            "pred_masks":  {0: "batch_size"},
            "pred_logits": {0: "batch_size"},
        },
        export_params=True,
        do_constant_folding=True,  # optimize constants at export time
    )
    print(f"Exported: {output_path}")
    return output_path
```

---

## 2. Verify ONNX Export

```python
# src/export_onnx.py (continued)
import onnx
import onnxruntime as ort
import numpy as np


def verify_onnx(pytorch_model, onnx_path: str, img_size: int = 512, tolerance: float = 1e-3):
    """Check ONNX outputs match PyTorch outputs."""

    dummy = torch.randn(1, 3, img_size, img_size)

    # PyTorch output
    pytorch_model.eval()
    with torch.no_grad():
        pt_out = pytorch_model(dummy.cuda())

    # ONNX output
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ort_out = session.run(None, {"image": dummy.numpy()})

    # Compare
    pt_masks = pt_out["pred_masks"].cpu().numpy()
    ort_masks = ort_out[0]

    max_diff = np.max(np.abs(pt_masks - ort_masks))
    print(f"Max difference (PyTorch vs ONNX): {max_diff:.6f}")

    if max_diff < tolerance:
        print("ONNX export verified.")
    else:
        print(f"WARNING: difference {max_diff:.6f} exceeds tolerance {tolerance}")

    return max_diff
```

---

## 3. INT8 Static Quantization

Static quantization requires a **calibration dataset** to compute activation ranges.
We prepared `data/calibration/` (200 images) in step 01.

```python
# src/quantize_int8.py
import numpy as np
import cv2
import onnxruntime
from onnxruntime.quantization import (
    quantize_static,
    CalibrationDataReader,
    QuantFormat,
    QuantType,
)


class MaskCalibrationReader(CalibrationDataReader):
    """Feed calibration images to compute INT8 activation ranges."""

    def __init__(self, calibration_dir: str, img_size: int = 512, n_images: int = 200):
        import os
        self.img_size = img_size
        self.images = [
            os.path.join(calibration_dir, f)
            for f in os.listdir(calibration_dir)
            if f.endswith(('.jpg', '.png'))
        ][:n_images]
        self.idx = 0
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def get_next(self):
        if self.idx >= len(self.images):
            return None

        img = cv2.imread(self.images[self.idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = img.transpose(2, 0, 1)           # HWC → CHW
        img = np.expand_dims(img, axis=0)      # add batch dim

        self.idx += 1
        return {"image": img}


def quantize_int8(
    fp32_onnx_path: str = "checkpoints/mask2former_fp32.onnx",
    int8_onnx_path: str = "checkpoints/mask2former_int8.onnx",
    calibration_dir: str = "data/calibration",
    img_size: int = 512,
):
    print("Starting INT8 static quantization...")
    print(f"Input:  {fp32_onnx_path}")
    print(f"Output: {int8_onnx_path}")

    calibration_reader = MaskCalibrationReader(calibration_dir, img_size)

    quantize_static(
        model_input=fp32_onnx_path,
        model_output=int8_onnx_path,
        calibration_data_reader=calibration_reader,
        quant_format=QuantFormat.QOperator,    # full INT8 operators
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        optimize_model=True,
    )

    print(f"INT8 model saved: {int8_onnx_path}")

    # Report size reduction
    import os
    fp32_size = os.path.getsize(fp32_onnx_path) / 1e6
    int8_size = os.path.getsize(int8_onnx_path) / 1e6
    print(f"Size: {fp32_size:.1f} MB (fp32) → {int8_size:.1f} MB (INT8) | {fp32_size/int8_size:.1f}x smaller")
```

---

## 4. Benchmark INT8 vs fp32

```python
# src/quantize_int8.py (continued)
import time

def benchmark(onnx_path: str, img_size: int = 512, n_runs: int = 50):
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    dummy = np.random.randn(1, 3, img_size, img_size).astype(np.float32)

    # Warmup
    for _ in range(5):
        session.run(None, {"image": dummy})

    # Benchmark
    start = time.perf_counter()
    for _ in range(n_runs):
        session.run(None, {"image": dummy})
    elapsed = (time.perf_counter() - start) / n_runs * 1000

    print(f"{onnx_path.split('/')[-1]}: {elapsed:.1f} ms/image")
    return elapsed
```

---

## 5. FiftyOne — Visual Evaluation fp32 vs INT8

Run this locally after pulling the ONNX models from Drive:

```python
# src/evaluate_onnx.py
import fiftyone as fo
import fiftyone.utils.coco as fouc
import onnxruntime as ort
import numpy as np
import cv2


def run_onnx_inference(session, img_path: str, img_size: int = 512):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    img_resized = cv2.resize(img, (img_size, img_size))
    img_norm = (img_resized.astype(np.float32) / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    inp = img_norm.transpose(2, 0, 1)[np.newaxis]

    pred_masks, pred_logits = session.run(None, {"image": inp})
    return pred_masks, pred_logits, (h, w)


def compare_fp32_vs_int8(
    fp32_path: str,
    int8_path: str,
    val_dir: str = "data/processed/valid",
):
    dataset = fo.Dataset("fp32_vs_int8_comparison")
    fp32_session = ort.InferenceSession(fp32_path)
    int8_session = ort.InferenceSession(int8_path)

    import os
    for img_file in os.listdir(f"{val_dir}/images")[:50]:  # compare on 50 images
        img_path = f"{val_dir}/images/{img_file}"
        sample = fo.Sample(filepath=img_path)

        fp32_masks, fp32_logits, (h, w) = run_onnx_inference(fp32_session, img_path)
        int8_masks, int8_logits, _      = run_onnx_inference(int8_session, img_path)

        sample["fp32_predictions"] = fo.Detections(
            detections=masks_to_fo_detections(fp32_masks, fp32_logits, h, w)
        )
        sample["int8_predictions"] = fo.Detections(
            detections=masks_to_fo_detections(int8_masks, int8_logits, h, w)
        )
        dataset.add_sample(sample)

    session = fo.launch_app(dataset)
    # In UI: toggle between fp32_predictions and int8_predictions
    # to visually compare mask quality
    return session
```

---

## 6. Run in Colab

```python
# Colab — after training
from export_onnx import export_to_onnx, verify_onnx
from quantize_int8 import quantize_int8, benchmark

# Step 1: Export
export_to_onnx("checkpoints/best_model.pth", "checkpoints/mask2former_fp32.onnx")

# Step 2: Verify
verify_onnx(model, "checkpoints/mask2former_fp32.onnx")

# Step 3: Quantize
quantize_int8(
    fp32_onnx_path="checkpoints/mask2former_fp32.onnx",
    int8_onnx_path="checkpoints/mask2former_int8.onnx",
    calibration_dir="data/calibration",
)

# Step 4: Benchmark
benchmark("checkpoints/mask2former_fp32.onnx")
benchmark("checkpoints/mask2former_int8.onnx")

# Step 5: Push to Drive
!dvc add checkpoints/mask2former_fp32.onnx checkpoints/mask2former_int8.onnx
!dvc push
!git add checkpoints/*.dvc
!git commit -m "model: add fp32 and int8 onnx exports"
!git push
```

---

## 7. Expected Results

```
mask2former_fp32.onnx:  ~300 ms/image (CPU)  | ~300 MB
mask2former_int8.onnx:  ~90 ms/image  (CPU)  | ~75 MB  | mAP drop ~1-2%
```

INT8 at ~90ms is fast enough for a ROS2 node running at ~10 Hz.

---

## Summary

| Step | Tool | Output |
|---|---|---|
| Export fp32 | `torch.onnx.export` | `mask2former_fp32.onnx` |
| Verify | `onnxruntime` | max diff < 1e-3 |
| Calibrate + Quantize | `onnxruntime.quantization` | `mask2former_int8.onnx` |
| Visual eval | FiftyOne | fp32 vs INT8 comparison |
| Storage | DVC → Google Drive | versioned ONNX artifacts |

**Next:** [05 — Model Registry](05_model_registry.md)
