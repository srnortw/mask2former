# 04 — ONNX Export + INT8 Quantization

## What We Actually Built (Build Log)

### Pipeline

```
best_model.pth (PyTorch fp32)
  ↓
[Step 1] Export to ONNX fp32 — opset 16, TorchScript exporter, dynamo=False
  ↓
[Step 2] Verify — max diff PyTorch vs ONNX < 0.01
  ↓
[Step 3] Selective static INT8 — Conv/MatMul/Gemm only, val set calibration
  ↓
mask2former_fp32.onnx (284 MB) + mask2former_int8.onnx (82 MB)
  ↓
HuggingFace Hub + Google Drive backup
```

---

### Key Decisions Made During Implementation

**Opset 16 (not 17 or 13)**
- Opset 13: failed — some Mask2Former ops require opset 14+
- Opset 17+: triggers dynamo exporter → detached initializers → breaks quantizer
- Opset 16: TorchScript exporter, all ops supported, quantization compatible

**`dynamo=False` in `torch.onnx.export`**
- Even with opset 16, newer PyTorch versions try the dynamo exporter if `onnxscript` is installed
- `dynamo=False` forces TorchScript exporter unconditionally

**ONNX Wrapper for HuggingFace model**
- HuggingFace models use dict kwargs and return dataclass objects — ONNX tracing can't handle these
- `Mask2FormerONNXWrapper` accepts a plain tensor and returns `(masks_queries_logits, class_queries_logits)`

**Selective static INT8 (not full static, not dynamic)**

| Approach | Tried | Result |
|---|---|---|
| Full static (all ops) | ❌ | `RuntimeError: Invalid model with unknown initializers` — deformable attention produces intermediate tensors the quantizer can't resolve |
| Dynamic INT8 | ❌ | 0.8x speedup (slower) — Colab CPU lacks Intel VNNI instructions |
| Selective static (Conv/MatMul/Gemm) | ✅ | Works — backbone quantized, attention stays fp32 |

**Correct type pairing (per Q-ViT / FQ-ViT literature)**
- `weight_type = QInt8` — weights can be negative
- `activation_type = QUInt8` — post-Softmax/GELU values are in [0,1], unsigned; QInt8 wastes half its range on negatives
- Previous incorrect attempt used QInt8 for both

**Val set as calibration data**
- `_get_calibration_dir()` auto-selects: val → test → train
- Val set (318 images, capped at 200) — domain-specific, not seen during weight updates
- Better than random samples or ImageNet — activation ranges match real inference

---

### Bugs Fixed During Build

| Error | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: onnxscript` | New torch ONNX requires it | `pip install onnxscript` added to Cell 14 |
| `RuntimeError: Invalid model with unknown initializers` | Dynamo exporter detaches initializers | `dynamo=False` + opset 16 |
| `Exception: Incomplete symbolic shape inference` | Swin window attention has dynamic shapes | Skip deformable attention nodes entirely |
| `TypeError: quantize_static() got unexpected arg optimize_model` | Removed in recent onnxruntime | Removed the argument |
| Benchmark 0.8–0.9x speedup | Colab CPU lacks Intel VNNI | Expected — real speedup on Jetson/VNNI hardware |

---

### Source Files Written

| File | Purpose |
|---|---|
| `src/export_onnx.py` | ONNX wrapper, fp32 export, verify |
| `src/quantize_int8.py` | `MaskCalibrationReader`, selective static INT8, benchmark |

---

### Results

```
Export:
  mask2former_fp32.onnx: 284.0 MB
  Max diff PyTorch vs ONNX: 0.0007 (masks), 0.000003 (logits) ✅

Quantization (selective static, Conv/MatMul/Gemm, QInt8/QUInt8):
  mask2former_int8.onnx: 82.4 MB  (3.4x smaller)

Benchmark on Colab CPU:
  fp32: ~1275 ms/image
  INT8: ~1375 ms/image  ← no speedup on x86 without Intel VNNI
  Size: 3.4x reduction ← real benefit for storage + transfer
```

**Note on CPU speedup:** INT8 speedup only materializes on hardware with native INT8 instructions:

| Hardware | Expected INT8 speedup |
|---|---|
| Intel with VNNI (Ice Lake+) | 2–4x |
| ARM Cortex-A with NEON | 2–3x |
| Jetson Nano/Xavier + TensorRT | 3–6x |
| x86 without VNNI (Colab CPU) | ~1x (no gain) |

---

### Colab Cells

| Cell | Purpose |
|---|---|
| 14 | Install `onnx onnxruntime onnxruntime-tools onnxscript` |
| 15 | Download `best_model.pth` from HF Hub (after restart) |
| 16 | Export fp32 ONNX (opset 16, `dynamo=False`) + verify |
| 17 | Selective static INT8 + benchmark |
| 18 | Push `fp32.onnx` + `int8.onnx` → HF Hub + Drive |

---

### How to Run Inference with the INT8 Model

```python
import onnxruntime as ort
import numpy as np
import cv2

session = ort.InferenceSession(
    "checkpoints/mask2former_int8.onnx",
    providers=["CPUExecutionProvider"],
)

img = cv2.imread("image.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (512, 512)).astype(np.float32) / 255.0
img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
inp = img.transpose(2, 0, 1)[np.newaxis]  # NCHW

masks_logits, class_logits = session.run(None, {"pixel_values": inp})
# masks_logits: [1, 100, H, W]  — 100 query masks
# class_logits: [1, 100, num_classes+1]  — class scores per query
```

---

## Summary

| Step | Tool | Output |
|---|---|---|
| Export fp32 | `torch.onnx.export` opset 16, `dynamo=False` | `mask2former_fp32.onnx` (284 MB) |
| Verify | `onnxruntime` | max diff < 0.001 ✅ |
| Quantize | selective static QInt8/QUInt8, Conv/MatMul/Gemm | `mask2former_int8.onnx` (82 MB) |
| Storage | HF Hub + Google Drive | versioned artifacts |

**Next:** [05 — Model Registry](05_model_registry.md)
