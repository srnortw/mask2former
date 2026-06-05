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
| `src/evaluate_onnx.py` | Fast pycocotools mAP on **valid + test** (Colab + CLI) |
| `src/fiftyone_onnx_review.py` | Local FiftyOne visual QA (25 samples/split, optional App) |

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

### When to evaluate what (recommendation)

| Stage | Tool | Split | Purpose |
|-------|------|-------|---------|
| **Phase 3 training** | `src/evaluate.py` (PyTorch + pycocotools) | **valid** | Pick `best_model.pth`, MLflow mAP — **keep as-is** |
| **Phase 4 after Cells 16–17** | `evaluate_onnx.py` (pycocotools) | **valid + test** | Confirm fp32 export + INT8 before HF push |

**Cell 18** — `run_phase4_report()`: mAP, AP50, AP75, AR, per-class AP, plots + JSON → Drive `checkpoints/eval_reports/`.  
**Cell 19** — `fiftyone_onnx_review.py`: visual **ground_truth vs predictions** on Drive `data/raw/{valid,test}`.

**Order in Colab:** Cell 16 (fp32) → Cell 17 (INT8) → Cell 18 (mAP) → **Cell 19 (FiftyOne)** → Cell 20 (push to HF).

---

### Colab Cells

| Cell | Purpose |
|---|---|
| 14 | Install `onnx onnxruntime onnxruntime-tools onnxscript` |
| 15 | Download `best_model.pth` from HF Hub (after restart) |
| 16 | Export fp32 ONNX (opset 16, `dynamo=False`) + verify |
| 17 | Selective static INT8 + benchmark |
| 18 | **mAP eval:** valid + test (`evaluate_onnx.py`) |
| 19 | **FiftyOne:** ground_truth vs predictions_fp32/int8 (25 samples/split) |
| 20 | Push `fp32.onnx` + `int8.onnx` → HF Hub + Drive |

**Cell 18** (`run_phase4_report` — data on Drive `DATA_DIR/raw`):

```python
from src.evaluate_onnx import run_phase4_report

phase4_report = run_phase4_report(
    fp32_onnx_path=os.path.join(CKPT, "mask2former_fp32.onnx"),
    int8_onnx_path=os.path.join(CKPT, "mask2former_int8.onnx"),
    raw_dir=os.path.join(os.environ["DATA_DIR"], "raw"),  # Drive images
    splits=["valid", "test"],
    report_dir=os.path.join(CKPT, "eval_reports"),
    max_samples=50,
    show_plots=True,
)
# → metrics_by_split.png, per_class_ap_valid.png, per_class_ap_test.png, phase4_eval_report.json
```

Local CLI (no FiftyOne required):

```bash
.venv/bin/python -m src.evaluate_onnx \
  --fp32 checkpoints/mask2former_fp32.onnx \
  --int8 checkpoints/mask2former_int8.onnx \
  --splits valid test
```

**Cell 19** (`src/fiftyone_onnx_review.py`):

```python
from src.fiftyone_onnx_review import run_visual_review

fo_result = run_visual_review(
    fp32_onnx_path=fp32_path,
    int8_onnx_path=int8_path,
    splits=["valid", "test"],
    raw_dir=RAW,
    max_samples_per_split=25,
    launch_app=True,
)
fo_session = fo_result["session"]  # keep alive in Colab
```

Local CLI (same tool):

```bash
.venv/bin/pip install fiftyone
.venv/bin/python -m src.fiftyone_onnx_review \
  --fp32 checkpoints/mask2former_fp32.onnx \
  --int8 checkpoints/mask2former_int8.onnx \
  --splits valid test \
  --max-samples 25
```

---

### Evaluation details

- **Splits:** `valid`, `test` under `data/raw/` (Drive: `mask2former-mlops/data/raw/{valid,test}`).
- **Metrics:** COCO mask mAP via `pycocotools` — aligned with `evaluate.py` (1-indexed `category_id`, same threshold/postprocess).
- **Speed:** No FiftyOne overhead; direct cv2 read + ONNX + RLE (no base64 roundtrip).
- **FiftyOne (Cell 19):** Colab + local; 25 samples/split; labels from COCO JSON; App fields: `ground_truth`, `predictions_fp32`, `predictions_int8`.

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
| Evaluate | `evaluate_onnx.py` (valid + test) | mAP report; optional local FiftyOne visual QA |
| Storage | HF Hub + Google Drive | versioned artifacts |

**Next:** [05 — Model Registry](05_model_registry.md)
