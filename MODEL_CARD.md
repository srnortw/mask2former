---
language: en
tags:
  - instance-segmentation
  - mask2former
  - swin-transformer
  - onnx
  - lane-detection
  - robotics
  - ros2
license: apache-2.0
---

# Mask2Former — Lane Detection Instance Segmentation

Custom Mask2Former with Swin-Small backbone, fine-tuned for lane marking instance segmentation.
Trained as part of a personal MLOps pipeline for ROS2 robot perception.

## Model Details

| Property | Value |
|---|---|
| Architecture | Mask2Former |
| Backbone | Swin-Small (69M params) |
| Pretrained weights | `facebook/mask2former-swin-small-coco-instance` |
| Task | Instance segmentation |
| Classes | 5 (lane markings: left/right boundary dashed/solid) |
| Input size | 512 × 512 |
| Framework | HuggingFace `transformers` |
| Training strategy | 3-phase gradual freezing + CosineAnnealingWarmRestarts |

## Training Details

| Phase | Epochs | Trainable | LR |
|---|---|---|---|
| 1 | 0–15 | Transformer decoder only | 1e-4 |
| 2 | 15–30 | + Pixel decoder | 5e-5 |
| 3 | 30–50 | Full network | 1e-5 |

- **Dataset:** Lane Detection Segmentation (Roboflow, 1610 images, 5 classes)
- **Compute:** Google Colab Pro (T4 GPU)
- **Augmentation:** Albumentations (flip, affine, color jitter, gaussian noise)
- **Experiment tracking:** MLflow → DagsHub

## Files

| File | Description | Size |
|---|---|---|
| `best_model.pth` | PyTorch checkpoint (best mAP) | ~270 MB |
| `mask2former_fp32.onnx` | ONNX export, opset 16 | 284 MB |
| `mask2former_int8.onnx` | Selective static INT8 (Conv/MatMul/Gemm) | 82 MB |

## Quantization

Selective static INT8 quantization:
- `weight_type = QInt8` — weights can be negative
- `activation_type = QUInt8` — post-Softmax/GELU values are in [0,1]
- Only `Conv`, `MatMul`, `Gemm` ops quantized (deformable attention stays fp32)
- Calibrated on validation set (318 domain-specific images)
- Size: 3.4x smaller than fp32

## Usage

```python
import onnxruntime as ort
import numpy as np
import cv2

session = ort.InferenceSession(
    "mask2former_int8.onnx",
    providers=["CPUExecutionProvider"],
)

img = cv2.imread("image.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (512, 512)).astype(np.float32) / 255.0
img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
inp = img.transpose(2, 0, 1)[np.newaxis]  # NCHW

masks_logits, class_logits = session.run(None, {"pixel_values": inp})
# masks_logits: [1, 100, H, W]
# class_logits: [1, 100, num_classes+1]

scores = class_logits[0].softmax(-1)[:, :-1].max(-1)
keep = scores > 0.5
```

## Download

```python
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="srnortw/mask2former-lane-seg",
    filename="mask2former_int8.onnx",
    token="your-hf-token",  # only needed for private repos
)
```
