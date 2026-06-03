# 05 — Model Registry

## Overview

```
best_model.pth          → MLflow Model Registry (DagsHub)  ← training artifact
mask2former_fp32.onnx   → Hugging Face Hub                 ← deployment artifact
mask2former_int8.onnx   → Hugging Face Hub                 ← deployment artifact
MODEL_CARD.md           → HF Hub README                    ← model documentation
```

| Registry | Purpose | Access |
|---|---|---|
| **MLflow Model Registry** | Link models to experiments, stage lifecycle (Staging → Production) | DagsHub UI |
| **Hugging Face Hub** | Store deployment-ready ONNX models, download from any machine | FastAPI server, ROS2 node |

---

## What We Actually Built

### Source Files Written

| File | Purpose |
|---|---|
| `src/register_model.py` | MLflow registration, stage transitions, HF model card push |
| `MODEL_CARD.md` | HF Hub README with architecture, training, usage details |

### Key Decisions

**MLflow on DagsHub (not Render.com)**
- `tracking_uri = https://dagshub.com/srnortw/mask2former.mlflow`
- Auth via `MLFLOW_TRACKING_USERNAME` + `MLFLOW_TRACKING_PASSWORD` env vars

**`run_id` continuity across phases**
- Each training phase cell saves `mlflow_run_id` into `last_checkpoint.pth`
- `register_model.py` reads it automatically — no manual copy-paste needed

**Stage lifecycle**
```
Registered → Staging (auto) → Production (manual, after ROS2 validation)
```
- Auto-transitions to Staging on registration
- Promote to Production only after testing in ROS2 node

---

## 1. MLflow Model Registry

### Register model (Colab Cell 20)

```python
from register_model import register_in_mlflow, push_model_card
import torch

# Read run_id from checkpoint
last_ckpt = torch.load('checkpoints/last_checkpoint.pth', map_location='cpu')
run_id = last_ckpt['mlflow_run_id']

version = register_in_mlflow(
    run_id=run_id,
    checkpoint_path='checkpoints/best_model.pth',
    fp32_onnx_path='checkpoints/mask2former_fp32.onnx',
    int8_onnx_path='checkpoints/mask2former_int8.onnx',
)
# → Logs ONNX + .pth artifacts, then mlflow.pytorch.log_model(name="model")
# → Registers via models:/<model_id> (MLflow 3 LoggedModel; runs:/.../model does not work)
# → Tags version with backbone, num_classes, hf_repo, quantization
# → Auto-transitions to Staging
```

### Promote to Production (after ROS2 validation)

```python
from register_model import promote_to_production
promote_to_production(version='1')
# → Archives previous Production version
# → Sets this version as Production
```

### View in DagsHub

```
https://dagshub.com/srnortw/mask2former.mlflow/#/models
```

---

## 2. Hugging Face Hub

### What's stored

| File | When pushed | Cell |
|---|---|---|
| `best_model.pth` | After training | Cell 12 |
| `mask2former_fp32.onnx` | After ONNX export | Cell 18 |
| `mask2former_int8.onnx` | After quantization | Cell 18 |
| `README.md` (MODEL_CARD.md) | After model registry | Cell 20 |

### Repo URL
```
https://huggingface.co/srnortw/mask2former-lane-seg
```

### Download on any machine

```python
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="srnortw/mask2former-lane-seg",
    filename="mask2former_int8.onnx",
    token="your-hf-token",
)
```

---

## 3. Version Tagging

After a full 50-epoch training run is complete and validated:

```bash
# Tag the git commit that produced the model
git tag -a v1.0 -m "lane-seg model: swin-small, 50ep, mAP=X.XX"
git push origin v1.0
```

This links:
```
git tag v1.0
  → MLflow run_id (in last_checkpoint.pth)
    → MLflow model version 1 (Staging/Production)
      → HF Hub srnortw/mask2former-lane-seg
        → mask2former_int8.onnx (deployed to FastAPI + ROS2)
```

---

## 4. Complete Artifact Lineage

```
Roboflow dataset v1 (1610 images, 5 classes)
  → DVC + Google Drive (data/raw/)
    → MLflow experiment 'mask2former-swin'
      → run_id: enthused-bass-906 (train metrics, loss curves)
        → MLflow model 'mask2former-lane-seg' v1 (Staging → Production)
          → HF Hub: srnortw/mask2former-lane-seg
            → mask2former_int8.onnx (82 MB, selective static INT8)
              → FastAPI Docker container
                → ROS2 node /perception/instance_masks
```

---

## Summary

| Registry | What's stored | Stage |
|---|---|---|
| MLflow (DagsHub) | `.pth` + ONNX links, experiment connection | Staging → Production |
| HF Hub | `fp32.onnx`, `int8.onnx`, `MODEL_CARD.md` | Always latest |
| DVC + Drive | Raw data, all checkpoints | Version-locked |
| Git tags | Code version linked to model | `v1.0`, `v1.1`, ... |

**Next:** [06 — Serving](06_serving.md)
