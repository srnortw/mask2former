# 05 — Model Registry

## Overview

After training and quantization, models are registered in two places:

```
checkpoints/best_model.pth          → MLflow Model Registry   (training artifacts)
checkpoints/mask2former_fp32.onnx   → Hugging Face Hub        (deployment artifacts)
checkpoints/mask2former_int8.onnx   → Hugging Face Hub        (deployment artifacts)
```

| Registry | Purpose | Access |
|---|---|---|
| **MLflow Model Registry** | Link models to experiments, track stages (Staging/Production) | Internal team |
| **Hugging Face Hub** | Store deployment-ready ONNX models, version them publicly or privately | Any machine, Colab, ROS2 |

---

## 1. MLflow Model Registry

The MLflow tracking server is already running on Render.com from step 03.

### Register model from training run

```python
# src/train.py — add inside the mlflow.start_run() block after training

import mlflow.pytorch

# Log and register the PyTorch model
mlflow.pytorch.log_model(
    pytorch_model=model,
    artifact_path="mask2former_swin",
    registered_model_name="mask2former-instance-seg",   # creates/updates registry entry
)

# Also log ONNX artifacts
mlflow.log_artifact("checkpoints/mask2former_fp32.onnx", artifact_path="onnx")
mlflow.log_artifact("checkpoints/mask2former_int8.onnx", artifact_path="onnx")
```

### Transition model stages via CLI or Python

```python
from mlflow.tracking import MlflowClient

client = MlflowClient(tracking_uri="https://your-render-app.onrender.com")

# Get latest version
versions = client.get_latest_versions("mask2former-instance-seg")
latest_version = versions[0].version

# Promote to Staging after verification
client.transition_model_version_stage(
    name="mask2former-instance-seg",
    version=latest_version,
    stage="Staging",
)

# Promote to Production after full validation
client.transition_model_version_stage(
    name="mask2former-instance-seg",
    version=latest_version,
    stage="Production",
)
```

### Stage lifecycle

```
None → Staging → Production → Archived
         ↑           ↑
    after eval   after ROS2 test
```

---

## 2. Hugging Face Hub

Hugging Face Hub stores the ONNX models — accessible from any machine without DVC setup.
This is what the FastAPI server and ROS2 node will pull at startup.

### Setup (one time)

```bash
# Install in local venv
pip install huggingface_hub

# Login
huggingface-cli login
# Paste your HF token from https://huggingface.co/settings/tokens
```

### Push ONNX models from Colab

```python
# In Colab — after quantization
from huggingface_hub import HfApi, create_repo

api = HfApi()
repo_id = "your-hf-username/mask2former-instance-seg"

# Create repo (private by default)
create_repo(repo_id, repo_type="model", private=True, exist_ok=True)

# Upload models
api.upload_file(
    path_or_fileobj="checkpoints/mask2former_fp32.onnx",
    path_in_repo="mask2former_fp32.onnx",
    repo_id=repo_id,
)

api.upload_file(
    path_or_fileobj="checkpoints/mask2former_int8.onnx",
    path_in_repo="mask2former_int8.onnx",
    repo_id=repo_id,
)

# Upload model card with metadata
api.upload_file(
    path_or_fileobj="MODEL_CARD.md",
    path_in_repo="README.md",
    repo_id=repo_id,
)

print(f"Models available at: https://huggingface.co/{repo_id}")
```

### Model Card template (MODEL_CARD.md)

```markdown
---
language: en
tags:
  - instance-segmentation
  - mask2former
  - swin-transformer
  - onnx
  - robotics
  - ros2
license: apache-2.0
---

# Mask2Former Instance Segmentation

Custom Mask2Former with Swin-Tiny backbone, fine-tuned for [your use case].

## Model Details
- Backbone: Swin-Tiny
- Training: Google Colab Pro, 50 epochs, 1200 images
- Gradual freezing: 3 phases
- LR schedule: CosineAnnealingWarmRestarts

## Files
- `mask2former_fp32.onnx` — full precision, ONNX opset 17
- `mask2former_int8.onnx` — INT8 static quantization, ~4x smaller

## Performance
| Model | mAP | Size | CPU Inference |
|---|---|---|---|
| fp32 | XX.X | ~300 MB | ~300 ms |
| int8 | XX.X | ~75 MB  | ~90 ms  |
```

---

## 3. Download Models from Hub (any machine)

```python
# FastAPI server, ROS2 node, or new Colab session
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="your-hf-username/mask2former-instance-seg",
    filename="mask2former_int8.onnx",
    token="your-hf-token",          # only needed for private repos
)
print(f"Model downloaded to: {model_path}")
```

---

## 4. Version Tagging Strategy

Use git tags to link code version to model version:

```bash
# After pushing a new model to HF Hub
git tag -a v1.0 -m "first trained model, mAP=0.XX, swin-tiny, 50ep"
git push origin v1.0
```

In MLflow, add the git tag as a model tag:

```python
client.set_model_version_tag(
    name="mask2former-instance-seg",
    version=latest_version,
    key="git_tag",
    value="v1.0",
)
client.set_model_version_tag(
    name="mask2former-instance-seg",
    version=latest_version,
    key="hf_repo",
    value="your-hf-username/mask2former-instance-seg",
)
```

---

## 5. Complete Artifact Lineage

```
Roboflow dataset v1
  → DVC commit abc123 (data/processed)
    → MLflow run_id xyz789 (train metrics, .pth artifact)
      → MLflow model version 1 (Staging → Production)
        → HF Hub: mask2former_int8.onnx @ v1.0
          → Docker image: mask2former-api:v1.0
            → ROS2 deployment
```

---

## Summary

| Registry | What's stored | When to use |
|---|---|---|
| MLflow Model Registry | `.pth`, experiment link, stage | Track training runs, promote to prod |
| Hugging Face Hub | `.onnx` (fp32 + int8) | Download for serving and ROS2 |
| DVC + Google Drive | raw data, checkpoints | Full artifact versioning |
| Git tags | code version | Link to model version |

**Next:** [06 — Serving](06_serving.md)
