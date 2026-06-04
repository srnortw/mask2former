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

**MLflow 3 on Colab (DagsHub)**
- Colab ships **MLflow 3.x** — model registry API changed vs MLflow 2
- `log_artifact("best_model.pth")` alone **cannot** be registered
- Must use `mlflow.pytorch.log_model(..., name="model")` → register via `models:/<model_id>` URI
- `register_model.py` version banner: `mlflow3-v2` (confirm Cell 4 git sync loaded latest code)

---

## What We Actually Did (Build Log)

### Successful registration (Colab Cell 20)

| Item | Value |
|------|--------|
| MLflow run | `4008ed1b292a4c1589e95daebf99d760` (`brawny-shoat-230`) |
| Experiment | `mask2former-swin` on DagsHub |
| Registered model | `mask2former-lane-seg` **version 1** |
| Stage | **Staging** (auto) |
| LoggedModel URI | `models:/m-df4feed12bbf4699934fe5e90f091a5b` |
| HF model card | `README.md` uploaded to `srnortw/mask2former-lane-seg` |

Artifacts logged into the **existing training run** (not a new run):

| Artifact path | File |
|---------------|------|
| `onnx/fp32` | `mask2former_fp32.onnx` |
| `onnx/int8` | `mask2former_int8.onnx` |
| `checkpoints/` | `best_model.pth` |
| `model/` | PyTorch flavor (for registry) |

Paths on Drive: `/content/drive/MyDrive/mask2former-mlops/checkpoints/`

### Issues hit and fixes

| Error | Cause | Fix |
|-------|--------|-----|
| `Unable to find logged_model ... checkpoints` | Registered from `runs:/.../checkpoints` (raw `.pth` artifact) | Log with `mlflow.pytorch.log_model`, register from `model/` |
| `Unable to find logged_model ... model` + `artifact_path` deprecated | Used `artifact_path="model"` (MLflow 2 style) | Use `name="model"` + `registered_model_name` or `models:/` URI |
| Stale code in Colab kernel | Old `register_model.py` cached after `git pull` | Cell 20: `git pull` + `importlib.reload`; Cell 4: `reset --hard` |
| `class_predictor` MISMATCH on load | COCO 81-class head vs 5 lane classes | Expected — checkpoint weights still load correctly |

### Colab prerequisites for Cell 20

1. Cells **1–4** (secrets, Drive mount, `git sync`)
2. Cell **5** if first-time clone / Roboflow data
3. ONNX on Drive: `mask2former_fp32.onnx`, `mask2former_int8.onnx` (Cells 14–18)
4. `last_checkpoint.pth` with `mlflow_run_id` on Drive

---

## 1. MLflow Model Registry

### Register model (Colab Cell 20)

```python
# Cell 20 pulls latest register_model.py and reloads the module
import sys, os, torch, subprocess, importlib
ROOT = os.environ['PROJECT_ROOT']
sys.path.insert(0, f'{ROOT}/src')
os.chdir(ROOT)

subprocess.run(['git', '-C', ROOT, 'fetch', 'origin', 'main'], check=True)
subprocess.run(['git', '-C', ROOT, 'reset', '--hard', 'origin/main'], check=True)

import register_model as _reg
importlib.reload(_reg)
from register_model import register_in_mlflow, push_model_card, REGISTRY_CODE_VERSION

CKPT = os.environ['CHECKPOINT_DIR']
last_ckpt = torch.load(os.path.join(CKPT, 'last_checkpoint.pth'), map_location='cpu')
run_id = last_ckpt['mlflow_run_id']

version = register_in_mlflow(
    run_id=run_id,
    checkpoint_path=os.path.join(CKPT, 'best_model.pth'),
    fp32_onnx_path=os.path.join(CKPT, 'mask2former_fp32.onnx'),
    int8_onnx_path=os.path.join(CKPT, 'mask2former_int8.onnx'),
)

push_model_card(repo_id=os.environ['HF_REPO_ID'], hf_token=os.environ['HF_TOKEN'])
```

Expected output (abbreviated):

```
register_model.py [mlflow3-v2]
  Logged: .../mask2former_fp32.onnx → onnx/fp32
  Logged: .../mask2former_int8.onnx → onnx/int8
  Reusing logged model on this run: models:/m-...
Registered as version 1
Version 1 → Staging
Model card uploaded → https://huggingface.co/srnortw/mask2former-lane-seg
```

`transition_model_version_stage` may show a **FutureWarning** (MLflow 3 deprecates stages in favor of aliases) — Staging still works.

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

| File | When pushed | Colab cell |
|---|---|---|
| `best_model.pth` | After training | 13 |
| `mask2former_fp32.onnx` | After ONNX export | 18 |
| `mask2former_int8.onnx` | After quantization | 18 |
| `README.md` (`MODEL_CARD.md`) | After model registry | 20 |

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
  → Drive: mask2former-mlops/data/raw/
    → MLflow experiment 'mask2former-swin' (DagsHub)
      → run_id: 4008ed1b... (brawny-shoat-230, training ~epoch 18)
        → MLflow model 'mask2former-lane-seg' v1 (Staging)
          → HF Hub: srnortw/mask2former-lane-seg
            → mask2former_int8.onnx (~82 MB, selective static INT8)
              → Phase 06: FastAPI Docker (mask2former-api)
                → Phase 09: ROS2 /perception/instance_masks
```

---

## Summary

| Registry | What's stored | Status |
|---|---|---|
| MLflow (DagsHub) | `mask2former-lane-seg` v1, run artifacts, LoggedModel | **Staging** (done) |
| HF Hub | `fp32.onnx`, `int8.onnx`, `best_model.pth`, `README.md` | **Done** |
| Drive | `mask2former-mlops/checkpoints/` (outside git repo) | **Done** |
| Git tags | Code version linked to model | Optional (`v1.0` after full 50ep train) |

**Next:** [06 — Serving](06_serving.md) — implemented (`api/main.py` + Docker)
