# 03 — Training

## What We Actually Did (Build Log)

### Key Decisions

**Backbone: Swin-Small** (mid-tier)
- Swin-Tiny: 47M params, ~8GB VRAM
- **Swin-Small: 69M params, ~10GB VRAM ← chosen**
- Swin-Base: 107M params, ~14GB VRAM
- Colab T4 has 15GB — Swin-Small fits comfortably with batch size 4

**Framework: HuggingFace `transformers`** instead of detectron2
- Same Mask2Former architecture
- `pip install transformers` — no complex detectron2 build
- Native HF Hub integration for model push
- Pretrained checkpoint: `facebook/mask2former-swin-small-coco-instance`

**MLflow: DagsHub** instead of Render.com
- Render.com free tier (512MB RAM) → OOM with recent MLflow versions
- DagsHub: free, persistent, purpose-built for ML experiment tracking
- Tracking URI: `https://dagshub.com/srnortw/mask2former.mlflow`

### Source Files Written

| File | Purpose |
|---|---|
| `src/models/mask2former.py` | Model build, phase freezing, optimizer, scheduler |
| `src/train.py` | Full training loop with gradual freezing + MLflow |
| `src/evaluate.py` | COCO segmentation evaluation (mask AP) |
| `notebooks/train_colab.ipynb` | 10-cell Colab notebook, ready to run |

### Colab Secrets to Add

Before running the notebook, add these in Colab → 🔑 Secrets:

| Secret | Value |
|---|---|
| `ROBOFLOW_API_KEY` | from `.env` |
| `HF_TOKEN` | from `.env` |
| `MONGO_URI` | from `.env` |
| `MLFLOW_TRACKING_URI` | `https://dagshub.com/srnortw/mask2former.mlflow` |
| `MLFLOW_TRACKING_PASSWORD` | DagsHub token from `.env` |

### HF Repo ID
`srnortw/mask2former-lane-seg` (private)

---

## Overview

```
Mask2Former + Swin-T/S/B backbone (pretrained ImageNet/COCO)
  ↓
Phase 1: Freeze backbone + pixel decoder → train transformer decoder only
  ↓
Phase 2: Unfreeze pixel decoder → train decoder + pixel decoder
  ↓
Phase 3: Unfreeze all → full fine-tune with low LR
  ↓
CosineAnnealingWarmRestarts throughout all phases
  ↓
MLflow logs metrics → Render.com tracking server
  ↓
Best checkpoint → DVC push → Google Drive
```

---

## 1. Colab Setup

```python
# Cell 1 — mount drive and clone repo
from google.colab import drive
drive.mount('/content/drive')

!git clone https://github.com/your-user/mask2former.git
%cd mask2former

# Override DVC remote to colab path
!dvc remote modify gdrive url "/content/drive/MyDrive/mask2former-mlops"
!dvc pull data/processed
```

```python
# Cell 2 — install dependencies
!pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.1/index.html
!pip install git+https://github.com/facebookresearch/Mask2Former.git
!pip install mlflow pymongo albumentations pycocotools
```

```python
# Cell 3 — verify GPU
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"torch: {torch.__version__}")
```

---

## 2. Model Setup

```python
# src/models/mask2former.py
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
import torch.nn as nn


def build_mask2former_config(
    num_classes: int,
    backbone: str = "swin_tiny",   # swin_tiny | swin_small | swin_base
    weights: str = "COCO-InstanceSegmentation",
):
    cfg = get_cfg()

    # Add Mask2Former defaults
    from mask2former import add_maskformer2_config
    add_maskformer2_config(cfg)

    # Choose backbone config
    backbone_configs = {
        "swin_tiny":  "configs/coco/instance-segmentation/maskformer2_swin_tiny_bs16_50ep.yaml",
        "swin_small": "configs/coco/instance-segmentation/maskformer2_swin_small_bs16_50ep.yaml",
        "swin_base":  "configs/coco/instance-segmentation/maskformer2_swin_base_384_bs16_50ep.yaml",
    }

    cfg.merge_from_file(backbone_configs[backbone])

    # Custom dataset settings
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = num_classes
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 1e-4
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "new_baselines/mask2former_swin_T_bs16_50ep.pkl"
    )

    return cfg
```

---

## 3. Gradual Freezing

### Architecture layers

```
Mask2Former:
  model.backbone                           ← Swin Transformer
  model.sem_seg_head.pixel_decoder         ← Multi-Scale Deformable Attention FPN
  model.sem_seg_head.predictor             ← Transformer Decoder (queries → masks)
```

```python
# src/models/mask2former.py (continued)

def set_phase(model, phase: int):
    """
    Phase 1: Only transformer decoder trains (backbone + pixel decoder frozen)
    Phase 2: Pixel decoder unfreezes (backbone still frozen)
    Phase 3: Everything trains (full fine-tune)
    """
    if phase == 1:
        # Freeze backbone
        for param in model.backbone.parameters():
            param.requires_grad = False
        # Freeze pixel decoder
        for param in model.sem_seg_head.pixel_decoder.parameters():
            param.requires_grad = False
        # Transformer decoder stays trainable
        for param in model.sem_seg_head.predictor.parameters():
            param.requires_grad = True

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Phase 1 | Trainable params: {trainable:,}")

    elif phase == 2:
        # Unfreeze pixel decoder
        for param in model.sem_seg_head.pixel_decoder.parameters():
            param.requires_grad = True

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Phase 2 | Trainable params: {trainable:,}")

    elif phase == 3:
        # Unfreeze everything
        for param in model.parameters():
            param.requires_grad = True

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Phase 3 | Trainable params: {trainable:,}")


def get_optimizer(model, lr: float, weight_decay: float = 1e-4):
    """Only pass trainable params to optimizer."""
    return torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
```

---

## 4. Cosine Warm Restarts Scheduler

```python
# src/train.py
import torch

def build_scheduler(optimizer, T_0: int = 10, T_mult: int = 2, eta_min: float = 1e-6):
    """
    CosineAnnealingWarmRestarts:
      - Decays LR from base to eta_min over T_0 epochs
      - Restarts with same base LR
      - Each restart period doubles (T_mult=2): 10, 20, 40, ...
    """
    return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=T_0,
        T_mult=T_mult,
        eta_min=eta_min,
    )
```

### LR curve visualization (what it looks like)

```
LR
│\        /\          /\
│ \      /  \        /  \
│  \    /    \      /    \
│   \  /      \    /      \
│    \/        \  /        \
│               \/          ...
└────────────────────────────── epochs
    0   10  20     40        80
    ↑   ↑           ↑
 start restart    restart (period doubled)
```

---

## 5. Full Training Loop

```python
# src/train.py
import mlflow
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Phase schedule (tune based on your dataset)
PHASE_SCHEDULE = {
    1: {"start": 0,  "end": 15, "lr": 1e-4},   # decoder only
    2: {"start": 15, "end": 30, "lr": 5e-5},   # + pixel decoder
    3: {"start": 30, "end": 50, "lr": 1e-5},   # full fine-tune
}

def train(cfg, train_loader, val_loader, mlflow_uri: str):
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("mask2former-swin")

    model = build_model(cfg).cuda()
    set_phase(model, phase=1)
    optimizer = get_optimizer(model, lr=PHASE_SCHEDULE[1]["lr"])
    scheduler = build_scheduler(optimizer, T_0=10, T_mult=2)

    current_phase = 1
    best_map = 0.0

    with mlflow.start_run():
        # Log hyperparameters
        mlflow.log_params({
            "backbone": "swin_tiny",
            "batch_size": cfg.SOLVER.IMS_PER_BATCH,
            "phase1_epochs": PHASE_SCHEDULE[1]["end"],
            "phase2_epochs": PHASE_SCHEDULE[2]["end"] - PHASE_SCHEDULE[2]["start"],
            "phase3_epochs": PHASE_SCHEDULE[3]["end"] - PHASE_SCHEDULE[3]["start"],
            "T_0": 10, "T_mult": 2,
        })

        for epoch in range(PHASE_SCHEDULE[3]["end"]):

            # Phase transitions
            if epoch == PHASE_SCHEDULE[2]["start"] and current_phase == 1:
                print(f"\n--- Transitioning to Phase 2 (epoch {epoch}) ---")
                set_phase(model, phase=2)
                optimizer = get_optimizer(model, lr=PHASE_SCHEDULE[2]["lr"])
                scheduler = build_scheduler(optimizer, T_0=10, T_mult=2)
                current_phase = 2

            elif epoch == PHASE_SCHEDULE[3]["start"] and current_phase == 2:
                print(f"\n--- Transitioning to Phase 3 (epoch {epoch}) ---")
                set_phase(model, phase=3)
                optimizer = get_optimizer(model, lr=PHASE_SCHEDULE[3]["lr"])
                scheduler = build_scheduler(optimizer, T_0=10, T_mult=2)
                current_phase = 3

            # Training
            train_loss = train_one_epoch(model, train_loader, optimizer)
            scheduler.step(epoch)

            # Validation
            val_map = evaluate(model, val_loader)

            current_lr = optimizer.param_groups[0]["lr"]

            # Log to MLflow
            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_map": val_map,
                "learning_rate": current_lr,
                "phase": current_phase,
            }, step=epoch)

            print(f"Epoch {epoch:03d} | Phase {current_phase} | Loss: {train_loss:.4f} | mAP: {val_map:.4f} | LR: {current_lr:.2e}")

            # Save best checkpoint
            if val_map > best_map:
                best_map = val_map
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_map": val_map,
                    "phase": current_phase,
                }, "checkpoints/best_model.pth")

                # DVC track and push
                os.system("dvc add checkpoints/best_model.pth")
                os.system("dvc push")
                mlflow.log_artifact("checkpoints/best_model.pth")
                print(f"  ↑ New best: {best_map:.4f}")

    return best_map
```

---

## 6. Training One Epoch

```python
def train_one_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0.0

    for batch_idx, (images, targets) in enumerate(loader):
        images = images.cuda()

        # Convert targets to detectron2 format
        batched_inputs = prepare_batched_inputs(images, targets)

        loss_dict = model(batched_inputs)
        losses = sum(loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
        optimizer.step()

        total_loss += losses.item()

    return total_loss / len(loader)
```

---

## 7. params.yaml — Training Parameters

```yaml
# params.yaml
training:
  backbone: swin_tiny
  total_epochs: 50
  img_size: 512
  batch_size: 4
  weight_decay: 1e-4

  phases:
    phase1: {start: 0,  end: 15, lr: 1e-4}
    phase2: {start: 15, end: 30, lr: 5e-5}
    phase3: {start: 30, end: 50, lr: 1e-5}

  scheduler:
    T_0: 10
    T_mult: 2
    eta_min: 1e-6

mlflow:
  tracking_uri: "https://your-render-app.onrender.com"
  experiment_name: "mask2former-swin"
```

---

## 8. MLflow Tracking Server (Render.com)

Free setup — takes ~10 minutes:

1. Create account at [render.com](https://render.com)
2. New Web Service → connect GitHub repo
3. Build command: `pip install mlflow`
4. Start command: `mlflow server --host 0.0.0.0 --port $PORT --backend-store-uri sqlite:///mlflow.db`
5. Copy the service URL (e.g. `https://mlflow-xyz.onrender.com`)
6. Set `mlflow.tracking_uri` in `params.yaml`

---

## 9. Checkpoint Strategy

```
checkpoints/
├── best_model.pth         ← DVC tracked, pushed to Drive after each improvement
├── phase1_final.pth       ← optional: save at end of each phase
├── phase2_final.pth
└── phase3_final.pth       ← final model for ONNX export
```

```bash
# After training completes
dvc add checkpoints/
git add checkpoints.dvc
git commit -m "model: training complete, best mAP=0.XX"
dvc push
```

---

## 10. Phase Summary for 1200 Images

| Phase | Epochs | LR | Frozen | Goal |
|---|---|---|---|---|
| 1 | 1–15 | 1e-4 | backbone + pixel decoder | learn instance queries fast |
| 2 | 16–30 | 5e-5 | backbone only | adapt feature pyramid |
| 3 | 31–50 | 1e-5 | nothing | full fine-tune |

**Expected training time on Colab Pro (T4):** ~3-5 hours for 50 epochs with 1200 images at batch size 4.

---

## Summary

| Component | Tool/Approach |
|---|---|
| Architecture | Mask2Former + Swin-T backbone |
| Freezing | 3-phase gradual unfreeze |
| Optimizer | AdamW |
| LR schedule | CosineAnnealingWarmRestarts (T_0=10, T_mult=2) |
| Experiment tracking | MLflow → Render.com |
| Checkpointing | DVC → Google Drive |

**Next:** [04 — ONNX Export + INT8 Quantization](04_onnx_quantization.md)
