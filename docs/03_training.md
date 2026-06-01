# 03 — Training

## What We Actually Built (Build Log)

### Key Decisions Made During Implementation

**Framework: HuggingFace `transformers`** (not detectron2)
- Original plan used detectron2 — dropped it due to complex CUDA build requirements on Colab
- HuggingFace `transformers` provides the same Mask2Former architecture via `pip install transformers`
- Pretrained checkpoint: `facebook/mask2former-swin-small-coco-instance`
- Native HF Hub integration for model push

**Backbone: Swin-Small** (mid-tier balance)
| Backbone | Params | VRAM | Colab T4 (15GB) |
|---|---|---|---|
| Swin-Tiny | 47M | ~8GB | fits easily |
| **Swin-Small** | **69M** | **~10GB** | **chosen — fits with batch 4** |
| Swin-Base | 107M | ~14GB | tight |

**MLflow: DagsHub** (not Render.com)
- Render.com free tier (512MB RAM) → OOM with recent MLflow versions
- DagsHub: free, persistent, built for MLOps
- Tracking URI: `https://dagshub.com/srnortw/mask2former.mlflow`

**Checkpoint strategy: resume-safe**
- `last_checkpoint.pth` saved **every epoch** with full optimizer + scheduler state
- On restart, training auto-resumes from last epoch — no epochs lost if Colab disconnects
- `best_model.pth` saved on mAP improvement
- `phaseN_final.pth` saved at end of each phase

---

### Source Files Written

| File | Purpose |
|---|---|
| `src/models/mask2former.py` | Model build, `build_model_from_checkpoint`, 3-phase gradual freezing |
| `src/train.py` | Full training loop: gradual freezing + AMP + MLflow + checkpoint resume |
| `src/evaluate.py` | COCO segmentation evaluation (mask AP via `pycocotools`) |
| `notebooks/train_colab.ipynb` | 10-cell Colab notebook |

---

### Bugs Fixed During Build

| Error | Cause | Fix |
|---|---|---|
| `AttributeError: mask_embedder` | HF model doesn't have that attr | Removed from `set_phase()` |
| `AttributeError: ann_file` | Dataset didn't store the path | Added `self.ann_file = ann_file` to `__init__` |
| `TypeError: keep loop` | `range(keep.sum())` indexes by count not position | Replaced with `keep.nonzero(as_tuple=True)[0]` |
| `TypeError: scores.values[q]` | `scores` is already a plain tensor, not a named tuple | Changed to `scores[q].item()` |
| `ModuleNotFoundError: config_loader` | Wrong `sys.path` in Colab | Added `sys.path.insert(0, '/content/mask2former/src')` |
| Private repo clone failure | No auth token in URL | Injected `GITHUB_TOKEN` into clone URL |

---

### HuggingFace Layer Structure (actual, for reference)

```
Mask2FormerForUniversalSegmentation
  model.pixel_level_module.encoder       ← Swin backbone
  model.pixel_level_module.decoder       ← Pixel decoder (FPN)
  model.transformer_module               ← Transformer decoder (queries → masks)
  class_predictor                        ← Classification head
```

This differs from the detectron2 structure (`model.backbone`, `model.sem_seg_head.pixel_decoder`, etc.).

---

### Gradual Freezing — Phase Breakdown

```python
# Phase 1: Only transformer decoder + class head train
model.model.pixel_level_module.encoder   → frozen
model.model.pixel_level_module.decoder   → frozen
model.model.transformer_module           → trainable
model.class_predictor                    → trainable

# Phase 2: Pixel decoder unfreezes
model.model.pixel_level_module.decoder   → trainable

# Phase 3: Full fine-tune
all parameters                           → trainable
```

| Phase | Epochs | LR | Trainable | Goal |
|---|---|---|---|---|
| 1 | 0–15 | 1e-4 | ~30% of params | Teach queries task-specific patterns fast |
| 2 | 15–30 | 5e-5 | ~60% of params | Adapt feature pyramid to lane data |
| 3 | 30–50 | 1e-5 | 100% of params | Full end-to-end fine-tuning |

---

### Colab Notebook — Cell Structure

| Cell | Purpose |
|---|---|
| 1 | Verify GPU (CUDA, VRAM, torch version) |
| 2 | Install dependencies (transformers, mlflow, albumentations, etc.) |
| 3 | Load secrets from Colab secret manager |
| 4 | Clone repo + download Roboflow dataset |
| 5 | Load config + build DataLoaders + batch shape check |
| 6 | Build model + verify phase freezing + sanity forward pass |
| 7 | Test MLflow connection to DagsHub |
| 8 | **Run training** (`train(cfg)`) |
| 9 | Backup checkpoints to Google Drive |
| 10 | Push `best_model.pth` to Hugging Face Hub |

### Colab Secrets Required

| Secret | Value |
|---|---|
| `ROBOFLOW_API_KEY` | Roboflow private key |
| `HF_TOKEN` | Hugging Face write token |
| `MLFLOW_TRACKING_URI` | `https://dagshub.com/srnortw/mask2former.mlflow` |
| `MLFLOW_TRACKING_PASSWORD` | DagsHub access token |
| `GITHUB_TOKEN` | Fine-grained PAT for private repo clone |
| `MONGO_URI` | MongoDB Atlas connection string |

---

### Training Results (First Run — Interrupted at Epoch 18)

```
Phase 1 | Trainable: 12,386,502 / 68,943,670 (18.0%)
Phase 2 | Trainable: ~41M / 69M (60%)

Epoch 001 | Phase 1 | Loss: 48.5962 | mAP: 0.0000 | LR: 1.00e-04
...
Epoch 018 | Phase 2 | Loss: 18.0492 | mAP: 0.0000 | LR: 3.66e-05
```

**Why mAP = 0.000 at epoch 18:**
- `score_threshold: 0.5` in `config.yaml` — model confidence scores are below 0.5 this early
- Loss dropping 48 → 18 confirms training is working correctly
- Real mAP numbers expected from Phase 3 (epochs 30–50) when full network trains together
- Lower `score_threshold` to 0.1–0.2 to see non-zero mAP earlier

**Status:** Checkpoint pushed to `srnortw/mask2former-lane-seg` on HuggingFace for pipeline testing.
Training can be resumed from `last_checkpoint.pth` (saved at epoch 18).

---

### Checkpoint Resume — How It Works

```python
# On restart, train.py detects last_checkpoint.pth and resumes:
last_ckpt_path = os.path.join(ckpt_dir, "last_checkpoint.pth")
if os.path.exists(last_ckpt_path):
    ckpt = torch.load(last_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    start_epoch   = ckpt["epoch"] + 1
    current_phase = ckpt["phase"]
    best_map      = ckpt["best_map"]
```

In Colab, copy checkpoints to Drive after each session:
```python
!cp -r /content/mask2former/checkpoints /content/drive/MyDrive/mask2former-mlops/checkpoints
```

---

### MLflow Experiment

- **DagsHub URL:** https://dagshub.com/srnortw/mask2former.mlflow
- **Experiment:** `mask2former-swin` (experiment ID: 0)
- **Runs logged:** `connection-test` (validation run), `enthused-bass-906` (training run)
- **Metrics tracked per epoch:** `train_loss`, `val_map`, `learning_rate`, `phase`

---

## Summary

| Component | Tool / Approach |
|---|---|
| Architecture | Mask2Former + Swin-Small (HuggingFace transformers) |
| Pretrained weights | `facebook/mask2former-swin-small-coco-instance` |
| Freezing | 3-phase gradual unfreeze |
| Optimizer | AdamW (weight decay 1e-4) |
| LR schedule | CosineAnnealingWarmRestarts (T_0=10, T_mult=2) |
| Mixed precision | `torch.amp.autocast` + `GradScaler` |
| Experiment tracking | MLflow → DagsHub |
| Checkpointing | `last_checkpoint.pth` every epoch + `best_model.pth` on improvement |
| Model registry | Hugging Face Hub (`srnortw/mask2former-lane-seg`) |

**Next:** [04 — ONNX Export + INT8 Quantization](04_onnx_quantization.md)
