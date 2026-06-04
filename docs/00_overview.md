# Mask2Former MLOps Pipeline — Overview

## Goal

**Robotic perception** pipeline: end-to-end MLOps for **lane instance segmentation** using **Mask2Former (Swin-Small)**, trained on a Roboflow dataset, deployed for a **ROS2** robot (camera → masks → `/perception` topics). Computer vision is the method; the project target is autonomous driving / lane perception on-robot.

---

## Workspaces (`mask2former-mlops`)

The name **`mask2former-mlops`** is used consistently for cloud resources (not the git repo folder name):

| Where | What |
|-------|------|
| **Google Drive** | `mask2former-mlops/` — `data/raw/`, `checkpoints/`, git clone at `mask2former-mlops/mask2former/` |
| **MongoDB Atlas** | Project **`mask2former-mlops`** — cluster **`mask2former-cluster`**, DB **`mask2former`** |
| **GitHub** | Repo `srnortw/mask2former` (code only) |

Atlas project ID: `6a21b995cbf3f23e5981be8f` (set `ATLAS_PROJECT_ID` in `.env`).  
Do not use the old **Computer_vision** Atlas project / **Cluster0** for this pipeline.

---

## System

| Component | Details |
|---|---|
| OS | Ubuntu 24.04 LTS |
| CPU | Intel i5-8300H (8 threads) |
| RAM | 16 GB |
| GPU (local) | NVIDIA GTX 1050 Mobile |
| Local Disk | 239 GB SSD |
| Cloud Storage | Google Drive 200 GB (GNOME mounted) |
| Training | Google Colab Pro |

---

## Pipeline Stages

```
[01] Data Management
      Roboflow download → FiftyOne QA → DVC versioning → Google Drive

[02] Data Pipeline
      PyTorch Dataset + Albumentations → DataLoader (shuffle + augment every epoch)

[03] Training
      Mask2Former + Swin backbone
      Gradual freezing (3 phases)
      CosineAnnealingWarmRestarts
      MLflow experiment tracking

[04] ONNX Export + INT8 Quantization
      .pth → ONNX (fp32) → INT8 → FiftyOne mAP (valid+test, fp32 vs INT8)
      Phase 3 evaluate.py = PyTorch on val during training

[05] Model Registry
      Hugging Face Hub + MLflow Model Registry

[06] Serving
      FastAPI + Docker (ONNX INT8 inference)

[07] Monitoring
      Evidently AI (drift) + MongoDB Atlas project mask2former-mlops (prediction logs)

[08] CI/CD
      GitHub Actions (test → build → push Docker image)

[09] ROS2 Integration (Mode A done)
      Jazzy HTTP client → Docker FastAPI → /perception topics
      Mode B embedded ONNX + on-robot deploy — later
```

---

## Tools Summary

| Layer | Tool | Cost |
|---|---|---|
| Dataset | Roboflow | Free tier |
| Dataset exploration | FiftyOne | Open-source |
| Data versioning | DVC | Open-source |
| Data storage | Google Drive (DVC remote) | Already owned |
| Training compute | Google Colab Pro | Already paying |
| Experiment tracking | MLflow | Open-source (self-hosted) |
| Model registry | Hugging Face Hub | Free |
| Export & quantization | ONNX + ONNXRuntime | Open-source |
| Serving | FastAPI + Docker | Open-source |
| Database | MongoDB Atlas (`mask2former-mlops`) | Free tier (512 MB) |
| Monitoring | Evidently AI | Open-source |
| CI/CD | GitHub Actions | Free tier |
| Code + pipeline | GitHub + DVC | Free |

**Total extra cost: $0**

---

## Project Structure

```
mask2former/
├── docs/                        # this folder — pipeline documentation
│   ├── 00_overview.md
│   ├── 01_data_management.md
│   ├── 02_data_pipeline.md
│   ├── 03_training.md
│   ├── 04_onnx_quantization.md
│   ├── 05_model_registry.md
│   ├── 06_serving.md
│   ├── 07_monitoring.md
│   ├── 08_cicd.md
│   ├── 09_ros2.md
│   └── 10_project_requirements.md
├── data/                        # DVC tracked → Google Drive
│   ├── raw/                     # Roboflow COCO download
│   ├── processed/               # FiftyOne filtered
│   └── calibration/             # INT8 calibration samples
├── configs/                     # Mask2Former detectron2 configs
├── src/
│   ├── data/
│   │   ├── roboflow_download.py
│   │   ├── fiftyone_filter.py
│   │   ├── dataset.py           # PyTorch Dataset class
│   │   └── transforms.py        # Albumentations pipeline
│   ├── models/
│   │   └── mask2former.py       # model setup + freezing logic
│   ├── train.py
│   ├── evaluate.py
│   ├── export_onnx.py
│   ├── quantize_int8.py
│   ├── evaluate_onnx_fiftyone.py  # Phase 4 FiftyOne (valid/test)
│   ├── mongo_logger.py
│   └── monitoring/drift_report.py
├── api/
│   ├── main.py                  # FastAPI app
│   ├── Dockerfile
│   └── requirements-api.txt
├── src/inference.py             # ONNX preprocess + postprocess (serving / ROS2)
├── docker-compose.yml
├── scripts/
│   ├── setup_atlas_mongo.sh     # Atlas project mask2former-mlops
│   ├── seed_predictions.sh      # batch /predict → MongoDB
│   ├── run_drift_report.sh      # Evidently drift (uses .venv)
│   ├── visualize_predict.py     # INPUT.jpg OUTPUT.jpg → API overlay
│   ├── predict_image.sh         # wrapper for visualize_predict
│   ├── build_ros2.sh            # colcon + .venv
│   ├── run_ros2_mode_a.sh
│   └── test_ros2_mode_a_once.sh
├── requirements-monitoring.txt
├── tests/test_inference.py
├── tests/test_mongo_logger.py
├── ros2_ws/                     # Phase 09 — Jazzy; build via scripts/build_ros2.sh (.venv)
│   └── src/mask2former_ros/     # Mode A: segmentation_client (HTTP → Docker API)
├── requirements-ros2-client.txt # .venv/bin/pip install -r …
├── notebooks/                   # Colab training notebooks
├── tests/
├── .dvc/
├── dvc.yaml                     # DVC pipeline stages
├── params.yaml                  # hyperparameters
├── requirements.txt             # local venv only (no torch)
└── .github/workflows/
    ├── ci.yml                   # lint + pytest + docker build
    └── docker.yml               # push to GHCR on v* tags
├── requirements-ci.txt          # GitHub Actions test deps
├── pyproject.toml               # ruff + pytest config
```

---

## Quick Links

- [01 — Data Management](01_data_management.md)
- [02 — Data Pipeline](02_data_pipeline.md)
- [03 — Training](03_training.md)
- [04 — ONNX + INT8 Quantization](04_onnx_quantization.md)
- [05 — Model Registry](05_model_registry.md)
- [06 — Serving](06_serving.md)
- [07 — Monitoring](07_monitoring.md)
- [08 — CI/CD](08_cicd.md)
- [09 — ROS2 Integration](09_ros2.md)
