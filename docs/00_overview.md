# Mask2Former MLOps Pipeline — Overview

## Goal

End-to-end MLOps pipeline for **instance segmentation** using **Mask2Former with Swin Transformer backbone**, trained on a custom dataset from Roboflow, and deployed for use in a **ROS2** robotics project.

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
      .pth → ONNX (fp32) → INT8 static quantization
      FiftyOne visual evaluation

[05] Model Registry
      Hugging Face Hub + MLflow Model Registry

[06] Serving
      FastAPI + Docker (ONNX INT8 inference)

[07] Monitoring
      Evidently AI (drift) + MongoDB Atlas (prediction logs)

[08] CI/CD
      GitHub Actions (test → build → push Docker image)

[09] ROS2 Integration
      segmentation_node subscribing to camera topic
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
| Database | MongoDB Atlas | Free tier (512 MB) |
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
│   └── mongo_logger.py
├── api/
│   ├── main.py                  # FastAPI app
│   ├── Dockerfile
│   └── requirements-api.txt
├── src/inference.py             # ONNX preprocess + postprocess (serving / ROS2)
├── docker-compose.yml
├── tests/test_inference.py
├── ros2_ws/
│   └── src/mask2former_ros/
│       ├── segmentation_node.py
│       └── package.xml
├── notebooks/                   # Colab training notebooks
├── tests/
├── .dvc/
├── dvc.yaml                     # DVC pipeline stages
├── params.yaml                  # hyperparameters
├── requirements.txt             # local venv only (no torch)
└── .github/
    └── workflows/
        └── ci.yml
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
