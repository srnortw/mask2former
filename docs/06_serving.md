# 06 — Serving

## Overview

```
mask2former_int8.onnx (Hugging Face Hub or LOCAL_ONNX_PATH)
  ↓
src/inference.py  →  preprocess + ONNX run + postprocess (aligned with evaluate.py)
  ↓
api/main.py       →  FastAPI  GET /health  POST /predict
  ↓
Docker / docker-compose  →  port 8000
  ↓
ROS2 (Phase 09)   →  HTTP client or embedded ONNX
```

---

## What We Actually Built

**Commit:** `890a386` — `feat(serving): Phase 06 FastAPI + Docker for ONNX INT8 inference`

| File | Purpose |
|------|---------|
| `src/inference.py` | Preprocess, ONNX session, postprocess (numpy only; reusable in ROS2) |
| `api/main.py` | FastAPI app: lifespan loads model, `/health`, `/predict` |
| `api/requirements-api.txt` | Serving deps (no PyTorch) |
| `api/Dockerfile` | `python:3.12-slim` + OpenCV + ONNX Runtime |
| `docker-compose.yml` | One-service compose on port 8000 |
| `scripts/visualize_predict.py` | Calls `/predict`, draws masks, saves overlay JPG |
| `tests/test_inference.py` | Unit tests for `postprocess_instances` (no ONNX file needed) |

### Design choices

- **Inference logic lives in `src/inference.py`**, not inline in `main.py` — same postprocessing as `evaluate.py`, so ROS2 can import it without FastAPI.
- **Model at startup:** downloads `mask2former_int8.onnx` from `srnortw/mask2former-lane-seg` via `huggingface_hub`, or uses `LOCAL_ONNX_PATH` to skip download.
- **No PyTorch in the API image** — only `onnxruntime`, `opencv-python-headless`, FastAPI.
- **Categories:** hardcoded Roboflow lane classes in `DEFAULT_CATEGORIES`; override with env `CATEGORIES` JSON if needed.

### ONNX I/O (matches `export_onnx.py`)

| | Name | Shape |
|--|------|-------|
| Input | `pixel_values` | `[1, 3, 512, 512]` |
| Output | `masks_queries_logits` | `[1, Q, H, W]` |
| Output | `class_queries_logits` | `[1, Q, num_classes+1]` |

Postprocessing: sigmoid on mask queries, softmax on class logits (no-object class dropped), `CONF_THRESH` filter, resize masks to original image size.

### API response (`POST /predict`)

Each instance in JSON:

| Field | Description |
|-------|-------------|
| `instance_id` | Index in response list |
| `category_id` | Class index (0–4 for lane dataset) |
| `category_name` | e.g. `Lane-Markings`, `Left-Boundary-Solid` |
| `score` | Max class probability after softmax |
| `mask_b64` | Base64-encoded binary mask (`uint8`, flattened) |
| `mask_shape` | `[height, width]` of original image |
| `bbox` | `[x, y, w, h]` in pixel coordinates |

Top-level: `image_id`, `instances`, `inference_ms` (ONNX run only), `model`.

### Local test (2026-06-04)

| Step | Result |
|------|--------|
| `docker compose build && docker compose up -d` | Image builds; model pulls from HF on first start (~45s) |
| `curl http://localhost:8000/health` | `{"status":"ok","loaded":true,"hf_repo":"srnortw/mask2former-lane-seg",...}` |
| `POST /predict` on val image | **19 instances** detected |
| Latency (CPU, Docker) | **~1.7–2.4 s** end-to-end per image (includes preprocess + postprocess + 19 masks) |
| `scripts/visualize_predict.py` | Overlay saved to `reports/predict_overlay.jpg` |

> Model was trained to ~epoch 18/50 — masks work but scores can be low; lower `CONF_THRESH` if you need more instances.

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_REPO_ID` | `srnortw/mask2former-lane-seg` | HF model repo |
| `HF_TOKEN` | (empty) | HF token if repo is private |
| `MODEL_FILE` | `mask2former_int8.onnx` | ONNX filename in repo |
| `LOCAL_ONNX_PATH` | (empty) | If set, skip HF download |
| `IMG_SIZE` | `512` | Input resize |
| `CONF_THRESH` | `0.5` | Score threshold |
| `CATEGORIES` | (built-in) | Optional JSON map of class id → name |
| `MONGO_URI` | (empty) | Phase 07: Atlas project **mask2former-mlops** — log prediction metadata |
| `ATLAS_PROJECT_ID` | `6a21b995cbf3f23e5981be8f` | Atlas project (documentation / setup script) |
| `ATLAS_CLUSTER` | `mask2former-cluster` | Atlas cluster name |

---

## 1. Run with Docker (recommended)

```bash
cd ~/Desktop/mask2former
source .env          # HF_TOKEN, etc.

docker compose build
docker compose up -d

# Wait for model download on first start (~30–60s), then:
curl -s http://localhost:8000/health | python3 -m json.tool

curl -s -X POST http://localhost:8000/predict \
  -F "file=@data/raw/valid/<your_image>.jpg" | python3 -m json.tool
```

Stop:

```bash
docker compose down
```

---

## 2. Visualize predictions

With the API running:

```bash
source .venv/bin/activate
pip install requests opencv-python-headless  # if not already installed

python scripts/visualize_predict.py data/raw/valid/<image>.jpg \
  -o reports/predict_overlay.jpg
```

Opens the image with colored instance masks and labels. Uses `http://localhost:8000/predict` by default (`--url` to override).

---

## 3. Run locally without Docker

```bash
cd ~/Desktop/mask2former
source .venv/bin/activate
pip install -r api/requirements-api.txt

export HF_REPO_ID=srnortw/mask2former-lane-seg
export HF_TOKEN=your_hf_token   # if needed
# Or skip download:
# export LOCAL_ONNX_PATH=/path/to/mask2former_int8.onnx

PYTHONPATH=. uvicorn api.main:app --host 0.0.0.0 --port 8000
```

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/predict \
  -F "file=@data/raw/valid/<image>.jpg" | python3 -m json.tool
```

---

## 4. Docker (manual build)

```bash
docker build -f api/Dockerfile -t mask2former-api .
docker run -p 8000:8000 \
  -e HF_REPO_ID=srnortw/mask2former-lane-seg \
  -e HF_TOKEN=$HF_TOKEN \
  mask2former-api:latest
```

`api/Dockerfile` copies `src/inference.py`, `src/mongo_logger.py` (Phase 07), and `api/main.py`; sets `PYTHONPATH=/app`.

---

## 5. Unit tests

```bash
source .venv/bin/activate
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_inference.py -q
```

Tests `postprocess_instances` with synthetic logits — no ONNX file or GPU required.

---

## 6. Python client (ROS2 / scripts)

```python
import requests

def predict_masks(image_path: str, server_url: str = "http://localhost:8000"):
    with open(image_path, "rb") as f:
        response = requests.post(
            f"{server_url}/predict",
            files={"file": f},
            timeout=120,
        )
    response.raise_for_status()
    return response.json()

result = predict_masks("frame.jpg")
print(f"{len(result['instances'])} instances in {result['inference_ms']:.1f} ms (ONNX only)")
for inst in result["instances"]:
    print(f"  [{inst['category_name']}] score={inst['score']:.3f} bbox={inst['bbox']}")
```

Decode a mask:

```python
import base64
import numpy as np

inst = result["instances"][0]
h, w = inst["mask_shape"]
mask = np.frombuffer(base64.b64decode(inst["mask_b64"]), dtype=np.uint8).reshape(h, w)
```

---

## 7. docker-compose.yml

```yaml
services:
  mask2former-api:
    build:
      context: .
      dockerfile: api/Dockerfile
    ports:
      - "8000:8000"
    environment:
      - HF_REPO_ID=srnortw/mask2former-lane-seg
      - HF_TOKEN=${HF_TOKEN:-}
      - MODEL_FILE=mask2former_int8.onnx
      - IMG_SIZE=512
      - CONF_THRESH=0.5
      - MONGO_URI=${MONGO_URI:-}   # mask2former-mlops / mask2former-cluster
    restart: unless-stopped
```

---

## Summary

| Component | Details |
|---|---|
| Framework | FastAPI + uvicorn |
| Runtime | ONNX Runtime INT8 (`mask2former_int8.onnx`) |
| Model source | Hugging Face Hub at container startup |
| Container | Docker (`python:3.12-slim`) |
| Input | `multipart/form-data` image upload |
| Output | JSON instances with `mask_b64`, scores, bboxes |
| End-to-end latency (local CPU, Docker) | ~1.7–2.4 s/image (19 instances) |
| ONNX-only time | Reported in `inference_ms` field |

**Next:** [07 — Monitoring](07_monitoring.md)
