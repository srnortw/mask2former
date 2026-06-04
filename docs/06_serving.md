# 06 — Serving

## Overview

```
mask2former_int8.onnx (Hugging Face Hub — Phase 05)
  ↓
src/inference.py  →  preprocess + ONNX Runtime + postprocess (same logic as evaluate.py)
  ↓
api/main.py       →  FastAPI  GET /health  POST /predict
  ↓
Docker / docker-compose  →  port 8000
  ↓
Phase 07  →  optional MongoDB logging on each /predict
  ↓
Phase 09  →  ROS2 HTTP client or embedded ONNX
```

---

## What We Actually Built

**Commit:** `890a386` — `feat(serving): Phase 06 FastAPI + Docker for ONNX INT8 inference`

| File | Purpose |
|------|---------|
| `src/inference.py` | Preprocess, ONNX session, postprocess — numpy/onnxruntime only (reused by ROS2 later) |
| `api/main.py` | FastAPI lifespan, `/health`, `/predict` |
| `api/requirements-api.txt` | fastapi, uvicorn, onnxruntime, opencv, huggingface_hub, pydantic |
| `api/Dockerfile` | `python:3.12-slim`, copies `inference.py` + `main.py` |
| `docker-compose.yml` | Local deploy on port 8000 |
| `scripts/visualize_predict.py` | POST to API, draw masks → `reports/predict_overlay.jpg` |
| `tests/test_inference.py` | Unit tests for `postprocess_instances` (no ONNX file) |

**Later (Phase 07, same API):** `mongo_logger.py` added to Dockerfile and `main.py` for optional MongoDB logging — serving still works without it.

### Design choices

- **Inference in `src/inference.py`**, not buried in `main.py` — ROS2 can import the same postprocess without FastAPI.
- **Model from Hugging Face** at startup: `srnortw/mask2former-lane-seg` / `mask2former_int8.onnx` (Phase 04–05 artifact), or `LOCAL_ONNX_PATH` to skip download.
- **No PyTorch in the API image** — smaller Docker image; only ONNX Runtime + OpenCV + FastAPI.
- **Lane class names** — `DEFAULT_CATEGORIES` in `inference.py` (Roboflow 5 classes); override with env `CATEGORIES` JSON.
- **Correct ONNX tensor names** — input `pixel_values` (not `image`); outputs `masks_queries_logits`, `class_queries_logits` (aligned with `export_onnx.py`).

### ONNX I/O (matches `export_onnx.py`)

| | Name | Shape |
|--|------|-------|
| Input | `pixel_values` | `[1, 3, 512, 512]` |
| Output | `masks_queries_logits` | `[1, Q, H, W]` |
| Output | `class_queries_logits` | `[1, Q, num_classes+1]` |

Postprocessing: sigmoid on masks → softmax on classes (drop no-object) → `CONF_THRESH` → resize masks to original size → `mask_b64` + bbox.

### API response (`POST /predict`)

| Field | Description |
|-------|-------------|
| `instance_id` | Index in response |
| `category_id` | 0–4 (lane dataset) |
| `category_name` | e.g. `Lane-Markings` |
| `score` | Max class probability |
| `mask_b64` | Base64 `uint8` mask, flattened |
| `mask_shape` | `[height, width]` |
| `bbox` | `[x, y, w, h]` pixels |

Response also includes: `image_id`, `instances`, `inference_ms` (ONNX only), `model`.

### `/health` response

```json
{
  "status": "ok",
  "model": "mask2former_int8.onnx",
  "hf_repo": "srnortw/mask2former-lane-seg",
  "loaded": true,
  "mongodb": true
}
```

`mongodb` is `true` when Phase 07 `MONGO_URI` is set and Atlas is reachable; otherwise `false` — API still serves predictions.

### Build log — issues and fixes

| Issue | Fix |
|-------|-----|
| Early doc draft used wrong ONNX input name `image` | Implementation uses `pixel_values` per exported model |
| Early doc used `mask_rle` | Actual field is `mask_b64` + `mask_shape` |
| Postprocess diverged from training eval | Shared logic with `evaluate.py` in `src/inference.py` |
| First Docker run slow | Normal: HF downloads ~50–90 MB ONNX on first `docker compose up` (~30–60s) |
| Low mAP at epoch 18/50 | Serving works; lower `CONF_THRESH` if you need more instances |

### Verification (2026-06-04) — Phase 06 complete

| Check | Result |
|-------|--------|
| `docker compose build && docker compose up -d` | Image builds and starts |
| `curl http://localhost:8000/health` | `status: ok`, `loaded: true` |
| `POST /predict` (val image) | **19 instances** |
| Latency (CPU, Docker) | **~1.7–2.4 s** end-to-end; `inference_ms` ~ONNX slice only |
| `scripts/visualize_predict.py` | Overlay → `reports/predict_overlay.jpg` |
| `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/test_inference.py` | Pass |
| GitHub `main` | Pushed in `890a386` |

> Training was at ~epoch 18/50 when tested — masks visible; scores can be low until Phase 3 training finishes.

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_REPO_ID` | `srnortw/mask2former-lane-seg` | HF repo (Phase 05) |
| `HF_TOKEN` | (empty) | If repo is private |
| `MODEL_FILE` | `mask2former_int8.onnx` | ONNX file in repo |
| `LOCAL_ONNX_PATH` | (empty) | Skip HF download if set |
| `IMG_SIZE` | `512` | Input resize |
| `CONF_THRESH` | `0.5` | Score threshold (env default) |
| `?conf_threshold=` | — | Per-request override on `POST /predict` (used by ROS2 client) |
| `CATEGORIES` | built-in lane map | Optional JSON override |
| `MONGO_URI` | (empty) | Phase 07 — optional prediction logging |

---

## 1. Run with Docker (recommended)

```bash
cd ~/Desktop/mask2former
source .env    # HF_TOKEN if needed; MONGO_URI for Phase 07 logging

docker compose build
docker compose up -d

# First start: wait for HF model download (~30–60s)
curl -s http://localhost:8000/health | python3 -m json.tool

curl -s -X POST http://localhost:8000/predict \
  -F "file=@data/raw/valid/<image>.jpg" | python3 -m json.tool
```

Stop:

```bash
docker compose down
```

---

## 2. Visualize predictions

Uses local **`.venv`** for `requests` + OpenCV (not the API container):

```bash
cd ~/Desktop/mask2former
.venv/bin/pip install requests opencv-python-headless   # if missing

.venv/bin/python scripts/visualize_predict.py data/raw/valid/<image>.jpg reports/my_overlay.jpg
# or: ./scripts/predict_image.sh <image>.jpg reports/my_overlay.jpg
# optional stricter filter: ... reports/out.jpg 0.65
```

Default URL: `http://localhost:8000/predict` (`--url` to override).  
Input/output paths: `visualize_predict.py INPUT.jpg OUTPUT.jpg` or `./scripts/predict_image.sh INPUT.jpg OUTPUT.jpg [conf_threshold]`.

---

## 3. Run locally without Docker (`.venv`)

```bash
cd ~/Desktop/mask2former
source .venv/bin/activate
pip install -r api/requirements-api.txt

export HF_REPO_ID=srnortw/mask2former-lane-seg
export HF_TOKEN=your_hf_token
# export LOCAL_ONNX_PATH=/path/to/mask2former_int8.onnx

PYTHONPATH=. uvicorn api.main:app --host 0.0.0.0 --port 8000
```

---

## 4. Docker image layout

```
api/Dockerfile
  ├── requirements-api.txt  → pip install in image
  ├── src/inference.py
  ├── src/mongo_logger.py   # Phase 07 (optional at runtime)
  └── api/main.py → /app/main.py

ENV PYTHONPATH=/app
CMD uvicorn main:app --host 0.0.0.0 --port 8000
```

Manual build:

```bash
docker build -f api/Dockerfile -t mask2former-api .
docker run -p 8000:8000 \
  -e HF_REPO_ID=srnortw/mask2former-lane-seg \
  -e HF_TOKEN=$HF_TOKEN \
  mask2former-api:latest
```

---

## 5. Unit tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/test_inference.py -q
```

Synthetic logits only — no ONNX file, no GPU.

---

## 6. Python client (ROS2 / scripts)

```python
import requests

def predict_masks(image_path: str, server_url: str = "http://localhost:8000"):
    with open(image_path, "rb") as f:
        r = requests.post(f"{server_url}/predict", files={"file": f}, timeout=120)
    r.raise_for_status()
    return r.json()
```

Decode mask:

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
      - MONGO_URI=${MONGO_URI:-}
    restart: unless-stopped
```

---

## Summary

| Component | Details |
|-----------|---------|
| Framework | FastAPI + uvicorn |
| Runtime | ONNX Runtime INT8 |
| Model | `mask2former_int8.onnx` from HF Hub (Phase 05) |
| Deploy | Docker Compose, port 8000 |
| Input | `multipart/form-data` image |
| Output | JSON + `mask_b64`, scores, bboxes |
| Latency (local CPU) | ~1.7–2.4 s/image (19 instances, full pipeline) |

**Status:** Phase 06 **complete** (2026-06-04). Phase 07 adds optional MongoDB on top of the same API.

**Next:** [07 — Monitoring](07_monitoring.md)
