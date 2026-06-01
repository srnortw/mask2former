# 06 — Serving

## Overview

```
mask2former_int8.onnx (pulled from Hugging Face Hub at startup)
  ↓
FastAPI app  →  POST /predict  →  returns instance masks as JSON
  ↓
Docker container (portable, runs on local machine or robot's compute)
  ↓
ROS2 node calls HTTP endpoint  OR  loads ONNX directly (see 09_ros2.md)
```

---

## 1. FastAPI Inference Server

```python
# api/main.py
import io
import os
import time
import base64
import logging
import numpy as np
import cv2
import onnxruntime as ort
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from huggingface_hub import hf_hub_download
from typing import List
from contextlib import asynccontextmanager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Model config ---
HF_REPO_ID  = os.getenv("HF_REPO_ID",  "your-hf-username/mask2former-instance-seg")
HF_TOKEN    = os.getenv("HF_TOKEN",    "")
MODEL_FILE  = os.getenv("MODEL_FILE",  "mask2former_int8.onnx")
IMG_SIZE    = int(os.getenv("IMG_SIZE", "512"))
CONF_THRESH = float(os.getenv("CONF_THRESH", "0.5"))

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# --- Global session ---
session: ort.InferenceSession = None

CATEGORIES = {}   # loaded at startup from metadata


@asynccontextmanager
async def lifespan(app: FastAPI):
    global session, CATEGORIES
    logger.info(f"Downloading model: {HF_REPO_ID}/{MODEL_FILE}")
    model_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=MODEL_FILE,
        token=HF_TOKEN or None,
    )
    session = ort.InferenceSession(
        model_path,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    logger.info("Model loaded.")
    yield
    session = None


app = FastAPI(
    title="Mask2Former Instance Segmentation API",
    version="1.0.0",
    lifespan=lifespan,
)


# --- Request / Response schemas ---
class InstancePrediction(BaseModel):
    instance_id: int
    category_id: int
    category_name: str
    score: float
    mask_rle: str        # base64 encoded binary mask
    bbox: List[float]    # [x, y, w, h] in pixel coords


class PredictResponse(BaseModel):
    image_id: str
    instances: List[InstancePrediction]
    inference_ms: float
    model: str


# --- Preprocessing ---
def preprocess(image_bytes: bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = img.shape[:2]

    resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    normalized = (resized.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    tensor = normalized.transpose(2, 0, 1)[np.newaxis]   # [1, 3, H, W]

    return tensor, orig_h, orig_w


# --- Postprocessing ---
def postprocess(pred_masks, pred_logits, orig_h: int, orig_w: int):
    instances = []
    scores = pred_logits.max(axis=-1)
    class_ids = pred_logits.argmax(axis=-1)

    for i, (mask, score, class_id) in enumerate(zip(pred_masks[0], scores[0], class_ids[0])):
        if score < CONF_THRESH:
            continue

        # Resize mask back to original image size
        mask_resized = cv2.resize(mask.astype(np.float32), (orig_w, orig_h))
        binary_mask = (mask_resized > 0.5).astype(np.uint8)

        # Bounding box from mask
        ys, xs = np.where(binary_mask)
        if len(xs) == 0:
            continue
        x, y = int(xs.min()), int(ys.min())
        w = int(xs.max()) - x
        h = int(ys.max()) - y

        # Encode mask as base64
        mask_bytes = binary_mask.tobytes()
        mask_b64 = base64.b64encode(mask_bytes).decode("utf-8")

        instances.append(InstancePrediction(
            instance_id=i,
            category_id=int(class_id),
            category_name=CATEGORIES.get(int(class_id), str(class_id)),
            score=float(score),
            mask_rle=mask_b64,
            bbox=[float(x), float(y), float(w), float(h)],
        ))

    return instances


# --- Endpoints ---
@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_FILE}


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    if session is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    image_bytes = await file.read()
    tensor, orig_h, orig_w = preprocess(image_bytes)

    t0 = time.perf_counter()
    pred_masks, pred_logits = session.run(None, {"image": tensor})
    inference_ms = (time.perf_counter() - t0) * 1000

    instances = postprocess(pred_masks, pred_logits, orig_h, orig_w)

    return PredictResponse(
        image_id=file.filename,
        instances=instances,
        inference_ms=round(inference_ms, 2),
        model=MODEL_FILE,
    )
```

---

## 2. Dockerfile

```dockerfile
# api/Dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy app
COPY api/main.py .

# Expose
EXPOSE 8000

# Start
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```
# api/requirements-api.txt
fastapi
uvicorn[standard]
onnxruntime
opencv-python-headless
numpy
huggingface_hub
pydantic
```

---

## 3. Build & Run

```bash
# Build Docker image
docker build -f api/Dockerfile -t mask2former-api:latest .

# Run locally
docker run -p 8000:8000 \
  -e HF_REPO_ID="your-hf-username/mask2former-instance-seg" \
  -e HF_TOKEN="your-hf-token" \
  -e CONF_THRESH="0.5" \
  mask2former-api:latest
```

### Test the endpoint

```bash
# Health check
curl http://localhost:8000/health

# Predict
curl -X POST http://localhost:8000/predict \
  -F "file=@test_image.jpg" \
  | python3 -m json.tool
```

---

## 4. Python Client Example

```python
# For ROS2 node or any client
import requests
import json

def predict_masks(image_path: str, server_url: str = "http://localhost:8000"):
    with open(image_path, "rb") as f:
        response = requests.post(
            f"{server_url}/predict",
            files={"file": f},
            timeout=10.0,
        )
    response.raise_for_status()
    return response.json()

result = predict_masks("frame.jpg")
print(f"Found {len(result['instances'])} instances in {result['inference_ms']:.1f}ms")
for inst in result["instances"]:
    print(f"  [{inst['category_name']}] score={inst['score']:.3f} bbox={inst['bbox']}")
```

---

## 5. docker-compose.yml (with monitoring)

```yaml
# docker-compose.yml
version: "3.9"

services:
  mask2former-api:
    build:
      context: .
      dockerfile: api/Dockerfile
    ports:
      - "8000:8000"
    environment:
      - HF_REPO_ID=your-hf-username/mask2former-instance-seg
      - HF_TOKEN=${HF_TOKEN}
      - CONF_THRESH=0.5
    restart: unless-stopped

  mlflow:
    image: ghcr.io/mlflow/mlflow:latest
    ports:
      - "5000:5000"
    command: mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db
    volumes:
      - mlflow_data:/mlflow.db
    restart: unless-stopped

volumes:
  mlflow_data:
```

---

## 6. Environment Variables

```bash
# .env (never commit this file)
HF_TOKEN=hf_xxxxxxxxxxxx
ROBOFLOW_API_KEY=xxxxxxxxxx
MONGO_URI=mongodb+srv://user:pass@cluster.mongodb.net/mask2former
MLFLOW_TRACKING_URI=https://your-render-app.onrender.com
```

---

## Summary

| Component | Details |
|---|---|
| Framework | FastAPI |
| Runtime | ONNX Runtime (INT8 model) |
| Model download | Hugging Face Hub at startup |
| Container | Docker (python:3.12-slim) |
| Input | multipart/form-data image upload |
| Output | JSON with masks (base64 RLE), scores, bboxes |
| Inference speed | ~90ms/image on CPU |

**Next:** [07 — Monitoring](07_monitoring.md)
