"""
FastAPI server for Mask2Former ONNX INT8 inference.
Model pulled from Hugging Face Hub at startup (or LOCAL_ONNX_PATH).
"""

import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.inference import (  # noqa: E402
    create_session,
    load_categories,
    resolve_model_path,
    run_inference,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HF_REPO_ID = os.getenv("HF_REPO_ID", "srnortw/mask2former-lane-seg")
HF_TOKEN = os.getenv("HF_TOKEN", "")
MODEL_FILE = os.getenv("MODEL_FILE", "mask2former_int8.onnx")
LOCAL_ONNX_PATH = os.getenv("LOCAL_ONNX_PATH", "")
IMG_SIZE = int(os.getenv("IMG_SIZE", "512"))
CONF_THRESH = float(os.getenv("CONF_THRESH", "0.5"))

session = None
categories = {}


class InstancePrediction(BaseModel):
    instance_id: int
    category_id: int
    category_name: str
    score: float
    mask_b64: str
    mask_shape: List[int]
    bbox: List[float]


class PredictResponse(BaseModel):
    image_id: str
    instances: List[InstancePrediction]
    inference_ms: float
    model: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    global session, categories
    logger.info("Loading ONNX model: %s/%s", HF_REPO_ID, MODEL_FILE)
    model_path = resolve_model_path(
        hf_repo_id=HF_REPO_ID,
        model_file=MODEL_FILE,
        hf_token=HF_TOKEN or None,
        local_path=LOCAL_ONNX_PATH or None,
    )
    session = create_session(model_path)
    categories = load_categories()
    logger.info("Model ready: %s", model_path)
    yield
    session = None


app = FastAPI(
    title="Mask2Former Lane Segmentation API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_FILE,
        "hf_repo": HF_REPO_ID,
        "loaded": session is not None,
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    if session is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        instances, inference_ms = run_inference(
            session,
            image_bytes,
            img_size=IMG_SIZE,
            score_threshold=CONF_THRESH,
            categories=categories,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    return PredictResponse(
        image_id=file.filename or "upload",
        instances=instances,
        inference_ms=round(inference_ms, 2),
        model=MODEL_FILE,
    )
