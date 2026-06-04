FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY api/requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY src/__init__.py src/__init__.py
COPY src/inference.py src/inference.py
COPY api/main.py main.py

ENV PYTHONPATH=/app
ENV HF_REPO_ID=srnortw/mask2former-lane-seg
ENV MODEL_FILE=mask2former_int8.onnx
ENV IMG_SIZE=512
ENV CONF_THRESH=0.5

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
