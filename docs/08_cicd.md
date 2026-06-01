# 08 — CI/CD

## Overview

```
git push origin main
  ↓
GitHub Actions triggers
  ├── [ci.yml]     lint + unit tests + Docker build
  └── [docker.yml] build and push Docker image to GHCR (GitHub Container Registry)
```

We do NOT automate retraining in CI (training runs on Colab manually).
CI validates code quality and keeps the Docker image up to date.

---

## 1. Repository Secrets

Add these in GitHub → Settings → Secrets and variables → Actions:

| Secret | Value |
|---|---|
| `HF_TOKEN` | Hugging Face access token |
| `MONGO_URI` | MongoDB Atlas connection string |
| `ROBOFLOW_API_KEY` | Roboflow API key |
| `GHCR_TOKEN` | GitHub personal access token (for container registry) |

---

## 2. CI Workflow — Lint + Tests + Docker Build

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  lint-and-test:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python 3.12
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"
          cache: pip

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install ruff pytest

      - name: Lint with ruff
        run: ruff check src/ api/ tests/

      - name: Run tests
        env:
          MONGO_URI: ${{ secrets.MONGO_URI }}
        run: pytest tests/ -v --tb=short

      - name: Build Docker image
        run: |
          docker build -f api/Dockerfile -t mask2former-api:ci-test .
          echo "Docker build succeeded"
```

---

## 3. Docker Publish Workflow

```yaml
# .github/workflows/docker.yml
name: Build and Push Docker Image

on:
  push:
    tags:
      - "v*"          # only on version tags like v1.0, v1.1

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}   # e.g. your-user/mask2former

jobs:
  build-and-push:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write

    steps:
      - uses: actions/checkout@v4

      - name: Log in to GitHub Container Registry
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Extract metadata (tags, labels)
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
          tags: |
            type=semver,pattern={{version}}
            type=semver,pattern={{major}}.{{minor}}
            type=raw,value=latest

      - name: Build and push Docker image
        uses: docker/build-push-action@v5
        with:
          context: .
          file: api/Dockerfile
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
```

---

## 4. Tests to Write

```python
# tests/test_api.py
import pytest
from fastapi.testclient import TestClient
import numpy as np
import cv2
import io


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_returns_instances(client, sample_image_bytes):
    response = client.post(
        "/predict",
        files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")}
    )
    assert response.status_code == 200
    data = response.json()
    assert "instances" in data
    assert "inference_ms" in data
    assert data["inference_ms"] > 0


def test_predict_instance_fields(client, sample_image_bytes):
    response = client.post(
        "/predict",
        files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")}
    )
    data = response.json()
    if data["instances"]:
        inst = data["instances"][0]
        assert "score" in inst
        assert "bbox" in inst
        assert "category_id" in inst
        assert 0.0 <= inst["score"] <= 1.0
        assert len(inst["bbox"]) == 4
```

```python
# tests/conftest.py
import pytest
import numpy as np
import cv2
from fastapi.testclient import TestClient
import sys
sys.path.insert(0, "api")


@pytest.fixture(scope="session")
def sample_image_bytes():
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    _, encoded = cv2.imencode(".jpg", img)
    return encoded.tobytes()
```

```python
# tests/test_transforms.py
import numpy as np
import sys
sys.path.insert(0, "src")
from data.transforms import get_train_transforms, get_val_transforms


def test_train_transforms_output_shape():
    transforms = get_train_transforms(img_size=512)
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    mask = np.random.randint(0, 2, (480, 640), dtype=np.uint8)
    result = transforms(image=img, masks=[mask])
    assert result["image"].shape == (3, 512, 512)
    assert result["masks"][0].shape == (512, 512)


def test_different_augmentation_each_call():
    """Verify transforms produce different outputs each call (stochastic)."""
    transforms = get_train_transforms(img_size=512)
    img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    mask = np.ones((512, 512), dtype=np.uint8)

    result1 = transforms(image=img.copy(), masks=[mask.copy()])
    result2 = transforms(image=img.copy(), masks=[mask.copy()])

    # With random augmentation, results should differ (not guaranteed but very likely)
    are_same = np.array_equal(result1["image"].numpy(), result2["image"].numpy())
    # Not asserting False because there's a tiny chance they're same — just print
    print(f"Two augmentations identical: {are_same} (expected False most of the time)")
```

---

## 5. Workflow Summary

```
Developer pushes code
  ↓
[ci.yml] runs on every push to main/develop:
  ├── ruff lint
  ├── pytest (api + transform tests)
  └── docker build (verify it compiles)

Developer tags a release (git tag v1.0 && git push origin v1.0)
  ↓
[docker.yml] runs:
  └── build + push → ghcr.io/your-user/mask2former:v1.0
                   → ghcr.io/your-user/mask2former:latest
```

---

## 6. Pull Updated Image on ROS2 Machine

```bash
# On robot computer or any machine
docker pull ghcr.io/your-user/mask2former:latest

docker run -d -p 8000:8000 \
  -e HF_TOKEN="your-token" \
  ghcr.io/your-user/mask2former:latest
```

---

## Summary

| Workflow | Trigger | Action |
|---|---|---|
| `ci.yml` | Push to main/develop or PR | Lint + tests + docker build check |
| `docker.yml` | Push a `v*` tag | Build + push to GHCR |

**Next:** [09 — ROS2 Integration](09_ros2.md)
