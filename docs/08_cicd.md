# 08 — CI/CD

## Overview

```
git push / PR → main
  ↓
.github/workflows/ci.yml
  ├── ruff lint (serving + monitoring + tests)
  ├── pytest (unit tests, no ONNX download)
  └── docker build (verify api/Dockerfile)

git tag v* (e.g. v1.0.0)
  ↓
.github/workflows/docker.yml
  └── build + push → ghcr.io/srnortw/mask2former
```

Training stays on **Colab** — CI does not retrain.

---

## What We Actually Built

| File | Purpose |
|------|---------|
| `.github/workflows/ci.yml` | Lint, unit tests, Docker build on push/PR to `main` |
| `.github/workflows/docker.yml` | Push API image to GHCR on `v*` tags (+ manual dispatch) |
| `requirements-ci.txt` | Minimal deps for CI (numpy, opencv, pymongo, pytest, ruff) |
| `pyproject.toml` | Ruff + pytest config (`pythonpath = ["."]`) |

### Design choices

- **CI scope** — Ruff/tests cover **serving + monitoring** paths (`api/`, `src/inference.py`, `mongo_logger`, `tests/`), not full training stack (torch, fiftyone, etc.).
- **No ONNX in pytest** — `test_inference.py` and `test_mongo_logger.py` use mocks/synthetic tensors; no HF download in CI.
- **No `MONGO_URI` in CI** — Mongo tests mock `MongoClient`; no Atlas secret required for green CI.
- **Docker build only** — `ci.yml` verifies `api/Dockerfile` compiles; does not run the container.
- **GHCR on version tags** — `docker.yml` runs on `v*` tags; uses built-in `GITHUB_TOKEN` (no extra `GHCR_TOKEN` secret).

### CI job steps (`ci.yml`)

| Step | Command / action |
|------|------------------|
| Python 3.12 | `actions/setup-python@v5` + pip cache |
| Install | `pip install -r requirements-ci.txt` |
| Lint | `ruff check api/ tests/ src/inference.py src/mongo_logger.py src/monitoring/ scripts/visualize_predict.py` |
| Test | `pytest tests/test_inference.py tests/test_mongo_logger.py` |
| Docker | `docker build -f api/Dockerfile -t mask2former-api:ci .` |

### Repository secrets (optional)

| Secret | Needed for CI? | Used for |
|--------|----------------|----------|
| `GITHUB_TOKEN` | Auto (GHCR push) | `docker.yml` package write |
| `HF_TOKEN` | No | Runtime when pulling model in deployed container |
| `MONGO_URI` | No | Phase 07 at runtime only |
| `ROBOFLOW_API_KEY` | No | Colab / local data download |

Add secrets in GitHub → **Settings → Secrets and variables → Actions** when you deploy or need private HF repo access.

### Publish image (release)

```bash
git tag v1.0.0
git push origin v1.0.0
```

Image: `ghcr.io/srnortw/mask2former:v1.0.0` and `:latest`

Pull on any machine:

```bash
docker pull ghcr.io/srnortw/mask2former:latest
docker run -d -p 8000:8000 \
  -e HF_TOKEN="$HF_TOKEN" \
  -e MONGO_URI="$MONGO_URI" \
  ghcr.io/srnortw/mask2former:latest
```

### Local CI (same as GitHub)

```bash
cd ~/Desktop/mask2former
.venv/bin/pip install -r requirements-ci.txt

.venv/bin/ruff check api/ tests/ \
  src/inference.py src/mongo_logger.py src/monitoring/ \
  scripts/visualize_predict.py

PYTHONPATH=. .venv/bin/pytest tests/test_inference.py tests/test_mongo_logger.py -q

docker build -f api/Dockerfile -t mask2former-api:ci .
```

Use **`.venv/bin/pip`** / **`.venv/bin/pytest`** — not system Python.

### Verification

After push to `main`, check **Actions** tab on GitHub for green **CI** workflow.

---

## 1. Workflow files

### `ci.yml` (every push / PR to `main`)

See `.github/workflows/ci.yml` in the repo.

### `docker.yml` (`v*` tags)

See `.github/workflows/docker.yml` in the repo.

---

## 2. Tests in CI

| Test file | What it checks |
|-----------|----------------|
| `tests/test_inference.py` | `postprocess_instances` (synthetic logits) |
| `tests/test_mongo_logger.py` | `PredictionLogger` with mocked MongoDB |

Future (not in CI): full `/predict` integration test would require downloading ONNX in the runner (~90 MB) — skipped for speed.

---

## 3. Optional secrets for production deploy

| Secret | Purpose |
|--------|---------|
| `HF_TOKEN` | Private HF model repo |
| `MONGO_URI` | Atlas **mask2former-mlops** logging |
| `ROBOFLOW_API_KEY` | Data pipeline (Colab/local) |

---

## Summary

| Workflow | Trigger | Action |
|----------|---------|--------|
| `ci.yml` | Push / PR → `main` | Ruff + pytest + Docker build |
| `docker.yml` | Tag `v*` or manual | Push image to GHCR |

**Status:** Phase 08 implemented — verify first green run on GitHub Actions after push.

**Next:** [09 — ROS2 Integration](09_ros2.md)
