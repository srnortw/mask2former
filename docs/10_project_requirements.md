# 10 — Project Requirements & Account Setup

Everything needed before writing a single line of code.

---

## Accounts to Create

### 1. GitHub
**URL:** https://github.com  
**Plan:** Free  
**Why:** Code hosting, CI/CD (GitHub Actions), Docker image registry (GHCR)

- Create account
- Repository: [`srnortw/mask2former`](https://github.com/srnortw/mask2former) (public)
- `GITHUB_TOKEN` optional in Colab (only needed if you `git push` from the notebook)

```bash
# Authenticate GitHub CLI (already installed)
gh auth login
```

---

### 2. Google Account (Colab + Drive)
**URL:** https://colab.research.google.com  
**Plan:** Colab Pro (already paying) + Google Drive 200GB (already have)  
**Why:** GPU training, data + checkpoint storage

- Already set up — GNOME Drive mount confirmed at:
  ```
  /run/user/1000/gvfs/google-drive:host=gmail.com,user=srnortw/
  ```
- Folder on Drive: `mask2former-mlops/` with subfolders `mask2former/` (git), `data/raw/`, `checkpoints/`

---

### 3. Roboflow
**URL:** https://roboflow.com  
**Plan:** Free (up to 10k images, 3 workspaces)  
**Why:** Dataset hosting, annotation, COCO format export

- Create account
- Create workspace and project (instance segmentation type)
- Upload and annotate images (or import existing annotations)
- Export as **COCO Segmentation** format
- Go to **Settings → API Keys** → copy your API key
- Save as `ROBOFLOW_API_KEY`

---

### 4. Hugging Face
**URL:** https://huggingface.co  
**Plan:** Free (unlimited private models)  
**Why:** Model registry for ONNX artifacts (fp32 + INT8)

- Create account
- Go to **Settings → Access Tokens** → New token → Role: `write`
- Save as `HF_TOKEN`
- Create a new model repository: `your-username/mask2former-instance-seg` (private)

```bash
# Activate venv (already created and huggingface_hub installed)
source ~/Desktop/mask2former/.venv/bin/activate

# Login — note: CLI command is now 'hf', not 'huggingface-cli' (deprecated)
hf auth login
# paste your HF_TOKEN when prompted
```

---

### 5. MongoDB Atlas
**URL:** https://cloud.mongodb.com  
**Plan:** Free (M0 cluster, 512 MB)  
**Why:** Store prediction metadata, drift monitoring data (Phase 07)  
**Note:** Atlas CLI already installed (`atlascli 1.55.0`)

- Atlas project: **`mask2former-mlops`** (id `6a21b995cbf3f23e5981be8f`)
- Cluster: **`mask2former-cluster`** (M0, `EU_CENTRAL_1`)
- DB user: `mask2former_api` → database `mask2former`
- Or run: `./scripts/setup_atlas_mongo.sh` after `atlas auth login`
- Do **not** use old **Cluster0** (stuck `UPDATING` in separate CV project)

- Create account (if new)
- Create project `mask2former-mlops` + M0 cluster (script does this)
- Go to **Database Access** → Add user → password auth
- Go to **Network Access** → Add IP → `0.0.0.0/0` (allow all, or restrict to your IP)
- Go to **Clusters → Connect** → copy connection string
- Save as `MONGO_URI` for project **mask2former-mlops**, cluster **mask2former-cluster** (see `.env.example`)

```bash
# Authenticate Atlas CLI
atlas auth login
```

---

### 6. Render.com (MLflow tracking server)
**URL:** https://render.com  
**Plan:** Free web service (spins down after 15min inactivity)  
**Why:** Host MLflow UI accessible from Colab during training

- Create account (sign in with GitHub)
- New → **Web Service** → connect your GitHub repo
- Settings:
  - **Build command:** `pip install mlflow`
  - **Start command:** `mlflow server --host 0.0.0.0 --port $PORT --backend-store-uri sqlite:///mlflow.db`
  - **Instance type:** Free
- Copy the service URL: `https://your-app.onrender.com`
- Save as `MLFLOW_TRACKING_URI`

> Free tier sleeps after 15 minutes. Send a request to wake it up before training starts.

---

## Software Prerequisites (Local Machine)

Already installed on your system:

| Tool | Version | Status |
|---|---|---|
| Python | 3.12.3 | ✅ installed |
| DVC | 3.67.1 | ✅ installed |
| Docker | 29.1.3 | ✅ installed |
| GitHub CLI (`gh`) | 2.45.0 | ✅ installed |
| Atlas CLI | 1.55.0 | ✅ installed |
| Google Drive (GNOME) | — | ✅ mounted |
| `torch` | 2.11.0 | ✅ installed globally |
| `onnxruntime` | 1.25.1 | ✅ installed globally |
| `roboflow` | 1.3.7 | ✅ installed globally |

Still needed (install in project venv):

```bash
cd ~/Desktop/mask2former
python3 -m venv .venv
source .venv/bin/activate

pip install \
  dvc \
  roboflow \
  fiftyone \
  mlflow \
  pymongo \
  evidently \
  albumentations \
  pycocotools \
  onnx \
  onnxruntime \
  fastapi \
  uvicorn \
  huggingface_hub \
  python-dotenv
```

---

## Environment Variables Checklist

Copy `.env.example` → `.env` and fill in secrets. The example documents Atlas project **mask2former-mlops**.

A `.env` file already exists in the project root at `mask2former/.env`.  
**Open it and replace every placeholder with your real values** after completing the account setup steps above.

```bash
# open the file
nano ~/Desktop/mask2former/.env
```

Here is every variable and exactly where to get it:

| Variable | Where to get it | Example value |
|---|---|---|
| `ROBOFLOW_API_KEY` | Roboflow → Settings → API Keys | `abc123xyz` |
| `ROBOFLOW_WORKSPACE` | Roboflow → your workspace name in the URL | `my-workspace` |
| `ROBOFLOW_PROJECT` | Roboflow → project name in the URL | `my-project` |
| `ROBOFLOW_VERSION` | Roboflow → dataset version number | `1` |
| `HF_TOKEN` | Hugging Face → Settings → Access Tokens | `hf_xxxx...` |
| `HF_REPO_ID` | Your HF username + repo name | `yourname/mask2former-instance-seg` |
| `MONGO_URI` | Atlas **mask2former-mlops** → **mask2former-cluster** → Connect | `mongodb+srv://mask2former_api:...@mask2former-cluster.*.mongodb.net/mask2former` |
| `MONGO_DB_NAME` | database in that cluster (default: `mask2former`) | `mask2former` |
| `MONGO_COLLECTION_PREDICTIONS` | collection name (default: `predictions`) | `predictions` |
| `MONGO_COLLECTION_DRIFT` | collection name (default: `drift_reports`) | `drift_reports` |
| `ATLAS_PROJECT_ID` | Atlas → Project Settings → ID | `6a21b995cbf3f23e5981be8f` |
| `ATLAS_CLUSTER` | cluster name in mask2former-mlops | `mask2former-cluster` |
| `MLFLOW_TRACKING_URI` | Render.com → your service URL | `https://your-app.onrender.com` |
| `MLFLOW_EXPERIMENT_NAME` | your choice | `mask2former-swin` |
| `GITHUB_TOKEN` | GitHub → Settings → Developer Settings → Tokens | `ghp_xxxx...` |
| `GITHUB_REPO` | your GitHub username + repo name | `yourname/mask2former` |

> `.env` is listed in `.gitignore` — it will never be accidentally committed.

Add to `.gitignore` (already done):
```
.env
.venv/
__pycache__/
*.pyc
checkpoints/*.pth
checkpoints/*.onnx
```

Load in Python:
```python
from dotenv import load_dotenv
load_dotenv()
import os

mongo_uri = os.environ["MONGO_URI"]
```

Or use the project's config loader which handles this automatically:
```python
from src.config_loader import load_config
cfg = load_config()   # reads .env + config.yaml together
```

---

## GitHub Repository Secrets

After creating the repo, go to **Settings → Secrets and variables → Actions** and add:

| Secret Name | Value |
|---|---|
| `HF_TOKEN` | Hugging Face write token |
| `MONGO_URI` | MongoDB Atlas connection string |
| `ROBOFLOW_API_KEY` | Roboflow API key |

These are used by GitHub Actions CI/CD workflows.

---

## Colab Secrets

In Colab Pro, go to **🔑 Secrets (left sidebar)** and add:

| Secret Name | Value |
|---|---|
| `HF_TOKEN` | Hugging Face write token |
| `GITHUB_TOKEN` | Fine-grained PAT for private repo clone |
| `MONGO_URI` | MongoDB Atlas connection string |
| `ROBOFLOW_API_KEY` | Roboflow API key |
| `MLFLOW_TRACKING_URI` | `https://dagshub.com/srnortw/mask2former.mlflow` |
| `MLFLOW_TRACKING_PASSWORD` | DagsHub token |

Access in notebook:
```python
from google.colab import userdata
import os

os.environ["HF_TOKEN"]            = userdata.get("HF_TOKEN")
os.environ["MONGO_URI"]           = userdata.get("MONGO_URI")
os.environ["ROBOFLOW_API_KEY"]    = userdata.get("ROBOFLOW_API_KEY")
os.environ["MLFLOW_TRACKING_URI"] = userdata.get("MLFLOW_TRACKING_URI")
```

---

## Setup Order (Step by Step)

```
1.  Create GitHub account + repo              → get GITHUB_TOKEN
2.  gh auth login                             → authenticate CLI
3.  Create Roboflow account + project         → get ROBOFLOW_API_KEY
4.  Create Hugging Face account               → get HF_TOKEN
5.  MongoDB Atlas project **mask2former-mlops** + cluster **mask2former-cluster** → `MONGO_URI` (see `setup_atlas_mongo.sh`)
6.  Create Render.com account + MLflow service → get MLFLOW_TRACKING_URI
7.  Create Drive folder: mask2former-mlops    → (already have Drive)
8.  cd ~/Desktop/mask2former
9.  python3 -m venv .venv && source .venv/bin/activate
10. pip install (deps above)
11. git init && dvc init
12. dvc remote add -d gdrive "<gnome-drive-path>/mask2former-mlops"
13. Fill in .env with real values (file already exists, just replace placeholders)
14. Add secrets to GitHub repo settings
15. Add secrets to Colab
16. git remote add origin https://github.com/your-user/mask2former.git
17. git push -u origin main
```

---

## Summary

| Account | Free? | Required for |
|---|---|---|
| GitHub | Yes | Code, CI/CD, Docker registry |
| Google (Colab Pro) | Already paying | GPU training |
| Google Drive | Already have | Data + checkpoint storage |
| Roboflow | Yes | Dataset download |
| Hugging Face | Yes | ONNX model registry |
| MongoDB Atlas | Yes (512 MB) | Prediction logging |
| Render.com | Yes | MLflow tracking UI |

**Total new accounts to create: 5**  
**Total extra monthly cost: $0**
