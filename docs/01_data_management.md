# 01 — Data Management

## Overview

```
Roboflow (download COCO segmentation format)
  ↓
DVC (version control for data)
  ↓
Google Drive 200GB via rclone FUSE mount (remote storage)
  ↓
FiftyOne (visual QA, filter bad samples)  ← next step
```

---

## What We Actually Did (Build Log)

### Step 1 — Create venv and install dependencies

```bash
cd ~/Desktop/mask2former
python3 -m venv .venv
source .venv/bin/activate

pip install dvc python-dotenv roboflow fiftyone albumentations pycocotools "dvc[gdrive]"
```

---

### Step 2 — Initialize Git + GitHub repo + DVC

```bash
# Configure git identity
git config user.name "Serkan"
git config user.email "srnortw@users.noreply.github.com"

# Init git and make first commit
git init
git add .gitignore config.yaml docs/ src/
git commit -m "init: project structure, config, pipeline docs"

# Create private GitHub repo and push (using gh CLI — already authenticated)
gh repo create mask2former --private --source=. --remote=origin --push

# Init DVC
dvc init
```

GitHub repo: https://github.com/srnortw/mask2former

---

### Step 3 — Download Dataset from Roboflow

Project: **Lane Detection (Segmentation)**
- Workspace: `test-mfeql`
- Project slug: `lane-detection-segmentation-edyqp-fibkz`
- Version: `1`
- Format: `coco-segmentation`

> **Important:** You must generate at least one version in Roboflow before downloading.
> Go to your project → click **"Generate New Version"** → skip augmentations → Generate.

```bash
source .venv/bin/activate
python src/data/roboflow_download.py
```

The script reads all values from `.env` via `config_loader.py` — no hardcoded credentials.

**Dataset downloaded to:** `data/raw/`

```
data/raw/
├── train/
│   ├── *.jpg  (1141 images)
│   └── _annotations.coco.json
├── valid/
│   ├── *.jpg  (318 images)
│   └── _annotations.coco.json
└── test/
    ├── *.jpg  (151 images)
    └── _annotations.coco.json
```

**Verified dataset stats:**

| Split | Images | Annotations |
|---|---|---|
| Train | 1,141 | 3,026 |
| Valid | 318 | 853 |
| Test | 151 | 385 |
| **Total** | **1,610** | **4,264** |

**Classes (5):**
- `Lane-Markings`
- `Left Boundary -Dashed-`
- `Left Boundary -Solid-`
- `Right Boundary -Dashed-`
- `Right Boundary -Solid-`

`config.yaml` updated: `model.num_classes: 5`

---

### Step 4 — Google Drive Setup with rclone

**Why rclone instead of GNOME mount:**
GNOME GVFS mount auto-disconnects during bulk file transfers and DVC's config parser misreads the comma in the GVFS path (`google-drive:host=gmail.com,user=srnortw`). rclone stays connected and is designed for bulk transfers.

**Install rclone:**
```bash
sudo apt install rclone
```

**Configure rclone with Google Drive (one-time setup):**
```bash
rclone config create gdrive drive scope=drive
# Browser opens automatically → sign in → allow access
```

**Verify connection:**
```bash
rclone lsd gdrive:
# Should show your Drive folders including mask2former-mlops
```

**Mount Google Drive as FUSE filesystem:**
```bash
mkdir -p ~/rclone-gdrive
rclone mount gdrive:mask2former-mlops ~/rclone-gdrive --daemon --vfs-cache-mode writes
```

> Run this mount command each time before `dvc push/pull`.
> Check if already mounted: `ls ~/rclone-gdrive`

---

### Step 5 — DVC Remote + Push Data to Drive

```bash
# Set rclone FUSE mount as DVC remote
dvc remote add -d gdrive "$HOME/rclone-gdrive"

# Version the raw dataset
dvc add data/raw

# Push 1617 files to Google Drive
dvc push
# Output: 1617 files pushed ✅
```

**Commit DVC metadata to git:**
```bash
git add data/raw.dvc data/.gitignore .dvc/config .dvc/.gitignore .dvcignore
git commit -m "feat: data pipeline — roboflow download, DVC tracking, rclone Drive remote"
git push
```

> DVC `.dvc` files are tiny metadata files (~1KB) that track the dataset version.
> Actual image files live on Google Drive — never in git.

---

### Google Drive folder layout (`mask2former-mlops`)

| Path on Drive | Purpose |
|---------------|---------|
| `files/` | DVC remote storage (dataset cache — from `dvc push`) |
| `checkpoints/` | Optional manual backups (Colab Cells 11/18) |
| ~~`mask2former/`~~ | **Not used** — was a one-time whole-repo copy; safe to delete |

Check contents: `rclone lsd gdrive:mask2former-mlops`

We do **not** sync the full project to Drive. Code → GitHub. Colab → `git pull` + Roboflow + HF Hub.

### Colab: code via Git, data via Roboflow

Colab does not use `dvc pull` for this project. Sync scripts with `git pull`; download data with Roboflow (notebook Cell 4).

---

## Next Step — FiftyOne Visual QA

Before training, launch FiftyOne to inspect annotations:

```bash
source .venv/bin/activate
python src/data/fiftyone_filter.py
# Opens browser UI at http://localhost:5151
```

**What to look for:**
- Annotations not aligned with lane boundaries
- Images with missing labels
- Blurry or corrupted frames
- Wrong class assigned to a lane

**Tag bad samples as `delete` in UI, then export filtered dataset:**
```python
# After review in browser:
export_filtered(dataset, "train", cfg)
export_filtered(dataset, "valid", cfg)
# → saves to data/processed/

# Then version processed data
# dvc add data/processed && dvc push
```

---

## Source Files Written

| File | Purpose |
|---|---|
| `src/data/roboflow_download.py` | Download dataset from Roboflow using config.yaml values |
| `src/data/dataset.py` | PyTorch Dataset + DataLoader builder |
| `src/data/transforms.py` | Albumentations train/val pipelines (driven by config.yaml) |
| `src/data/fiftyone_filter.py` | FiftyOne QA, filtering, calibration set prep |
| `src/config_loader.py` | Loads `.env` + `config.yaml` → dot-accessible namespace |

---

## Summary

| Step | Tool | Status |
|---|---|---|
| Virtual environment | Python venv | ✅ done |
| Code repository | GitHub (`srnortw/mask2former`) | ✅ done |
| Dataset download | Roboflow SDK → `data/raw/` | ✅ done |
| Data versioning | DVC | ✅ done |
| Drive storage | rclone → Google Drive 200GB | ✅ 1617 files pushed |
| Visual QA | FiftyOne | ⏳ next step |

**Next:** [02 — Data Pipeline](02_data_pipeline.md)
