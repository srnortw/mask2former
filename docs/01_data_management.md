# 01 — Data Management

## Overview

```
Roboflow (download COCO format)
  ↓
FiftyOne (visual QA, filter bad samples)
  ↓
DVC (version control for data)
  ↓
Google Drive 200GB (GNOME mounted, remote storage)
```

---

## 1. Environment Setup (local venv)

```bash
cd ~/Desktop/mask2former
python3 -m venv .venv
source .venv/bin/activate

pip install roboflow fiftyone dvc pymongo
```

---

## 2. Project & DVC Initialization

```bash
git init
dvc init
git add .dvc .gitignore
git commit -m "init: dvc setup"
```

### Set Google Drive as DVC remote

Google Drive is already mounted via GNOME at:
```
/run/user/1000/gvfs/google-drive:host=gmail.com,user=srnortw/
```

```bash
# Create a folder on Drive for this project first (via Files app or Drive web)
# Then point DVC to it:
DRIVE_PATH="/run/user/1000/gvfs/google-drive:host=gmail.com,user=srnortw/mask2former-mlops"
dvc remote add -d gdrive "$DRIVE_PATH"
dvc remote modify gdrive type local   # GNOME mount looks like local filesystem to DVC

git add .dvc/config
git commit -m "config: add google drive dvc remote"
```

> **Note:** This path is only available while logged into GNOME desktop. In Colab, use `drive.mount('/content/drive')` and set the DVC remote to the equivalent path.

---

## 3. Dataset Download from Roboflow

```python
# src/data/roboflow_download.py
from roboflow import Roboflow
import os

def download_dataset(api_key: str, workspace: str, project: str, version: int):
    rf = Roboflow(api_key=api_key)
    proj = rf.workspace(workspace).project(project)
    dataset = proj.version(version).download(
        "coco-segmentation",        # COCO instance segmentation format
        location="data/raw"
    )
    return dataset.location

if __name__ == "__main__":
    download_dataset(
        api_key=os.environ["ROBOFLOW_API_KEY"],
        workspace="your-workspace",
        project="your-project",
        version=1
    )
```

Expected output structure after download:
```
data/raw/
├── train/
│   ├── images/           # .jpg files
│   └── _annotations.coco.json
├── valid/
│   ├── images/
│   └── _annotations.coco.json
└── test/
    ├── images/
    └── _annotations.coco.json
```

---

## 4. FiftyOne — Visual QA & Filtering

FiftyOne launches a browser UI to inspect every image and annotation.

```python
# src/data/fiftyone_filter.py
import fiftyone as fo
import fiftyone.utils.coco as fouc

def load_and_inspect(split: str = "train"):
    dataset = fo.Dataset.from_dir(
        dataset_type=fo.types.COCODetectionDataset,
        data_path=f"data/raw/{split}/images",
        labels_path=f"data/raw/{split}/_annotations.coco.json",
        name=f"mask2former_{split}",
        label_types=["segmentations"],
    )
    session = fo.launch_app(dataset)
    return session, dataset


def export_filtered(dataset, output_dir: str):
    """Export filtered samples back to COCO format."""
    dataset.export(
        export_dir=output_dir,
        dataset_type=fo.types.COCODetectionDataset,
        label_field="ground_truth",
    )
```

### What to look for in FiftyOne UI

- Incorrect or missing polygon annotations
- Images with no labels (background only — keep a small %)
- Severely blurry or corrupted images
- Wrong class labels
- Annotations that don't align with object boundaries

### Tag and delete bad samples

```python
# In FiftyOne UI: tag bad samples as "delete"
# Then in code:
bad_samples = dataset.match_tags("delete")
dataset.remove_samples(bad_samples)
print(f"Remaining: {len(dataset)} samples")
```

---

## 5. Version Data with DVC

```bash
mkdir -p data/raw data/processed data/calibration

# After downloading and filtering:
dvc add data/raw
dvc add data/processed

git add data/raw.dvc data/processed.dvc data/.gitignore
git commit -m "data: add raw and filtered dataset v1"

# Push data to Google Drive
dvc push
```

> `.dvc` files are small metadata files tracked by git.
> Actual data lives on Google Drive.

### Pull data on a new machine (e.g. Colab)

```bash
# In Colab:
from google.colab import drive
drive.mount('/content/drive')

!git clone https://github.com/your-user/mask2former.git
%cd mask2former

# Override remote to point to Colab Drive mount
!dvc remote modify gdrive url "/content/drive/MyDrive/mask2former-mlops"
!dvc pull
```

---

## 6. Calibration Data for INT8 Quantization

Select ~200 representative images from the training set for ONNX INT8 calibration:

```python
# src/data/prepare_calibration.py
import shutil, random, os

def prepare_calibration(src_dir: str, dst_dir: str, n: int = 200):
    os.makedirs(dst_dir, exist_ok=True)
    images = [f for f in os.listdir(src_dir) if f.endswith(('.jpg', '.png'))]
    selected = random.sample(images, min(n, len(images)))
    for img in selected:
        shutil.copy(os.path.join(src_dir, img), os.path.join(dst_dir, img))
    print(f"Calibration set: {len(selected)} images → {dst_dir}")
```

```bash
dvc add data/calibration
git add data/calibration.dvc
git commit -m "data: add calibration set for int8 quantization"
dvc push
```

---

## 7. DVC Pipeline Stage (dvc.yaml)

```yaml
# dvc.yaml
stages:
  download_data:
    cmd: python src/data/roboflow_download.py
    deps:
      - src/data/roboflow_download.py
    outs:
      - data/raw

  filter_data:
    cmd: python src/data/fiftyone_filter.py
    deps:
      - src/data/fiftyone_filter.py
      - data/raw
    outs:
      - data/processed

  prepare_calibration:
    cmd: python src/data/prepare_calibration.py
    deps:
      - src/data/prepare_calibration.py
      - data/processed
    outs:
      - data/calibration
```

---

## 8. Dataset Stats

After downloading from Roboflow:

```python
from pycocotools.coco import COCO

coco = COCO("data/raw/train/_annotations.coco.json")
print(f"Images:      {len(coco.imgs)}")
print(f"Annotations: {len(coco.anns)}")
print(f"Categories:  {[c['name'] for c in coco.loadCats(coco.getCatIds())]}")
```

---

## Summary

| Step | Tool | Output |
|---|---|---|
| Download | Roboflow SDK | `data/raw/` COCO JSON |
| Visual QA | FiftyOne | `data/processed/` cleaned |
| Calibration prep | Python | `data/calibration/` 200 imgs |
| Version | DVC | `.dvc` files in git |
| Storage | Google Drive (GNOME) | Full data on Drive |

**Next:** [02 — Data Pipeline](02_data_pipeline.md)
