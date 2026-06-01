# 02 — Data Pipeline

## What We Actually Did (Build Log)

### Fixes Applied During Build

**1. `ShiftScaleRotate` deprecated → replaced with `A.Affine`**
Albumentations deprecated `ShiftScaleRotate` in favour of the more flexible `Affine` transform.

**2. `GaussNoise var_limit` → replaced with `std_range`**
New Albumentations API uses `std_range` (normalized 0-1) instead of `var_limit`.

**3. Roboflow flat structure — no `images/` subfolder**
Roboflow downloads images directly into `data/raw/train/` not `data/raw/train/images/`.
Added `_img_dir()` helper that auto-detects the correct path.

**4. `data/processed` fallback**
`build_dataloaders()` now falls back to `data/raw` automatically if FiftyOne filtering hasn't been run yet.

**5. Graceful handling of unreadable images**
Added None check for `cv2.imread()` — returns a black image instead of crashing.

### Verified Results

```
Train: 1141 samples | 285 batches
Val:   318 samples  | 80 batches

Batch shape:         torch.Size([4, 3, 512, 512])
Batch dtype:         torch.float32
Sample masks shape:  torch.Size([2, 512, 512])
Pixel value range:   [-2.118, 2.640]  (ImageNet normalized)
Different aug per call: True ✅
```

---

## Overview

PyTorch data pipeline that mirrors tf.data behavior:
- **Shuffles** every epoch (different order each time)
- **Different augmentation** per image per epoch (random state changes each call)
- **Parallel loading** with multiple workers
- **Mask-safe transforms** — Albumentations keeps masks in sync with images

```
data/processed/
  ↓
Mask2FormerDataset (PyTorch Dataset)
  ↓  __getitem__: loads image + masks + labels
Albumentations transforms  ← runs fresh every call → different aug each epoch
  ↓
DataLoader (shuffle=True, num_workers=4)
  ↓
Training loop
```

---

## 1. Albumentations Transform Pipelines

```python
# src/data/transforms.py
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ImageNet normalization stats (Swin backbone pretrained on ImageNet)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def get_train_transforms(img_size: int = 512):
    return A.Compose([
        # Geometric — applied to image AND masks simultaneously
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.2,
            rotate_limit=15,
            border_mode=0,          # constant padding (black)
            p=0.5
        ),
        A.RandomResizedCrop(
            height=img_size,
            width=img_size,
            scale=(0.7, 1.0),
            p=0.5
        ),
        A.PadIfNeeded(
            min_height=img_size,
            min_width=img_size,
            border_mode=0
        ),
        A.Resize(height=img_size, width=img_size),

        # Photometric — applied to image only (masks are unaffected)
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.GaussNoise(var_limit=(5.0, 30.0), p=0.2),
        A.RandomShadow(p=0.2),              # useful for outdoor/robot scenes

        # Normalize + convert to tensor
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ],
    # tells Albumentations to also transform segmentation masks
    additional_targets={
        "masks": "masks"
    })


def get_val_transforms(img_size: int = 512):
    return A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ],
    additional_targets={"masks": "masks"})
```

---

## 2. PyTorch Dataset

```python
# src/data/dataset.py
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask


class Mask2FormerDataset(Dataset):
    def __init__(self, img_dir: str, ann_file: str, transforms=None):
        self.img_dir = img_dir
        self.coco = COCO(ann_file)
        self.img_ids = list(self.coco.imgs.keys())
        self.transforms = transforms

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.coco.imgs[img_id]

        # Load image (BGR → RGB)
        img_path = os.path.join(self.img_dir, img_info["file_name"])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load annotations
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        masks = []
        labels = []
        for ann in anns:
            if ann.get("segmentation"):
                rle = coco_mask.frPyObjects(
                    ann["segmentation"],
                    img_info["height"],
                    img_info["width"]
                )
                mask = coco_mask.decode(rle)
                if len(mask.shape) == 3:
                    mask = mask[:, :, 0]
                masks.append(mask.astype(np.uint8))
                labels.append(ann["category_id"])

        # Apply transforms — THIS runs every call with new random state
        # so every epoch, each image gets DIFFERENT augmentation
        if self.transforms and len(masks) > 0:
            augmented = self.transforms(image=image, masks=masks)
            image = augmented["image"]          # tensor [C, H, W]
            masks = augmented["masks"]          # list of H x W arrays
        elif self.transforms:
            augmented = self.transforms(image=image, masks=[np.zeros_like(image[:,:,0])])
            image = augmented["image"]
            masks = []

        # Convert masks to tensor [N, H, W]
        if masks:
            masks = torch.stack([torch.as_tensor(m, dtype=torch.uint8) for m in masks])
        else:
            masks = torch.zeros((0, image.shape[1], image.shape[2]), dtype=torch.uint8)

        labels = torch.as_tensor(labels, dtype=torch.long)

        target = {
            "masks": masks,          # [N, H, W]
            "labels": labels,        # [N]
            "image_id": torch.tensor([img_id]),
        }

        return image, target
```

---

## 3. Custom Collate Function

Variable number of instances per image means we can't use the default collate:

```python
# src/data/dataset.py (continued)
def collate_fn(batch):
    images, targets = zip(*batch)
    # stack images into [B, C, H, W]
    images = torch.stack(images, dim=0)
    # targets stay as list — each element has different N (number of instances)
    return images, list(targets)
```

---

## 4. DataLoader — The tf.data Equivalent

```python
# src/data/dataset.py (continued)
from torch.utils.data import DataLoader
from .transforms import get_train_transforms, get_val_transforms

def build_dataloaders(
    data_root: str = "data/processed",
    img_size: int = 512,
    batch_size: int = 4,
    num_workers: int = 4,
):
    train_dataset = Mask2FormerDataset(
        img_dir=f"{data_root}/train/images",
        ann_file=f"{data_root}/train/_annotations.coco.json",
        transforms=get_train_transforms(img_size),   # augmentation applied per sample per epoch
    )

    val_dataset = Mask2FormerDataset(
        img_dir=f"{data_root}/valid/images",
        ann_file=f"{data_root}/valid/_annotations.coco.json",
        transforms=get_val_transforms(img_size),     # no augmentation for val
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,           # new random order every epoch
        num_workers=num_workers,
        pin_memory=True,        # faster CPU → GPU transfer
        collate_fn=collate_fn,
        drop_last=True,         # avoid batch size 1 causing batchnorm issues
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader
```

### How shuffling + augmentation works per epoch

```
Epoch 1:
  DataLoader picks random order: [img_47, img_3, img_201, img_88, ...]
  img_47 → __getitem__ → Albumentations(random_seed_A) → flipped + bright+0.15 + crop(x1,y1,x2,y2)

Epoch 2:
  DataLoader picks NEW random order: [img_180, img_92, img_47, img_5, ...]
  img_47 → __getitem__ → Albumentations(random_seed_B) → NOT flipped + bright-0.08 + crop(x3,y3,x4,y4)
```

Same behavior as `tf.data.Dataset.shuffle(reshuffle_each_iteration=True)` + `.map(augment_fn)`.

---

## 5. Colab Usage

```python
# In Colab notebook — install deps first
!pip install albumentations pycocotools

# Clone and pull data
!git clone https://github.com/your-user/mask2former.git
%cd mask2former
!dvc remote modify gdrive url "/content/drive/MyDrive/mask2former-mlops"
!dvc pull data/processed

# Build dataloaders
import sys
sys.path.insert(0, "src")
from data.dataset import build_dataloaders

train_loader, val_loader = build_dataloaders(
    data_root="data/processed",
    img_size=512,
    batch_size=4,
    num_workers=2,   # Colab has limited CPUs
)

print(f"Train batches: {len(train_loader)}")
print(f"Val batches:   {len(val_loader)}")

# Verify a batch
images, targets = next(iter(train_loader))
print(f"Image batch shape: {images.shape}")         # [4, 3, 512, 512]
print(f"Masks in sample 0: {targets[0]['masks'].shape}")  # [N, 512, 512]
```

---

## 6. params.yaml — Data Parameters

```yaml
# params.yaml
data:
  img_size: 512
  batch_size: 4
  num_workers: 2       # 2 for Colab, 4 for local
  train_dir: data/processed/train
  val_dir: data/processed/valid
  calibration_dir: data/calibration
```

---

## Summary

| Feature | Implementation |
|---|---|
| Shuffle every epoch | `DataLoader(shuffle=True)` |
| Different aug per epoch | Albumentations in `__getitem__` (new random state each call) |
| Mask-safe transforms | `additional_targets={"masks": "masks"}` in Albumentations |
| Parallel loading | `num_workers=4` |
| Variable instances | custom `collate_fn` |

**Next:** [03 — Training](03_training.md)
