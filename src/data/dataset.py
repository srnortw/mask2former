import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask

from .transforms import get_train_transforms, get_val_transforms


class Mask2FormerDataset(Dataset):
    def __init__(self, img_dir: str, ann_file: str, transforms=None):
        self.img_dir = img_dir
        self.ann_file = ann_file
        self.coco = COCO(ann_file)
        self.img_ids = list(self.coco.imgs.keys())
        self.transforms = transforms

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.coco.imgs[img_id]

        img_path = os.path.join(self.img_dir, img_info["file_name"])
        image = cv2.imread(img_path)
        if image is None:
            # fallback: return black image if file is missing/corrupt
            h = w = self.img_dir and 512
            image = np.zeros((512, 512, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        masks, labels = [], []
        for ann in anns:
            if ann.get("segmentation"):
                rle = coco_mask.frPyObjects(
                    ann["segmentation"],
                    img_info["height"],
                    img_info["width"],
                )
                mask = coco_mask.decode(rle)
                if len(mask.shape) == 3:
                    mask = mask[:, :, 0]
                masks.append(mask.astype(np.uint8))
                labels.append(ann["category_id"])

        if self.transforms and len(masks) > 0:
            augmented = self.transforms(image=image, masks=masks)
            image = augmented["image"]
            masks = augmented["masks"]
        elif self.transforms:
            h, w = image.shape[:2]
            augmented = self.transforms(image=image, masks=[np.zeros((h, w), dtype=np.uint8)])
            image = augmented["image"]
            masks = []

        if masks:
            masks = torch.stack([torch.as_tensor(m, dtype=torch.uint8) for m in masks])
        else:
            masks = torch.zeros(
                (0, self.transforms and image.shape[1] or image.shape[0],
                    self.transforms and image.shape[2] or image.shape[1]),
                dtype=torch.uint8,
            )

        target = {
            "masks":    masks,
            "labels":   torch.as_tensor(labels, dtype=torch.long),
            "image_id": torch.tensor([img_id]),
        }
        return image, target


def collate_fn(batch):
    images, targets = zip(*batch)
    return torch.stack(images, dim=0), list(targets)


def _resolve_data_root(cfg):
    """Use processed data if available, fall back to raw."""
    root = cfg.data.processed_dir
    ann = os.path.join(root, cfg.data.train_subdir, cfg.data.ann_filename)
    if not os.path.exists(ann):
        root = cfg.data.raw_dir
        print("data/processed not found — using data/raw (run FiftyOne filter first)")
    return root


def _img_dir(root, split):
    """Roboflow puts images directly in split folder, not in split/images/."""
    images_subdir = os.path.join(root, split, "images")
    if os.path.isdir(images_subdir):
        return images_subdir
    return os.path.join(root, split)


def build_dataloaders(cfg):
    root = _resolve_data_root(cfg)
    dl = cfg.data.dataloader

    train_dataset = Mask2FormerDataset(
        img_dir=_img_dir(root, cfg.data.train_subdir),
        ann_file=os.path.join(root, cfg.data.train_subdir, cfg.data.ann_filename),
        transforms=get_train_transforms(cfg),
    )
    val_dataset = Mask2FormerDataset(
        img_dir=_img_dir(root, cfg.data.val_subdir),
        ann_file=os.path.join(root, cfg.data.val_subdir, cfg.data.ann_filename),
        transforms=get_val_transforms(cfg),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=dl.batch_size,
        shuffle=True,
        num_workers=dl.num_workers,
        pin_memory=dl.pin_memory,
        collate_fn=collate_fn,
        drop_last=dl.drop_last,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=dl.batch_size,
        shuffle=False,
        num_workers=dl.num_workers,
        pin_memory=dl.pin_memory,
        collate_fn=collate_fn,
    )

    print(f"Train: {len(train_dataset)} samples | {len(train_loader)} batches")
    print(f"Val:   {len(val_dataset)} samples | {len(val_loader)} batches")
    return train_loader, val_loader
