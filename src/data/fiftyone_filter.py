import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import fiftyone as fo
from src.config_loader import load_config


def load_split(cfg, split: str):
    raw = cfg.data.raw_dir
    name = f"mask2former_{split}"

    if fo.dataset_exists(name):
        fo.delete_dataset(name)

    dataset = fo.Dataset.from_dir(
        dataset_type=fo.types.COCODetectionDataset,
        data_path=os.path.join(raw, split, "images"),
        labels_path=os.path.join(raw, split, cfg.data.ann_filename),
        name=name,
        label_types=["segmentations"],
    )
    print(f"Loaded {split}: {len(dataset)} samples")
    return dataset


def launch_review(split: str = "train"):
    cfg = load_config()
    dataset = load_split(cfg, split)
    print("\nFiftyOne UI launched. Review annotations in your browser.")
    print("Tag bad samples as 'delete', then run export_filtered() to save cleaned data.\n")
    session = fo.launch_app(dataset, remote=False)
    return session, dataset


def export_filtered(dataset, split: str, cfg=None):
    if cfg is None:
        cfg = load_config()

    bad = dataset.match_tags("delete")
    n_bad = len(bad)
    dataset.remove_samples(bad)
    print(f"Removed {n_bad} bad samples. Remaining: {len(dataset)}")

    out_dir = os.path.join(cfg.data.processed_dir, split)
    dataset.export(
        export_dir=out_dir,
        dataset_type=fo.types.COCODetectionDataset,
        label_field="ground_truth",
    )
    print(f"Exported filtered dataset → {out_dir}")
    return out_dir


def prepare_calibration(cfg=None, n: int = None):
    import shutil, random
    if cfg is None:
        cfg = load_config()
    if n is None:
        n = cfg.data.calibration.n_images

    src = os.path.join(cfg.data.processed_dir, cfg.data.train_subdir, "images")
    dst = cfg.data.calibration_dir
    os.makedirs(dst, exist_ok=True)

    images = [f for f in os.listdir(src) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    selected = random.sample(images, min(n, len(images)))
    for img in selected:
        shutil.copy(os.path.join(src, img), os.path.join(dst, img))

    print(f"Calibration set: {len(selected)} images → {dst}")
    return dst


if __name__ == "__main__":
    session, dataset = launch_review("train")
    input("Press Enter after tagging bad samples in the browser...")
    cfg = load_config()
    export_filtered(dataset, "train", cfg)
    export_filtered(load_split(cfg, "valid"), "valid", cfg)
    prepare_calibration(cfg)
