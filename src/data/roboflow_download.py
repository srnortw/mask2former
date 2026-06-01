import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from roboflow import Roboflow
from src.config_loader import load_config


def download_dataset(cfg=None):
    if cfg is None:
        cfg = load_config()

    rf = Roboflow(api_key=cfg.services.roboflow.api_key)
    project = rf.workspace(cfg.services.roboflow.workspace).project(cfg.services.roboflow.project)
    dataset = project.version(int(cfg.services.roboflow.version)).download(
        cfg.services.roboflow.format,
        location=cfg.data.raw_dir,
    )
    print(f"Dataset downloaded to: {dataset.location}")
    return dataset.location


if __name__ == "__main__":
    download_dataset()
