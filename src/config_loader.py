"""
Loads config.yaml and resolves ${VAR_NAME} placeholders from .env
Usage:
    from config_loader import load_config
    cfg = load_config()
    print(cfg.training.total_epochs)
    print(cfg.services.mongodb.uri)
"""

import os
import re
import yaml
from pathlib import Path
from dotenv import load_dotenv
from types import SimpleNamespace


def _resolve_env_vars(obj):
    """Recursively resolve ${VAR_NAME} placeholders using environment variables."""
    if isinstance(obj, str):
        matches = re.findall(r'\$\{(\w+)\}', obj)
        for var in matches:
            value = os.environ.get(var, "")
            if not value:
                print(f"  WARNING: env var '{var}' not set (referenced in config.yaml)")
            obj = obj.replace(f"${{{var}}}", value)
        return obj
    elif isinstance(obj, dict):
        return {k: _resolve_env_vars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_resolve_env_vars(i) for i in obj]
    return obj


def _dict_to_namespace(d):
    """Recursively convert dict to dot-accessible SimpleNamespace."""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [_dict_to_namespace(i) for i in d]
    return d


def load_config(
    config_path: str = "config.yaml",
    env_path: str = ".env",
) -> SimpleNamespace:
    # Load .env into environment
    env_file = Path(env_path)
    if env_file.exists():
        load_dotenv(env_file)
    else:
        print(f"WARNING: .env file not found at '{env_path}'")

    # Load yaml
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"config.yaml not found at '{config_path}'")

    with open(config_file, "r") as f:
        raw = yaml.safe_load(f)

    # Resolve ${VAR_NAME} → actual values from environment
    resolved = _resolve_env_vars(raw)

    # Convert to dot-accessible namespace
    return _dict_to_namespace(resolved)


if __name__ == "__main__":
    cfg = load_config()

    print("=== Config loaded ===")
    print(f"Backbone:          {cfg.model.backbone}")
    print(f"Total epochs:      {cfg.training.total_epochs}")
    print(f"Batch size:        {cfg.data.dataloader.batch_size}")
    print(f"Image size:        {cfg.data.dataloader.img_size}")
    print(f"Phase1 LR:         {cfg.learning_rate.phase1}")
    print(f"Phase2 LR:         {cfg.learning_rate.phase2}")
    print(f"Phase3 LR:         {cfg.learning_rate.phase3}")
    print(f"Scheduler T_0:     {cfg.scheduler.T_0}")
    print(f"Scheduler T_mult:  {cfg.scheduler.T_mult}")
    print(f"Phase1 epochs:     {cfg.freezing.phase1.start_epoch} → {cfg.freezing.phase1.end_epoch}")
    print(f"Phase2 epochs:     {cfg.freezing.phase2.start_epoch} → {cfg.freezing.phase2.end_epoch}")
    print(f"Phase3 epochs:     {cfg.freezing.phase3.start_epoch} → {cfg.freezing.phase3.end_epoch}")
    print(f"MLflow URI:        {cfg.services.mlflow.tracking_uri}")
    print(f"MongoDB DB:        {cfg.services.mongodb.db_name}")
