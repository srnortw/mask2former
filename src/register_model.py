import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient

from config_loader import load_config


def register_in_mlflow(
    run_id: str,
    checkpoint_path: str,
    fp32_onnx_path: str,
    int8_onnx_path: str,
    cfg=None,
) -> str:
    """
    Register the trained model in MLflow Model Registry and log ONNX artifacts.
    Returns the registered model version number.
    """
    if cfg is None:
        cfg = load_config()

    os.environ["MLFLOW_TRACKING_USERNAME"] = cfg.services.mlflow.tracking_username
    os.environ["MLFLOW_TRACKING_PASSWORD"] = cfg.services.mlflow.tracking_password
    mlflow.set_tracking_uri(cfg.services.mlflow.tracking_uri)

    client = MlflowClient()
    model_name = "mask2former-lane-seg"

    # Log ONNX artifacts into the existing training run
    print(f"Logging ONNX artifacts to run {run_id}...")
    with mlflow.start_run(run_id=run_id):
        for path, name in [
            (fp32_onnx_path, "onnx/fp32"),
            (int8_onnx_path, "onnx/int8"),
            (checkpoint_path, "checkpoints"),
        ]:
            if os.path.exists(path):
                mlflow.log_artifact(path, artifact_path=name)
                print(f"  Logged: {path} → {name}")

    # Register model from the run's checkpoint artifact
    model_uri = f"runs:/{run_id}/checkpoints"
    print(f"\nRegistering model '{model_name}' from {model_uri}...")
    result = mlflow.register_model(model_uri=model_uri, name=model_name)
    version = result.version
    print(f"Registered as version {version}")

    # Tag the version with useful metadata
    client.set_model_version_tag(model_name, version, "backbone", cfg.model.backbone)
    client.set_model_version_tag(model_name, version, "num_classes", str(cfg.model.num_classes))
    client.set_model_version_tag(model_name, version, "hf_repo", os.environ.get("HF_REPO_ID", ""))
    client.set_model_version_tag(model_name, version, "onnx_opset", "16")
    client.set_model_version_tag(model_name, version, "quantization", "selective_static_QInt8_QUInt8")

    # Auto-transition to Staging
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage="Staging",
        archive_existing_versions=False,
    )
    print(f"Version {version} → Staging")

    return version


def promote_to_production(version: str, cfg=None):
    """Promote a model version to Production after ROS2 validation."""
    if cfg is None:
        cfg = load_config()

    os.environ["MLFLOW_TRACKING_USERNAME"] = cfg.services.mlflow.tracking_username
    os.environ["MLFLOW_TRACKING_PASSWORD"] = cfg.services.mlflow.tracking_password
    mlflow.set_tracking_uri(cfg.services.mlflow.tracking_uri)

    client = MlflowClient()
    model_name = "mask2former-lane-seg"

    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage="Production",
        archive_existing_versions=True,  # archive previous production version
    )
    print(f"Version {version} → Production (previous versions archived)")


def push_model_card(repo_id: str, hf_token: str, cfg=None):
    """Upload MODEL_CARD.md as the HuggingFace Hub README."""
    from huggingface_hub import HfApi

    if cfg is None:
        cfg = load_config()

    card_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "MODEL_CARD.md"
    )
    if not os.path.exists(card_path):
        print("MODEL_CARD.md not found — skipping model card upload")
        return

    api = HfApi(token=hf_token)
    api.upload_file(
        path_or_fileobj=card_path,
        path_in_repo="README.md",
        repo_id=repo_id,
    )
    print(f"Model card uploaded → https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    cfg = load_config()

    # Run ID must be passed — find it from last_checkpoint.pth or training output
    ckpt_dir = cfg.model.checkpoint_dir
    last_ckpt = os.path.join(ckpt_dir, "last_checkpoint.pth")

    import torch
    if os.path.exists(last_ckpt):
        ckpt = torch.load(last_ckpt, map_location="cpu")
        run_id = ckpt.get("mlflow_run_id")
        if not run_id:
            raise ValueError("mlflow_run_id not found in checkpoint — re-run training first")
    else:
        raise FileNotFoundError(f"No checkpoint found at {last_ckpt}")

    version = register_in_mlflow(
        run_id=run_id,
        checkpoint_path=os.path.join(ckpt_dir, "best_model.pth"),
        fp32_onnx_path=cfg.onnx.output.fp32,
        int8_onnx_path=cfg.onnx.output.int8,
        cfg=cfg,
    )

    push_model_card(
        repo_id=os.environ.get("HF_REPO_ID", ""),
        hf_token=os.environ.get("HF_TOKEN", ""),
        cfg=cfg,
    )

    print(f"\nModel registry complete. Version {version} in Staging.")
    print(f"Run promote_to_production('{version}') after ROS2 validation.")
