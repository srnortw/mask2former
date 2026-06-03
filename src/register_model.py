import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient

from config_loader import load_config
from models.mask2former import build_model_from_checkpoint


REGISTRY_CODE_VERSION = "mlflow3-v2"  # printed so Colab users can confirm git pull worked


def _logged_model_to_uri(lm) -> str | None:
    model_id = getattr(lm, "model_id", None)
    if model_id:
        return f"models:/{model_id}"
    uri = getattr(lm, "model_uri", None)
    if uri and uri.startswith("models:/"):
        return uri
    return None


def _find_logged_model_uri(client: MlflowClient, run_id: str, name: str = "model") -> str | None:
    """MLflow 3: models live as LoggedModel entities (models:/<id>), not runs:/.../model."""
    run = client.get_run(run_id)
    exp_id = run.info.experiment_id
    candidates: list[tuple[int, str]] = []

    def _collect(logged_models, label: str):
        for lm in logged_models or []:
            uri = _logged_model_to_uri(lm)
            if not uri:
                continue
            ts = getattr(lm, "creation_timestamp", 0) or 0
            lm_name = getattr(lm, "name", None)
            if lm_name == name:
                candidates.append((ts, uri))
            elif label == "any":
                candidates.append((ts, uri))

    try:
        _collect(
            client.search_logged_models(
                experiment_ids=[exp_id],
                filter_string=f"source_run_id = '{run_id}'",
                max_results=50,
            ),
            "named",
        )
    except Exception as e:
        print(f"  (search_logged_models client: {e})")

    try:
        _collect(
            mlflow.search_logged_models(
                experiment_ids=[exp_id],
                filter_string=f"source_run_id = '{run_id}'",
                max_results=50,
            ),
            "named",
        )
    except Exception as e:
        print(f"  (search_logged_models fluent: {e})")

    if candidates:
        candidates.sort(key=lambda x: x[0])
        return candidates[-1][1]
    return None


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

    # MLflow 3: log_model uses name= (not artifact_path=); never register runs:/.../model.
    print(f"register_model.py [{REGISTRY_CODE_VERSION}]")
    print(f"Logging artifacts to run {run_id}...")
    model_info = None
    model_uri = None
    with mlflow.start_run(run_id=run_id):
        for path, subdir in [
            (fp32_onnx_path, "onnx/fp32"),
            (int8_onnx_path, "onnx/int8"),
        ]:
            if os.path.exists(path):
                mlflow.log_artifact(path, artifact_path=subdir)
                print(f"  Logged: {path} → {subdir}")

        if os.path.exists(checkpoint_path):
            mlflow.log_artifact(checkpoint_path, artifact_path="checkpoints")
            print(f"  Logged: {checkpoint_path} → checkpoints/")
        else:
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        model_uri = _find_logged_model_uri(client, run_id)
        if model_uri:
            print(f"  Reusing logged model on this run: {model_uri}")
        else:
            print(f"  Logging PyTorch model for registry from {checkpoint_path}...")
            model = build_model_from_checkpoint(checkpoint_path)
            model_info = mlflow.pytorch.log_model(
                model,
                name="model",
                registered_model_name=model_name,
                pip_requirements=["torch", "transformers", "torchvision"],
            )
            del model
            model_uri = model_info.model_uri
            print(f"  Logged model: {model_uri}")

    if model_info is not None and model_info.registered_model_version is not None:
        version = str(model_info.registered_model_version)
        print(f"\nRegistered as version {version} (via log_model + registered_model_name)")
    else:
        if not model_uri:
            model_uri = _find_logged_model_uri(client, run_id)
        if not model_uri:
            raise RuntimeError(
                "No LoggedModel found for this run. Run Cell 4 (git pull) then re-run Cell 20."
            )
        if model_uri.startswith("runs:/"):
            raise RuntimeError(
                f"Refusing runs:/ URI ({model_uri}). Pull latest code and reload register_model."
            )
        print(f"\nRegistering '{model_name}' from {model_uri}...")
        result = mlflow.register_model(model_uri=model_uri, name=model_name)
        version = str(result.version)
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
