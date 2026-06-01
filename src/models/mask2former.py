import torch
import torch.nn as nn
from transformers import (
    Mask2FormerForUniversalSegmentation,
    Mask2FormerConfig,
    SwinConfig,
)


# Pretrained HF checkpoints per backbone size
PRETRAINED_CHECKPOINTS = {
    "swin_tiny":  "facebook/mask2former-swin-tiny-coco-instance",
    "swin_small": "facebook/mask2former-swin-small-coco-instance",
    "swin_base":  "facebook/mask2former-swin-base-coco-instance",
}


def build_model(cfg) -> Mask2FormerForUniversalSegmentation:
    backbone = cfg.model.backbone
    num_classes = cfg.model.num_classes
    checkpoint = PRETRAINED_CHECKPOINTS[backbone]

    print(f"Loading Mask2Former ({backbone}) from: {checkpoint}")
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        checkpoint,
        num_labels=num_classes,
        ignore_mismatched_sizes=True,   # class head is replaced for custom num_classes
    )
    return model


def build_model_from_checkpoint(checkpoint_path: str) -> Mask2FormerForUniversalSegmentation:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    cfg_dict = checkpoint["config"]
    num_classes = cfg_dict["num_classes"]
    backbone = cfg_dict["backbone"]

    base = PRETRAINED_CHECKPOINTS[backbone]
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        base,
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


# ---------------------------------------------------------------------------
# Gradual freezing
# ---------------------------------------------------------------------------
# HuggingFace Mask2Former layer structure:
#   model.pixel_level_module.encoder        ← Swin backbone
#   model.pixel_level_module.decoder        ← Pixel decoder (FPN)
#   model.transformer_module                ← Transformer decoder (queries)
# ---------------------------------------------------------------------------

def set_phase(model: Mask2FormerForUniversalSegmentation, phase: int):
    if phase == 1:
        # Freeze backbone + pixel decoder — train transformer decoder only
        for param in model.model.pixel_level_module.encoder.parameters():
            param.requires_grad = False
        for param in model.model.pixel_level_module.decoder.parameters():
            param.requires_grad = False
        for param in model.model.transformer_module.parameters():
            param.requires_grad = True
        for param in model.class_predictor.parameters():
            param.requires_grad = True
        for param in model.mask_embedder.parameters():
            param.requires_grad = True

    elif phase == 2:
        # Unfreeze pixel decoder
        for param in model.model.pixel_level_module.decoder.parameters():
            param.requires_grad = True

    elif phase == 3:
        # Unfreeze everything
        for param in model.parameters():
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Phase {phase} | Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")


def get_optimizer(model: nn.Module, lr: float, weight_decay: float):
    return torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )


def build_scheduler(optimizer, T_0: int, T_mult: int, eta_min: float):
    return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=T_0,
        T_mult=T_mult,
        eta_min=eta_min,
    )
