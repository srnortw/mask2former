import os
import sys
import torch
import mlflow
import mlflow.pytorch

sys.path.insert(0, os.path.dirname(__file__))

from config_loader import load_config
from data.dataset import build_dataloaders
from models.mask2former import (
    build_model,
    set_phase,
    get_optimizer,
    build_scheduler,
)
from evaluate import evaluate


def train_one_epoch(model, loader, optimizer, device, scaler, cfg):
    model.train()
    total_loss = 0.0
    log_every  = cfg.training.log_every_n_batches

    for batch_idx, (images, targets) in enumerate(loader):
        images = images.to(device)

        # Build HuggingFace-style pixel_mask and labels
        mask_labels  = [t["masks"].to(device).float() for t in targets]
        class_labels = [t["labels"].to(device) for t in targets]

        with torch.amp.autocast("cuda", enabled=cfg.training.mixed_precision):
            outputs = model(
                pixel_values=images,
                mask_labels=mask_labels,
                class_labels=class_labels,
            )
            loss = outputs.loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            cfg.learning_rate.grad_clip_norm
        )
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        total_loss += loss.item()

        if (batch_idx + 1) % log_every == 0:
            print(f"  batch {batch_idx+1}/{len(loader)}  loss={loss.item():.4f}")

    return total_loss / len(loader)


def train(cfg=None):
    if cfg is None:
        cfg = load_config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    train_loader, val_loader = build_dataloaders(cfg)

    # Model
    model = build_model(cfg).to(device)
    if cfg.training.compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)

    # MLflow
    mlflow.set_tracking_uri(cfg.services.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.services.mlflow.experiment_name)

    # Phase schedule from config
    phases = {
        1: cfg.freezing.phase1,
        2: cfg.freezing.phase2,
        3: cfg.freezing.phase3,
    }
    lr_map = {
        1: cfg.learning_rate.phase1,
        2: cfg.learning_rate.phase2,
        3: cfg.learning_rate.phase3,
    }

    # AMP scaler
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.training.mixed_precision)

    # Init phase 1
    current_phase = 1
    set_phase(model, phase=1)
    optimizer = get_optimizer(model, lr_map[1], cfg.learning_rate.weight_decay)
    scheduler = build_scheduler(
        optimizer,
        cfg.scheduler.T_0,
        cfg.scheduler.T_mult,
        cfg.scheduler.eta_min,
    )

    best_map   = 0.0
    total_epochs = cfg.training.total_epochs
    os.makedirs(cfg.model.checkpoint_dir, exist_ok=True)

    with mlflow.start_run():
        mlflow.log_params({
            "backbone":     cfg.model.backbone,
            "num_classes":  cfg.model.num_classes,
            "batch_size":   cfg.data.dataloader.batch_size,
            "img_size":     cfg.data.dataloader.img_size,
            "total_epochs": total_epochs,
            "phase1_end":   phases[1].end_epoch,
            "phase2_end":   phases[2].end_epoch,
            "phase3_end":   phases[3].end_epoch,
            "T_0":          cfg.scheduler.T_0,
            "T_mult":       cfg.scheduler.T_mult,
            "lr_phase1":    lr_map[1],
            "lr_phase2":    lr_map[2],
            "lr_phase3":    lr_map[3],
        })

        for epoch in range(total_epochs):

            # Phase transitions
            if epoch == phases[2].start_epoch and current_phase == 1:
                print(f"\n=== Phase 2 starts (epoch {epoch}) ===")
                set_phase(model, phase=2)
                optimizer = get_optimizer(model, lr_map[2], cfg.learning_rate.weight_decay)
                scheduler = build_scheduler(optimizer, cfg.scheduler.T_0, cfg.scheduler.T_mult, cfg.scheduler.eta_min)
                current_phase = 2

            elif epoch == phases[3].start_epoch and current_phase == 2:
                print(f"\n=== Phase 3 starts (epoch {epoch}) ===")
                set_phase(model, phase=3)
                optimizer = get_optimizer(model, lr_map[3], cfg.learning_rate.weight_decay)
                scheduler = build_scheduler(optimizer, cfg.scheduler.T_0, cfg.scheduler.T_mult, cfg.scheduler.eta_min)
                current_phase = 3

            # Train
            print(f"\nEpoch {epoch+1}/{total_epochs} | Phase {current_phase}")
            train_loss = train_one_epoch(model, train_loader, optimizer, device, scaler, cfg)
            scheduler.step(epoch)

            current_lr = optimizer.param_groups[0]["lr"]

            # Evaluate every N epochs
            val_map = 0.0
            if (epoch + 1) % cfg.evaluation.eval_every_n_epochs == 0:
                val_map = evaluate(model, val_loader, device, cfg.evaluation.score_threshold)

            # Log to MLflow
            mlflow.log_metrics({
                "train_loss":    train_loss,
                "val_map":       val_map,
                "learning_rate": current_lr,
                "phase":         current_phase,
            }, step=epoch)

            print(f"Epoch {epoch+1:03d} | Phase {current_phase} | Loss: {train_loss:.4f} | mAP: {val_map:.4f} | LR: {current_lr:.2e}")

            # Save best checkpoint
            if val_map > best_map or epoch == 0:
                if val_map > best_map:
                    best_map = val_map
                ckpt_path = os.path.join(cfg.model.checkpoint_dir, "best_model.pth")
                torch.save({
                    "epoch":             epoch,
                    "model_state_dict":  model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_map":           val_map,
                    "phase":             current_phase,
                    "config": {
                        "backbone":    cfg.model.backbone,
                        "num_classes": cfg.model.num_classes,
                    },
                }, ckpt_path)
                mlflow.log_artifact(ckpt_path)
                print(f"  → Checkpoint saved (best mAP: {best_map:.4f})")

            # Save phase-end checkpoints
            if cfg.training.save_phase_end:
                for phase_id, phase_cfg in phases.items():
                    if epoch + 1 == phase_cfg.end_epoch:
                        path = os.path.join(cfg.model.checkpoint_dir, f"phase{phase_id}_final.pth")
                        torch.save({"epoch": epoch, "model_state_dict": model.state_dict()}, path)
                        print(f"  → Phase {phase_id} final checkpoint saved")

    print(f"\nTraining complete. Best mAP: {best_map:.4f}")
    return best_map


if __name__ == "__main__":
    train()
