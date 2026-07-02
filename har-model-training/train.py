"""
train.py — Fine-tune a SlowFast-R50 model on a custom security surveillance dataset.

Usage:
    python train.py                          # Train with default config
    python train.py --epochs 50 --batch_size 4 --lr 5e-4
    python train.py --resume output/checkpoints/last_model.pth
    python train.py --data_root ./data       # Custom data directory

The training strategy:
  1. Freeze backbone, train only the new head for N warmup epochs.
  2. Unfreeze backbone, fine-tune the whole model with a lower LR for the
     backbone and a higher LR for the head.
  3. Use class-weighted sampling to handle dataset imbalance.
  4. Early stopping based on validation macro-F1.

Expected data structure:
    data/
      train/
        normal/   *.avi / *.mp4
        fight/    ...
        ...
      test/
        normal/   ...
        fight/    ...
        ...

Classes are auto-discovered from the folder names in data/train/.
You can train with 2, 3, or more classes — just have the corresponding folders.
"""

import os
import sys
import argparse
import time

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.config import (
    ModelConfig, TrainConfig, ACTION_CLASSES, NUM_CLASSES, refresh_classes,
)
from data.dataset import build_dataloaders
from models.slowfast_model import SlowFastSecurityModel
from utils.metrics import (
    compute_metrics, print_metrics, plot_confusion_matrix,
    plot_training_curves, save_checkpoint, load_checkpoint,
    EarlyStopping,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train SlowFast Security Model")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--data_root", type=str, default=None,
                        help="Path to data directory (default: ./data)")
    parser.add_argument("--freeze_epochs", type=int, default=None,
                        help="Number of epochs to freeze backbone")
    return parser.parse_args()


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    scaler: GradScaler,
    epoch: int,
) -> float:
    """Train for one epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(loader, desc=f"Train Epoch {epoch}", leave=False)
    for (slow_clips, fast_clips), labels in pbar:
        slow_clips = slow_clips.to(device, non_blocking=True)
        fast_clips = fast_clips.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        with autocast(device_type="cuda" if "cuda" in device else "cpu"):
            logits = model([slow_clips, fast_clips])
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: str,
) -> tuple:
    """Validate the model. Returns (avg_loss, all_preds, all_labels)."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_preds = []
    all_labels = []

    for (slow_clips, fast_clips), labels in tqdm(loader, desc="Validating", leave=False):
        slow_clips = slow_clips.to(device, non_blocking=True)
        fast_clips = fast_clips.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model([slow_clips, fast_clips])
        loss = criterion(logits, labels)

        total_loss += loss.item()
        num_batches += 1

        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss, np.array(all_preds), np.array(all_labels)


def main():
    args = parse_args()

    # ── Configuration ──
    model_cfg = ModelConfig()
    train_cfg = TrainConfig()

    # Override config with CLI args
    if args.data_root:
        train_cfg.data_root = args.data_root
        # Re-discover classes from the new data root
        refresh_classes(args.data_root)
    if args.epochs:
        train_cfg.epochs = args.epochs
    if args.batch_size:
        train_cfg.batch_size = args.batch_size
    if args.lr:
        train_cfg.learning_rate = args.lr
    if args.output_dir:
        train_cfg.output_dir = args.output_dir
        train_cfg.checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
        train_cfg.log_dir = os.path.join(args.output_dir, "logs")
    if args.freeze_epochs is not None:
        model_cfg.freeze_backbone_epochs = args.freeze_epochs

    # Update num_classes from (possibly refreshed) global
    from configs.config import ACTION_CLASSES as CURRENT_CLASSES, NUM_CLASSES as CURRENT_NUM
    model_cfg.num_classes = CURRENT_NUM

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Classes ({CURRENT_NUM}): {CURRENT_CLASSES}")
    print(f"Config: {train_cfg.epochs} epochs, batch={train_cfg.batch_size}, "
          f"lr={train_cfg.learning_rate}")
    print(f"Data root: {train_cfg.data_root}")

    # ── Create output directories ──
    os.makedirs(train_cfg.checkpoint_dir, exist_ok=True)
    os.makedirs(train_cfg.log_dir, exist_ok=True)

    # ── Build data loaders ──
    print("\nLoading datasets...")
    train_loader, val_loader = build_dataloaders(train_cfg)

    # ── Build model ──
    print("\n Building SlowFast model...")
    model = SlowFastSecurityModel(
        num_classes=model_cfg.num_classes,
        pretrained=model_cfg.pretrained,
        dropout_rate=model_cfg.dropout_rate,
    )
    model.to(device)

    # ── Loss function ──
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # ── Mixed precision scaler ──
    scaler = GradScaler("cuda")

    # ── Training state ──
    start_epoch = 0
    best_f1 = 0.0
    train_losses, val_losses, val_f1s = [], [], []
    early_stopping = EarlyStopping(patience=train_cfg.early_stopping_patience, mode="max")

    # ── Resume from checkpoint ──
    if args.resume and os.path.isfile(args.resume):
        optimizer_tmp = torch.optim.AdamW(model.parameters(), lr=train_cfg.learning_rate)
        start_epoch = load_checkpoint(model, optimizer_tmp, args.resume, device)
        del optimizer_tmp

    # ══════════════════════════════════════════════════════════════════
    #  Phase 1: Frozen backbone — train only the classification head
    # ══════════════════════════════════════════════════════════════════
    freeze_epochs = model_cfg.freeze_backbone_epochs

    if freeze_epochs > 0 and start_epoch < freeze_epochs:
        print(f"\n{'='*60}")
        print(f"  Phase 1: Frozen backbone (epochs 1–{freeze_epochs})")
        print(f"{'='*60}")
        model.freeze_backbone()

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=train_cfg.learning_rate,
            weight_decay=train_cfg.weight_decay,
        )

        for epoch in range(start_epoch + 1, freeze_epochs + 1):
            train_loss = train_one_epoch(
                model, train_loader, criterion, optimizer, device, scaler, epoch
            )
            val_loss, val_preds, val_labels = validate(model, val_loader, criterion, device)

            metrics = compute_metrics(val_labels, val_preds)
            print_metrics(metrics, epoch)

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_f1s.append(metrics["macro_f1"])

            if metrics["macro_f1"] > best_f1:
                best_f1 = metrics["macro_f1"]
                save_checkpoint(
                    model, optimizer, epoch, metrics,
                    os.path.join(train_cfg.checkpoint_dir, "best_model.pth"),
                )

            save_checkpoint(
                model, optimizer, epoch, metrics,
                os.path.join(train_cfg.checkpoint_dir, "last_model.pth"),
            )

        start_epoch = freeze_epochs

    # ══════════════════════════════════════════════════════════════════
    #  Phase 2: Full fine-tuning with differential learning rates
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  Phase 2: Full fine-tuning (epochs {start_epoch+1}–{train_cfg.epochs})")
    print(f"{'='*60}")
    model.unfreeze_backbone()

    param_groups = model.get_optimizer_param_groups(
        lr=train_cfg.learning_rate, lr_backbone_factor=0.1
    )
    optimizer = torch.optim.AdamW(
        param_groups, weight_decay=train_cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=train_cfg.lr_step_size, gamma=train_cfg.lr_gamma
    )

    for epoch in range(start_epoch + 1, train_cfg.epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler, epoch
        )
        val_loss, val_preds, val_labels = validate(model, val_loader, criterion, device)
        scheduler.step()

        metrics = compute_metrics(val_labels, val_preds)
        elapsed = time.time() - t0
        print(f"[Epoch {epoch}/{train_cfg.epochs}] "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"val_f1={metrics['macro_f1']:.4f}  ({elapsed:.1f}s)")
        print_metrics(metrics, epoch)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_f1s.append(metrics["macro_f1"])

        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            save_checkpoint(
                model, optimizer, epoch, metrics,
                os.path.join(train_cfg.checkpoint_dir, "best_model.pth"),
            )

        save_checkpoint(
            model, optimizer, epoch, metrics,
            os.path.join(train_cfg.checkpoint_dir, "last_model.pth"),
        )

        if early_stopping(metrics["macro_f1"]):
            break

    # ══════════════════════════════════════════════════════════════════
    #  Final evaluation & plots
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  Training complete! Best macro F1: {best_f1:.4f}")
    print(f"{'='*60}")

    from models.slowfast_model import load_model_for_inference
    best_model = load_model_for_inference(
        os.path.join(train_cfg.checkpoint_dir, "best_model.pth"),
        num_classes=model_cfg.num_classes,
        device=device,
    )

    _, final_preds, final_labels = validate(best_model, val_loader, criterion, device)
    final_metrics = compute_metrics(final_labels, final_preds)
    print_metrics(final_metrics)

    plot_confusion_matrix(
        final_labels, final_preds,
        save_path=os.path.join(train_cfg.output_dir, "confusion_matrix.png"),
    )
    plot_training_curves(
        train_losses, val_losses, val_f1s,
        save_path=os.path.join(train_cfg.output_dir, "training_curves.png"),
    )

    print(f"\nAll outputs saved to {train_cfg.output_dir}/")


if __name__ == "__main__":
    main()