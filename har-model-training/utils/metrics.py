"""
Utility functions: metrics computation, confusion matrix plotting,
checkpoint management, and early stopping.
"""

import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
from typing import Dict, List, Optional

from configs.config import ACTION_CLASSES, NUM_CLASSES


# ══════════════════════════════════════════════════════════════════════
#  Metrics
# ══════════════════════════════════════════════════════════════════════

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = None,
) -> Dict:
    """Compute accuracy, per-class precision/recall/F1, and macro averages."""
    if class_names is None:
        class_names = ACTION_CLASSES

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(class_names))),
        average=None, zero_division=0,
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0,
    )

    metrics = {
        "accuracy": float(acc),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "per_class": {},
    }
    for i, name in enumerate(class_names):
        metrics["per_class"][name] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]) if support is not None else 0,
        }

    return metrics


def print_metrics(metrics: Dict, epoch: Optional[int] = None):
    """Pretty-print metrics to console."""
    header = f"Epoch {epoch}" if epoch is not None else "Evaluation"
    print(f"\n{'─'*50}")
    print(f"  {header} Results")
    print(f"{'─'*50}")
    print(f"  Accuracy:        {metrics['accuracy']:.4f}")
    print(f"  Macro Precision: {metrics['macro_precision']:.4f}")
    print(f"  Macro Recall:    {metrics['macro_recall']:.4f}")
    print(f"  Macro F1:        {metrics['macro_f1']:.4f}")
    print(f"{'─'*50}")
    print(f"  {'Class':<22} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Sup':>6}")
    print(f"  {'─'*46}")
    for name, vals in metrics["per_class"].items():
        print(f"  {name:<22} {vals['precision']:>6.3f} {vals['recall']:>6.3f} "
              f"{vals['f1']:>6.3f} {vals['support']:>6d}")
    print()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = None,
    save_path: str = "confusion_matrix.png",
    normalize: bool = True,
):
    """Plot and save a confusion matrix."""
    if class_names is None:
        class_names = ACTION_CLASSES

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    if normalize:
        cm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

    fig, ax = plt.subplots(figsize=(max(8, len(class_names) * 1.5), max(6, len(class_names) * 1.2)))
    sns.heatmap(
        cm, annot=True, fmt=".2f" if normalize else "d",
        xticklabels=class_names, yticklabels=class_names,
        cmap="Blues", ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (Normalized)" if normalize else "Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"✓ Confusion matrix saved to {save_path}")


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    val_f1s: List[float],
    save_path: str = "training_curves.png",
):
    """Plot training/validation loss and F1 curves."""
    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, train_losses, "b-", label="Train Loss")
    ax1.plot(epochs, val_losses, "r-", label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, val_f1s, "g-", label="Val Macro F1")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Macro F1")
    ax2.set_title("Validation Macro F1")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"✓ Training curves saved to {save_path}")


# ══════════════════════════════════════════════════════════════════════
#  Checkpoint management
# ══════════════════════════════════════════════════════════════════════

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict,
    path: str,
):
    """Save model checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }, path)
    print(f"Checkpoint saved: {path}")


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    path: str,
    device: str = "cuda",
) -> int:
    """Load checkpoint, return the epoch number."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint.get("epoch", 0)
    print(f"✓ Checkpoint loaded from {path} (epoch {epoch})")
    return epoch


# ══════════════════════════════════════════════════════════════════════
#  Early Stopping
# ══════════════════════════════════════════════════════════════════════

class EarlyStopping:
    """Stop training when a monitored metric stops improving."""

    def __init__(self, patience: int = 7, min_delta: float = 0.001, mode: str = "max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.should_stop = False

    def __call__(self, value: float) -> bool:
        if self.best_value is None:
            self.best_value = value
            return False

        if self.mode == "max":
            improved = value > self.best_value + self.min_delta
        else:
            improved = value < self.best_value - self.min_delta

        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                print(f"Early stopping triggered (no improvement for {self.patience} epochs)")

        return self.should_stop