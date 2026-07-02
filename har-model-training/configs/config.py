"""
Configuration for the Security Action Recognition System.
All hyperparameters, paths, and class definitions are centralized here.

Classes are auto-discovered from the folders present in DATA_ROOT/train/.
The defaults below (normal, fight, vandalism) are used only as
fallback when the data directory doesn't exist yet.
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Tuple


# ──────────────────────────────────────────────────────────────────────
# Data root – single folder-based dataset
# ──────────────────────────────────────────────────────────────────────
DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

# ──────────────────────────────────────────────────────────────────────
# Default action classes (used as fallback if data dir doesn't exist)
# ──────────────────────────────────────────────────────────────────────
DEFAULT_ACTION_CLASSES = [
    "normal",       # 0
    "fight",        # 1
    "vandalism",    # 2
]


def discover_classes(data_root: str = DATA_ROOT) -> List[str]:
    """
    Auto-discover classes from folder names in data_root/train/.
    Convention: 'normal' always gets index 0; the rest are sorted alphabetically.
    If the train directory doesn't exist, fall back to DEFAULT_ACTION_CLASSES.
    """
    train_dir = os.path.join(data_root, "train")
    if not os.path.isdir(train_dir):
        return list(DEFAULT_ACTION_CLASSES)

    folders = [
        d for d in sorted(os.listdir(train_dir))
        if os.path.isdir(os.path.join(train_dir, d)) and not d.startswith(".")
    ]

    if not folders:
        return list(DEFAULT_ACTION_CLASSES)

    # Put "normal" first if it exists, then the rest alphabetically
    if "normal" in folders:
        folders.remove("normal")
        folders = ["normal"] + sorted(folders)
    else:
        folders = sorted(folders)

    return folders


# ── Discover classes at import time ──
ACTION_CLASSES: List[str] = discover_classes()
NUM_CLASSES: int = len(ACTION_CLASSES)
CLASS_TO_IDX: Dict[str, int] = {c: i for i, c in enumerate(ACTION_CLASSES)}
IDX_TO_CLASS: Dict[int, str] = {i: c for i, c in enumerate(ACTION_CLASSES)}


def refresh_classes(data_root: str = DATA_ROOT):
    """
    Re-discover classes and update the module-level variables.
    Useful if you change the data directory at runtime.
    """
    global ACTION_CLASSES, NUM_CLASSES, CLASS_TO_IDX, IDX_TO_CLASS
    ACTION_CLASSES = discover_classes(data_root)
    NUM_CLASSES = len(ACTION_CLASSES)
    CLASS_TO_IDX = {c: i for i, c in enumerate(ACTION_CLASSES)}
    IDX_TO_CLASS = {i: c for i, c in enumerate(ACTION_CLASSES)}


# ──────────────────────────────────────────────────────────────────────
# Dataclass configs
# ──────────────────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    """SlowFast model configuration."""
    model_name: str = "slowfast_r50"
    pretrained: bool = True
    num_classes: int = NUM_CLASSES
    dropout_rate: float = 0.5
    freeze_backbone_epochs: int = 5


@dataclass
class TrainConfig:
    """Training hyperparameters."""
    # Paths
    data_root: str = DATA_ROOT
    output_dir: str = "./output"
    checkpoint_dir: str = "./output/checkpoints"
    log_dir: str = "./output/logs"

    # Video clip parameters
    clip_duration_sec: float = 2.0
    num_frames_slow: int = 8
    num_frames_fast: int = 32
    sampling_rate_slow: int = 8
    sampling_rate_fast: int = 2
    crop_size: int = 224

    # Training
    batch_size: int = 8
    num_workers: int = 4
    epochs: int = 30
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    lr_warmup_epochs: int = 3
    lr_step_size: int = 10
    lr_gamma: float = 0.1

    # Augmentation
    horizontal_flip_prob: float = 0.5
    color_jitter: float = 0.2
    random_crop_scale: tuple = (0.8, 1.0)

    # Class imbalance
    use_class_weights: bool = True

    # Validation (not used when test/ folder exists)
    val_split: float = 0.2
    early_stopping_patience: int = 7

    # Video extensions to scan
    video_extensions: Tuple[str, ...] = (".avi", ".mp4", ".mkv", ".mov")