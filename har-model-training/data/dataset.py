"""
Dataset module for loading and preprocessing video clips into SlowFast format.

Expected directory structure
─────────────────────────────
    data/
      train/
        normal/     video1.avi  video2.mp4 ...
        fight/      ...
        vandalism/  ...
      test/
        normal/     ...
        fight/      ...
        vandalism/  ...

Classes are auto-discovered from the folder names in data/train/.
You can train with 2, 3, or more classes — just include the folders you need.
"""

import os
import random
from typing import List, Tuple, Optional, Callable

import av
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms as T

from configs.config import (
    TrainConfig, ACTION_CLASSES, CLASS_TO_IDX, NUM_CLASSES,
)


# ══════════════════════════════════════════════════════════════════════
#  Video I/O utilities
# ══════════════════════════════════════════════════════════════════════

def decode_video_pyav(video_path: str, num_frames: int, fps: Optional[float] = None) -> np.ndarray:
    """
    Decode *num_frames* uniformly-sampled RGB frames from a video file using PyAV.
    Returns shape (T, H, W, 3) uint8.
    """
    container = av.open(video_path)
    stream = container.streams.video[0]
    total_frames = stream.frames
    if total_frames == 0:
        duration = float(stream.duration * stream.time_base)
        avg_fps = float(stream.average_rate) if stream.average_rate else 25.0
        total_frames = int(duration * avg_fps)

    total_frames = max(total_frames, num_frames)

    indices = np.linspace(0, total_frames - 1, num_frames, dtype=np.int64)
    indices_set = set(indices.tolist())

    frames = []
    frame_idx = 0
    for frame in container.decode(video=0):
        if frame_idx in indices_set:
            img = frame.to_ndarray(format="rgb24")
            frames.append(img)
        frame_idx += 1
        if len(frames) == num_frames:
            break

    container.close()

    while len(frames) < num_frames:
        frames.append(frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8))

    return np.stack(frames, axis=0)


# ══════════════════════════════════════════════════════════════════════
#  SlowFast-specific frame packing
# ══════════════════════════════════════════════════════════════════════

def pack_frames_slowfast(
    frames: np.ndarray,
    num_frames_slow: int = 8,
    num_frames_fast: int = 32,
    crop_size: int = 224,
    spatial_transform: Optional[Callable] = None,
) -> List[torch.Tensor]:
    """
    Given T frames (T, H, W, 3), produce the two-pathway input for SlowFast:
      - slow_pathway: (3, num_frames_slow, crop_size, crop_size)
      - fast_pathway: (3, num_frames_fast, crop_size, crop_size)

    Returns [slow_tensor, fast_tensor].
    """
    T_total = frames.shape[0]

    slow_indices = np.linspace(0, T_total - 1, num_frames_slow, dtype=np.int64)
    fast_indices = np.linspace(0, T_total - 1, num_frames_fast, dtype=np.int64)

    slow_frames = frames[slow_indices]
    fast_frames = frames[fast_indices]

    slow_tensor = _frames_to_tensor(slow_frames, crop_size, spatial_transform)
    fast_tensor = _frames_to_tensor(fast_frames, crop_size, spatial_transform)

    return [slow_tensor, fast_tensor]


def _frames_to_tensor(
    frames: np.ndarray,
    crop_size: int,
    spatial_transform: Optional[Callable] = None,
) -> torch.Tensor:
    """Convert (T, H, W, 3) uint8 → (3, T, crop_size, crop_size) float tensor."""
    num_t = frames.shape[0]

    tensors = []
    for t in range(num_t):
        img = torch.from_numpy(frames[t]).permute(2, 0, 1).float() / 255.0
        img = T.functional.resize(img, [crop_size, crop_size], antialias=True)
        tensors.append(img)

    tensor = torch.stack(tensors, dim=1)  # (3, T, H, W)

    if spatial_transform is not None:
        tensor = spatial_transform(tensor)

    # Normalize with Kinetics mean/std
    mean = torch.tensor([0.45, 0.45, 0.45]).view(3, 1, 1, 1)
    std = torch.tensor([0.225, 0.225, 0.225]).view(3, 1, 1, 1)
    tensor = (tensor - mean) / std

    return tensor


# ══════════════════════════════════════════════════════════════════════
#  Spatial augmentation for training
# ══════════════════════════════════════════════════════════════════════

class SpatialAugmentation:
    """
    Augmentations applied consistently across all frames of a clip.
    Operates on tensor of shape (3, T, H, W).
    """

    def __init__(self, cfg: TrainConfig, is_train: bool = True):
        self.is_train = is_train
        self.cfg = cfg

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if not self.is_train:
            return tensor

        if random.random() < self.cfg.horizontal_flip_prob:
            tensor = torch.flip(tensor, dims=[-1])

        if self.cfg.color_jitter > 0:
            jitter = self.cfg.color_jitter
            brightness = 1.0 + random.uniform(-jitter, jitter)
            contrast = 1.0 + random.uniform(-jitter, jitter)
            tensor = torch.clamp(tensor * brightness, 0, 1)
            mean = tensor.mean(dim=(0, 2, 3), keepdim=True)
            tensor = torch.clamp((tensor - mean) * contrast + mean, 0, 1)

        return tensor


# ══════════════════════════════════════════════════════════════════════
#  Unified Video Dataset
# ══════════════════════════════════════════════════════════════════════

class ActionVideoDataset(Dataset):
    """
    Folder-based video dataset for SlowFast.

    Expects:
        data_root/{split}/{class_name}/video.ext

    Classes are mapped using the global CLASS_TO_IDX dictionary,
    which is auto-discovered from data_root/train/ at import time.
    """

    def __init__(
        self,
        data_root: str,
        train_cfg: TrainConfig,
        split: str = "train",
        is_train: bool = True,
    ):
        super().__init__()
        self.train_cfg = train_cfg
        self.is_train = is_train
        self.spatial_aug = SpatialAugmentation(train_cfg, is_train)
        self.num_frames_decode = max(train_cfg.num_frames_fast, train_cfg.num_frames_slow * 4)

        self.samples: List[Tuple[str, int]] = []
        self._load_from_folders(data_root, split, train_cfg.video_extensions)

        print(f"[{split}] Loaded {len(self.samples)} clips ({self._class_distribution()})")

    def _load_from_folders(self, data_root: str, split: str, video_extensions: Tuple[str, ...]):
        """Scan data_root/{split}/{class_name}/ for video files."""
        split_dir = os.path.join(data_root, split)
        if not os.path.isdir(split_dir):
            print(f"⚠ Split directory not found: {split_dir}")
            return

        for class_name in sorted(os.listdir(split_dir)):
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.isdir(class_dir) or class_name.startswith("."):
                continue

            if class_name not in CLASS_TO_IDX:
                print(f"⚠ Folder '{class_name}' in {split_dir} not in CLASS_TO_IDX — skipping. "
                      f"Known classes: {list(CLASS_TO_IDX.keys())}")
                continue

            label_idx = CLASS_TO_IDX[class_name]
            for fname in sorted(os.listdir(class_dir)):
                if fname.lower().endswith(video_extensions):
                    fpath = os.path.join(class_dir, fname)
                    self.samples.append((fpath, label_idx))

    def _class_distribution(self) -> str:
        from collections import Counter
        counts = Counter(label for _, label in self.samples)
        parts = [f"{ACTION_CLASSES[k]}:{v}" for k, v in sorted(counts.items())]
        return ", ".join(parts)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        video_path, label = self.samples[idx]

        try:
            frames = decode_video_pyav(video_path, self.num_frames_decode)
        except Exception as e:
            print(f"⚠ Error decoding {video_path}: {e}. Returning zeros.")
            h = w = self.train_cfg.crop_size
            frames = np.zeros((self.num_frames_decode, h, w, 3), dtype=np.uint8)

        slow_tensor, fast_tensor = pack_frames_slowfast(
            frames,
            num_frames_slow=self.train_cfg.num_frames_slow,
            num_frames_fast=self.train_cfg.num_frames_fast,
            crop_size=self.train_cfg.crop_size,
            spatial_transform=self.spatial_aug,
        )

        label_tensor = torch.tensor(label, dtype=torch.long)
        return [slow_tensor, fast_tensor], label_tensor


# ══════════════════════════════════════════════════════════════════════
#  Build DataLoaders
# ══════════════════════════════════════════════════════════════════════

def build_dataloaders(
    train_cfg: TrainConfig,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build train and val/test DataLoaders.

    Looks for data_root/train/ and data_root/test/.
    If test/ doesn't exist, splits train/ using train_cfg.val_split.
    """
    data_root = train_cfg.data_root

    train_dataset = ActionVideoDataset(data_root, train_cfg, split="train", is_train=True)

    # Use test/ as the validation set
    test_dir = os.path.join(data_root, "test")
    if os.path.isdir(test_dir):
        val_dataset = ActionVideoDataset(data_root, train_cfg, split="test", is_train=False)
    else:
        # Fall back to a random split of train
        n_val = int(len(train_dataset) * train_cfg.val_split)
        n_train = len(train_dataset) - n_val
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [n_train, n_val]
        )
        print(f"⚠ No test/ folder found — split train into {n_train} train + {n_val} val")

    print(f"\n{'='*60}")
    print(f"Dataset: {len(train_dataset)} train clips, {len(val_dataset)} val/test clips")
    print(f"Classes: {ACTION_CLASSES} ({NUM_CLASSES} classes)")
    print(f"{'='*60}\n")

    # --- Weighted sampler for class imbalance ---
    sampler = None
    shuffle = True
    if train_cfg.use_class_weights:
        labels = _extract_labels(train_dataset)
        if labels is not None:
            class_counts = np.bincount(labels, minlength=NUM_CLASSES).astype(float)
            class_counts = np.maximum(class_counts, 1.0)
            weights_per_class = 1.0 / class_counts
            sample_weights = [weights_per_class[l] for l in labels]
            sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
            shuffle = False

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=train_cfg.num_workers,
        collate_fn=slowfast_collate,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=train_cfg.num_workers,
        collate_fn=slowfast_collate,
        pin_memory=True,
    )

    return train_loader, val_loader


def _extract_labels(dataset) -> Optional[np.ndarray]:
    """Extract all labels from a dataset (handles Subset and ActionVideoDataset)."""
    labels = []
    try:
        if hasattr(dataset, "samples"):
            # ActionVideoDataset
            labels = [lbl for _, lbl in dataset.samples]
        elif hasattr(dataset, "dataset") and hasattr(dataset, "indices"):
            # torch Subset from random_split
            for idx in dataset.indices:
                _, lbl = dataset.dataset.samples[idx]
                labels.append(lbl)
        else:
            return None
    except Exception:
        return None
    return np.array(labels)


def slowfast_collate(batch):
    """Custom collate: batch list of ([slow, fast], label)."""
    slow_clips = torch.stack([item[0][0] for item in batch])
    fast_clips = torch.stack([item[0][1] for item in batch])
    labels = torch.stack([item[1] for item in batch])
    return [slow_clips, fast_clips], labels