#!/usr/bin/env python3
"""
Human Action Recognition (HAR) System using SlowFast R50.

Classifies video clips from the live camera feed into three categories:
  - normal:    Ordinary activity
  - fight:     Physical altercations
  - vandalism: Intentional property destruction

The model is a SlowFast R50 (PyTorchVideo) fine-tuned on a custom
3-class dataset.  It expects two-pathway input:
  slow_pathway: (B, 3, 8, 224, 224)
  fast_pathway: (B, 3, 32, 224, 224)

Integration notes
─────────────────
• The system accumulates frames in a ring buffer.
• Every `clip_interval` frames it packs the buffer into a SlowFast
  clip and runs inference.
• Results are returned as a list of dicts (same shape as
  fire_detection_system results) so the merging thread can treat
  them uniformly.
"""

import os
import time
import threading
import numpy as np
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import torch
import torch.nn as nn

import logging

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
#  SlowFast helpers (inlined so the module is self-contained)
# ──────────────────────────────────────────────────────────────────────

def _build_slowfast_model(num_classes: int, pretrained: bool, dropout_rate: float) -> nn.Module:
    """Build a SlowFast-R50 model with a custom classification head."""
    try:
        model = torch.hub.load(
            "facebookresearch/pytorchvideo",
            model="slowfast_r50",
            pretrained=pretrained,
        )
    except Exception:
        from pytorchvideo.models.slowfast import create_slowfast
        model = create_slowfast(
            slowfast_channel_reduction_ratio=(8,),
            slowfast_conv_channel_fusion_ratio=2,
            slowfast_fusion_conv_kernel_size=(7, 1, 1),
            slowfast_fusion_conv_stride=(4, 1, 1),
            model_depth=50,
            model_num_class=400 if pretrained else num_classes,
            dropout_rate=0.0,
            head_pool_kernel_sizes=((8, 7, 7), (32, 7, 7)),
        )

    # Replace classification head
    if hasattr(model, "blocks"):
        head_block = model.blocks[-1]
        if hasattr(head_block, "proj"):
            in_features = head_block.proj.in_features
            head_block.proj = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features, num_classes),
            )
        elif hasattr(head_block, "output_proj"):
            in_features = head_block.output_proj.in_features
            head_block.output_proj = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features, num_classes),
            )
    return model


class _SlowFastWrapper(nn.Module):
    """Thin wrapper matching the training code's SlowFastSecurityModel."""

    def __init__(self, num_classes: int, pretrained: bool, dropout_rate: float):
        super().__init__()
        self.model = _build_slowfast_model(num_classes, pretrained, dropout_rate)
        self.num_classes = num_classes

    def forward(self, x):
        return self.model(x)


def _load_model(checkpoint_path: str, num_classes: int, device: str) -> _SlowFastWrapper:
    """Load a trained SlowFast checkpoint."""
    model = _SlowFastWrapper(num_classes=num_classes, pretrained=False, dropout_rate=0.0)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model


def _frames_to_tensor(frames: np.ndarray, crop_size: int) -> torch.Tensor:
    """Convert (T, H, W, 3) uint8 RGB → (3, T, crop_size, crop_size) float tensor."""
    tensors = []
    for t in range(frames.shape[0]):
        img = torch.from_numpy(frames[t]).permute(2, 0, 1).float() / 255.0
        img = torch.nn.functional.interpolate(
            img.unsqueeze(0), size=(crop_size, crop_size), mode="bilinear", align_corners=False
        ).squeeze(0)
        tensors.append(img)
    tensor = torch.stack(tensors, dim=1)  # (3, T, H, W)
    mean = torch.tensor([0.45, 0.45, 0.45]).view(3, 1, 1, 1)
    std = torch.tensor([0.225, 0.225, 0.225]).view(3, 1, 1, 1)
    return (tensor - mean) / std


def _pack_slowfast(frames: np.ndarray, num_slow: int = 8, num_fast: int = 32,
                   crop_size: int = 224) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pack RGB frames into [slow_pathway, fast_pathway] tensors."""
    T = frames.shape[0]
    slow_idx = np.linspace(0, T - 1, num_slow, dtype=np.int64)
    fast_idx = np.linspace(0, T - 1, num_fast, dtype=np.int64)
    slow_tensor = _frames_to_tensor(frames[slow_idx], crop_size)
    fast_tensor = _frames_to_tensor(frames[fast_idx], crop_size)
    return slow_tensor, fast_tensor


# ──────────────────────────────────────────────────────────────────────
#  Severity / alert config
# ──────────────────────────────────────────────────────────────────────

_ALERT_CONFIG = {
    "normal":    {"severity": "low",      "color": (0, 200, 0),   "label": "Normal"},
    "fight":     {"severity": "critical", "color": (0, 0, 255),   "label": "FIGHTING"},
    "fighting":  {"severity": "critical", "color": (0, 0, 255),   "label": "FIGHTING"},
    "vandalism": {"severity": "high",     "color": (0, 50, 255),  "label": "VANDALISM"},
}

_DEFAULT_ALERT = {"severity": "medium", "color": (0, 165, 255), "label": "ALERT"}


# ══════════════════════════════════════════════════════════════════════
#  Public API  –  HumanActionRecognitionSystem
# ══════════════════════════════════════════════════════════════════════

class HumanActionRecognitionSystem:
    """
    Real-time human action recognition for the security system.

    Usage mirrors FireDetectionSystem:
        har = HumanActionRecognitionSystem(model_path="models/har/best_model.pth")
        results = har.process_frame(frame)        # feed every frame
        annotated = har.draw_detections(frame, results)
    """

    # Class-level constants  (match your training config)
    NUM_CLASSES = 3
    CLASS_NAMES = {0: "normal", 1: "fight", 2: "vandalism"}
    IDX_TO_CLASS = CLASS_NAMES  # alias

    def __init__(
        self,
        model_path: str = "models/har/best_model.pth",
        device: str = "auto",
        confidence_threshold: float = 0.5,
        clip_interval_frames: int = 30,
        buffer_size: int = 128,
        crop_size: int = 224,
        clip_duration_sec: float = 2.0,
    ):
        """
        Args:
            model_path:             Path to the trained SlowFast checkpoint.
            device:                 'cuda', 'cpu', or 'auto'.
            confidence_threshold:   Minimum softmax probability to trigger an alert.
            clip_interval_frames:   Run inference every N frames.
            buffer_size:            Max frames kept in ring buffer.
            crop_size:              Spatial crop fed to SlowFast.
            clip_duration_sec:      Temporal span of the clip fed to SlowFast,
                                    in seconds. MUST match the training config
                                    (2.0 s). The clip is built from frames within
                                    this many seconds of wall-clock — so it stays
                                    correctly scaled regardless of how many frames
                                    the upstream pipeline drops (real-time playback
                                    means 1 wall-second ≈ 1 video-second).
        """
        os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)

        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"HAR model not found at: {model_path}\n"
                "Please place your trained best_model.pth at that path."
            )

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Loading HAR SlowFast model from: {model_path}  (device={self.device})")
        self.model = _load_model(model_path, self.NUM_CLASSES, self.device)
        logger.info("HAR model loaded successfully!")

        # Inference parameters
        self.confidence_threshold = confidence_threshold
        self.clip_interval = clip_interval_frames
        self.crop_size = crop_size
        self.clip_duration_sec = clip_duration_sec

        # Frame ring-buffer: stores (timestamp, RGB-frame) so the clip can be
        # cut to a fixed *time* window (clip_duration_sec) rather than a fixed
        # frame count — keeping the temporal scale matched to training even when
        # the pipeline drops frames under load.
        self.frame_buffer: deque = deque(maxlen=buffer_size)
        self.frame_count = 0

        # Current prediction (persisted between clips)
        self.current_action = "normal"
        self.current_confidence = 0.0
        self.last_inference_time = 0.0

        # Alert cooldown
        self.alert_cooldown = 5.0  # seconds
        self.last_alert_time: Dict[str, datetime] = {}

        # Statistics
        self.stats = {
            "total_inferences": 0,
            "fight_detections": 0,
            "vandalism_detections": 0,
            "normal_detections": 0,
            "critical_alerts": 0,
        }

    # ── frame-by-frame API (called from the processing thread) ───────

    def process_frame(self, frame: np.ndarray) -> List[Dict]:
        """
        Accumulate *frame* and periodically run SlowFast inference.

        Returns a list of detection dicts (may be empty when no inference
        runs or the scene is classified as *normal*).  The dict shape is
        compatible with the fire-detection results so that the merging
        thread can treat them uniformly.
        """
        # Store BGR→RGB converted copy, resized to uniform dimensions
        # so that np.stack() works even with multi-camera inputs.
        # Tag with wall-clock time so the clip can be cut to a fixed time window.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.crop_size, self.crop_size))
        now_ts = time.time()
        self.frame_buffer.append((now_ts, rgb))
        self.frame_count += 1

        # Only run inference every clip_interval frames & when we have enough
        if (self.frame_count % self.clip_interval != 0 or
                len(self.frame_buffer) < 8):
            return self._make_result_list(frame)

        # ── Build clip from the last `clip_duration_sec` seconds ────
        # Keep only frames within the temporal window the model was trained on,
        # so a rapid action (fight) isn't smeared over a long span and confused
        # with a slower one (vandalism).
        cutoff = now_ts - self.clip_duration_sec
        window = [f for (t, f) in self.frame_buffer if t >= cutoff]
        if len(window) < 8:                      # too few -> use what we have
            window = [f for (_, f) in self.frame_buffer]
        # Sample 64 frames uniformly from the time window
        n_sample = min(64, len(window))
        indices = np.linspace(0, len(window) - 1, n_sample, dtype=np.int64)
        clip = np.stack([window[i] for i in indices])

        slow_t, fast_t = _pack_slowfast(clip, crop_size=self.crop_size)
        slow_t = slow_t.unsqueeze(0).to(self.device)
        fast_t = fast_t.unsqueeze(0).to(self.device)

        start = time.time()
        with torch.no_grad():
            logits = self.model([slow_t, fast_t])
            probs = torch.softmax(logits, dim=1)
            conf, pred_idx = probs.max(dim=1)

        self.last_inference_time = time.time() - start
        action = self.CLASS_NAMES.get(pred_idx.item(), f"class_{pred_idx.item()}")
        confidence = conf.item()

        self.stats["total_inferences"] += 1

        if confidence >= self.confidence_threshold:
            self.current_action = action
            self.current_confidence = confidence

            # Update per-class stats
            if action == "fight" or action == "fighting":
                self.stats["fight_detections"] += 1
            elif action == "vandalism":
                self.stats["vandalism_detections"] += 1
            else:
                self.stats["normal_detections"] += 1

        return self._make_result_list(frame)

    def _make_result_list(self, frame: np.ndarray) -> List[Dict]:
        """
        Package the current prediction into a result list.
        Non-normal actions above threshold become alert entries.
        """
        if self.current_action == "normal":
            # Return a single "normal" entry (low severity, no alert)
            return [{
                "class": "normal",
                "confidence": self.current_confidence,
                "severity": "low",
                "alert": False,
                "timestamp": datetime.now(),
                "action_label": "Normal Activity",
            }]

        # Non-normal action detected
        alert_cfg = _ALERT_CONFIG.get(self.current_action, _DEFAULT_ALERT)
        severity = alert_cfg["severity"]

        # Cooldown logic
        should_alert = False
        now = datetime.now()
        if self.current_action not in self.last_alert_time:
            should_alert = True
            self.last_alert_time[self.current_action] = now
        else:
            elapsed = (now - self.last_alert_time[self.current_action]).total_seconds()
            if elapsed > self.alert_cooldown:
                should_alert = True
                self.last_alert_time[self.current_action] = now

        if severity == "critical" and should_alert:
            self.stats["critical_alerts"] += 1

        return [{
            "class": self.current_action,
            "confidence": self.current_confidence,
            "severity": severity,
            "alert": should_alert,
            "timestamp": datetime.now(),
            "action_label": alert_cfg["label"],
        }]

    # ── Drawing overlay on frame ─────────────────────────────────────

    @staticmethod
    def draw_detections(frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Draw an action-recognition HUD overlay on the frame.

        Unlike fire/plate which draw per-box, HAR is a *scene-level*
        classification, so we draw a status banner at the top of the frame.
        """
        if not detections:
            return frame

        output = frame.copy()
        det = detections[0]  # scene-level → only one entry
        action = det.get("class", "normal")
        confidence = det.get("confidence", 0.0)
        severity = det.get("severity", "low")
        label = det.get("action_label", action.upper())

        alert_cfg = _ALERT_CONFIG.get(action, _DEFAULT_ALERT)
        color = alert_cfg["color"]

        h, w = output.shape[:2]

        if action != "normal":
            # Semi-transparent banner at top
            overlay = output.copy()
            banner_h = 50
            cv2.rectangle(overlay, (0, 0), (w, banner_h), color, -1)
            cv2.addWeighted(overlay, 0.6, output, 0.4, 0, output)

            # Text
            text = f"  {label}  |  Confidence: {confidence:.0%}  |  Severity: {severity.upper()}"
            cv2.putText(output, text, (10, 33),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Flashing border for critical
            if severity == "critical":
                border_thickness = 4
                cv2.rectangle(output, (0, 0), (w - 1, h - 1), color, border_thickness)
        else:
            # Small green badge bottom-left
            badge_text = f"HAR: Normal ({confidence:.0%})"
            (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(output, (5, h - th - 15), (tw + 15, h - 5), (0, 0, 0), -1)
            cv2.putText(output, badge_text, (10, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)

        return output

    # ── Statistics ────────────────────────────────────────────────────

    def get_statistics(self) -> Dict:
        return {
            "model_loaded": self.model is not None,
            "device": self.device,
            "confidence_threshold": self.confidence_threshold,
            "clip_interval": self.clip_interval,
            "buffer_size": len(self.frame_buffer),
            "current_action": self.current_action,
            "current_confidence": self.current_confidence,
            "last_inference_time_ms": self.last_inference_time * 1000,
            **self.stats,
        }

    def reset_statistics(self):
        self.stats = {k: 0 for k in self.stats}
        self.last_alert_time.clear()
        self.current_action = "normal"
        self.current_confidence = 0.0


# ──────────────────────────────────────────────────────────────────────
#  Quick standalone demo
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("Human Action Recognition System - Demo")
    print("=" * 50)

    try:
        har = HumanActionRecognitionSystem()

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera")
            sys.exit(1)

        print("Camera opened. Press 'q' to quit, 's' for stats.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = har.process_frame(frame)
            output = har.draw_detections(frame, results)

            cv2.imshow("HAR Demo", output)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                stats = har.get_statistics()
                print("\nHAR Statistics:")
                for k, v in stats.items():
                    print(f"  {k}: {v}")

        cap.release()
        cv2.destroyAllWindows()

    except FileNotFoundError as e:
        print(f"\n{e}")
        print("\nPlace your trained model at: models/har/best_model.pth")