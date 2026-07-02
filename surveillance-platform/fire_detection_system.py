#!/usr/bin/env python3
"""
Fire and Smoke Detection System.

Model: TommyNgx/YOLOv10-Fire-and-Smoke-Detection (HuggingFace)
  https://huggingface.co/TommyNgx/YOLOv10-Fire-and-Smoke-Detection

False-positive mitigation layers (in order):
  1. Per-class confidence thresholds (smoke is stricter than fire).
  2. Size sanity filters (too-small/too-large boxes are rejected).
  3. Color verification in HSV (fire must have warm pixels;
     smoke must be low-saturation grey-ish).
  4. Multi-frame IoU-tracked confirmation (N consecutive frames).
  5. Per-location alert cooldown.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
from ultralytics import YOLO


class FireDetectionSystem:
    """Fire and smoke detection using a YOLOv10 HuggingFace model."""

    def __init__(self, model_path: str = "models/fire_and_smoke/best.pt"):
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Fire detection model not found at {model_path}")

        print(f"Loading fire detection model from: {model_path}")
        self.model = YOLO(model_path)

        # Normalize class names from the model metadata to lowercase.
        # HuggingFace model uses {0: 'Fire', 1: 'Smoke'}.
        raw_names = getattr(self.model, 'names', {0: 'fire', 1: 'smoke'})
        self.class_names = {int(k): str(v).lower() for k, v in raw_names.items()}

        # Per-class confidence thresholds. Smoke is much more prone to
        # false positives (grey walls, clouds, steam) so demand more.
        self.conf_thresholds = {
            'fire': 0.40,
            'smoke': 0.55,
        }
        self.default_conf = 0.40
        self.iou_threshold = 0.45  # NMS IoU

        # Size sanity filters: reject detections that are either negligible
        # (noise, reflections) or span almost the whole frame (runaway misclass).
        self.min_area_ratio = 0.003   # 0.3% of frame
        self.max_area_ratio = 0.85    # 85% of frame

        # Alert cooldown per location (seconds)
        self.alert_cooldown = 3.0
        self.last_alert_time = {}

        # Multi-frame confirmation
        self.min_consecutive_frames = 6
        self.frame_history = {}
        self.current_frame_number = 0
        self.iou_match_threshold = 0.3

        # Statistics
        self.stats = {
            'total_detections': 0,
            'fire_detections': 0,
            'smoke_detections': 0,
            'rejected_by_size': 0,
            'rejected_by_color': 0,
        }

        print("Fire detection system initialized successfully!")

    # ─────────────────────── Main entry ───────────────────────
    def process_frame(self, frame: np.ndarray) -> List[Dict]:
        """Run detection on a single frame and return a list of detections."""
        h, w = frame.shape[:2]
        frame_area = h * w
        detections = []
        self.current_frame_number += 1

        # Use the lowest per-class threshold so we still get candidates
        # that we then filter per-class below.
        min_conf = min(self.conf_thresholds.values())

        results = self.model.predict(
            frame,
            conf=min_conf,
            iou=self.iou_threshold,
            verbose=False,
        )[0]

        now = datetime.now()
        matched_keys = set()

        if results.boxes is not None and len(results.boxes) > 0:
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 <= x1 or y2 <= y1:
                    continue

                cls = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = self.class_names.get(cls, f'unknown_{cls}')

                # Filter 1: per-class confidence threshold
                if conf < self.conf_thresholds.get(class_name, self.default_conf):
                    continue

                # Filter 2: size sanity
                area = (x2 - x1) * (y2 - y1)
                area_ratio = area / frame_area
                if area_ratio < self.min_area_ratio or area_ratio > self.max_area_ratio:
                    self.stats['rejected_by_size'] += 1
                    continue

                # Filter 3: color verification in HSV space
                roi = frame[y1:y2, x1:x2]
                if not self._color_plausible(roi, class_name):
                    self.stats['rejected_by_color'] += 1
                    continue

                # Filter 4: multi-frame IoU-tracked confirmation
                consecutive = self._update_frame_history(
                    class_name, (x1, y1, x2, y2), matched_keys
                )
                confirmed = consecutive >= self.min_consecutive_frames

                # Alert (with per-location cooldown) once confirmed
                should_alert = False
                if confirmed:
                    alert_key = f"{class_name}_{x1//40}_{y1//40}"  # coarser key
                    last = self.last_alert_time.get(alert_key)
                    if last is None or (now - last).total_seconds() > self.alert_cooldown:
                        should_alert = True
                        self.last_alert_time[alert_key] = now

                # Stats
                self.stats['total_detections'] += 1
                if class_name == 'fire':
                    self.stats['fire_detections'] += 1
                elif class_name == 'smoke':
                    self.stats['smoke_detections'] += 1

                detections.append({
                    'bbox': (x1, y1, x2, y2),
                    'class': class_name,
                    'confidence': conf,
                    'timestamp': now,
                    'alert': should_alert,
                    'confirmed': confirmed,
                    'consecutive_frames': consecutive,
                    'area_ratio': area_ratio,
                })

        # Expire tracking entries not seen this frame (streak broken)
        expired = [k for k, v in self.frame_history.items()
                   if v['last_seen_frame'] < self.current_frame_number]
        for k in expired:
            del self.frame_history[k]

        return detections

    # ─────────────────────── Filters ───────────────────────
    @staticmethod
    def _color_plausible(roi: np.ndarray, class_name: str) -> bool:
        """
        Heuristic color check in HSV.

        Fire: should have a meaningful fraction of warm, saturated pixels
              (red-orange-yellow hues).
        Smoke: should be low-saturation and roughly neutral brightness
               (grey / white / dark grey).
        """
        if roi.size == 0:
            return False
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        total = roi.shape[0] * roi.shape[1]

        if class_name == 'fire':
            # Warm hues wrap around 0: 0-25 and 160-179; require decent saturation & brightness.
            warm = ((h <= 25) | (h >= 160)) & (s >= 90) & (v >= 120)
            return (warm.sum() / total) >= 0.12  # ≥12% warm pixels

        if class_name == 'smoke':
            greyish = (s <= 60)          # low saturation
            not_too_dark = (v >= 40)     # reject black patches
            not_uniform_sky = v.std() >= 6  # reject perfectly uniform sky/wall
            return (((greyish & not_too_dark).sum() / total) >= 0.55) and bool(not_uniform_sky)

        return True

    def _update_frame_history(self, class_name: str,
                              bbox: Tuple[int, int, int, int],
                              matched_keys: set) -> int:
        """Match detection against tracked history via IoU; return streak length."""
        best_key, best_iou = None, 0.0
        for key, entry in self.frame_history.items():
            if key in matched_keys or not key.startswith(class_name + "_"):
                continue
            iou = self._compute_iou(bbox, entry['bbox'])
            if iou > best_iou:
                best_iou, best_key = iou, key

        if best_key is not None and best_iou >= self.iou_match_threshold:
            self.frame_history[best_key]['count'] += 1
            self.frame_history[best_key]['last_seen_frame'] = self.current_frame_number
            self.frame_history[best_key]['bbox'] = bbox
            matched_keys.add(best_key)
            return self.frame_history[best_key]['count']

        new_key = f"{class_name}_{self.current_frame_number}_{id(bbox)}"
        self.frame_history[new_key] = {
            'bbox': bbox,
            'count': 1,
            'last_seen_frame': self.current_frame_number,
        }
        matched_keys.add(new_key)
        return 1

    @staticmethod
    def _compute_iou(a: Tuple[int, int, int, int],
                     b: Tuple[int, int, int, int]) -> float:
        x1, y1 = max(a[0], b[0]), max(a[1], b[1])
        x2, y2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        if inter == 0:
            return 0.0
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        return inter / (area_a + area_b - inter)

    # ─────────────────────── Drawing ───────────────────────
    @staticmethod
    def draw_detections(frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Draw confirmed detections. Unconfirmed candidates are not drawn."""
        output = frame.copy()
        class_colors = {
            'fire': (0, 0, 255),        # red
            'smoke': (160, 160, 160),   # grey
        }

        for det in detections:
            if not det.get('confirmed'):
                continue
            x1, y1, x2, y2 = det['bbox']
            cls = det['class']
            conf = det['confidence']
            color = class_colors.get(cls, (0, 255, 255))
            thickness = 4 if det.get('alert') else 2

            cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)

            label = f"{cls.upper()} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            ty = y1 - 8 if y1 > 25 else y2 + th + 8
            cv2.rectangle(output, (x1, ty - th - 4), (x1 + tw + 8, ty + 4), color, -1)
            cv2.putText(output, label, (x1 + 4, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Top banner if any fire alert fired this frame
        if any(d.get('alert') and d['class'] == 'fire' for d in detections):
            msg = "FIRE ALERT"
            (mw, mh), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 3)
            x = (output.shape[1] - mw) // 2
            cv2.rectangle(output, (x - 20, 10), (x + mw + 20, 20 + mh + 10), (0, 0, 255), -1)
            cv2.putText(output, msg, (x, 20 + mh),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)

        return output

    # ─────────────────────── Stats ───────────────────────
    def get_statistics(self) -> Dict:
        return {
            'model_loaded': self.model is not None,
            'conf_thresholds': self.conf_thresholds,
            'min_area_ratio': self.min_area_ratio,
            'max_area_ratio': self.max_area_ratio,
            'min_consecutive_frames': self.min_consecutive_frames,
            'alert_cooldown': self.alert_cooldown,
            'active_alerts': len(self.last_alert_time),
            **self.stats,
        }

    def reset_statistics(self):
        self.stats = {
            'total_detections': 0,
            'fire_detections': 0,
            'smoke_detections': 0,
            'rejected_by_size': 0,
            'rejected_by_color': 0,
        }
        self.last_alert_time.clear()
        self.frame_history.clear()
        self.current_frame_number = 0


if __name__ == "__main__":
    import sys
    fire_system = FireDetectionSystem()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open camera"); sys.exit(1)
    print("'q' to quit, 's' for stats")
    while True:
        ret, frame = cap.read()
        if not ret: break
        dets = fire_system.process_frame(frame)
        out = fire_system.draw_detections(frame, dets)
        cv2.imshow("Fire & Smoke", out)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'): break
        if k == ord('s'): print(fire_system.get_statistics())
    cap.release(); cv2.destroyAllWindows()
