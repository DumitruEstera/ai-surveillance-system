#!/usr/bin/env python3
"""
Weapon Detection System using YOLOv8
Model: https://github.com/jztchl/realtime-weapon-detection
YOLOv8 model trained for weapon detection
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from ultralytics import YOLO
import os


class WeaponDetectionSystem:
    """Weapon detection using YOLOv8 trained model"""

    def __init__(self, model_path: str = "models/weapon/best.pt"):
        """
        Initialize Weapon Detection System

        Args:
            model_path: Path to the YOLOv8 weapon detection model
        """
        # Check if model exists
        if not Path(model_path).exists():
            print(f"⚠  Model not found at: {model_path}")
            print("Please download the model from:")
            print("   https://github.com/jztchl/realtime-weapon-detection")
            print(f"   And place 'best.pt' at: {model_path}")
            raise FileNotFoundError(f"Weapon detection model not found at {model_path}")

        print(f"Loading weapon detection model from: {model_path}")
        self.model = YOLO(model_path)

        # Get class names from the model itself
        self.class_names = self.model.names if self.model.names else {0: 'weapon'}
        print(f"   Model classes: {self.class_names}")

        # Base model confidence threshold (kept low so candidates reach the
        # stricter post-filter below).
        self.confidence_threshold = 0.3
        self.iou_threshold = 0.45

        # Stricter post-filter confidence: a detection must exceed this to
        # be considered. Raised to suppress phone-edge false positives.
        self.min_confidence = 0.7

        # Alert cooldown (seconds) - to avoid spamming alerts
        self.alert_cooldown = 3.0
        self.last_alert_time = {}

        # Multi-frame temporal voting: a detection must persist across this
        # many consecutive frames before it is considered confirmed and
        # allowed to raise an alert.
        self.min_consecutive_frames = 6
        self.frame_history = {}  # key -> {'bbox', 'count', 'last_seen_frame'}
        self.current_frame_number = 0
        self.iou_match_threshold = 0.3  # IoU to treat two boxes as the same track

        # Statistics tracking
        self.stats = {
            'total_detections': 0,
            'critical_alerts': 0
        }

        print("Weapon detection system initialized successfully!")

    def process_frame(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect weapons in a frame

        Args:
            frame: Input frame (BGR format)

        Returns:
            List of detection dictionaries
        """
        h, w = frame.shape[:2]
        detections = []
        self.current_frame_number += 1

        # Run detection with the model
        results = self.model.predict(
            frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            verbose=False
        )[0]

        current_time = datetime.now()

        # Track which history entries were matched this frame so each
        # tracked box is consumed by at most one current detection.
        matched_keys = set()

        # Process detections
        if results.boxes is not None and len(results.boxes) > 0:
            for box in results.boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Ensure coordinates are within frame bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                # Get class and confidence
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                # Model has a single class: 'weapon'.
                class_name = self.class_names.get(cls, 'weapon')

                # --- Confidence filter ---
                # Drop low-confidence candidates. Phones seen from the side
                # can score low-to-mid confidence as weapons, so the bar
                # is deliberately higher than the raw model threshold.
                if conf < self.min_confidence:
                    continue

                # Calculate detection area
                detection_area = (x2 - x1) * (y2 - y1)
                frame_area = w * h

                # Calculate severity - weapons are always high priority
                severity = self._calculate_severity(class_name, conf, detection_area, frame_area)

                # --- Multi-frame temporal voting ---
                # Require the same detection to persist across several
                # consecutive frames before it can raise an alert.
                consecutive_count, track_key = self._update_frame_history(
                    class_name, (x1, y1, x2, y2), matched_keys
                )
                confirmed = consecutive_count >= self.min_consecutive_frames

                # Unconfirmed candidates are tracked silently — don't surface
                # them to the UI, since showing a "WEAPON" box for a phone
                # is exactly the false alarm we're trying to eliminate.
                if not confirmed:
                    continue

                # Alert cooldown keyed by track, not by exact bbox (bboxes
                # jitter every frame, which would defeat a bbox-keyed cooldown).
                should_alert = False
                if track_key not in self.last_alert_time:
                    should_alert = True
                    self.last_alert_time[track_key] = current_time
                else:
                    time_diff = (current_time - self.last_alert_time[track_key]).total_seconds()
                    if time_diff > self.alert_cooldown:
                        should_alert = True
                        self.last_alert_time[track_key] = current_time

                # Update statistics
                self.stats['total_detections'] += 1
                if severity == 'critical':
                    self.stats['critical_alerts'] += 1

                # Create detection dictionary
                detection = {
                    'bbox': (x1, y1, x2, y2),
                    'class': class_name,
                    'confidence': conf,
                    'timestamp': current_time,
                    'alert': should_alert,
                    'confirmed': confirmed,
                    'consecutive_frames': consecutive_count,
                    'severity': severity,
                    'area_ratio': detection_area / frame_area
                }

                detections.append(detection)

        # Expire tracks that were not matched this frame — the streak breaks
        # as soon as the detection disappears for one frame.
        expired = [k for k, v in self.frame_history.items()
                   if v['last_seen_frame'] < self.current_frame_number]
        for k in expired:
            del self.frame_history[k]

        return detections

    def _update_frame_history(self, class_name: str,
                              bbox: Tuple[int, int, int, int],
                              matched_keys: set) -> Tuple[int, str]:
        """
        Match `bbox` against existing tracks of the same class via IoU and
        increment its consecutive counter, or open a new track.

        Returns (consecutive_count, track_key).
        """
        best_key: Optional[str] = None
        best_iou = 0.0

        for key, entry in self.frame_history.items():
            if key in matched_keys:
                continue
            if not key.startswith(class_name + "_"):
                continue
            iou = self._compute_iou(bbox, entry['bbox'])
            if iou > best_iou:
                best_iou = iou
                best_key = key

        if best_key is not None and best_iou >= self.iou_match_threshold:
            self.frame_history[best_key]['count'] += 1
            self.frame_history[best_key]['last_seen_frame'] = self.current_frame_number
            self.frame_history[best_key]['bbox'] = bbox
            matched_keys.add(best_key)
            return self.frame_history[best_key]['count'], best_key

        new_key = f"{class_name}_{id(bbox)}_{self.current_frame_number}"
        self.frame_history[new_key] = {
            'bbox': bbox,
            'count': 1,
            'last_seen_frame': self.current_frame_number,
        }
        matched_keys.add(new_key)
        return 1, new_key

    @staticmethod
    def _compute_iou(box_a: Tuple[int, int, int, int],
                     box_b: Tuple[int, int, int, int]) -> float:
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        if inter == 0:
            return 0.0

        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        return inter / (area_a + area_b - inter)

    def _calculate_severity(self, class_name: str, confidence: float,
                           detection_area: int, frame_area: int) -> str:
        """
        Calculate severity level based on detection parameters.
        Weapons are always treated as high-priority threats.
        """
        if confidence > 0.7:
            return 'critical'
        elif confidence > 0.5:
            return 'high'
        elif confidence > 0.4:
            return 'medium'
        else:
            return 'low'

    @staticmethod
    def draw_detections(frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Draw detection boxes and labels on frame

        Args:
            frame: Input frame
            detections: List of detection dictionaries

        Returns:
            Frame with drawn detections
        """
        output = frame.copy()

        # Define colors for different severity levels
        severity_colors = {
            'critical': (0, 0, 255),      # Red
            'high': (0, 50, 200),          # Dark Red
            'medium': (0, 100, 255),       # Orange-Red
            'low': (0, 165, 255)           # Orange
        }

        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            class_name = detection['class']
            confidence = detection['confidence']
            severity = detection['severity']
            area_ratio = detection.get('area_ratio', 0)

            # Choose color based on severity
            color = severity_colors.get(severity, (0, 0, 255))

            # Make critical alerts more visible
            thickness = 4 if severity in ('critical', 'high') else 2

            # Draw rectangle
            cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)

            # Prepare labels
            label = "WEAPON DETECTED"
            conf_label = f"Conf: {confidence:.2f}"
            severity_label = f"Severity: {severity.upper()}"

            # Calculate text size for background
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = 2

            # Draw background rectangle for text
            text_y = y1 - 10
            if text_y < 60:
                text_y = y2 + 20
                label_positions = [
                    (text_y, label),
                    (text_y + 20, conf_label),
                    (text_y + 40, severity_label),
                ]
            else:
                label_positions = [
                    (text_y, label),
                    (text_y - 20, conf_label),
                    (text_y - 40, severity_label),
                ]

            # Draw background for all text
            max_text_width = max([cv2.getTextSize(text, font, font_scale, font_thickness)[0][0]
                                 for _, text in label_positions])

            bg_y1 = min([y for y, _ in label_positions]) - 5
            bg_y2 = max([y for y, _ in label_positions]) + 5
            cv2.rectangle(output, (x1, bg_y1), (x1 + max_text_width + 10, bg_y2),
                         color, -1)

            # Draw all text labels
            for y_pos, text in label_positions:
                cv2.putText(output, text, (x1 + 5, y_pos),
                           font, font_scale, (255, 255, 255), font_thickness)

        return output

    def get_statistics(self) -> Dict:
        """Get detection statistics"""
        return {
            'model_loaded': self.model is not None,
            'confidence_threshold': self.confidence_threshold,
            'iou_threshold': self.iou_threshold,
            'alert_cooldown': self.alert_cooldown,
            'min_confidence': self.min_confidence,
            'min_consecutive_frames': self.min_consecutive_frames,
            'active_tracks': len(self.frame_history),
            'active_alerts': len(self.last_alert_time),
            **self.stats
        }

    def reset_statistics(self):
        """Reset detection statistics"""
        self.stats = {
            'total_detections': 0,
            'critical_alerts': 0
        }
        self.last_alert_time.clear()
        self.frame_history.clear()
        self.current_frame_number = 0
