#!/usr/bin/env python3
"""
License Plate Recognition System - FIXED FOR SMALL PLATES

This version automatically handles small plate crops by:
1. Increasing detection padding (0.05 → 0.20)
2. Upscaling crops to 250+ pixels wide
3. Using better interpolation

Works with EasyOCR (or switch to PaddleOCR for even better results)
"""

from __future__ import annotations

import re
import cv2
import numpy as np
import torch
import easyocr
from collections import Counter, deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from ultralytics import YOLO
from database_manager import DatabaseManager


# Romanian plate formats:
#   - County plates: 1-2 letters + 2-3 digits + 3 letters  (B123ABC, CJ12ABC)
#   - Bucharest can have 3 digits (B123ABC)
ROMANIAN_PLATE_RE = re.compile(r'^[A-Z]{1,2}\d{2,3}[A-Z]{3}$')


def strip_eu_band(text: str) -> str:
    """
    Remove a leading EU country-band code that OCR occasionally reads as part of
    the plate. On Romanian plates the blue strip on the left carries "RO" under
    the EU stars; when the crop includes that strip and is upscaled, EasyOCR can
    transcribe it, yielding e.g. "ROB95GSV" instead of "B95GSV".

    No Romanian county code is "RO" (county prefixes are B, AB, CJ, IF, SB, ...),
    so a leading "RO" followed by a plausible plate body is always the band, not
    part of the number. We strip it only when at least 5 characters remain, so a
    genuine short reading is never truncated.
    """
    if text.startswith("RO") and len(text) - 2 >= 5:
        return text[2:]
    return text


# Valid county codes (41 counties + B for Bucharest). Used to pick the correct
# number from the OCR text, avoiding the EU band or other stray characters being
# mistaken for the county prefix.
RO_COUNTIES = {
    "AB", "AR", "AG", "BC", "BH", "BN", "BT", "BV", "BR", "BZ", "CS", "CL", "CJ",
    "CT", "CV", "DB", "DJ", "GL", "GR", "GJ", "HR", "HD", "IL", "IS", "IF", "MM",
    "MH", "MS", "NT", "OT", "PH", "SM", "SJ", "SB", "SV", "TR", "TM", "TL", "VS",
    "VL", "VN", "B",
}

# Pattern of a Romanian plate number, searched inside the text (lookahead to allow
# overlapping matches, e.g. both "OB999NDY" and "B999NDY").
_RO_PLATE_SEARCH = re.compile(r'(?=([A-Z]{1,2})(\d{2,3})([A-Z]{3}))')


def extract_ro_plate(text: str) -> str:
    """
    Extract the Romanian plate number from a possibly noisy OCR text.
    OCR sometimes adds characters around the plate (the "RO" EU band, dealer
    stickers, county badges). We look for the substring that matches the Romanian
    plate structure and whose county code is valid; if several exist, we pick the
    leftmost. If none is found, we fall back to stripping a leading EU band.
    """
    for m in _RO_PLATE_SEARCH.finditer(text):
        county, digits, suffix = m.groups()
        if county in RO_COUNTIES:
            return county + digits + suffix
    return strip_eu_band(text)


# ============================================================================
# HELPER UTILITIES
# ============================================================================

def expand_bbox(bbox: Tuple[int, int, int, int], pad_ratio: float,
                w: int, h: int) -> Tuple[int, int, int, int]:
    """Expand bounding box with padding"""
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    pad_x, pad_y = int(bw * pad_ratio), int(bh * pad_ratio)
    return (max(0, x1 - pad_x), max(0, y1 - pad_y),
            min(w, x2 + pad_x), min(h, y2 + pad_y))


def upscale_if_small(crop: np.ndarray, 
                     target_width: int = 250,
                     max_scale: float = 3.0) -> np.ndarray:
    """
    Upscale crop if too small for reliable OCR
    
    CRITICAL FIX: Your plates are 116-151px wide (TOO SMALL)
    This upscales them to 250px for much better OCR results
    
    Args:
        crop: Original plate crop
        target_width: Target width in pixels (250 is good for OCR)
        max_scale: Maximum upscaling factor (3.0 = 3x max)
    
    Returns:
        Upscaled crop
    """
    if crop.size == 0:
        return crop
    
    h, w = crop.shape[:2]
    
    if w < target_width:
        # Calculate scale factor
        scale = target_width / w
        scale = min(scale, max_scale)  # Don't upscale too much
        
        # New dimensions
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Upscale with high-quality interpolation
        upscaled = cv2.resize(crop, (new_w, new_h), 
                             interpolation=cv2.INTER_CUBIC)
        
        #print(f"Upscaled plate: {w}×{h} → {new_w}×{new_h} (scale: {scale:.2f}x)")
        return upscaled
    
    return crop


def preprocess_plate(img: np.ndarray) -> np.ndarray:
    """
    Stronger preprocessing tuned for EasyOCR on license plates.

    Pipeline:
      1. Grayscale
      2. Bilateral filter - denoise while preserving character edges
      3. CLAHE - local contrast boost (handles shadows / uneven light)
      4. Unsharp mask - sharpen characters after upscaling

    Note: we deliberately do NOT binarize. EasyOCR's CRNN performs better
    on continuous-tone grayscale than on thresholded images.
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    gray = cv2.bilateralFilter(gray, 11, 17, 17)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.0)
    gray = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)

    return gray


# ============================================================================
# PLATE TRACKER - temporal majority vote across frames
# ============================================================================

def _iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    xa1, ya1, xa2, ya2 = a
    xb1, yb1, xb2, yb2 = b
    ix1, iy1 = max(xa1, xb1), max(ya1, yb1)
    ix2, iy2 = min(xa2, xb2), min(ya2, yb2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = (xa2 - xa1) * (ya2 - ya1) + (xb2 - xb1) * (yb2 - yb1) - inter
    return inter / union if union else 0.0


class PlateTracker:
    """
    Tracks plate detections across frames by IoU and emits a stabilized
    reading only after the same text wins a majority vote over several
    frames. Filters non-Romanian-format readings out of the vote.
    """

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 15,
                 min_votes: int = 3, history: int = 10):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_votes = min_votes
        self.history = history
        self.tracks: Dict[int, Dict] = {}
        self._next_id = 0

    def update(self, bbox: Tuple[int, int, int, int], text: str,
               conf: float) -> Tuple[Optional[str], float]:
        best_id, best_iou = None, self.iou_threshold
        for tid, t in self.tracks.items():
            i = _iou(bbox, t["bbox"])
            if i > best_iou:
                best_id, best_iou = tid, i

        if best_id is None:
            best_id = self._next_id
            self._next_id += 1
            self.tracks[best_id] = {
                "bbox": bbox,
                "readings": deque(maxlen=self.history),
                "age": 0,
                "emitted": None,
                "emitted_conf": 0.0,
            }

        t = self.tracks[best_id]
        t["bbox"] = bbox
        t["age"] = 0

        if text and conf >= 0.4 and ROMANIAN_PLATE_RE.match(text):
            t["readings"].append((text, conf))

        if len(t["readings"]) >= self.min_votes:
            scores: Counter = Counter()
            for txt, c in t["readings"]:
                scores[txt] += c
            best_text, total_conf = scores.most_common(1)[0]
            votes = sum(1 for txt, _ in t["readings"] if txt == best_text)
            if votes >= self.min_votes:
                t["emitted"] = best_text
                t["emitted_conf"] = total_conf / votes

        return t["emitted"], t["emitted_conf"]

    def age_tracks(self) -> None:
        for tid in list(self.tracks):
            self.tracks[tid]["age"] += 1
            if self.tracks[tid]["age"] > self.max_age:
                del self.tracks[tid]


# ============================================================================
# MAIN SYSTEM
# ============================================================================

class LicensePlateRecognitionSystem:
    """
    License Plate Recognition with SMALL PLATE FIX
    
    Key changes from original:
    1. Increased padding: 0.05 → 0.20 (captures more context)
    2. Automatic upscaling: crops upscaled to 250+ pixels
    3. Better interpolation: INTER_CUBIC for quality
    """

    def __init__(self,
                 db: DatabaseManager,
                 plate_detector_weights: str | Path = "models/license_plate/best.pt",
                 ocr_lang: List[str] = ['en'],
                 use_gpu: bool = True,
                 target_plate_width: int = 250,
                 min_votes: int = 3,
                 tracker_max_age: int = 15):
        """
        Initialize with small plate fixes
        
        Args:
            db: Database config dict
            plate_detector_weights: Path to YOLO weights
            ocr_lang: OCR languages
            use_gpu: Use GPU acceleration
            target_plate_width: Target width for upscaling (250 recommended)
        """
        self.db = db
        self.target_plate_width = target_plate_width

        self.db_manager = DatabaseManager(**self.db)
        self.db_manager.connect()

        # Check GPU
        use_gpu = torch.cuda.is_available() and use_gpu
        device = "cuda" if use_gpu else "cpu"
        
        print(f"System using: {device}")
        if use_gpu:
            print(f"   GPU: {torch.cuda.get_device_name(0)}")

        # Load YOLO
        plate_path = Path(plate_detector_weights)
        if not plate_path.exists():
            raise FileNotFoundError(f"Weights not found: {plate_path}")
        
        print(f"Loading YOLO from: {plate_path}")
        self.plate_detector = YOLO(str(plate_path))
        if use_gpu:
            self.plate_detector.to('cuda')
        
        # Initialize EasyOCR
        print(f"Initializing EasyOCR (GPU: {use_gpu})...")
        self.reader = easyocr.Reader(
            ocr_lang,
            gpu=use_gpu,
            verbose=False
        )
        
        self.tracker = PlateTracker(min_votes=min_votes,
                                    max_age=tracker_max_age)

        print("System ready with SMALL PLATE FIX")
        print(f"   Will upscale plates to {target_plate_width}+ pixels")
        print(f"    Temporal voting: need {min_votes} consistent reads")

    def process_frame(self, frame: np.ndarray) -> List[Dict]:
        """
        Process frame with SMALL PLATE FIXES
        """
        h, w = frame.shape[:2]
        outs = []

        # Detect plates
        detection_results = self.plate_detector.predict(
            frame, 
            imgsz=640, 
            conf=0.25,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            verbose=False
        )[0].boxes
        
        for plate in detection_results:
            x1, y1, x2, y2 = map(int, plate.xyxy[0])
            
            # FIX 1: INCREASE PADDING (0.05 → 0.20)
            x1, y1, x2, y2 = expand_bbox((x1, y1, x2, y2), 0.20, w, h)
            crop = frame[y1:y2, x1:x2]
            
            if crop.size == 0:
                continue
            
            # FIX 2: UPSCALE IF TOO SMALL
            crop = upscale_if_small(crop, target_width=self.target_plate_width)
            
            # Preprocess
            pre = preprocess_plate(crop)

            # OCR with allowlist + internal upscaling
            raw_text, raw_conf = self._run_ocr(pre)

            # Fallback on original color crop if low confidence
            if not raw_text or raw_conf < 0.4:
                alt_text, alt_conf = self._run_ocr(crop)
                if alt_conf > raw_conf:
                    raw_text, raw_conf = alt_text, alt_conf

            # Push this per-frame reading through the tracker.
            # The tracker filters by Romanian plate format and only emits
            # a stable result after enough consistent votes.
            stable_text, stable_conf = self.tracker.update(
                (x1, y1, x2, y2), raw_text, raw_conf
            )

            # Lookup only once we have a stabilized reading
            owner = "Unknown car"
            authorised = False
            if stable_text:
                owner_name = self.db_manager.lookup_owner_by_plate(stable_text)
                if owner_name:
                    owner = owner_name
                    authorised = True
                print(f"Stable plate: {stable_text} (conf: {stable_conf:.2f})")

            plate_number = stable_text if stable_text else "UNKNOWN"

            outs.append({
                "timestamp": datetime.now(),
                "plate_number": plate_number,
                "vehicle_type": None,
                "is_authorized": authorised,
                "bbox": (x1, y1, x2, y2),
                "plate": plate_number,
                "confidence": float(stable_conf),
                "owner": owner,
                "authorised": authorised,
                "raw_reading": raw_text,
                "raw_confidence": float(raw_conf),
            })

        # Age tracks so stale cars drop out
        self.tracker.age_tracks()
        return outs

    def _run_ocr(self, img: np.ndarray) -> Tuple[str, float]:
        """Run EasyOCR with plate-specific constraints and L->R assembly."""
        try:
            results = self.reader.readtext(
                img,
                allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                detail=1,
                paragraph=False,
                text_threshold=0.6,
                low_text=0.3,
                mag_ratio=1.5,
            )
        except Exception as e:
            print(f"OCR error: {e}")
            return "", 0.0

        if not results:
            return "", 0.0

        # Sort detections left-to-right by top-left x coordinate
        results.sort(key=lambda r: r[0][0][0])

        text_parts: List[str] = []
        confs: List[float] = []
        for (_bbox, detected_text, confidence) in results:
            cleaned = ''.join(
                c for c in detected_text.upper() if c.isalnum()
            )
            if cleaned:
                text_parts.append(cleaned)
                confs.append(confidence)

        if not text_parts:
            return "", 0.0

        text = extract_ro_plate(''.join(text_parts))
        conf = float(np.mean(confs))
        return text, conf
    
    def register_plate(self, plate_number: str, vehicle_type: str = None,
                       owner_name: str = None, owner_id: str = None,
                       is_authorized: bool = True, expiry_date: datetime = None,
                       notes: str = None) -> bool:
        """Register new plate"""
        try:
            plate_id = self.db_manager.add_license_plate(
                plate_number=plate_number,
                vehicle_type=vehicle_type,
                owner_name=owner_name,
                owner_id=owner_id,
                is_authorized=is_authorized,
                expiry_date=expiry_date,
                notes=notes
            )
            print(f"Registered plate ID: {plate_id}")
            return True
        except Exception as e:
            print(f"Failed to register: {e}")
            return False

    @staticmethod
    def draw_outputs(frame: np.ndarray, outs: List[Dict]) -> np.ndarray:
        """Draw detections"""
        for o in outs:
            x1, y1, x2, y2 = o["bbox"]
            colour = (0, 255, 0) if o["authorised"] else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
            
            for i, line in enumerate([o["owner"], o["plate"],
                                      f"conf:{o['confidence']:.2f}"]):
                cv2.putText(frame, line, (x1, y2 + 22 + 20 * i),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
        return frame


# ============================================================================
# DEMO
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("LICENSE PLATE RECOGNITION - FIXED FOR SMALL PLATES")
    print("="*70)
    print("\nFixes applied:")
    print("   1. Increased padding: 0.05 → 0.20 (more context)")
    print("   2. Auto-upscaling: plates scaled to 250+ pixels")
    print("   3. Better interpolation: INTER_CUBIC quality")
    print("\nExpected improvement:")
    print("   Before: 116×33 pixels → OCR fails")
    print("   After:  250×70 pixels → OCR succeeds!")
    print("="*70 + "\n")
    
    db_config = {
        'host': 'localhost',
        'database': 'facial_recognition',
        'user': 'postgres',
        'password': 'incorect'
    }
    
    lpr = LicensePlateRecognitionSystem(db_config)
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Could not open camera")
        exit(1)
    
    print("Camera opened")
    print("\nPress 'Q' to quit\n")
    
    frame_count = 0
    
    while True:
        ret, frm = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Process every 2nd frame for performance
        if frame_count % 2 == 0:
            detections = lpr.process_frame(frm)
            vis = LicensePlateRecognitionSystem.draw_outputs(frm.copy(), detections)
        else:
            vis = frm
        
        cv2.putText(vis, "FIXED VERSION - Upscaling enabled", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("License Plate Recognition (FIXED) - Press Q to quit", vis)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    
    cap.release()
    cv2.destroyAllWindows()