#!/usr/bin/env python3
"""
Step 3 (video test) — extract plates from clips for annotation.

You do not draw any bounding box. This script runs the YOLO detector on every frame
of every clip, groups detections of the same plate into "tracks" by overlap (IoU),
and saves, for each track, the clearest crops + a montage — so you can READ the
number and fill it in manually.

Put the clips (.mp4/.avi/.mov/.mkv) in:  video_eval/clips/
Then run:
    python plate_03_video_extract.py
Output goes to video_eval/<clip_name>/track_XX/ crops + montaj.png, and a
video_ground_truth.csv file to fill in (the plate_number column).
"""

import os
import sys

_CUDNN8 = os.path.expanduser("~/.local/cudnn8_pkg/nvidia/cudnn/lib")
if os.path.isdir(_CUDNN8) and _CUDNN8 not in os.environ.get("LD_LIBRARY_PATH", ""):
    os.environ["LD_LIBRARY_PATH"] = _CUDNN8 + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    os.execv(sys.executable, [sys.executable] + sys.argv)

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = Path(os.environ.get("SURVEILLANCE_APP_DIR",
                                   _THIS_DIR.parents[1] / "surveillance-platform"))
sys.path.insert(0, str(_PROJECT_DIR))
from license_plate_recognition_system import expand_bbox, upscale_if_small, _iou  # noqa: E402
from ultralytics import YOLO  # noqa: E402

VIDEO_DIR = _THIS_DIR / "video_eval"
CLIPS_DIR = VIDEO_DIR / "clips"
GT_CSV = VIDEO_DIR / "video_ground_truth.csv"
YOLO_WEIGHTS = _PROJECT_DIR / "models" / "license_plate" / "best.pt"
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def _sharpness(img: np.ndarray) -> float:
    """Laplacian variance — higher = sharper (less blur)."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


class _SpatialTracker:
    """Group detections of the same plate over consecutive frames, by IoU."""
    def __init__(self, iou_thr: float = 0.3, max_age: int = 20):
        self.iou_thr = iou_thr
        self.max_age = max_age
        self.tracks: Dict[int, Dict] = {}
        self._next = 0

    def update(self, boxes: List[Tuple[int, int, int, int]], frame, fidx: int):
        for t in self.tracks.values():
            t["age"] += 1
        for box in boxes:
            best_id, best = None, self.iou_thr
            for tid, t in self.tracks.items():
                i = _iou(box, t["bbox"])
                if i > best:
                    best_id, best = tid, i
            if best_id is None:
                best_id = self._next; self._next += 1
                self.tracks[best_id] = {"bbox": box, "age": 0, "crops": []}
            t = self.tracks[best_id]
            t["bbox"] = box; t["age"] = 0
            x1, y1, x2, y2 = box
            crop = frame[y1:y2, x1:x2]
            if crop.size:
                t["crops"].append({"frame": fidx, "crop": crop,
                                   "area": (x2 - x1) * (y2 - y1),
                                   "sharp": _sharpness(crop)})
        for tid in [tid for tid, t in self.tracks.items() if t["age"] > self.max_age]:
            del self.tracks[tid]


def _best_crops(crops: List[Dict], k: int = 8) -> List[np.ndarray]:
    """The best k crops: prioritize large area and sharpness."""
    if not crops:
        return []
    amax = max(c["area"] for c in crops) or 1
    smax = max(c["sharp"] for c in crops) or 1
    ranked = sorted(crops, key=lambda c: 0.5 * c["area"] / amax + 0.5 * c["sharp"] / smax,
                    reverse=True)
    out, seen = [], set()
    for c in ranked:
        if c["frame"] in seen:
            continue
        seen.add(c["frame"])
        out.append(upscale_if_small(c["crop"], target_width=320))
        if len(out) >= k:
            break
    return out


def _montage(crops: List[np.ndarray], width: int = 320) -> np.ndarray:
    rows = []
    for c in crops:
        h = int(c.shape[0] * width / c.shape[1])
        rows.append(cv2.resize(c, (width, h)))
    pad = np.full((6, width, 3), 255, np.uint8)
    out = []
    for r in rows:
        out.append(r); out.append(pad)
    return np.vstack(out) if out else np.zeros((1, width, 3), np.uint8)


def process_video(detector, device: str, video: Path, stride: int,
                  min_track_len: int) -> List[Dict]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print(f"[WARN] Nu pot deschide {video.name}")
        return []
    tracker = _SpatialTracker()
    fidx = 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    print(f"[VIDEO] {video.name} (~{total} cadre, stride={stride})")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if fidx % stride == 0:
            h, w = frame.shape[:2]
            boxes = detector.predict(frame, imgsz=640, conf=0.25, device=device,
                                     verbose=False)[0].boxes
            dets = []
            for b in boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                dets.append(expand_bbox((x1, y1, x2, y2), 0.20, w, h))
            tracker.update(dets, frame, fidx)
        fidx += 1
    cap.release()

    out_dir = VIDEO_DIR / video.stem
    tracks_info = []
    tid_out = 0
    for tid, t in sorted(tracker.tracks.items(),
                         key=lambda kv: -len(kv[1]["crops"])):
        if len(t["crops"]) < min_track_len:
            continue
        tid_out += 1
        label = f"track_{tid_out:02d}"
        tdir = out_dir / label
        tdir.mkdir(parents=True, exist_ok=True)
        best = _best_crops(t["crops"], k=8)
        for i, c in enumerate(best):
            cv2.imwrite(str(tdir / f"crop_{i:02d}.jpg"), c)
        cv2.imwrite(str(out_dir / f"{label}_montaj.png"), _montage(best))
        tracks_info.append({"clip": video.stem, "track": label,
                            "n_frames": len(t["crops"])})
        print(f"   {label}: {len(t['crops'])} apariții -> {len(best)} decupaje în {tdir}")
    return tracks_info


def main() -> None:
    ap = argparse.ArgumentParser(description="Extrage plăcuțele din clipuri video pentru adnotare.")
    ap.add_argument("--stride", type=int, default=2, help="Procesează 1 din N cadre (implicit 2).")
    ap.add_argument("--min-track-len", type=int, default=3,
                    help="Minim apariții ca un traseu să fie reținut (implicit 3).")
    ap.add_argument("--clips-dir", type=str, default=None,
                    help="Folder cu clipurile video (implicit video_eval/clips/).")
    args = ap.parse_args()

    clips_dir = Path(args.clips_dir).expanduser() if args.clips_dir else CLIPS_DIR
    clips_dir.mkdir(parents=True, exist_ok=True)
    clips = sorted(p for p in clips_dir.iterdir() if p.suffix.lower() in VIDEO_EXTS)
    if not clips:
        raise SystemExit(f"Pune clipuri video în {clips_dir} și reia.")
    if not YOLO_WEIGHTS.exists():
        raise SystemExit(f"Lipsesc ponderile YOLO: {YOLO_WEIGHTS}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INIT] Device: {device} | {len(clips)} clipuri")
    detector = YOLO(str(YOLO_WEIGHTS))

    all_tracks = []
    for clip in clips:
        all_tracks += process_video(detector, device, clip, args.stride, args.min_track_len)

    # CSV to fill in: one row per detected track.
    existing = {}
    if GT_CSV.exists():
        for r in csv.DictReader(open(GT_CSV, encoding="utf-8")):
            existing[(r["clip"], r["track"])] = r.get("plate_number", "")
    with open(GT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["clip", "track", "n_frames", "plate_number"])
        for t in all_tracks:
            w.writerow([t["clip"], t["track"], t["n_frames"],
                        existing.get((t["clip"], t["track"]), "")])

    print(f"\n[OK] {len(all_tracks)} trasee de plăcuță extrase.")
    print(f">>> Deschide montajele '{VIDEO_DIR}/<clip>/track_XX_montaj.png', citește numărul")
    print(f"    și completează coloana 'plate_number' în {GT_CSV}")
    print( "    (lasă gol traseele care nu sunt plăcuțe reale sau sunt ilizibile).")
    print( "    Apoi rulează: python plate_04_video_evaluate.py")


if __name__ == "__main__":
    main()
