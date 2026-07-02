#!/usr/bin/env python3
"""
Step 4 (video test) — evaluate the FULL system, with temporal voting.

Run AFTER you fill video_ground_truth.csv (each plate's number, per track, produced
by plate_03_video_extract.py). For each clip:

  - run detection + pipeline reading on every frame;
  - group readings of the same plate into tracks and apply temporal VOTING
    (confidence-weighted majority vote, filtered on the RO plate format) --
    exactly the mechanism from `PlateTracker`;
  - compare the voted result with the real number AND with the best single-frame
    reading, to show how much the system gains from multi-frame processing.

Run:
    python plate_04_video_evaluate.py
"""

import os
import sys

_CUDNN8 = os.path.expanduser("~/.local/cudnn8_pkg/nvidia/cudnn/lib")
if os.path.isdir(_CUDNN8) and _CUDNN8 not in os.environ.get("LD_LIBRARY_PATH", ""):
    os.environ["LD_LIBRARY_PATH"] = _CUDNN8 + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    os.execv(sys.executable, [sys.executable] + sys.argv)

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = Path(os.environ.get("SURVEILLANCE_APP_DIR",
                                   _THIS_DIR.parents[1] / "surveillance-platform"))
sys.path.insert(0, str(_PROJECT_DIR))
from license_plate_recognition_system import expand_bbox, _iou, ROMANIAN_PLATE_RE  # noqa: E402
from ultralytics import YOLO  # noqa: E402
import easyocr  # noqa: E402
# Reuse the pipeline reading and metrics from the image script (the shim above
# already set LD_LIBRARY_PATH, so the import does not re-execute the process).
from plate_02_evaluate import read_pipeline, normalize, cer  # noqa: E402

VIDEO_DIR = _THIS_DIR / "video_eval"
CLIPS_DIR = VIDEO_DIR / "clips"
GT_CSV = VIDEO_DIR / "video_ground_truth.csv"
RESULTS_DIR = VIDEO_DIR / "results"
YOLO_WEIGHTS = _PROJECT_DIR / "models" / "license_plate" / "best.pt"
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def _vote(readings: List[Tuple[str, float]], min_votes: int) -> Tuple[str, float]:
    """Confidence-weighted majority vote over readings in a valid RO format --
    mirrors the logic in `PlateTracker.update`."""
    valid = [(t, c) for t, c in readings if c >= 0.4 and ROMANIAN_PLATE_RE.match(t)]
    if len(valid) < min_votes:
        return "", 0.0
    scores: Counter = Counter()
    for t, c in valid:
        scores[t] += c
    best_text, total = scores.most_common(1)[0]
    votes = sum(1 for t, _ in valid if t == best_text)
    if votes < min_votes:
        return "", 0.0
    return best_text, total / votes


def process_clip(detector, reader, device: str, video: Path, stride: int,
                 iou_thr: float = 0.3, max_age: int = 20) -> List[Dict]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print(f"[WARN] Nu pot deschide {video.name}")
        return []
    tracks: Dict[int, Dict] = {}
    finished: List[Dict] = []
    nxt = 0
    fidx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if fidx % stride == 0:
            h, w = frame.shape[:2]
            for t in tracks.values():
                t["age"] += 1
            boxes = detector.predict(frame, imgsz=640, conf=0.25, device=device,
                                     verbose=False)[0].boxes
            for b in boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                box = expand_bbox((x1, y1, x2, y2), 0.20, w, h)
                best_id, best = None, iou_thr
                for tid, t in tracks.items():
                    i = _iou(box, t["bbox"])
                    if i > best:
                        best_id, best = tid, i
                if best_id is None:
                    best_id = nxt; nxt += 1
                    tracks[best_id] = {"bbox": box, "age": 0, "readings": []}
                t = tracks[best_id]
                t["bbox"] = box; t["age"] = 0
                text, conf = read_pipeline(reader, frame, box)
                if text:
                    t["readings"].append((text, conf))
            for tid in [tid for tid, t in tracks.items() if t["age"] > max_age]:
                finished.append(tracks.pop(tid))   # retired, but kept for the report
        fidx += 1
    cap.release()

    # Label the tracks as in plate_03 (by number of appearances, descending).
    real = finished + list(tracks.values())
    real.sort(key=lambda t: -len(t["readings"]))
    out = []
    for i, t in enumerate(real, 1):
        if not t["readings"]:
            continue
        voted, vconf = _vote(t["readings"], min_votes=3)
        best_text, best_conf = max(t["readings"], key=lambda r: r[1])
        out.append({"clip": video.stem, "track": f"track_{i:02d}",
                    "n_read": len(t["readings"]),
                    "voted": normalize(voted), "single": normalize(best_text)})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluează recunoașterea plăcuțelor pe video, cu votare temporală.")
    ap.add_argument("--stride", type=int, default=2, help="Procesează 1 din N cadre (implicit 2).")
    ap.add_argument("--clips-dir", type=str, default=None,
                    help="Folder cu clipurile video (același ca la plate_03; implicit video_eval/clips/).")
    args = ap.parse_args()

    if not GT_CSV.exists():
        raise SystemExit(f"Lipsește {GT_CSV} — rulează întâi plate_03_video_extract.py și adnotează.")
    gt = {}
    for r in csv.DictReader(open(GT_CSV, encoding="utf-8")):
        num = (r.get("plate_number") or "").strip()
        if num:
            gt[(r["clip"], r["track"])] = normalize(num)
    if not gt:
        raise SystemExit("Niciun număr completat în video_ground_truth.csv.")

    clips_dir = Path(args.clips_dir).expanduser() if args.clips_dir else CLIPS_DIR
    clips = sorted(p for p in clips_dir.iterdir() if p.suffix.lower() in VIDEO_EXTS)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INIT] Device: {device} | {len(clips)} clipuri | {len(gt)} trasee adnotate")
    detector = YOLO(str(YOLO_WEIGHTS))
    reader = easyocr.Reader(["en"], gpu=(device == "cuda"), verbose=False)

    rows = []
    for clip in clips:
        for t in process_clip(detector, reader, device, clip, args.stride):
            key = (t["clip"], t["track"])
            if key not in gt:
                continue                      # unannotated track (non-plate/illegible)
            t["gt"] = gt[key]
            rows.append(t)

    _report(rows)


def _report(rows: List[Dict]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    n = len(rows)
    if n == 0:
        raise SystemExit("Niciun traseu adnotat nu s-a potrivit cu rezultatele. "
                         "Verifică numele clipurilor și traseele.")

    vote_exact = sum(1 for r in rows if r["voted"] == r["gt"])
    single_exact = sum(1 for r in rows if r["single"] == r["gt"])
    vote_cer = sum(cer(r["voted"], r["gt"]) for r in rows) / n
    single_cer = sum(cer(r["single"], r["gt"]) for r in rows) / n

    print(f"\n==================== EVALUARE PE VIDEO ({n} plăcuțe) ====================")
    header = f"{'Metodă':<32}{'Exact':>10}{'CER mediu':>12}"
    print(header); print("-" * len(header))
    print(f"{'Un singur cadru (fără votare)':<32}{single_exact}/{n} ({single_exact/n:>4.0%}){single_cer:>11.3f}")
    print(f"{'Sistem complet (cu votare)':<32}{vote_exact}/{n} ({vote_exact/n:>4.0%}){vote_cer:>11.3f}")

    print("\n=== detaliu per plăcuță ===")
    print(f"{'clip/track':<26}{'real':<10}{'1 cadru':<10}{'votat':<10}")
    for r in rows:
        flag = "✓" if r["voted"] == r["gt"] else "✗"
        print(f"{r['clip'][:14]+'/'+r['track']:<26}{r['gt']:<10}{r['single']:<10}{r['voted']:<10}{flag}")

    with open(RESULTS_DIR / "video_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["clip", "track", "n_read", "gt", "single_frame", "voted",
                    "single_exact", "voted_exact"])
        for r in rows:
            w.writerow([r["clip"], r["track"], r["n_read"], r["gt"], r["single"], r["voted"],
                        int(r["single"] == r["gt"]), int(r["voted"] == r["gt"])])
    _plot(single_exact / n, vote_exact / n, single_cer, vote_cer)
    print(f"\n[CSV] {RESULTS_DIR / 'video_results.csv'}")


def _plot(single_ex, vote_ex, single_cer, vote_cer) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = ["Un singur cadru", "Cu votare temporală"]
    exact = [single_ex * 100, vote_ex * 100]
    chacc = [max(0, 1 - single_cer) * 100, max(0, 1 - vote_cer) * 100]
    x = np.arange(2); width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4.6))
    b1 = ax.bar(x - width / 2, exact, width, label="Potrivire exactă", color="#1f77b4")
    b2 = ax.bar(x + width / 2, chacc, width, label="Acuratețe la nivel de caracter", color="#2ca02c")
    ax.set_ylabel("Procent (%)"); ax.set_ylim(0, 105)
    ax.set_title("Recunoașterea plăcuțelor pe video: efectul votării temporale")
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.grid(True, axis="y", alpha=0.3); ax.legend()
    for bars in (b1, b2):
        for bar in bars:
            ax.annotate(f"{bar.get_height():.0f}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)
    fig.tight_layout()
    out = RESULTS_DIR / "video_votare.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"[PLOT] {out}")


if __name__ == "__main__":
    main()
