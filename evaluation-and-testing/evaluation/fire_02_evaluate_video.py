#!/usr/bin/env python3
"""
Fire and smoke detection evaluation on video (FIRESENSE), at the alarm level.

Tests the whole system, including the temporal "voting" -- confirming a detection
only if it persists over several consecutive frames (the most important filter
against false alarms). Compares:

  - RAW: any frame with a YOLO detection triggers the clip's alarm;
  - PIPELINE: alarm only if a detection passes the filters (per-class conf + size
    + color) AND is temporally confirmed (>= N consecutive frames).

FIRESENSE contains positive clips (with fire/smoke) and negative ones (without) --
the negatives measure the false-alarm rate. A clip is an "alarm" (predicted
positive) if the system would have triggered the alert at least once during it.

Run:
    python fire_02_evaluate_video.py
    python fire_02_evaluate_video.py --data-dir <path> --stride 1 --max-frames 1500
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
from typing import Dict, List

import cv2
from tqdm import tqdm

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = Path(os.environ.get("SURVEILLANCE_APP_DIR",
                                   _THIS_DIR.parents[1] / "surveillance-platform"))
sys.path.insert(0, str(_PROJECT_DIR))
from fire_detection_system import FireDetectionSystem  # noqa: E402

DEFAULT_DATA = Path("/mnt/data/documentatie_licenta/pipeline_testing/fire_and_smoke/FIRESENSE_dataset")
RESULTS_DIR = _THIS_DIR / "fire_eval" / "results_video"
MODEL_PATH = _PROJECT_DIR / "models" / "fire_and_smoke" / "best.pt"
VIDEO_EXTS = {".avi", ".mp4", ".mov", ".mkv"}


def _update_history(history, class_name, bbox, matched, frame_num, fs) -> int:
    """Replicates `FireDetectionSystem._update_frame_history` (IoU-based confirmation)."""
    best_key, best_iou = None, 0.0
    for key, entry in history.items():
        if key in matched or not key.startswith(class_name + "_"):
            continue
        i = fs._compute_iou(bbox, entry["bbox"])
        if i > best_iou:
            best_iou, best_key = i, key
    if best_key is not None and best_iou >= fs.iou_match_threshold:
        history[best_key]["count"] += 1
        history[best_key]["last_seen_frame"] = frame_num
        history[best_key]["bbox"] = bbox
        matched.add(best_key)
        return history[best_key]["count"]
    new_key = f"{class_name}_{frame_num}_{id(bbox)}"
    history[new_key] = {"bbox": bbox, "count": 1, "last_seen_frame": frame_num}
    matched.add(new_key)
    return 1


def process_clip(fs, video: Path, stride: int, max_frames: int) -> Dict:
    """Return {'raw': bool, 'pipeline': bool} -- whether the clip would trigger the alarm."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return {"raw": False, "pipeline": False, "ok": False}
    base_conf = min(fs.conf_thresholds.values())
    history: Dict = {}
    frame_num = 0
    processed = 0
    raw_alarm = pipe_alarm = False
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_num % stride == 0:
            processed += 1
            h, w = frame.shape[:2]
            farea = h * w
            res = fs.model.predict(frame, conf=base_conf, iou=fs.iou_threshold, verbose=False)[0]
            boxes = res.boxes if res.boxes is not None else []
            if len(boxes) > 0:
                raw_alarm = True
            matched = set()
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1, y1 = max(0, x1), max(0, y1); x2, y2 = min(w, x2), min(h, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                cls = int(box.cls[0]); conf = float(box.conf[0])
                name = fs.class_names.get(cls, f"unknown_{cls}")
                if conf < fs.conf_thresholds.get(name, fs.default_conf):
                    continue
                ar = ((x2 - x1) * (y2 - y1)) / farea
                if ar < fs.min_area_ratio or ar > fs.max_area_ratio:
                    continue
                if not fs._color_plausible(frame[y1:y2, x1:x2], name):
                    continue
                streak = _update_history(history, name, (x1, y1, x2, y2), matched, processed, fs)
                if streak >= fs.min_consecutive_frames:
                    pipe_alarm = True
            for k in [k for k, v in history.items() if v["last_seen_frame"] < processed]:
                del history[k]
            if max_frames and processed >= max_frames:
                break
        frame_num += 1
    cap.release()
    return {"raw": raw_alarm, "pipeline": pipe_alarm, "ok": True, "frames": processed}


def _collect(data_dir: Path) -> List[Dict]:
    clips = []
    for cat_dir in sorted(data_dir.iterdir()):
        if not cat_dir.is_dir():
            continue
        cat = "foc" if "fire" in cat_dir.name.lower() else (
              "fum" if "smoke" in cat_dir.name.lower() else cat_dir.name)
        for sub, gt in (("pos", True), ("neg", False)):
            d = cat_dir / sub
            if not d.is_dir():
                continue
            for v in sorted(d.iterdir()):
                if v.suffix.lower() in VIDEO_EXTS:
                    clips.append({"path": v, "cat": cat, "gt": gt})
    return clips


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluarea detecției de foc/fum pe video (FIRESENSE).")
    ap.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA))
    ap.add_argument("--stride", type=int, default=1, help="Procesează 1 din N cadre (implicit 1).")
    ap.add_argument("--max-frames", type=int, default=1500, help="Plafon de cadre procesate / clip.")
    args = ap.parse_args()

    clips = _collect(Path(args.data_dir))
    n_pos = sum(1 for c in clips if c["gt"]); n_neg = len(clips) - n_pos
    print(f"[INIT] {len(clips)} clipuri ({n_pos} pozitive, {n_neg} negative)")
    fs = FireDetectionSystem(model_path=str(MODEL_PATH))

    rows = []
    for c in tqdm(clips, desc="clipuri"):
        r = process_clip(fs, c["path"], args.stride, args.max_frames)
        rows.append({**c, **r})
    _report(rows)


def _rate(rows, pred_key, gt_val) -> str:
    sub = [r for r in rows if r["gt"] == gt_val and r.get("ok")]
    if not sub:
        return "0/0"
    hit = sum(1 for r in sub if r[pred_key])
    return f"{hit}/{len(sub)} ({hit/len(sub):.0%})"


def _report(rows: List[Dict]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("\n==================== DETECȚIE FOC/FUM PE VIDEO ====================")
    print("Detecție pe clipuri POZITIVE (cât de des alarmează corect):")
    h = f"{'Variantă':<24}{'Toate':>12}{'Foc':>12}{'Fum':>12}"
    print(h); print("-" * len(h))
    for key, lab in (("raw", "Brut (orice cadru)"), ("pipeline", "Pipeline + votare")):
        allp = _rate(rows, key, True)
        focp = _rate([r for r in rows if r["cat"] == "foc"], key, True)
        fump = _rate([r for r in rows if r["cat"] == "fum"], key, True)
        print(f"{lab:<24}{allp:>12}{focp:>12}{fump:>12}")

    print("\nAlarme FALSE pe clipuri NEGATIVE (mai mic = mai bine):")
    print(h); print("-" * len(h))
    for key, lab in (("raw", "Brut (orice cadru)"), ("pipeline", "Pipeline + votare")):
        alln = _rate(rows, key, False)
        focn = _rate([r for r in rows if r["cat"] == "foc"], key, False)
        fumn = _rate([r for r in rows if r["cat"] == "fum"], key, False)
        print(f"{lab:<24}{alln:>12}{focn:>12}{fumn:>12}")

    print("\n=== detaliu per clip ===")
    for r in sorted(rows, key=lambda r: (r["cat"], not r["gt"], r["path"].name)):
        g = "POZ" if r["gt"] else "NEG"
        print(f"  {r['cat']:<4} {g} {r['path'].name:<24} brut={'DA' if r['raw'] else 'nu':<3} "
              f"votare={'DA' if r['pipeline'] else 'nu'}")

    with open(RESULTS_DIR / "fire_video_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["clip", "categorie", "gt_pozitiv", "raw_alarma", "pipeline_alarma"])
        for r in rows:
            w.writerow([r["path"].name, r["cat"], int(r["gt"]), int(r["raw"]), int(r["pipeline"])])
    _plot(rows)
    print(f"\n[CSV/PLOT] {RESULTS_DIR}")


def _plot(rows: List[Dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    def pct(key, gt):
        sub = [r for r in rows if r["gt"] == gt and r.get("ok")]
        return 100 * sum(1 for r in sub if r[key]) / len(sub) if sub else 0

    labels = ["Detecție\n(clipuri pozitive)", "Alarmă falsă\n(clipuri negative)"]
    raw_v = [pct("raw", True), pct("raw", False)]
    pipe_v = [pct("pipeline", True), pct("pipeline", False)]
    x = np.arange(2); width = 0.38
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    b1 = ax.bar(x - width / 2, raw_v, width, label="Brut (orice cadru)", color="#d62728")
    b2 = ax.bar(x + width / 2, pipe_v, width, label="Pipeline + votare temporală", color="#1f77b4")
    ax.set_ylabel("Procent din clipuri (%)"); ax.set_ylim(0, 105)
    ax.set_title("Detecția de foc/fum pe video: efectul votării temporale")
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.grid(True, axis="y", alpha=0.3); ax.legend()
    for bars in (b1, b2):
        for bar in bars:
            ax.annotate(f"{bar.get_height():.0f}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "fire_video_comparatie.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
