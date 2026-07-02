#!/usr/bin/env python3
"""
Fire and smoke detection evaluation on images (D-Fire), at the alarm level.

Compares RAW DETECTION (YOLO model only) with the false-positive reduction
PIPELINE from `FireDetectionSystem` (per-class confidence thresholds + size filter
+ HSV color check). On static images the temporal confirmation is not applied
(tested separately on video).

The D-Fire set contains both positive images (with annotated fire/smoke) and
negative ones (nothing) -- the latter let us measure the false-positive rate.

An image is considered an "alarm" (positive) if the system produces at least one
fire or smoke detection. Ground truth: positive = the annotation file has at least
one box; negative = empty file.

Run:
    python fire_01_evaluate_images.py            # full test set
    python fire_01_evaluate_images.py --limit 600  # quick balanced sample
"""

import os
import sys

_CUDNN8 = os.path.expanduser("~/.local/cudnn8_pkg/nvidia/cudnn/lib")
if os.path.isdir(_CUDNN8) and _CUDNN8 not in os.environ.get("LD_LIBRARY_PATH", ""):
    os.environ["LD_LIBRARY_PATH"] = _CUDNN8 + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    os.execv(sys.executable, [sys.executable] + sys.argv)

import argparse
import csv
import random
from pathlib import Path
from typing import Dict, List

import cv2
from tqdm import tqdm

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = Path(os.environ.get("SURVEILLANCE_APP_DIR",
                                   _THIS_DIR.parents[1] / "surveillance-platform"))
sys.path.insert(0, str(_PROJECT_DIR))
from fire_detection_system import FireDetectionSystem  # noqa: E402

DEFAULT_DATA = Path("/mnt/data/documentatie_licenta/pipeline_testing/fire_and_smoke/D-Fire/archive/test")
RESULTS_DIR = _THIS_DIR / "fire_eval" / "results_images"
MODEL_PATH = _PROJECT_DIR / "models" / "fire_and_smoke" / "best.pt"


def _passes_pipeline(fs: FireDetectionSystem, boxes, frame, frame_area: float) -> bool:
    """True if any box passes the pipeline filters (per-class conf + size + color)."""
    h, w = frame.shape[:2]
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        cls = int(box.cls[0]); conf = float(box.conf[0])
        name = fs.class_names.get(cls, f"unknown_{cls}")
        if conf < fs.conf_thresholds.get(name, fs.default_conf):       # filter 1
            continue
        area_ratio = ((x2 - x1) * (y2 - y1)) / frame_area
        if area_ratio < fs.min_area_ratio or area_ratio > fs.max_area_ratio:  # filter 2
            continue
        if not fs._color_plausible(frame[y1:y2, x1:x2], name):         # filter 3
            continue
        return True
    return False


def _reject_stage(fs: FireDetectionSystem, boxes, frame, frame_area: float):
    """Return None if the pipeline accepts; otherwise the stage that blocked (diagnostic)."""
    h, w = frame.shape[:2]
    reached_conf = reached_size = False
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        x1, y1 = max(0, x1), max(0, y1); x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        cls = int(box.cls[0]); conf = float(box.conf[0])
        name = fs.class_names.get(cls, f"unknown_{cls}")
        if conf < fs.conf_thresholds.get(name, fs.default_conf):
            continue
        reached_conf = True
        ar = ((x2 - x1) * (y2 - y1)) / frame_area
        if ar < fs.min_area_ratio or ar > fs.max_area_ratio:
            continue
        reached_size = True
        if not fs._color_plausible(frame[y1:y2, x1:x2], name):
            continue
        return None
    if not reached_conf:
        return "prag_conf"
    if not reached_size:
        return "dimensiune"
    return "culoare"


def _gather(data_dir: Path, limit: int, seed: int):
    labels = sorted((data_dir / "labels").glob("*.txt"))
    items = []
    for lab in labels:
        img = data_dir / "images" / (lab.stem + ".jpg")
        if not img.exists():
            continue
        gt_pos = bool(open(lab).read().strip())
        items.append((img, gt_pos))
    if limit and limit < len(items):
        pos = [it for it in items if it[1]]
        neg = [it for it in items if not it[1]]
        rng = random.Random(seed)
        rng.shuffle(pos); rng.shuffle(neg)
        k = limit // 2
        items = pos[:k] + neg[:k]
        rng.shuffle(items)
    return items


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluarea detecției de foc/fum pe imagini (D-Fire).")
    ap.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA))
    ap.add_argument("--limit", type=int, default=0, help="0 = tot setul; altfel eșantion echilibrat.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    items = _gather(data_dir, args.limit, args.seed)
    n_pos = sum(1 for _, p in items if p); n_neg = len(items) - n_pos
    print(f"[INIT] {len(items)} imagini ({n_pos} pozitive, {n_neg} negative)")

    fs = FireDetectionSystem(model_path=str(MODEL_PATH))
    base_conf = min(fs.conf_thresholds.values())

    # Confusion counters for each variant.
    raw = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    pipe = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    from collections import Counter
    lost_reason = Counter()   # why the pipeline loses positives the raw model catches
    rows = []
    for img_path, gt_pos in tqdm(items, desc="evaluez"):
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        res = fs.model.predict(frame, conf=base_conf, iou=fs.iou_threshold, verbose=False)[0]
        boxes = res.boxes if res.boxes is not None else []
        raw_pos = len(boxes) > 0                                   # raw: any detection at base conf
        pipe_pos = _passes_pipeline(fs, boxes, frame, frame.shape[0] * frame.shape[1])
        if gt_pos and raw_pos and not pipe_pos:
            lost_reason[_reject_stage(fs, boxes, frame, frame.shape[0] * frame.shape[1])] += 1

        for d, pred in ((raw, raw_pos), (pipe, pipe_pos)):
            if gt_pos and pred: d["tp"] += 1
            elif gt_pos and not pred: d["fn"] += 1
            elif not gt_pos and pred: d["fp"] += 1
            else: d["tn"] += 1
        rows.append({"image": img_path.name, "gt": int(gt_pos),
                     "raw": int(raw_pos), "pipeline": int(pipe_pos)})

    _report(raw, pipe, rows, n_pos, n_neg)
    if lost_reason:
        total = sum(lost_reason.values())
        print(f"\n=== De ce pierde pipeline-ul {total} pozitive prinse de brut ===")
        for reason, cnt in lost_reason.most_common():
            print(f"   blocat la filtrul '{reason}': {cnt} ({cnt/total:.0%})")


def _metrics(c: Dict) -> Dict:
    tp, fp, tn, fn = c["tp"], c["fp"], c["tn"], c["fn"]
    tpr = tp / (tp + fn) if tp + fn else 0.0          # positive detection (recall)
    fpr = fp / (fp + tn) if fp + tn else 0.0          # false-positive rate
    prec = tp / (tp + fp) if tp + fp else 0.0
    acc = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) else 0.0
    f1 = 2 * prec * tpr / (prec + tpr) if prec + tpr else 0.0
    return {"tpr": tpr, "fpr": fpr, "prec": prec, "acc": acc, "f1": f1}


def _report(raw, pipe, rows, n_pos, n_neg) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    mr, mp = _metrics(raw), _metrics(pipe)

    print(f"\n==================== DETECȚIE FOC/FUM PE IMAGINI ====================")
    print(f"{n_pos} pozitive, {n_neg} negative\n")
    header = f"{'Variantă':<26}{'Detecție poz.':>14}{'Fals pozitiv':>14}{'Precizie':>10}{'F1':>8}{'Acuratețe':>11}"
    print(header); print("-" * len(header))
    for label, m, c in (("Brut (doar YOLO)", mr, raw), ("Pipeline (cu filtre)", mp, pipe)):
        print(f"{label:<26}{m['tpr']:>13.1%}{m['fpr']:>14.1%}{m['prec']:>9.1%}{m['f1']:>8.2f}{m['acc']:>10.1%}")
    print(f"\nConfuzie brut    : TP={raw['tp']} FP={raw['fp']} TN={raw['tn']} FN={raw['fn']}")
    print(f"Confuzie pipeline: TP={pipe['tp']} FP={pipe['fp']} TN={pipe['tn']} FN={pipe['fn']}")
    print(f"\n>>> Pipeline-ul reduce falsele pozitive de la {mr['fpr']:.1%} la {mp['fpr']:.1%} "
          f"(detecția pozitivelor: {mr['tpr']:.1%} -> {mp['tpr']:.1%}).")

    with open(RESULTS_DIR / "fire_images_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["image", "gt", "raw", "pipeline"])
        for r in rows: w.writerow([r["image"], r["gt"], r["raw"], r["pipeline"]])
    with open(RESULTS_DIR / "fire_images_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["varianta", "tpr", "fpr", "precizie", "f1", "acuratete"])
        for label, m in (("brut", mr), ("pipeline", mp)):
            w.writerow([label, f"{m['tpr']:.4f}", f"{m['fpr']:.4f}", f"{m['prec']:.4f}",
                        f"{m['f1']:.4f}", f"{m['acc']:.4f}"])
    _plot(mr, mp)
    print(f"[CSV/PLOT] {RESULTS_DIR}")


def _plot(mr, mp) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    labels = ["Detecție\npozitive", "Fals\npozitiv", "Precizie", "Acuratețe"]
    raw_v = [mr["tpr"] * 100, mr["fpr"] * 100, mr["prec"] * 100, mr["acc"] * 100]
    pipe_v = [mp["tpr"] * 100, mp["fpr"] * 100, mp["prec"] * 100, mp["acc"] * 100]
    x = np.arange(len(labels)); width = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.8))
    b1 = ax.bar(x - width / 2, raw_v, width, label="Brut (doar YOLO)", color="#d62728")
    b2 = ax.bar(x + width / 2, pipe_v, width, label="Pipeline (cu filtre)", color="#1f77b4")
    ax.set_ylabel("Procent (%)"); ax.set_ylim(0, 105)
    ax.set_title("Detecția de foc/fum pe imagini: brut vs. pipeline")
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.grid(True, axis="y", alpha=0.3); ax.legend()
    for bars in (b1, b2):
        for bar in bars:
            ax.annotate(f"{bar.get_height():.0f}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "fire_images_comparatie.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
