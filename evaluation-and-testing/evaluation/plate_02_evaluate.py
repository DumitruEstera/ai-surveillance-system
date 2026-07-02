#!/usr/bin/env python3
"""
Step 2 — license plate recognition evaluation.

Run AFTER you manually fill the `plate_number` column in
plate_eval/ground_truth.csv. It evaluates:

  A. YOLO DETECTION -- compares the predicted boxes with the real coordinates from
     the dataset (manifest.json), reporting detection rate (IoU >= 0.5) and mean IoU.

  B. OCR READING -- compares three variants, to see whether your processing
     pipeline helps over a raw EasyOCR read:
       1. PIPELINE      = exactly the chain from the real system (expand 20%% -> upscale 250px
                          -> grayscale/bilateral/CLAHE/unsharp -> EasyOCR with allowlist
                          and tuned params + color-crop fallback);
       2. RAW on crop   = EasyOCR on the YOLO-detected box, no processing, default
                          params (isolates the pipeline's contribution);
       3. RAW on image  = EasyOCR on the whole image, no YOLO (brute force).

  OCR metrics: exact match, CER (character error rate) and the percentage of reads
  in a valid Romanian format.

Run:
    python plate_02_evaluate.py
"""

# --------------------------------------------------------------------------- #
#  Environment fix: torch (cu121) needs libcudnn.so.8, but the venv has cuDNN 9.#
#  Prepend the local cuDNN 8 package to LD_LIBRARY_PATH BEFORE importing torch, #
#  re-executing the process once if needed.                                     #
# --------------------------------------------------------------------------- #
import os
import sys

_CUDNN8 = os.path.expanduser("~/.local/cudnn8_pkg/nvidia/cudnn/lib")
if os.path.isdir(_CUDNN8) and _CUDNN8 not in os.environ.get("LD_LIBRARY_PATH", ""):
    os.environ["LD_LIBRARY_PATH"] = _CUDNN8 + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    os.execv(sys.executable, [sys.executable] + sys.argv)

import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = Path(os.environ.get("SURVEILLANCE_APP_DIR",
                                   _THIS_DIR.parents[1] / "surveillance-platform"))
sys.path.insert(0, str(_PROJECT_DIR))

# Reuse the exact processing functions of the real system.
from license_plate_recognition_system import (  # noqa: E402
    expand_bbox, upscale_if_small, preprocess_plate, ROMANIAN_PLATE_RE, extract_ro_plate,
)
from ultralytics import YOLO  # noqa: E402
import easyocr  # noqa: E402

PLATE_DIR = _THIS_DIR / "plate_eval"
IMAGES_DIR = PLATE_DIR / "images"
MANIFEST_PATH = PLATE_DIR / "manifest.json"
GT_CSV = PLATE_DIR / "ground_truth.csv"
RESULTS_DIR = PLATE_DIR / "results"

YOLO_WEIGHTS = _PROJECT_DIR / "models" / "license_plate" / "best.pt"
OCR_ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


# --------------------------------------------------------------------------- #
#  Text utilities                                                              #
# --------------------------------------------------------------------------- #
def normalize(text: str) -> str:
    """Uppercase, alphanumeric characters only (apples-to-apples comparison)."""
    return re.sub(r"[^A-Z0-9]", "", (text or "").upper())


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(pred: str, gt: str) -> float:
    """Character Error Rate = edit distance / reference length."""
    gt = normalize(gt)
    pred = normalize(pred)
    if not gt:
        return 0.0 if not pred else 1.0
    return levenshtein(pred, gt) / len(gt)


def iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


# --------------------------------------------------------------------------- #
#  OCR                                                                          #
# --------------------------------------------------------------------------- #
def _assemble(results: List) -> Tuple[str, float]:
    """Assemble the EasyOCR detections left->right, keeping only alphanumerics."""
    if not results:
        return "", 0.0
    results = sorted(results, key=lambda r: r[0][0][0])
    parts, confs = [], []
    for (_bbox, text, conf) in results:
        cleaned = "".join(c for c in text.upper() if c.isalnum())
        if cleaned:
            parts.append(cleaned)
            confs.append(conf)
    if not parts:
        return "", 0.0
    return "".join(parts), float(np.mean(confs))


def ocr_pipeline_call(reader, img) -> Tuple[str, float]:
    """OCR call with allowlist and tuned params -- mirrors `_run_ocr`,
    including removing the 'RO' EU band from the text."""
    try:
        results = reader.readtext(img, allowlist=OCR_ALLOWLIST, detail=1,
                                  paragraph=False, text_threshold=0.6,
                                  low_text=0.3, mag_ratio=1.5)
    except Exception as e:
        print(f"OCR error: {e}")
        return "", 0.0
    text, conf = _assemble(results)
    return extract_ro_plate(text), conf


def ocr_raw_call(reader, img) -> Tuple[str, float]:
    """Raw EasyOCR call -- default params, no allowlist or preprocessing."""
    try:
        results = reader.readtext(img)
    except Exception as e:
        print(f"OCR error: {e}")
        return "", 0.0
    return _assemble(results)


def read_pipeline(reader, frame, box) -> Tuple[str, float]:
    """The full chain from `process_frame` (without temporal voting, irrelevant here)."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = expand_bbox(box, 0.20, w, h)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return "", 0.0
    crop = upscale_if_small(crop, target_width=250)
    pre = preprocess_plate(crop)
    text, conf = ocr_pipeline_call(reader, pre)
    if not text or conf < 0.4:
        alt_text, alt_conf = ocr_pipeline_call(reader, crop)
        if alt_conf > conf:
            text, conf = alt_text, alt_conf
    return text, conf


def read_raw_crop(reader, frame, box) -> Tuple[str, float]:
    """Raw EasyOCR on the detected box, no processing."""
    x1, y1, x2, y2 = box
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return "", 0.0
    return ocr_raw_call(reader, crop)


# --------------------------------------------------------------------------- #
#  Evaluation                                                                  #
# --------------------------------------------------------------------------- #
def load_ground_truth() -> Dict[str, str]:
    if not GT_CSV.exists():
        raise SystemExit(f"Lipsește {GT_CSV} — rulează mai întâi plate_01_download_select.py.")
    gt = {}
    with open(GT_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            num = (row.get("plate_number") or "").strip()
            if num:
                gt[row["image"].strip()] = num
    if not gt:
        raise SystemExit("Nicio plăcuță completată în ground_truth.csv. Completează coloana "
                         "plate_number și reia.")
    return gt


def main() -> None:
    gt = load_ground_truth()
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        manifest = {it["plate_id"] + ".jpg": it for it in json.load(f)["items"]}

    use_gpu = torch.cuda.is_available()
    device = "cuda" if use_gpu else "cpu"
    print(f"[INIT] Device: {device}")
    if not YOLO_WEIGHTS.exists():
        raise SystemExit(f"Lipsesc ponderile YOLO: {YOLO_WEIGHTS}")
    detector = YOLO(str(YOLO_WEIGHTS))
    reader = easyocr.Reader(["en"], gpu=use_gpu, verbose=False)
    print(f"[INIT] Evaluez {len(gt)} imagini adnotate.\n")

    records: List[Dict] = []
    for image_name, true_plate in tqdm(sorted(gt.items()), desc="evaluez"):
        img_path = IMAGES_DIR / image_name
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"[WARN] Nu pot citi {img_path}")
            continue

        gt_box = tuple(manifest[image_name]["gt_bbox"]) if image_name in manifest else None

        # --- YOLO detection ---
        boxes = detector.predict(frame, imgsz=640, conf=0.25, device=device,
                                 verbose=False)[0].boxes
        det_boxes = [tuple(map(int, b.xyxy[0])) for b in boxes]
        best_iou, best_box = 0.0, None
        if gt_box is not None:
            for b in det_boxes:
                i = iou(b, gt_box)
                if i > best_iou:
                    best_iou, best_box = i, b
        if best_box is None and det_boxes:           # no reference or IoU 0: take the first
            best_box = det_boxes[0]
        det_ok = best_iou >= 0.5

        # --- Three OCR reads ---
        if best_box is not None:
            pipe_txt, pipe_conf = read_pipeline(reader, frame, best_box)
            rawc_txt, rawc_conf = read_raw_crop(reader, frame, best_box)
        else:
            pipe_txt, pipe_conf, rawc_txt, rawc_conf = "", 0.0, "", 0.0
        rawf_txt, rawf_conf = ocr_raw_call(reader, frame)   # raw on the whole image

        gt_norm = normalize(true_plate)
        records.append({
            "image": image_name, "gt": gt_norm,
            "n_det": len(det_boxes), "best_iou": best_iou, "det_ok": det_ok,
            "pipe": normalize(pipe_txt), "pipe_conf": pipe_conf,
            "rawc": normalize(rawc_txt), "rawc_conf": rawc_conf,
            "rawf": normalize(rawf_txt), "rawf_conf": rawf_conf,
        })

    _report(records)


def _variant_stats(records, key, subset=None) -> Dict:
    rows = records if subset is None else [r for r in records if subset(r)]
    n = len(rows)
    if n == 0:
        return {"n": 0, "exact": 0.0, "cer": 0.0, "fmt": 0.0}
    exact = sum(1 for r in rows if r[key] == r["gt"]) / n
    mean_cer = sum(cer(r[key], r["gt"]) for r in rows) / n
    fmt = sum(1 for r in rows if ROMANIAN_PLATE_RE.match(r[key])) / n
    return {"n": n, "exact": exact, "cer": mean_cer, "fmt": fmt}


def _report(records: List[Dict]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    n = len(records)

    # --- A. Detection ---
    det_rate = sum(1 for r in records if r["det_ok"]) / n if n else 0.0
    mean_iou = sum(r["best_iou"] for r in records) / n if n else 0.0
    print("\n==================== A. DETECȚIA YOLO ====================")
    print(f"Imagini evaluate           : {n}")
    print(f"Rată de detecție (IoU>=0.5): {det_rate:.1%}")
    print(f"IoU mediu                  : {mean_iou:.3f}")

    # --- B. OCR: comparison ---
    variants = [("pipe", "PIPELINE (procesare completă)"),
                ("rawc", "BRUT pe crop (EasyOCR implicit)"),
                ("rawf", "BRUT pe imagine (fără YOLO)")]
    print("\n==================== B. COMPARAȚIE OCR (toate imaginile) ====================")
    header = f"{'Variantă':<34}{'Exact':>8}{'CER mediu':>12}{'Format RO':>12}"
    print(header); print("-" * len(header))
    all_stats = {}
    for key, label in variants:
        s = _variant_stats(records, key)
        all_stats[key] = s
        print(f"{label:<34}{s['exact']:>7.1%}{s['cer']:>12.3f}{s['fmt']:>11.1%}")

    # Subset: only images where YOLO detected the plate (isolates preprocessing)
    det_subset = lambda r: r["det_ok"]
    n_det = sum(1 for r in records if r["det_ok"])
    print(f"\n--- Doar imaginile cu plăcuță detectată corect ({n_det}/{n}) ---")
    print(header); print("-" * len(header))
    for key, label in variants[:2]:  # pipeline vs raw-on-crop (only meaningful with detection)
        s = _variant_stats(records, key, subset=det_subset)
        print(f"{label:<34}{s['exact']:>7.1%}{s['cer']:>12.3f}{s['fmt']:>11.1%}")

    _save_csv(records)
    _plot(all_stats, variants)

    # Short, automatic conclusion
    best = max(("pipe", "rawc", "rawf"), key=lambda k: all_stats[k]["exact"])
    label = dict(variants)[best]
    print(f"\n>>> Cea mai bună potrivire exactă: {label} ({all_stats[best]['exact']:.1%}).")
    print(f"    Rezultate detaliate: {RESULTS_DIR}")


def _save_csv(records: List[Dict]) -> None:
    with open(RESULTS_DIR / "plate_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "gt", "n_det", "best_iou", "det_ok",
                    "pipeline", "pipeline_exact", "pipeline_cer", "pipeline_conf",
                    "raw_crop", "rawcrop_exact", "rawcrop_cer", "raw_crop_conf",
                    "raw_full", "rawfull_exact", "rawfull_cer", "raw_full_conf"])
        for r in records:
            w.writerow([
                r["image"], r["gt"], r["n_det"], f"{r['best_iou']:.3f}", int(r["det_ok"]),
                r["pipe"], int(r["pipe"] == r["gt"]), f"{cer(r['pipe'], r['gt']):.3f}", f"{r['pipe_conf']:.3f}",
                r["rawc"], int(r["rawc"] == r["gt"]), f"{cer(r['rawc'], r['gt']):.3f}", f"{r['rawc_conf']:.3f}",
                r["rawf"], int(r["rawf"] == r["gt"]), f"{cer(r['rawf'], r['gt']):.3f}", f"{r['rawf_conf']:.3f}",
            ])
    print(f"\n[CSV] Rezultate per imagine: {RESULTS_DIR / 'plate_results.csv'}")


def _plot(stats: Dict, variants) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["Pipeline", "Brut pe crop", "Brut pe imagine"]
    keys = [v[0] for v in variants]
    exact = [stats[k]["exact"] * 100 for k in keys]
    char_acc = [max(0.0, 1 - stats[k]["cer"]) * 100 for k in keys]
    fmt = [stats[k]["fmt"] * 100 for k in keys]

    x = np.arange(len(labels))
    width = 0.27
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    b1 = ax.bar(x - width, exact, width, label="Potrivire exactă", color="#1f77b4")
    b2 = ax.bar(x, char_acc, width, label="Acuratețe la nivel de caracter", color="#2ca02c")
    b3 = ax.bar(x + width, fmt, width, label="Format RO valid", color="#ff7f0e")
    ax.set_ylabel("Procent (%)")
    ax.set_title("Recunoașterea plăcuțelor: pipeline vs. citire brută EasyOCR")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0, 105); ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    for bars in (b1, b2, b3):
        for bar in bars:
            ax.annotate(f"{bar.get_height():.0f}", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)
    fig.tight_layout()
    out = RESULTS_DIR / "comparatie_ocr.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[PLOT] Figură: {out}")


if __name__ == "__main__":
    main()
