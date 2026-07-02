#!/usr/bin/env python3
"""
Evaluation of the human action recognition model (SlowFast R50).

Simple model test: takes each test clip in turn, classifies it into one of the
three classes (normal / fight / vandalism) and checks whether the prediction
matches the ground-truth label (given by the folder the clip is in). There is no
extra pipeline -- the model is evaluated directly.

The test clips are long and of variable duration, while the model was trained on
~2-second segments. So each clip is scanned with a sliding window of
clip_duration_sec (2 s): each window is classified and the softmax probabilities
are averaged over the whole clip, yielding a single prediction per video
(clip-level aggregation, robust to noise).

Run:
    python har_01_evaluate.py
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
import numpy as np
import torch
from tqdm import tqdm

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = Path(os.environ.get("SURVEILLANCE_APP_DIR",
                                   _THIS_DIR.parents[1] / "surveillance-platform"))
sys.path.insert(0, str(_PROJECT_DIR))
from har_system import HumanActionRecognitionSystem, _pack_slowfast  # noqa: E402

DEFAULT_DATA = Path("/mnt/data/documentatie_licenta/pipeline_testing/har")
RESULTS_DIR = _THIS_DIR / "har_eval" / "results"
MODEL_PATH = _PROJECT_DIR / "models" / "har" / "best_model.pth"

# Ground-truth label by folder -> model class.
FOLDER_TO_CLASS = {"normal": "normal", "fighting": "fight", "vandalism": "vandalism"}
CLASSES = ["normal", "fight", "vandalism"]

CLIP_SEC = 2.0          # temporal window (as in training)
MAX_WINDOWS = 16        # cap on windows / clip (for very long clips)
CROP = 224


def _read_window(cap, start: int, length: int) -> np.ndarray:
    """Read `length` frames from `start`, resize them to 224 RGB."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = []
    for _ in range(length):
        ok, fr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        frames.append(cv2.resize(rgb, (CROP, CROP)))
    return np.stack(frames) if frames else np.empty((0, CROP, CROP, 3), np.uint8)


def classify_video(model, device: str, path: Path) -> Dict:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    win = max(8, int(round(CLIP_SEC * fps)))

    if n <= win:
        starts = [0]
    else:
        n_win = min(MAX_WINDOWS, max(1, int(n / fps / 1.0) - 1))   # ~1 window / second
        starts = list(np.linspace(0, n - win, n_win, dtype=int))

    prob_sum = np.zeros(len(CLASSES), dtype=np.float64)
    prob_max = np.zeros(len(CLASSES), dtype=np.float64)   # strongest window / class
    n_used = 0
    for s in starts:
        frames = _read_window(cap, int(s), win)
        if frames.shape[0] < 8:
            continue
        n_sample = min(64, frames.shape[0])
        idx = np.linspace(0, frames.shape[0] - 1, n_sample, dtype=np.int64)
        clip = frames[idx]
        slow_t, fast_t = _pack_slowfast(clip, crop_size=CROP)
        slow_t = slow_t.unsqueeze(0).to(device)
        fast_t = fast_t.unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model([slow_t, fast_t])
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        prob_sum += probs
        prob_max = np.maximum(prob_max, probs)
        n_used += 1
    cap.release()

    if n_used == 0:
        return {"pred": None, "conf": 0.0, "probs": [0, 0, 0],
                "pred_best": None, "probs_best": [0, 0, 0], "windows": 0}
    avg = prob_sum / n_used
    ai = int(avg.argmax())
    # "alarm" aggregation: the non-normal class with the strongest window, if it
    # exceeds the system's confidence threshold (0.5); otherwise normal.
    best_idx = 0
    best_nonnormal = max((1, 2), key=lambda k: prob_max[k])
    if prob_max[best_nonnormal] >= 0.5:
        best_idx = best_nonnormal
    return {"pred": CLASSES[ai], "conf": float(avg[ai]), "probs": avg.tolist(),
            "pred_best": CLASSES[best_idx], "probs_best": prob_max.tolist(),
            "windows": n_used}


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluarea modelului HAR (SlowFast) pe clipuri de test.")
    ap.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA))
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    videos = []
    for folder, cls in FOLDER_TO_CLASS.items():
        for v in sorted((data_dir / folder).glob("*.mp4")):
            videos.append((v, cls))
    print(f"[INIT] {len(videos)} clipuri de test")

    har = HumanActionRecognitionSystem(model_path=str(MODEL_PATH))
    device = har.device

    rows = []
    for path, gt in tqdm(videos, desc="clasific"):
        r = classify_video(har.model, device, path)
        rows.append({"video": path.name, "gt": gt, **r})
    _report(rows)


def _confusion(rows, key):
    idx = {c: i for i, c in enumerate(CLASSES)}
    cm = np.zeros((3, 3), dtype=int)
    for r in rows:
        if r[key] is None:
            continue
        cm[idx[r["gt"]], idx[r[key]]] += 1
    return cm


def _print_cm(cm, title):
    n = int(cm.sum()); correct = int(np.trace(cm))
    print(f"\n{title}: {correct}/{n} ({correct/n:.1%})")
    hdr = "real\\prezis  " + "".join(f"{c:>11}" for c in CLASSES) + f"{'  recall':>10}"
    print(hdr); print("-" * len(hdr))
    for i, c in enumerate(CLASSES):
        rt = cm[i].sum()
        rec = cm[i, i] / rt if rt else 0.0
        print(f"{c:<12}" + "".join(f"{cm[i, j]:>11}" for j in range(3)) + f"{rec:>10.0%}")


def _report(rows: List[Dict]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cm_avg = _confusion(rows, "pred")
    cm_best = _confusion(rows, "pred_best")

    print(f"\n==================== RECUNOAȘTEREA ACȚIUNILOR (SlowFast) ====================")
    print("Două moduri de agregare a ferestrelor de 2 s pe fiecare clip:")
    _print_cm(cm_avg, "A) Mediere pe tot clipul (medie softmax)")
    _print_cm(cm_best, "B) Cea mai puternică fereastră / alarmă (fidel sistemului real)")

    print("\n=== detaliu per clip ===")
    print(f"{'video':<24}{'real':<11}{'mediere':<11}{'alarmă':<11}  probs_medii[N,F,V]")
    for r in sorted(rows, key=lambda r: (r["gt"], r["video"])):
        oa = "✓" if r["pred"] == r["gt"] else "✗"
        ob = "✓" if r["pred_best"] == r["gt"] else "✗"
        p = ",".join(f"{x:.2f}" for x in r["probs"])
        print(f"{r['video']:<24}{r['gt']:<11}{str(r['pred'])+' '+oa:<11}{str(r['pred_best'])+' '+ob:<11}  [{p}]")

    with open(RESULTS_DIR / "har_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["video", "gt", "pred_mediere", "pred_alarma", "conf_medie",
                    "p_normal", "p_fight", "p_vandalism", "windows",
                    "corect_mediere", "corect_alarma"])
        for r in rows:
            w.writerow([r["video"], r["gt"], r["pred"], r["pred_best"], f"{r['conf']:.4f}",
                        *[f"{x:.4f}" for x in r["probs"]], r["windows"],
                        int(r["pred"] == r["gt"]), int(r["pred_best"] == r["gt"])])
    _plot(cm_avg, cm_best)
    print(f"\n[CSV/PLOT] {RESULTS_DIR}")


def _plot(cm_avg: np.ndarray, cm_best: np.ndarray) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, cm, title in ((axes[0], cm_avg, "A) Mediere pe clip"),
                          (axes[1], cm_best, "B) Cea mai puternică fereastră")):
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(3)); ax.set_yticks(range(3))
        ax.set_xticklabels(CLASSES); ax.set_yticklabels(CLASSES)
        ax.set_xlabel("Clasă prezisă"); ax.set_ylabel("Clasă reală")
        acc = np.trace(cm) / cm.sum() if cm.sum() else 0
        ax.set_title(f"{title}  (acuratețe {acc:.0%})")
        thr = cm.max() / 2 if cm.max() else 0
        for i in range(3):
            for j in range(3):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > thr else "black", fontsize=13)
    fig.suptitle("Recunoașterea acțiunilor — matrice de confuzie")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "har_confusion.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
