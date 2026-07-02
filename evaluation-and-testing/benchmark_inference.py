#!/usr/bin/env python3
"""
benchmark_inference.py — per-detector inference latency measurement.

Runs each AI module of the pipeline (face, plate, fire, weapon, HAR) on a set of
representative frames and reports the real per-frame latency (median / mean / p95).

Notes on what it does and does not measure:
  * Measures only inference time (forward pass + the pre/post-processing inside
    `process_frame`). Model initialization (GPU load) is not counted.
  * It does not need to trigger real alerts (fights, fire, weapons). A model's
    latency is almost independent of whether it finds something — it depends on
    resolution, architecture and the number of detected objects.
  * For realistic FACE and PLATE numbers use a video that actually contains a
    face / a plate (cost scales per object: FAISS recognition per face, EasyOCR
    per plate). For FIRE / WEAPON / HAR the content does not matter much.

Examples:
    # representative video, all modules
    python benchmark_inference.py --source /path/clip.mp4 --frames 200

    # only the modules that do not need PostgreSQL
    python benchmark_inference.py --source /path/clip.mp4 --modules fire weapon har

    # webcam
    python benchmark_inference.py --source 0

    # no real source (synthetic) — smoke-test only; face/plate numbers will be
    # underestimated (0 objects detected)
    python benchmark_inference.py --synthetic
"""

import argparse
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Callable, List, Optional

import cv2
import numpy as np

# This benchmark imports the application modules and their models. It expects the
# app to sit in a sibling "surveillance-platform" directory (override with the
# SURVEILLANCE_APP_DIR environment variable). We add it to sys.path and run from
# there so the default model paths ("models/...") resolve like in app.py.
SCRIPT_DIR = Path(__file__).resolve().parent
APP_DIR = Path(os.environ.get("SURVEILLANCE_APP_DIR",
                              SCRIPT_DIR.parent / "surveillance-platform"))
sys.path.insert(0, str(APP_DIR))
os.chdir(APP_DIR)

try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

# Same defaults as in app.py (overridable from the environment).
DB_CONFIG = {
    "host": os.environ.get("DB_HOST", "localhost"),
    "database": os.environ.get("DB_NAME", "facial_recognition"),
    "user": os.environ.get("DB_USER", "postgres"),
    "password": os.environ.get("DB_PASSWORD", "incorect"),
}


def cuda_sync():
    """Force GPU kernels to finish before stopping the timer.

    CUDA launches are asynchronous; without a sync we would measure kernel
    *launch* time, not *execution* time."""
    if _HAS_TORCH and torch.cuda.is_available():
        torch.cuda.synchronize()


# --------------------------------------------------------------------------- #
#                               Frame source                                  #
# --------------------------------------------------------------------------- #
class FrameSource:
    """Frame iterator over a video / webcam index / image directory.

    Loops forever (restarts from the beginning) so it can serve any number of
    frames the benchmark asks for."""

    def __init__(self, source: str, width: int, height: int):
        self.size = (width, height)
        self._images: Optional[List[Path]] = None
        self._img_idx = 0
        self.cap = None

        p = Path(source)
        if source.isdigit():
            self.cap = cv2.VideoCapture(int(source))
            self.kind = f"webcam[{source}]"
        elif p.is_dir():
            exts = {".jpg", ".jpeg", ".png", ".bmp"}
            self._images = sorted(f for f in p.iterdir() if f.suffix.lower() in exts)
            if not self._images:
                raise RuntimeError(f"Niciun fișier imagine în directorul {p}")
            self.kind = f"imagini[{len(self._images)}]"
        elif p.is_file():
            self.cap = cv2.VideoCapture(str(p))
            self.kind = f"video[{p.name}]"
        else:
            raise RuntimeError(f"Sursă invalidă: {source}")

        if self.cap is not None and not self.cap.isOpened():
            raise RuntimeError(f"Nu pot deschide sursa video: {source}")

    def read(self) -> np.ndarray:
        if self._images is not None:
            frame = cv2.imread(str(self._images[self._img_idx % len(self._images)]))
            self._img_idx += 1
        else:
            ok, frame = self.cap.read()
            if not ok:  # video exhausted -> restart from the beginning
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self.cap.read()
                if not ok:
                    raise RuntimeError("Nu pot citi cadre din sursă")
        if self.size[0] > 0 and self.size[1] > 0:
            frame = cv2.resize(frame, self.size)
        return frame

    def release(self):
        if self.cap is not None:
            self.cap.release()


class SyntheticSource:
    """Noise frames — smoke-test only (nothing real is detected)."""

    def __init__(self, width: int, height: int):
        self.size = (width or 640, height or 480)
        self.kind = "sintetic[zgomot]"

    def read(self) -> np.ndarray:
        return np.random.randint(0, 256, (self.size[1], self.size[0], 3), dtype=np.uint8)

    def release(self):
        pass


# --------------------------------------------------------------------------- #
#                           Detector initialization                           #
# --------------------------------------------------------------------------- #
def make_face():
    from facial_recognition_system import FacialRecognitionSystem
    sys_ = FacialRecognitionSystem(DB_CONFIG, camera_id="0")
    # process_frame -> (annotated_frame, results); we time the whole call.
    return lambda frame: sys_.process_frame(frame)


def make_plate():
    from license_plate_recognition_system import LicensePlateRecognitionSystem
    sys_ = LicensePlateRecognitionSystem(DB_CONFIG)
    return lambda frame: sys_.process_frame(frame)


def make_fire():
    from fire_detection_system import FireDetectionSystem
    sys_ = FireDetectionSystem()
    return lambda frame: sys_.process_frame(frame)


def make_weapon():
    from weapon_detection_system import WeaponDetectionSystem
    sys_ = WeaponDetectionSystem(model_path="models/weapon/best.pt")
    return lambda frame: sys_.process_frame(frame)


def make_har():
    from har_system import HumanActionRecognitionSystem
    # clip_interval_frames=1 => run SlowFast inference on every frame, provided
    # the buffer holds >=32 frames (pre-filled during warmup).
    sys_ = HumanActionRecognitionSystem(
        model_path="models/har/best_model.pth",
        device="auto",
        confidence_threshold=0.5,
        clip_interval_frames=1,
    )
    return sys_  # return the object: HAR needs special handling


MODULE_FACTORIES: dict = {
    "face": make_face,
    "plate": make_plate,
    "fire": make_fire,
    "weapon": make_weapon,
    "har": make_har,
}
NEEDS_DB = {"face", "plate"}


# --------------------------------------------------------------------------- #
#                              Measurement loop                               #
# --------------------------------------------------------------------------- #
def bench_callable(name: str, fn: Callable, src, frames: int, warmup: int) -> List[float]:
    """Time `fn(frame)` over `frames` frames, after `warmup` warmup runs (the
    first GPU inference is always much slower)."""
    for _ in range(warmup):
        fn(src.read())
    cuda_sync()

    times_ms: List[float] = []
    for _ in range(frames):
        frame = src.read()
        t0 = time.perf_counter()
        fn(frame)
        cuda_sync()
        times_ms.append((time.perf_counter() - t0) * 1000.0)
    return times_ms


def bench_har(name: str, har, src, frames: int, warmup: int) -> List[float]:
    """HAR is special: `process_frame` only accumulates into a ring buffer and
    runs SlowFast periodically. We pre-fill the buffer (>=64 frames) during warmup
    so every measured call triggers a full inference, then read `last_inference_time`
    reported by the module (the pure forward-pass time)."""
    for _ in range(max(warmup, 64)):
        har.process_frame(src.read())
    cuda_sync()

    times_ms: List[float] = []
    for _ in range(frames):
        har.process_frame(src.read())
        cuda_sync()
        # last_inference_time is set in har_system.py around the forward pass
        times_ms.append(har.last_inference_time * 1000.0)
    return times_ms


def summarize(times_ms: List[float]) -> dict:
    s = sorted(times_ms)
    n = len(s)
    p95 = s[min(n - 1, int(round(0.95 * (n - 1))))]
    return {
        "n": n,
        "median": statistics.median(s),
        "mean": statistics.fmean(s),
        "p95": p95,
        "min": s[0],
        "max": s[-1],
        "fps": 1000.0 / statistics.median(s) if s and s[len(s) // 2] > 0 else 0.0,
    }


# --------------------------------------------------------------------------- #
#                                   main                                       #
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Benchmark latență inferență per detector")
    ap.add_argument("--source", default=None,
                    help="video, director de imagini, sau index webcam (ex. 0)")
    ap.add_argument("--synthetic", action="store_true",
                    help="cadre de zgomot în loc de o sursă reală (doar smoke-test)")
    ap.add_argument("--modules", nargs="+", default=list(MODULE_FACTORIES),
                    choices=list(MODULE_FACTORIES),
                    help="ce module să măsoare (implicit: toate)")
    ap.add_argument("--frames", type=int, default=200, help="cadre măsurate / modul")
    ap.add_argument("--warmup", type=int, default=10, help="cadre de încălzire / modul")
    ap.add_argument("--width", type=int, default=640, help="lățime resize (0 = nativ)")
    ap.add_argument("--height", type=int, default=480, help="înălțime resize (0 = nativ)")
    args = ap.parse_args()

    if not args.synthetic and not args.source:
        ap.error("dă fie --source <cale/index>, fie --synthetic")

    # Environment info (essential to cite the numbers in the thesis).
    print("=" * 70)
    if _HAS_TORCH and torch.cuda.is_available():
        print(f"GPU      : {torch.cuda.get_device_name(0)}")
        print(f"CUDA     : {torch.version.cuda}  |  torch {torch.__version__}")
    else:
        print("GPU      : indisponibil — rulez pe CPU (cifrele NU sunt reprezentative)")
    print(f"Rezoluție: {args.width}x{args.height}")
    print(f"Cadre    : {args.frames} măsurate, {args.warmup} warmup, per modul")

    results: dict = {}
    for name in args.modules:
        print("\n" + "-" * 70)
        print(f"Modul: {name}")
        if name in NEEDS_DB:
            print(f"  (necesită PostgreSQL @ {DB_CONFIG['host']}/{DB_CONFIG['database']})")
        try:
            obj = MODULE_FACTORIES[name]()
        except Exception as e:
            print(f"  ⚠  SĂRIT — inițializare eșuată: {type(e).__name__}: {e}")
            continue

        # A fresh source per module (from the first frame).
        try:
            src = (SyntheticSource(args.width, args.height) if args.synthetic
                   else FrameSource(args.source, args.width, args.height))
        except Exception as e:
            print(f"  ⚠  SĂRIT — sursă invalidă: {e}")
            continue
        print(f"  Sursă: {src.kind}")

        try:
            if name == "har":
                times = bench_har(name, obj, src, args.frames, args.warmup)
            else:
                times = bench_callable(name, obj, src, args.frames, args.warmup)
        except Exception as e:
            print(f"  ⚠  EROARE la măsurare: {type(e).__name__}: {e}")
            src.release()
            continue
        src.release()

        results[name] = summarize(times)
        r = results[name]
        print(f"  mediană={r['median']:.1f} ms  medie={r['mean']:.1f} ms  "
              f"p95={r['p95']:.1f} ms  min={r['min']:.1f}  max={r['max']:.1f}  "
              f"(~{r['fps']:.0f} fps)")

    if not results:
        print("\nNiciun modul măsurat cu succes.")
        return

    # ── Summary table ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("REZUMAT (latență de inferență per cadru/clip)")
    print("=" * 70)
    hdr = f"{'modul':<10}{'mediană':>10}{'medie':>10}{'p95':>10}{'min':>9}{'max':>9}"
    print(hdr)
    print("-" * len(hdr))
    for name, r in results.items():
        print(f"{name:<10}{r['median']:>9.1f}{r['mean']:>10.1f}{r['p95']:>10.1f}"
              f"{r['min']:>9.1f}{r['max']:>9.1f}")

    # ── LaTeX table, ready to paste into the thesis ───────────────────────
    print("\n" + "=" * 70)
    print("Tabel LaTeX (booktabs):")
    print("=" * 70)
    label = {"face": "Recunoaștere facială", "plate": "Plăcuțe înmatriculare",
             "fire": "Detecție foc/fum", "weapon": "Detecție armă",
             "har": "Recunoaștere acțiuni (SlowFast)"}
    print(r"\begin{tabular}{lrrr}")
    print(r"\toprule")
    print(r"Detector & Mediană (ms) & p95 (ms) & Throughput (fps) \\")
    print(r"\midrule")
    for name, r in results.items():
        print(f"{label.get(name, name)} & {r['median']:.1f} & {r['p95']:.1f} & {r['fps']:.0f} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")


if __name__ == "__main__":
    sys.exit(main())
