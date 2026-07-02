#!/usr/bin/env python3
"""
Evaluation of the facial recognition module on the LFW set (Labeled Faces in the Wild).

The script builds a controlled test set from LFW (deepfunneled) and reproduces
exactly the decision logic of the real system (`facial_recognition_system.py`):

    image --> InsightFace buffalo_s --> 512-d L2-normalized embedding
          --> FaissIndex (IndexFlatL2) --> nearest person (L2)
          --> known if  L2 < recognition_threshold  AND  cos_sim >= min_confidence
              otherwise  Unknown

It produces three tables (known identification, unknown rejection, threshold study),
CSV files and a figure of the L2 threshold sweep.

Stages (can be run separately or all at once with --all):
    python evaluate_facial_lfw.py --download   # fetch LFW deepfunneled into cache/
    python evaluate_facial_lfw.py --build       # build the controlled dataset/
    python evaluate_facial_lfw.py --run         # enroll + evaluate + tables + CSV + figure
    python evaluate_facial_lfw.py --all         # all of the above

The controlled set follows the structure:
    dataset/
      enrollment/person_01/ ... person_12/   (5 enrollment images / person)
      test/known/person_01/ ... person_12/   (10 test images / person)
      test/unknown/                          (20 images, single-photo people)
"""

import argparse
import csv
import json
import os
import random
import shutil
import sys
import tarfile
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

# --------------------------------------------------------------------------- #
#  Reuse the real system's index class instead of reimplementing it, so the    #
#  search (IndexFlatL2 + idx->person_id mapping) is identical to production.    #
# --------------------------------------------------------------------------- #
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = Path(os.environ.get("SURVEILLANCE_APP_DIR",
                                   _THIS_DIR.parents[1] / "surveillance-platform"))
sys.path.insert(0, str(_PROJECT_DIR))
from faiss_index import FaissIndex  # noqa: E402

# --------------------------------------------------------------------------- #
#  Constants taken 1:1 from facial_recognition_system.py for fidelity.         #
# --------------------------------------------------------------------------- #
EMBEDDING_DIM = 512                 # buffalo_s / w600k_mbf
MODEL_NAME = "buffalo_s"
DET_SIZE = (640, 640)
RECOGNITION_THRESHOLD = 1.0         # L2 distance threshold (chosen in the real system)
MIN_CONFIDENCE = 0.4                # secondary cos_sim threshold (ArcFace guard)

# LFW source: the official UMass host (vis-www.cs.umass.edu) is often unavailable
# in DNS, so we use the figshare mirror that scikit-learn also uses. It serves the
# *funneled* variant (250x250, aligned, with background — just as suitable for
# InsightFace detection as deepfunneled).
LFW_URL = "https://ndownloader.figshare.com/files/5976015"  # -> lfwfunneled.tgz
LFW_TGZ_NAME = "lfw-funneled.tgz"
# Possible names of the folder produced after extraction (depends on the variant).
LFW_CANDIDATE_DIRS = ("lfw_funneled", "lfw-deepfunneled", "lfw-funneled", "lfw")
# Overridden by --lfw-dir; allows using an LFW already downloaded manually.
LFW_ROOT_OVERRIDE: Optional[Path] = None

# Thresholds evaluated explicitly in the thesis table.
THRESHOLDS_TABLE = [
    (0.8, "strict"),
    (1.0, "ales"),
    (1.2, "permisiv"),
]
# Range for the threshold-sweep figure.
THRESHOLD_SWEEP = np.round(np.arange(0.60, 1.45, 0.05), 2)

# Working paths (relative to this file). For "large" mode they are overridden in
# main() so the artifacts of the two experiments do not mix.
CACHE_DIR = _THIS_DIR / "cache"
DATASET_DIR = _THIS_DIR / "dataset"
RESULTS_DIR = _THIS_DIR / "results"
MANIFEST_PATH = DATASET_DIR / "manifest.json"

DATASET_DIR_LARGE = _THIS_DIR / "dataset_large"   # manifest only (no image copies)
RESULTS_DIR_LARGE = _THIS_DIR / "results_large"


# =========================================================================== #
#  Step 0 — downloading LFW                                                    #
# =========================================================================== #
def _find_lfw_root() -> Optional[Path]:
    """Locate the extracted LFW folder in cache/ (per-person with .jpg)."""
    if LFW_ROOT_OVERRIDE is not None:
        return LFW_ROOT_OVERRIDE
    if not CACHE_DIR.is_dir():
        return None
    for name in LFW_CANDIDATE_DIRS:
        p = CACHE_DIR / name
        if p.is_dir() and any(p.iterdir()):
            return p
    # Fallback: any subdirectory that contains person-subfolders with images.
    for p in sorted(CACHE_DIR.iterdir()):
        if p.is_dir():
            subdirs = [d for d in p.iterdir() if d.is_dir()]
            if subdirs and any(next(d.glob("*.jpg"), None) for d in subdirs[:10]):
                return p
    return None


def download_lfw() -> Path:
    """Download and extract LFW into cache/. Idempotent."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    lfw_root = _find_lfw_root()
    if lfw_root is not None:
        print(f"[LFW] Deja prezent în {lfw_root} — sar peste descărcare.")
        return lfw_root

    tgz_path = CACHE_DIR / LFW_TGZ_NAME
    if not tgz_path.exists():
        print(f"[LFW] Descarc {LFW_URL} ...")
        _download_with_progress(LFW_URL, tgz_path)
    else:
        print(f"[LFW] Arhiva există deja: {tgz_path}")

    print(f"[LFW] Dezarhivez {tgz_path.name} ...")
    with tarfile.open(tgz_path, "r:gz") as tar:
        tar.extractall(CACHE_DIR)
    lfw_root = _find_lfw_root()
    if lfw_root is None:
        raise RuntimeError("Dezarhivarea nu a produs un folder LFW recognoscibil.")
    print(f"[LFW] Gata: {lfw_root}")
    return lfw_root


def _download_with_progress(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:  # noqa: S310 (mirror public figshare)
        total = int(resp.headers.get("Content-Length", 0))
        bar = tqdm(total=total, unit="B", unit_scale=True, desc="download")
        with open(dest, "wb") as f:
            while True:
                chunk = resp.read(1 << 16)
                if not chunk:
                    break
                f.write(chunk)
                bar.update(len(chunk))
        bar.close()


# =========================================================================== #
#  Step 1 — building the controlled set                                        #
# =========================================================================== #
def build_dataset(num_known, enroll_per_person: int, test_per_person: int,
                  num_unknown, seed: int, copy_files: bool = True,
                  test_cap: Optional[int] = None) -> Dict:
    """
    Select people from LFW and prepare the test set. The manifest stores ABSOLUTE
    paths to the images.

    - `num_known` / `num_unknown`: int or "all" (all qualifying people).
    - `copy_files=True`  -> copy the images into the `dataset/` structure (controlled
      experiment, useful as visual evidence in the thesis).
    - `copy_files=False` -> copy nothing; reference images directly from the cache
      ("large" mode, with thousands of images).
    - `test_cap`: max test images per person (None = all remaining after enrollment).
      Bounds runtime and avoids domination by people with very many photos
      (e.g. Bush, 530 images).
    """
    lfw_root = _find_lfw_root()
    if lfw_root is None:
        raise SystemExit("LFW lipsește. Rulează --download sau indică --lfw-dir <cale>.")

    min_known_images = enroll_per_person + test_per_person  # ex. 5 + 10 = 15

    per_person: Dict[str, List[Path]] = {}
    for person_dir in sorted(lfw_root.iterdir()):
        if not person_dir.is_dir():
            continue
        imgs = sorted(person_dir.glob("*.jpg"))
        if imgs:
            per_person[person_dir.name] = imgs

    rng = random.Random(seed)

    # Known: people with enough images, sorted desc by photo count.
    known_candidates = sorted(
        (name for name, imgs in per_person.items() if len(imgs) >= min_known_images),
        key=lambda n: (-len(per_person[n]), n),
    )
    if num_known != "all":
        if len(known_candidates) < num_known:
            raise SystemExit(
                f"Doar {len(known_candidates)} persoane au >= {min_known_images} imagini, "
                f"dar s-au cerut {num_known}.")
        known_names = known_candidates[:num_known]
    else:
        known_names = known_candidates

    # Unknown: people with exactly one image.
    singletons = sorted(name for name, imgs in per_person.items() if len(imgs) == 1)
    rng.shuffle(singletons)
    if num_unknown != "all":
        if len(singletons) < num_unknown:
            raise SystemExit(
                f"Doar {len(singletons)} persoane au exact 1 imagine, "
                f"dar s-au cerut {num_unknown}.")
        unknown_names = singletons[:num_unknown]
    else:
        unknown_names = singletons

    if copy_files:
        if DATASET_DIR.exists():
            shutil.rmtree(DATASET_DIR)
        (DATASET_DIR / "test" / "known").mkdir(parents=True, exist_ok=True)
        (DATASET_DIR / "test" / "unknown").mkdir(parents=True, exist_ok=True)
        (DATASET_DIR / "enrollment").mkdir(parents=True, exist_ok=True)

    manifest = {
        "params": {
            "num_known": num_known, "enroll_per_person": enroll_per_person,
            "test_per_person": test_per_person, "num_unknown": num_unknown,
            "seed": seed, "copy_files": copy_files, "test_cap": test_cap,
        },
        "known": {}, "unknown": [],
    }

    for i, name in enumerate(known_names, start=1):
        label = f"person_{i:02d}"
        imgs = list(per_person[name])
        rng.shuffle(imgs)
        enroll_imgs = imgs[:enroll_per_person]
        rest = imgs[enroll_per_person:]
        test_imgs = rest[:test_cap] if test_cap else rest

        if copy_files:
            enr_dir = DATASET_DIR / "enrollment" / label
            tst_dir = DATASET_DIR / "test" / "known" / label
            enr_dir.mkdir(parents=True, exist_ok=True)
            tst_dir.mkdir(parents=True, exist_ok=True)
            enr_paths, tst_paths = [], []
            for p in enroll_imgs:
                shutil.copy(p, enr_dir / p.name); enr_paths.append(str(enr_dir / p.name))
            for p in test_imgs:
                shutil.copy(p, tst_dir / p.name); tst_paths.append(str(tst_dir / p.name))
        else:
            enr_paths = [str(p) for p in enroll_imgs]
            tst_paths = [str(p) for p in test_imgs]

        manifest["known"][label] = {
            "person_id": i, "lfw_name": name,
            "enrollment": enr_paths, "test": tst_paths,
        }

    if copy_files:
        unk_dir = DATASET_DIR / "test" / "unknown"
    for name in unknown_names:
        p = per_person[name][0]
        if copy_files:
            shutil.copy(p, unk_dir / p.name)
            path = str(unk_dir / p.name)
        else:
            path = str(p)
        manifest["unknown"].append({"lfw_name": name, "image_path": path})

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    n_test = sum(len(v["test"]) for v in manifest["known"].values())
    print(f"[BUILD] {len(known_names)} cunoscuți ({enroll_per_person} înrolare + {n_test} test în total) "
          f"+ {len(unknown_names)} necunoscuți. copy_files={copy_files}")
    print(f"[BUILD] Manifest: {MANIFEST_PATH}")
    return manifest


# =========================================================================== #
#  InsightFace — same initialization as in the real system                     #
# =========================================================================== #
def build_face_app():
    from insightface.app import FaceAnalysis
    print(f"[FACE] Inițializez InsightFace ({MODEL_NAME}, det_size={DET_SIZE}) ...")
    app = FaceAnalysis(
        name=MODEL_NAME,
        providers=[
            ("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC"}),
            "CPUExecutionProvider",
        ],
    )
    app.prepare(ctx_id=0, det_size=DET_SIZE)
    provider = app.models["recognition"].session.get_providers()[0] \
        if hasattr(app, "models") else "?"
    print(f"[FACE] Provider activ: {provider}")
    return app


def extract_embedding(app, img: np.ndarray) -> Optional[np.ndarray]:
    """
    Return the 512-d L2-normalized embedding if the image contains EXACTLY one face
    (identical to `extract_embedding_from_image` in the real system). None otherwise.
    """
    if img is None or img.size == 0:
        return None
    faces = app.get(img)
    if len(faces) != 1:
        return None
    return np.asarray(faces[0].normed_embedding, dtype=np.float32)


# =========================================================================== #
#  Step 2 — enrollment + evaluation                                            #
# =========================================================================== #
def enroll(app, index: FaissIndex, manifest: Dict) -> int:
    """Add the enrollment embeddings to the index (one per image, not averaged)."""
    n_added = 0
    for label, info in tqdm(manifest["known"].items(), desc="enroll"):
        person_id = info["person_id"]
        for path in info["enrollment"]:
            img = cv2.imread(path)
            emb = extract_embedding(app, img)
            if emb is None:
                print(f"[WARN] Înrolare: nicio față unică în {label}/{Path(path).name} — ignorat.")
                continue
            index.add_embedding(emb, person_id)
            n_added += 1
    print(f"[ENROLL] {n_added} embeddings adăugate în index "
          f"({index.get_statistics()['unique_persons']} persoane).")
    return n_added


def evaluate(app, index: FaissIndex, manifest: Dict) -> List[Dict]:
    """
    For each test image store (set, true_id, pred_id, L2). We search with an
    infinite threshold to get the raw distance; the threshold is applied later,
    which allows the threshold study without re-embedding.
    """
    records: List[Dict] = []

    # Known.
    for label, info in tqdm(manifest["known"].items(), desc="test/known"):
        for path in info["test"]:
            img = cv2.imread(path)
            emb = extract_embedding(app, img)
            rec = {"set": "known", "label": label, "true_id": info["person_id"],
                   "image": Path(path).name, "pred_id": None, "l2": None, "detected": emb is not None}
            if emb is not None:
                raw = index.search(emb, k=1, threshold=float("inf"))
                if raw:
                    rec["pred_id"], rec["l2"] = int(raw[0][0]), float(raw[0][1])
            records.append(rec)

    # Unknown.
    for item in tqdm(manifest["unknown"], desc="test/unknown"):
        path = item["image_path"]
        img = cv2.imread(path)
        emb = extract_embedding(app, img)
        rec = {"set": "unknown", "label": item["lfw_name"], "true_id": None,
               "image": Path(path).name, "pred_id": None, "l2": None, "detected": emb is not None}
        if emb is not None:
            raw = index.search(emb, k=1, threshold=float("inf"))
            if raw:
                rec["pred_id"], rec["l2"] = int(raw[0][0]), float(raw[0][1])
        records.append(rec)

    return records


# =========================================================================== #
#  Match decision — replicates _match_embedding + the min_confidence guard      #
# =========================================================================== #
def is_accepted(l2: Optional[float], threshold: float) -> bool:
    """True if the system would mark the face as known at the given threshold."""
    if l2 is None:
        return False
    if l2 >= threshold:
        return False
    cos_sim = 1.0 - (l2 ** 2) / 2.0
    return cos_sim >= MIN_CONFIDENCE


# =========================================================================== #
#  Reports                                                                     #
# =========================================================================== #
def report_known(records: List[Dict], threshold: float) -> Dict:
    """Per-person table: Tests | Correct | Wrong | Unknown | Accuracy."""
    known = [r for r in records if r["set"] == "known"]
    per_label: Dict[str, Dict] = defaultdict(
        lambda: {"tests": 0, "correct": 0, "wrong": 0, "unknown": 0, "nodetect": 0})

    for r in known:
        d = per_label[r["label"]]
        d["tests"] += 1
        if not r["detected"]:
            d["nodetect"] += 1
        elif is_accepted(r["l2"], threshold):
            if r["pred_id"] == r["true_id"]:
                d["correct"] += 1
            else:
                d["wrong"] += 1
        else:
            d["unknown"] += 1

    print(f"\n=== Identificarea persoanelor cunoscute (prag L2 = {threshold}) ===")
    print("(Accuracy = Corecte / Evaluate, unde Evaluate = Teste − Nedetect)")
    header = (f"{'Persoană':<12}{'Teste':>7}{'Nedetect':>9}{'Corecte':>9}"
              f"{'Greșite':>9}{'Unknown':>9}{'Accuracy':>10}")
    print(header)
    print("-" * len(header))
    tot = {"tests": 0, "correct": 0, "wrong": 0, "unknown": 0, "nodetect": 0}
    for label in sorted(per_label):
        d = per_label[label]
        evaluated = d["correct"] + d["wrong"] + d["unknown"]  # exclude detection failures
        acc = d["correct"] / evaluated if evaluated else 0.0
        print(f"{label:<12}{d['tests']:>7}{d['nodetect']:>9}{d['correct']:>9}"
              f"{d['wrong']:>9}{d['unknown']:>9}{acc:>9.1%}")
        for k in tot:
            tot[k] += d[k]
    evaluated = tot["correct"] + tot["wrong"] + tot["unknown"]
    overall = tot["correct"] / evaluated if evaluated else 0.0
    # Macro mean = mean of per-person accuracies (not weighted by photo count,
    # so it is not dominated by people with very many images).
    per_acc = []
    for d in per_label.values():
        ev = d["correct"] + d["wrong"] + d["unknown"]
        if ev:
            per_acc.append(d["correct"] / ev)
    macro = sum(per_acc) / len(per_acc) if per_acc else 0.0
    print("-" * len(header))
    print(f"{'TOTAL':<12}{tot['tests']:>7}{tot['nodetect']:>9}{tot['correct']:>9}"
          f"{tot['wrong']:>9}{tot['unknown']:>9}{overall:>9.1%}")
    print(f"Acuratețe micro (per imagine): {overall:.1%}   |   "
          f"macro (medie per persoană): {macro:.1%}   |   persoane: {len(per_label)}")

    return {"per_label": per_label, "total": tot, "accuracy": overall}


def report_unknown(records: List[Dict], threshold: float) -> Dict:
    """Table: Num images | Correct Unknown | False recognized | Unknown accuracy."""
    unk = [r for r in records if r["set"] == "unknown"]
    n = len(unk)
    nodetect = sum(1 for r in unk if not r["detected"])
    false_rec = sum(1 for r in unk if r["detected"] and is_accepted(r["l2"], threshold))
    correct_unknown = sum(1 for r in unk if r["detected"] and not is_accepted(r["l2"], threshold))
    evaluated = n - nodetect
    acc = correct_unknown / evaluated if evaluated else 0.0

    print(f"\n=== Respingerea persoanelor necunoscute (prag L2 = {threshold}) ===")
    header = f"{'Set':<12}{'Nr imagini':>12}{'Corect Unknown':>16}{'Fals recunoscute':>18}{'Unknown acc.':>14}"
    print(header)
    print("-" * len(header))
    print(f"{'unknown':<12}{n:>12}{correct_unknown:>16}{false_rec:>18}{acc:>13.1%}")
    if nodetect:
        print(f"(eșecuri de detecție excluse: {nodetect})")

    return {"n": n, "correct_unknown": correct_unknown, "false_recognized": false_rec,
            "nodetect": nodetect, "accuracy": acc}


def _known_accuracy_at(records: List[Dict], threshold: float) -> float:
    known = [r for r in records if r["set"] == "known" and r["detected"]]
    if not known:
        return 0.0
    correct = sum(1 for r in known if is_accepted(r["l2"], threshold) and r["pred_id"] == r["true_id"])
    return correct / len(known)


def _unknown_rejection_at(records: List[Dict], threshold: float) -> float:
    unk = [r for r in records if r["set"] == "unknown" and r["detected"]]
    if not unk:
        return 0.0
    rejected = sum(1 for r in unk if not is_accepted(r["l2"], threshold))
    return rejected / len(unk)


def report_threshold_study(records: List[Dict]) -> List[Dict]:
    """Table with the 3 thresholds + the data for the figure."""
    print("\n=== Studiul pragului de recunoaștere ===")
    header = f"{'Prag L2':>8}{'Known accuracy':>18}{'Unknown rejection':>20}{'Observații':>14}"
    print(header)
    print("-" * len(header))
    rows = []
    for thr, note in THRESHOLDS_TABLE:
        ka = _known_accuracy_at(records, thr)
        ur = _unknown_rejection_at(records, thr)
        print(f"{thr:>8.1f}{ka:>17.1%}{ur:>19.1%}{note:>14}")
        rows.append({"threshold": thr, "known_accuracy": ka,
                     "unknown_rejection": ur, "note": note})
    return rows


def plot_threshold_sweep(records: List[Dict]) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    thr = THRESHOLD_SWEEP
    ka = [_known_accuracy_at(records, t) * 100 for t in thr]
    ur = [_unknown_rejection_at(records, t) * 100 for t in thr]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(thr, ka, "o-", color="#1f77b4", label="Acuratețe cunoscuți")
    ax.plot(thr, ur, "s-", color="#d62728", label="Respingere necunoscuți")
    ax.axvline(RECOGNITION_THRESHOLD, color="gray", linestyle="--", linewidth=1)
    ax.annotate(f"prag ales = {RECOGNITION_THRESHOLD}",
                xy=(RECOGNITION_THRESHOLD, 5), xytext=(RECOGNITION_THRESHOLD + 0.02, 8),
                fontsize=9, color="gray")
    ax.set_xlabel("Prag distanță L2")
    ax.set_ylabel("Procent (%)")
    ax.set_title("Compromisul pragului de recunoaștere facială (LFW)")
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center right")
    fig.tight_layout()

    out = RESULTS_DIR / "prag_l2_facial.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[PLOT] Figură salvată: {out}")
    return out


# =========================================================================== #
#  Result persistence                                                          #
# =========================================================================== #
def save_csvs(records: List[Dict], known: Dict, unknown: Dict, thr_rows: List[Dict]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(RESULTS_DIR / "known_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["persoana", "teste", "corecte", "gresite", "unknown", "nodetect", "accuracy"])
        for label in sorted(known["per_label"]):
            d = known["per_label"][label]
            ev = d["correct"] + d["wrong"] + d["unknown"]
            acc = d["correct"] / ev if ev else 0.0
            w.writerow([label, d["tests"], d["correct"], d["wrong"], d["unknown"], d["nodetect"], f"{acc:.4f}"])
        t = known["total"]
        ev = t["correct"] + t["wrong"] + t["unknown"]
        w.writerow(["TOTAL", t["tests"], t["correct"], t["wrong"], t["unknown"], t["nodetect"],
                    f"{(t['correct']/ev if ev else 0):.4f}"])

    with open(RESULTS_DIR / "unknown_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["nr_imagini", "corect_unknown", "fals_recunoscute", "nodetect", "unknown_accuracy"])
        w.writerow([unknown["n"], unknown["correct_unknown"], unknown["false_recognized"],
                    unknown["nodetect"], f"{unknown['accuracy']:.4f}"])

    with open(RESULTS_DIR / "threshold_study.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["prag_l2", "known_accuracy", "unknown_rejection", "observatii"])
        for r in thr_rows:
            w.writerow([r["threshold"], f"{r['known_accuracy']:.4f}",
                        f"{r['unknown_rejection']:.4f}", r["note"]])

    # Raw records (useful for any further analysis).
    with open(RESULTS_DIR / "raw_records.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["set", "label", "true_id", "pred_id", "l2", "detected", "image"])
        for r in records:
            w.writerow([r["set"], r["label"], r["true_id"], r["pred_id"],
                        "" if r["l2"] is None else f"{r['l2']:.4f}", int(r["detected"]), r["image"]])

    print(f"[CSV] Rezultate scrise în {RESULTS_DIR}")


# =========================================================================== #
#  Orchestration                                                               #
# =========================================================================== #
def run_evaluation() -> None:
    if not MANIFEST_PATH.exists():
        raise SystemExit(f"Lipsește {MANIFEST_PATH} — rulează mai întâi --build.")
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        manifest = json.load(f)

    app = build_face_app()
    index = FaissIndex(dimension=EMBEDDING_DIM, index_type="FlatL2")

    enroll(app, index, manifest)
    records = evaluate(app, index, manifest)

    known = report_known(records, RECOGNITION_THRESHOLD)
    unknown = report_unknown(records, RECOGNITION_THRESHOLD)
    thr_rows = report_threshold_study(records)
    plot_threshold_sweep(records)
    save_csvs(records, known, unknown, thr_rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluarea recunoașterii faciale pe LFW.")
    ap.add_argument("--download", action="store_true", help="Descarcă LFW deepfunneled.")
    ap.add_argument("--build", action="store_true", help="Construiește setul controlat.")
    ap.add_argument("--run", action="store_true", help="Înrolare + evaluare + rapoarte.")
    ap.add_argument("--all", action="store_true", help="Toate etapele, în ordine.")
    ap.add_argument("--num-known", type=int, default=12, help="Număr persoane cunoscute.")
    ap.add_argument("--enroll-per-person", type=int, default=5, help="Imagini de înrolare / persoană.")
    ap.add_argument("--test-per-person", type=int, default=10, help="Imagini de test / persoană.")
    ap.add_argument("--num-unknown", type=int, default=20, help="Imagini cu persoane necunoscute.")
    ap.add_argument("--seed", type=int, default=42, help="Seed pentru reproductibilitate.")
    ap.add_argument("--lfw-dir", type=str, default=None,
                    help="Cale către un LFW deja extras (per-persoană cu .jpg). Sare peste descărcare.")
    ap.add_argument("--large", action="store_true",
                    help="Evaluare la scară mare: toate persoanele cu >=15 imagini ca galerie "
                         "și toți cei ~4069 singleton ca necunoscuți. Scrie în dataset_large/ și results_large/.")
    ap.add_argument("--test-cap", type=int, default=20,
                    help="(mod --large) Max imagini de test / persoană, ca să nu domine "
                         "persoanele cu foarte multe poze. Implicit 20.")
    args = ap.parse_args()

    if not (args.download or args.build or args.run or args.all):
        ap.error("Alege cel puțin o etapă: --download / --build / --run / --all")

    if args.lfw_dir:
        global LFW_ROOT_OVERRIDE
        LFW_ROOT_OVERRIDE = Path(args.lfw_dir).expanduser().resolve()
        if not LFW_ROOT_OVERRIDE.is_dir():
            ap.error(f"--lfw-dir nu există: {LFW_ROOT_OVERRIDE}")

    # In "large" mode redirect the artifacts so they do not overwrite the controlled experiment.
    if args.large:
        global DATASET_DIR, RESULTS_DIR, MANIFEST_PATH
        DATASET_DIR = DATASET_DIR_LARGE
        RESULTS_DIR = RESULTS_DIR_LARGE
        MANIFEST_PATH = DATASET_DIR_LARGE / "manifest.json"

    if (args.download or args.all) and not args.lfw_dir:
        download_lfw()
    if args.build or args.all:
        if args.large:
            build_dataset("all", args.enroll_per_person, args.test_per_person,
                          "all", args.seed, copy_files=False, test_cap=args.test_cap)
        else:
            build_dataset(args.num_known, args.enroll_per_person, args.test_per_person,
                          args.num_unknown, args.seed)
    if args.run or args.all:
        run_evaluation()


if __name__ == "__main__":
    main()
