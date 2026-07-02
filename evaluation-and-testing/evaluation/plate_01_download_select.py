#!/usr/bin/env python3
"""
Step 1 for evaluating license plate recognition.

Downloads the public Romanian plates dataset (RobertLucian/license-plate-dataset,
Pascal VOC annotated — only the plate coordinates, not the text) and selects 50
images where EXACTLY one annotation appears (i.e. practically a single car/plate),
to ease the later manual annotation of the plate number.

For each chosen image it produces:
  - a cleanly renamed copy (plate_01.jpg ... plate_50.jpg);
  - an enlarged plate crop (crops/plate_XX.jpg), so the number is easy to read;
  - an entry in ground_truth.csv (the plate_number column stays empty, to be filled manually);
  - the real plate coordinates in manifest.json (used for evaluating YOLO detection).

Run:
    python plate_01_download_select.py            # download + select 50
    python plate_01_download_select.py --num 50 --min-plate-width 40 --seed 42
"""

import argparse
import csv
import json
import random
import shutil
import tarfile
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
from tqdm import tqdm

_THIS_DIR = Path(__file__).resolve().parent
PLATE_DIR = _THIS_DIR / "plate_eval"
DATASET_CACHE = PLATE_DIR / "_dataset"           # the extracted dataset
IMAGES_OUT = PLATE_DIR / "images"
CROPS_OUT = PLATE_DIR / "crops"
MANIFEST_PATH = PLATE_DIR / "manifest.json"
GT_CSV = PLATE_DIR / "ground_truth.csv"

DATASET_URL = "https://github.com/RobertLucian/license-plate-dataset/archive/refs/heads/master.tar.gz"
DATASET_TGZ = PLATE_DIR / "license-plate-dataset.tar.gz"
EXTRACTED_ROOT_NAME = "license-plate-dataset-master"


# --------------------------------------------------------------------------- #
#  Download                                                                    #
# --------------------------------------------------------------------------- #
def download_dataset() -> Path:
    """Download and extract the dataset into plate_eval/_dataset/. Idempotent."""
    root = DATASET_CACHE / EXTRACTED_ROOT_NAME
    if root.is_dir() and (root / "dataset").is_dir():
        print(f"[DS] Deja prezent în {root} — sar peste descărcare.")
        return root

    PLATE_DIR.mkdir(parents=True, exist_ok=True)
    if not DATASET_TGZ.exists():
        print(f"[DS] Descarc {DATASET_URL} ...")
        req = urllib.request.Request(DATASET_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp:  # noqa: S310 (public GitHub source)
            total = int(resp.headers.get("Content-Length", 0))
            bar = tqdm(total=total, unit="B", unit_scale=True, desc="download")
            with open(DATASET_TGZ, "wb") as f:
                while True:
                    chunk = resp.read(1 << 16)
                    if not chunk:
                        break
                    f.write(chunk)
                    bar.update(len(chunk))
            bar.close()
    else:
        print(f"[DS] Arhiva există deja: {DATASET_TGZ}")

    print("[DS] Dezarhivez ...")
    DATASET_CACHE.mkdir(parents=True, exist_ok=True)
    with tarfile.open(DATASET_TGZ, "r:gz") as tar:
        tar.extractall(DATASET_CACHE)
    if not (root / "dataset").is_dir():
        raise RuntimeError(f"Structura așteptată lipsește în {root}")
    print(f"[DS] Gata: {root}")
    return root


# --------------------------------------------------------------------------- #
#  Pascal VOC annotation parsing                                               #
# --------------------------------------------------------------------------- #
def _parse_voc(xml_path: Path) -> Optional[Dict]:
    """Return {'size':(w,h), 'boxes':[(x1,y1,x2,y2), ...]} or None on error."""
    try:
        root = ET.parse(xml_path).getroot()
        size = root.find("size")
        w = int(float(size.findtext("width")))
        h = int(float(size.findtext("height")))
        boxes = []
        for obj in root.findall("object"):
            bb = obj.find("bndbox")
            x1 = int(float(bb.findtext("xmin")))
            y1 = int(float(bb.findtext("ymin")))
            x2 = int(float(bb.findtext("xmax")))
            y2 = int(float(bb.findtext("ymax")))
            boxes.append((x1, y1, x2, y2))
        return {"size": (w, h), "boxes": boxes}
    except Exception as e:
        print(f"[WARN] XML invalid {xml_path.name}: {e}")
        return None


def _collect_single_plate_candidates(root: Path, min_plate_width: int) -> List[Dict]:
    """All images with EXACTLY one annotation and plate width >= threshold."""
    candidates: List[Dict] = []
    for split in ("train", "valid"):
        annots_dir = root / "dataset" / split / "annots"
        images_dir = root / "dataset" / split / "images"
        if not annots_dir.is_dir():
            continue
        for xml_path in sorted(annots_dir.glob("*.xml")):
            parsed = _parse_voc(xml_path)
            if not parsed or len(parsed["boxes"]) != 1:
                continue
            box = parsed["boxes"][0]
            if (box[2] - box[0]) < min_plate_width:
                continue
            img_path = images_dir / (xml_path.stem + ".jpg")
            if not img_path.exists():
                continue
            candidates.append({
                "image": img_path, "bbox": list(box), "size": list(parsed["size"]),
                "orig_name": img_path.name,
            })
    return candidates


# --------------------------------------------------------------------------- #
#  Enlarged plate crop, for easy manual reading                                #
# --------------------------------------------------------------------------- #
def _make_readable_crop(img, bbox, pad_ratio: float = 0.15, target_w: int = 360):
    x1, y1, x2, y2 = bbox
    h, w = img.shape[:2]
    bw, bh = x2 - x1, y2 - y1
    px, py = int(bw * pad_ratio), int(bh * pad_ratio)
    x1, y1 = max(0, x1 - px), max(0, y1 - py)
    x2, y2 = min(w, x2 + px), min(h, y2 + py)
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return crop
    if crop.shape[1] < target_w:
        scale = target_w / crop.shape[1]
        crop = cv2.resize(crop, (int(crop.shape[1] * scale), int(crop.shape[0] * scale)),
                          interpolation=cv2.INTER_CUBIC)
    return crop


# --------------------------------------------------------------------------- #
#  Selection + writing                                                         #
# --------------------------------------------------------------------------- #
def build(num: int, min_plate_width: int, seed: int) -> None:
    root = download_dataset()
    candidates = _collect_single_plate_candidates(root, min_plate_width)
    print(f"[SELECT] {len(candidates)} imagini cu exact o plăcuță (lățime >= {min_plate_width}px).")
    if len(candidates) < num:
        raise SystemExit(f"Doar {len(candidates)} candidați, dar s-au cerut {num}. "
                         f"Scade --min-plate-width.")

    rng = random.Random(seed)
    rng.shuffle(candidates)
    chosen = candidates[:num]

    # Clean a previous selection run (keep the downloaded _dataset/).
    for d in (IMAGES_OUT, CROPS_OUT):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    manifest = {"params": {"num": num, "min_plate_width": min_plate_width, "seed": seed},
                "items": []}

    for i, c in enumerate(tqdm(chosen, desc="selectez"), start=1):
        plate_id = f"plate_{i:02d}"
        img = cv2.imread(str(c["image"]))
        if img is None:
            print(f"[WARN] Nu pot citi {c['image']}")
            continue
        cv2.imwrite(str(IMAGES_OUT / f"{plate_id}.jpg"), img)
        crop = _make_readable_crop(img, c["bbox"])
        if crop.size:
            cv2.imwrite(str(CROPS_OUT / f"{plate_id}.jpg"), crop)
        manifest["items"].append({
            "plate_id": plate_id, "orig_name": c["orig_name"],
            "gt_bbox": c["bbox"], "img_size": c["size"],
        })

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # ground_truth.csv — to be filled manually with the plate number.
    with open(GT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "plate_number"])
        for it in manifest["items"]:
            w.writerow([f"{it['plate_id']}.jpg", ""])

    print(f"\n[OK] {len(manifest['items'])} imagini scrise în {IMAGES_OUT}")
    print(f"[OK] Decupaje (pentru citire ușoară) în {CROPS_OUT}")
    print(f"[OK] Manifest (coordonate reale) în {MANIFEST_PATH}")
    print("\n>>> URMĂTORUL PAS (manual):")
    print(f"    Deschide decupajele din '{CROPS_OUT}' și completează numărul fiecărei")
    print(f"    plăcuțe în coloana 'plate_number' din '{GT_CSV}'.")
    print( "    Format recomandat: doar litere/cifre, fără spații (ex. B123ABC, CJ07XYZ).")
    print( "    Lasă gol orice plăcuță ilizibilă — va fi ignorată la evaluare.")
    print( "    Apoi rulează: python plate_02_evaluate.py")


def extend(min_plate_width: int, seed: int, replace: Optional[str], add: Optional[int]) -> None:
    """
    Grow an EXISTING set: keep the images and already-filled annotations,
    optionally replace one image (--replace plate_XX) and add new images from the
    unused candidates (plate width >= threshold). Does NOT rewrite existing annotations.
    """
    if not MANIFEST_PATH.exists() or not GT_CSV.exists():
        raise SystemExit("Nu există un set anterior. Rulează întâi selecția normală.")

    root = download_dataset()
    manifest = json.load(open(MANIFEST_PATH, encoding="utf-8"))
    items = manifest["items"]

    # The already-filled annotations, so we keep them.
    gt_values: Dict[str, str] = {}
    with open(GT_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            gt_values[row["image"]] = (row.get("plate_number") or "").strip()

    used_orig = {it["orig_name"] for it in items}
    pool = [c for c in _collect_single_plate_candidates(root, min_plate_width)
            if c["orig_name"] not in used_orig]
    rng = random.Random(seed)
    rng.shuffle(pool)
    print(f"[EXTEND] {len(pool)} candidați noi disponibili (lățime >= {min_plate_width}px).")

    need = (1 if replace else 0) + (add if add is not None else 0)
    if add is None:                         # default: take everything available
        add = len(pool) - (1 if replace else 0)
        need = len(pool)
    if len(pool) < need:
        raise SystemExit(f"Doar {len(pool)} candidați noi, dar s-au cerut {need}. "
                         f"Scade --min-plate-width.")

    def _write(plate_id: str, cand: Dict) -> Dict:
        img = cv2.imread(str(cand["image"]))
        cv2.imwrite(str(IMAGES_OUT / f"{plate_id}.jpg"), img)
        crop = _make_readable_crop(img, cand["bbox"])
        if crop.size:
            cv2.imwrite(str(CROPS_OUT / f"{plate_id}.jpg"), crop)
        return {"plate_id": plate_id, "orig_name": cand["orig_name"],
                "gt_bbox": cand["bbox"], "img_size": cand["size"]}

    cursor = 0
    # 1) In-place replacement of an existing image.
    if replace:
        target = next((it for it in items if it["plate_id"] == replace), None)
        if target is None:
            raise SystemExit(f"--replace {replace}: nu există în set.")
        new_item = _write(replace, pool[cursor]); cursor += 1
        target.update(new_item)
        gt_values[f"{replace}.jpg"] = ""        # re-annotated
        print(f"[EXTEND] Înlocuit {replace} (orig: {new_item['orig_name']}).")

    # 2) Add new images, continuing the numbering.
    next_idx = max(int(it["plate_id"].split("_")[1]) for it in items) + 1
    added = 0
    while added < add and cursor < len(pool):
        plate_id = f"plate_{next_idx:02d}"
        items.append(_write(plate_id, pool[cursor]))
        gt_values[f"{plate_id}.jpg"] = ""
        next_idx += 1; cursor += 1; added += 1
    print(f"[EXTEND] Adăugate {added} imagini noi.")

    # Reorder by numeric index and rewrite manifest + CSV (keeping the values).
    items.sort(key=lambda it: int(it["plate_id"].split("_")[1]))
    manifest["params"]["extended_to"] = len(items)
    manifest["params"]["extend_min_plate_width"] = min_plate_width
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    with open(GT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "plate_number"])
        for it in items:
            img = f"{it['plate_id']}.jpg"
            w.writerow([img, gt_values.get(img, "")])

    to_fill = sum(1 for it in items if not gt_values.get(f"{it['plate_id']}.jpg"))
    print(f"\n[OK] Set extins la {len(items)} imagini ({to_fill} de adnotat).")
    print(f"     Completează DOAR rândurile goale din {GT_CSV}")
    print(f"     (imaginile deja adnotate rămân neatinse). Decupaje în {CROPS_OUT}.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Descarcă datasetul de plăcuțe și selectează imagini cu o singură plăcuță.")
    ap.add_argument("--num", type=int, default=50, help="Câte imagini să selecteze (implicit 50).")
    ap.add_argument("--min-plate-width", type=int, default=180,
                    help="Lățime minimă a plăcuței în pixeli. Implicit 180: la această dimensiune "
                         "detectorul YOLO are ~99%% rată de detecție și plăcuța e ușor de citit. "
                         "Scade pragul (ex. 120) dacă vrei să testezi și plăcuțe mici/îndepărtate.")
    ap.add_argument("--seed", type=int, default=42, help="Seed pentru reproductibilitate.")
    ap.add_argument("--extend", action="store_true",
                    help="Mărește un set existent (păstrează imaginile și adnotările deja completate).")
    ap.add_argument("--replace", type=str, default=None,
                    help="(cu --extend) Înlocuiește o imagine, ex. plate_25.")
    ap.add_argument("--add", type=int, default=None,
                    help="(cu --extend) Câte imagini noi să adauge. Implicit: toți candidații disponibili.")
    args = ap.parse_args()

    if args.extend:
        # On extend, already-used candidates are excluded; the default threshold is
        # smaller (150px) because large plates are nearly exhausted after the first selection.
        mw = args.min_plate_width if args.min_plate_width != 180 else 150
        extend(mw, args.seed, args.replace, args.add)
    else:
        build(args.num, args.min_plate_width, args.seed)


if __name__ == "__main__":
    main()
