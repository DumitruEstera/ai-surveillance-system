#!/usr/bin/env python3
"""
02_relabel_and_merge.py
Relabels both datasets to 2 classes (0=weapon, 1=non-weapon) and merges
them into a single YOLO-format dataset at /dev/shm/estera/weapon_dataset/

Zenodo classes (all weapons):
    0: machete  -> 0 (weapon)
    1: knife    -> 0 (weapon)
    2: baseball_bat -> 0 (weapon)
    3: rifle    -> 0 (weapon)
    4: gun      -> 0 (weapon)

Roboflow classes:
    0: pistol      -> 0 (weapon)
    1: knife       -> 0 (weapon)
    2: smartphone  -> DROPPED
    3: billete     -> DROPPED
    4: monedero    -> DROPPED
    5: tarjeta     -> DROPPED

Single class output: 0 = weapon

Usage:
    python 02_relabel_and_merge.py
"""

import os
import glob
import shutil
import yaml
from pathlib import Path

BASE_DIR = "/dev/shm/estera"
ZENODO_DIR = os.path.join(BASE_DIR, "zenodo_raw")
ROBOFLOW_DIR = os.path.join(BASE_DIR, "roboflow_raw")
OUTPUT_DIR = os.path.join(BASE_DIR, "weapon_dataset")

# Zenodo: all 5 classes are weapons
ZENODO_WEAPON_IDS = {0, 1, 2, 3, 4}

# Roboflow: pistol=0, knife=1 are weapons; rest are dropped
ROBOFLOW_WEAPON_IDS = {0, 1}
ROBOFLOW_DROP_IDS = {2, 3, 4, 5}


def find_zenodo_structure(zenodo_dir):
    """Auto-detect the Zenodo dataset structure (train/valid/test with images/labels)."""
    # Walk to find a data.yaml or typical YOLO folder layout
    for root, dirs, files in os.walk(zenodo_dir):
        if "train" in dirs or "valid" in dirs or "test" in dirs:
            return root
        if "images" in dirs and "labels" in dirs:
            return os.path.dirname(root)
    # Fallback: maybe everything is flat
    return zenodo_dir


def find_roboflow_structure(roboflow_dir):
    """Auto-detect Roboflow dataset structure."""
    for root, dirs, files in os.walk(roboflow_dir):
        if "train" in dirs or "valid" in dirs or "test" in dirs:
            return root
        if "data.yaml" in files:
            return root
    return roboflow_dir


def relabel_file(src_label_path, weapon_ids):
    """Read a YOLO label file, keep only weapon classes, relabel all to 0."""
    if not os.path.exists(src_label_path):
        return []

    new_lines = []
    with open(src_label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            old_cls = int(parts[0])

            if old_cls in weapon_ids:
                parts[0] = "0"
                new_lines.append(" ".join(parts))

    return new_lines


def process_split(img_dir, lbl_dir, out_img_dir, out_lbl_dir,
                  weapon_ids, prefix):
    """Process one split (train/valid/test) from one dataset source."""
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    if not os.path.isdir(img_dir):
        print(f"  [SKIP] {img_dir} not found")
        return 0

    count = 0
    img_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    for img_path in sorted(glob.glob(os.path.join(img_dir, "*"))):
        ext = os.path.splitext(img_path)[1].lower()
        if ext not in img_extensions:
            continue

        stem = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(lbl_dir, stem + ".txt")

        new_lines = relabel_file(lbl_path, weapon_ids)

        # Keep images even if they have no annotations (background / negative samples)
        new_stem = f"{prefix}_{stem}"
        dst_img = os.path.join(out_img_dir, new_stem + ext)
        dst_lbl = os.path.join(out_lbl_dir, new_stem + ".txt")

        shutil.copy2(img_path, dst_img)
        with open(dst_lbl, "w") as f:
            f.write("\n".join(new_lines) + "\n" if new_lines else "")

        count += 1

    return count


def process_zenodo(zenodo_root, **kwargs):
    """Process the Zenodo dataset.
    
    Zenodo layout: <root>/images/{train,val,test}/ and <root>/labels/{train,val,test}/
    """
    print("\n=== Processing Zenodo dataset ===")

    # Find the folder containing both 'images' and 'labels' dirs
    dataset_root = None
    for root, dirs, files in os.walk(ZENODO_DIR):
        if "images" in dirs and "labels" in dirs:
            dataset_root = root
            break
    if dataset_root is None:
        print("  ERROR: Could not find images/labels folders in Zenodo dataset")
        return 0

    print(f"  Detected root: {dataset_root}")

    total = 0
    for split in ["train", "val", "test"]:
        img_dir = os.path.join(dataset_root, "images", split)
        lbl_dir = os.path.join(dataset_root, "labels", split)

        out_split = "valid" if split == "val" else split
        out_img = os.path.join(OUTPUT_DIR, out_split, "images")
        out_lbl = os.path.join(OUTPUT_DIR, out_split, "labels")

        n = process_split(img_dir, lbl_dir, out_img, out_lbl,
                          ZENODO_WEAPON_IDS, "zen")
        print(f"  {split}: {n} images")
        total += n

    print(f"  Total Zenodo: {total}")
    return total


def process_roboflow(roboflow_root, **kwargs):
    """Process the Roboflow dataset.
    
    Roboflow YOLOv8 layout is typically: <root>/{train,valid,test}/{images,labels}/
    """
    print("\n=== Processing Roboflow dataset ===")
    roboflow_root = find_roboflow_structure(ROBOFLOW_DIR)
    print(f"  Detected root: {roboflow_root}")

    # Check if Roboflow data.yaml exists to verify class order
    rf_yaml = os.path.join(roboflow_root, "data.yaml")
    if os.path.exists(rf_yaml):
        with open(rf_yaml) as f:
            rf_data = yaml.safe_load(f)
        names = rf_data.get("names", [])
        print(f"  Roboflow classes from data.yaml: {names}")
        # Rebuild weapon IDs based on actual class names
        weapon_names = {"pistol", "knife", "gun", "rifle", "machete", "weapon"}
        actual_weapon_ids = set()
        for i, name in enumerate(names):
            if name.lower() in weapon_names:
                actual_weapon_ids.add(i)
        if actual_weapon_ids:
            print(f"  Auto-detected weapon class IDs: {actual_weapon_ids}")
            weapon_ids = actual_weapon_ids
        else:
            print(f"  WARNING: No weapon classes auto-detected, using default {ROBOFLOW_WEAPON_IDS}")
            weapon_ids = ROBOFLOW_WEAPON_IDS
    else:
        weapon_ids = ROBOFLOW_WEAPON_IDS

    total = 0
    for split in ["train", "valid", "test"]:
        # Try layout: split/images/
        img_dir = os.path.join(roboflow_root, split, "images")
        lbl_dir = os.path.join(roboflow_root, split, "labels")

        # Try "val" variant
        if not os.path.isdir(img_dir):
            alt = "val" if split == "valid" else split
            img_dir = os.path.join(roboflow_root, alt, "images")
            lbl_dir = os.path.join(roboflow_root, alt, "labels")

        # Try flipped layout: images/split/
        if not os.path.isdir(img_dir):
            img_dir = os.path.join(roboflow_root, "images", split)
            lbl_dir = os.path.join(roboflow_root, "labels", split)

        out_split = "valid" if split == "val" else split
        out_img = os.path.join(OUTPUT_DIR, out_split, "images")
        out_lbl = os.path.join(OUTPUT_DIR, out_split, "labels")

        n = process_split(img_dir, lbl_dir, out_img, out_lbl,
                          weapon_ids, "rf")
        print(f"  {split}: {n} images")
        total += n

    print(f"  Total Roboflow: {total}")
    return total


def create_data_yaml():
    """Create the data.yaml file for YOLO training."""
    data = {
        "path": OUTPUT_DIR,
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": 1,
        "names": {0: "weapon"},
    }

    yaml_path = os.path.join(OUTPUT_DIR, "data.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    print(f"\ndata.yaml written to {yaml_path}")
    return yaml_path


def print_stats():
    """Print dataset statistics."""
    print("\n=== Merged Dataset Statistics ===")
    for split in ["train", "valid", "test"]:
        img_dir = os.path.join(OUTPUT_DIR, split, "images")
        lbl_dir = os.path.join(OUTPUT_DIR, split, "labels")
        if not os.path.isdir(img_dir):
            continue

        n_imgs = len([f for f in os.listdir(img_dir)
                      if os.path.splitext(f)[1].lower() in (".jpg", ".jpeg", ".png", ".bmp")])

        weapon_count = 0
        empty_count = 0
        for lbl_file in glob.glob(os.path.join(lbl_dir, "*.txt")):
            with open(lbl_file) as f:
                lines = [l.strip() for l in f if l.strip()]
            if not lines:
                empty_count += 1
                continue
            weapon_count += len(lines)

        print(f"  {split:6s}: {n_imgs} images | "
              f"weapon boxes: {weapon_count} | "
              f"background (empty label): {empty_count}")


def main():
    # Clean output
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)

    process_zenodo(ZENODO_DIR)
    process_roboflow(ROBOFLOW_DIR)
    create_data_yaml()
    print_stats()

    print(f"\n=== Merged dataset ready at: {OUTPUT_DIR} ===")


if __name__ == "__main__":
    main()