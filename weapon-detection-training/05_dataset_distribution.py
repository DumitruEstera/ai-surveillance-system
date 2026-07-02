#!/usr/bin/env python3
"""
05_dataset_distribution.py
Build a bar chart of the class distribution across the two raw datasets (before
relabeling) used to train the weapon detection model.

The per-category distribution cannot be reconstructed from the merged dataset,
because 02_relabel_and_merge.py collapses every class to id 0 ("weapon"), so we
count directly from the raw labels:
    - Zenodo "Dangerous Items":  0=machete, 1=knife, 2=baseball_bat, 3=rifle, 4=gun
    - Roboflow "SOHAS":          0=pistol, 1=knife (the rest are non-weapons)

The counted unit is the number of bounding boxes (instances) per category,
aggregated over all splits (train/valid/test).

Usage:
    python 05_dataset_distribution.py
    python 05_dataset_distribution.py --base-dir /dev/shm/estera --out distributie_clase.png
"""

import os
import glob
import argparse
from collections import Counter

import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# id -> class name mapping for each raw dataset
ZENODO_NAMES = {0: "machete", 1: "knife", 2: "baseball_bat", 3: "rifle", 4: "gun"}
# SOHAS has 6 classes: pistol and knife are weapons (kept in the model), the rest
# are non-weapons dropped at merge time. We show them all to illustrate the filter.
ROBOFLOW_ALL_NAMES = {0: "pistol", 1: "knife", 2: "smartphone",
                      3: "billete", 4: "monedero", 5: "tarjeta"}
# Weapon/non-weapon is decided by NAME, not id: the Roboflow export reorders
# classes alphabetically, so ids no longer match pistol/knife.
WEAPON_WORDS = {"pistol", "knife", "gun", "rifle", "machete", "baseball_bat", "weapon"}

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def find_dataset_root(base, with_images_labels=True):
    """Auto-detect the root of a YOLO dataset (same style as in 02_*)."""
    for root, dirs, _ in os.walk(base):
        if with_images_labels and "images" in dirs and "labels" in dirs:
            return root
        if "train" in dirs or "valid" in dirs or "val" in dirs or "test" in dirs:
            return root
    return base


def iter_label_files(root):
    """Return all .txt label files from any common YOLO layout."""
    seen = set()
    for path in glob.glob(os.path.join(root, "**", "*.txt"), recursive=True):
        # Skip files that are not labels (e.g. README, requirements)
        base = os.path.basename(path).lower()
        if base in ("readme.txt", "requirements.txt", "classes.txt"):
            continue
        if path not in seen:
            seen.add(path)
            yield path


def count_classes(root, id2name):
    """Count instances (boxes) per class, keeping only ids present in id2name."""
    counter = Counter()
    for lbl in iter_label_files(root):
        try:
            with open(lbl, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    cls = int(line.split()[0])
                    if cls in id2name:
                        counter[id2name[cls]] += 1
        except (ValueError, IndexError):
            continue
    return counter


def maybe_load_roboflow_names(root):
    """If SOHAS has a data.yaml, use the real class names/order from it."""
    yaml_path = os.path.join(root, "data.yaml")
    if not os.path.exists(yaml_path):
        return dict(ROBOFLOW_ALL_NAMES)
    with open(yaml_path) as f:
        names = yaml.safe_load(f).get("names", [])
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    found = {i: str(n) for i, n in enumerate(names)}
    return found or dict(ROBOFLOW_ALL_NAMES)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default="/dev/shm/estera",
                    help="Folderul ce contine zenodo_raw/ si roboflow_raw/")
    ap.add_argument("--out", default="distributie_clase_arme.png")
    args = ap.parse_args()

    zen_root = find_dataset_root(os.path.join(args.base_dir, "zenodo_raw"))
    rf_root = find_dataset_root(os.path.join(args.base_dir, "roboflow_raw"))

    print(f"Zenodo root:   {zen_root}")
    print(f"Roboflow root: {rf_root}")

    rf_names = maybe_load_roboflow_names(rf_root)
    zen_counts = count_classes(zen_root, ZENODO_NAMES)
    rf_counts = count_classes(rf_root, rf_names)

    # Build the lists for a single bar chart. Order: the 5 Zenodo classes first
    # (all weapons), then the 6 SOHAS ones. "kind" drives the bar color:
    # zen_weapon / rf_weapon / rf_dropped.
    bars = []  # (label, value, source, kind)
    for name in ZENODO_NAMES.values():
        bars.append((name, zen_counts.get(name, 0),
                     "Zenodo (Dangerous Items)", "zen_weapon"))
    for idx, name in rf_names.items():
        kind = "rf_weapon" if str(name).lower() in WEAPON_WORDS else "rf_dropped"
        bars.append((name, rf_counts.get(name, 0), "Roboflow (SOHAS)", kind))

    labels = [b[0] for b in bars]
    values = [b[1] for b in bars]
    sources = [b[2] for b in bars]
    kinds = [b[3] for b in bars]

    print("\n=== Distributie pe categorii (box-uri) ===")
    for lbl, val, src, kind in bars:
        flag = "" if kind != "rf_dropped" else "  (eliminata la merge)"
        print(f"  {lbl:14s} {val:7d}   [{src}]{flag}")
    n_weapon = sum(v for v, k in zip(values, kinds) if k != "rf_dropped")
    print(f"  {'TOTAL arme':14s} {n_weapon:7d}")
    print(f"  {'TOTAL':14s} {sum(values):7d}")

    # Disambiguate labels: "knife" appears in both datasets -> add a source suffix
    seen = Counter(labels)
    dup = {k for k, v in seen.items() if v > 1}
    x_labels = []
    for lbl, src in zip(labels, sources):
        if lbl in dup:
            tag = "Zenodo" if src.startswith("Zenodo") else "SOHAS"
            x_labels.append(f"{lbl}\n({tag})")
        else:
            x_labels.append(lbl)

    # Colors by kind: Zenodo weapons, SOHAS weapons, dropped SOHAS non-weapons
    color_map = {
        "zen_weapon": ("#2c7fb8", "Arme — Zenodo (Dangerous Items)"),
        "rf_weapon":  ("#d95f0e", "Arme — Roboflow (SOHAS)"),
        "rf_dropped": ("#bdbdbd", "Non-arme SOHAS"),
    }
    colors = [color_map[k][0] for k in kinds]

    fig, ax = plt.subplots(figsize=(11, 6))
    x = range(len(labels))
    bars_artist = ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.6)

    ax.set_xticks(list(x))
    ax.set_xticklabels(x_labels, fontsize=10)
    ax.set_ylabel("Numar de instante (bounding box-uri)", fontsize=11)
    ax.set_title("Distributia claselor in setul de date reunit pentru detectia de arme",
                 fontsize=12, pad=14)

    # Value labels above the bars
    for rect, val in zip(bars_artist, values):
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height(),
                f"{val:,}", ha="center", va="bottom", fontsize=9)

    # Manual legend by kind (only kinds present in the data)
    from matplotlib.patches import Patch
    present = [k for k in color_map if k in set(kinds)]
    legend = [Patch(facecolor=color_map[k][0], edgecolor="black",
                    label=color_map[k][1]) for k in present]
    ax.legend(handles=legend, title="Categorie", fontsize=9)

    ax.spines[["top", "right"]].set_visible(False)
    ax.margins(y=0.12)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"\nGrafic salvat in: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
