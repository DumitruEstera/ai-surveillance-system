"""
prepare_data.py — Dataset verifier and organizer.

Verifies that data/train/ and data/test/ contain the expected class folders
with video files. Can also create the directory structure for you.

Expected structure:
    data/
      train/
        normal/     *.avi / *.mp4
        fight/      ...
        vandalism/  ...
      test/
        normal/     ...
        fight/      ...
        vandalism/  ...

Usage:
    python prepare_data.py --verify                    # Check current setup
    python prepare_data.py --create                    # Create empty folder structure
    python prepare_data.py --create --classes normal fight vandalism
    python prepare_data.py --data_root /path/to/data   # Custom data location
"""

import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from configs.config import DATA_ROOT, DEFAULT_ACTION_CLASSES, discover_classes


VIDEO_EXTENSIONS = (".avi", ".mp4", ".mkv", ".mov")


def verify_data(data_root: str):
    """Check which classes and splits are properly set up."""
    print(f"\n{'='*60}")
    print(f"  Dataset Verification")
    print(f"  Data root: {data_root}")
    print(f"{'='*60}\n")

    if not os.path.isdir(data_root):
        print(f"✗ Data root not found: {data_root}")
        print(f"  Run: python prepare_data.py --create")
        return

    classes = discover_classes(data_root)
    print(f"Discovered classes: {classes}\n")

    total_videos = 0
    for split in ["train", "test"]:
        split_dir = os.path.join(data_root, split)
        if not os.path.isdir(split_dir):
            print(f"{split}/ — ✗ NOT FOUND")
            continue

        print(f"{split}/")
        split_total = 0
        for class_name in sorted(os.listdir(split_dir)):
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.isdir(class_dir) or class_name.startswith("."):
                continue

            count = sum(
                1 for f in os.listdir(class_dir)
                if f.lower().endswith(VIDEO_EXTENSIONS)
            )
            split_total += count
            status = "✓" if count > 0 else "⚠ EMPTY"
            marker = "" if class_name in classes else " (not in discovered classes)"
            print(f"   {status} {class_name}: {count} videos{marker}")

        total_videos += split_total
        print(f"   ─ subtotal: {split_total} videos\n")

    print(f"{'─'*60}")
    print(f"Total videos found: {total_videos}")
    if total_videos == 0:
        print(f"\n⚠ No videos found! Place your video files in the class folders.")
        print(f"  Example: {data_root}/train/fight/clip001.avi")
    print()


def create_structure(data_root: str, classes: list):
    """Create the empty folder structure."""
    print(f"\nCreating folder structure in: {data_root}")
    for split in ["train", "test"]:
        for cls in classes:
            path = os.path.join(data_root, split, cls)
            os.makedirs(path, exist_ok=True)
            print(f"   ✓ {split}/{cls}/")

    print(f"\n✓ Done! Place your video files (.avi / .mp4) in the folders above.")
    print(f"  Then run: python prepare_data.py --verify")


def main():
    parser = argparse.ArgumentParser(description="Dataset Preparation & Verification")
    parser.add_argument("--verify", action="store_true", help="Verify dataset setup")
    parser.add_argument("--create", action="store_true", help="Create empty folder structure")
    parser.add_argument("--data_root", type=str, default=DATA_ROOT,
                        help=f"Path to data directory (default: {DATA_ROOT})")
    parser.add_argument("--classes", nargs="+", default=None,
                        help="Class names for --create (default: normal fight vandalism)")
    args = parser.parse_args()

    if args.create:
        classes = args.classes or list(DEFAULT_ACTION_CLASSES)
        create_structure(args.data_root, classes)
        return

    # Default action: verify
    verify_data(args.data_root)


if __name__ == "__main__":
    main()