#!/usr/bin/env python3
"""
04_evaluate_test.py
Evaluates the fine-tuned YOLOv8 weapon model on the held-out TEST split.

Unlike the metrics reported during training (which are computed on the `valid`
split every epoch), this script runs a one-off evaluation on the `test` split
declared in data.yaml -- data the model never saw during training, validation
or early stopping. It reports precision, recall, mAP@0.5 and mAP@0.5:0.95.

Reuses the exact cluster paths from 03_train_yolo.py:
    - dataset:  /dev/shm/estera/weapon_dataset/data.yaml
    - weights:  ~/test/weapon_detection2/weapon_yolo/weights/best.pt

Prerequisite: the dataset must exist on disk. Since /dev/shm is tmpfs and is
wiped on reboot, re-run 01_download_datasets.py and 02_relabel_and_merge.py
first if the folder is gone.

Usage:
    python 04_evaluate_test.py
    python 04_evaluate_test.py --weights ~/test/weapon_detection2/best.pt
    python 04_evaluate_test.py --split valid   # sanity-check against training numbers
"""

import os
import argparse
import subprocess


DATA_YAML = "/dev/shm/estera/weapon_dataset/data.yaml"
SAVE_DIR = os.path.expanduser("~/test/weapon_detection2")
DEFAULT_WEIGHTS = os.path.join(SAVE_DIR, "weapon_yolo", "weights", "best.pt")


def pick_gpu():
    """Pick the GPU with the most free memory (same logic as 03_train_yolo.py)."""
    try:
        import pynvml
        pynvml.nvmlInit()
        best_gpu = 0
        best_free = 0
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            info = pynvml.nvmlDeviceGetMemoryInfo(h)
            if info.free > best_free:
                best_free = info.free
                best_gpu = i
        pynvml.nvmlShutdown()
        print(f"Selected GPU {best_gpu} ({best_free / 1e9:.1f} GB free)")
        return best_gpu
    except Exception:
        # Fallback: parse nvidia-smi
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index,memory.free",
                 "--format=csv,noheader,nounits"],
                text=True,
            )
            best_gpu, best_free = 0, 0
            for line in out.strip().split("\n"):
                idx, free = line.split(",")
                idx, free = int(idx.strip()), int(free.strip())
                if free > best_free:
                    best_free = free
                    best_gpu = idx
            print(f"Selected GPU {best_gpu} ({best_free} MiB free)")
            return best_gpu
        except Exception:
            print("Could not detect GPUs, defaulting to GPU 0")
            return 0


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the weapon YOLOv8 model on the test split")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS,
                        help=f"Path to model weights (default: {DEFAULT_WEIGHTS})")
    parser.add_argument("--data", default=DATA_YAML,
                        help=f"Path to data.yaml (default: {DATA_YAML})")
    parser.add_argument("--split", default="test", choices=["test", "valid", "train"],
                        help="Dataset split to evaluate on (default: test)")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--conf", type=float, default=0.001,
                        help="Confidence threshold for evaluation (default: 0.001, "
                             "the standard for mAP computation)")
    parser.add_argument("--iou", type=float, default=0.7,
                        help="NMS IoU threshold (default: 0.7)")
    parser.add_argument("--gpu", type=int, default=None,
                        help="Force a specific GPU index")
    args = parser.parse_args()

    # Fail early with a clear message if inputs are missing
    if not os.path.exists(args.weights):
        print(f"ERROR: weights not found: {args.weights}")
        return
    if not os.path.exists(args.data):
        print(f"ERROR: data.yaml not found: {args.data}")
        print("       The dataset may have been wiped from /dev/shm. Re-run")
        print("       01_download_datasets.py and 02_relabel_and_merge.py first.")
        return

    gpu_id = args.gpu if args.gpu is not None else pick_gpu()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    from ultralytics import YOLO

    print(f"\nLoading weights: {args.weights}")
    print(f"Dataset        : {args.data}")
    print(f"Split          : {args.split}")

    model = YOLO(args.weights)

    metrics = model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        conf=args.conf,
        iou=args.iou,
        device=0,  # after CUDA_VISIBLE_DEVICES this is the selected GPU
        project=SAVE_DIR,
        name=f"eval_{args.split}",
        exist_ok=True,
        plots=True,   # writes PR curve, confusion matrix, etc.
        verbose=True,
    )

    # box.mp = mean precision, box.mr = mean recall (single class -> the weapon class)
    precision = metrics.box.mp
    recall = metrics.box.mr
    map50 = metrics.box.map50
    map5095 = metrics.box.map

    print("\n" + "=" * 48)
    print(f"  Rezultate pe partitia '{args.split}'")
    print("=" * 48)
    print(f"  {'Precizie (precision)':30s} {precision:.3f}")
    print(f"  {'Sensibilitate (recall)':30s} {recall:.3f}")
    print(f"  {'mAP@0.5':30s} {map50:.3f}")
    print(f"  {'mAP@0.5:0.95':30s} {map5095:.3f}")
    print("=" * 48)
    print(f"\nPloturi si rezultate salvate in: "
          f"{os.path.join(SAVE_DIR, f'eval_{args.split}')}")


if __name__ == "__main__":
    main()
