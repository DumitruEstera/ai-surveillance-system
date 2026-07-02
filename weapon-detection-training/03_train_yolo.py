#!/usr/bin/env python3
"""
03_train_yolo.py
Fine-tune a YOLOv8 model (pretrained on COCO) on the merged weapon dataset.

Automatically picks the first available GPU (least memory used).
Saves results + best.pt / last.pt to ~/test/weapon_detection2/

Usage:
    python 03_train_yolo.py
    python 03_train_yolo.py --model yolov8m.pt --epochs 150 --batch 32 --imgsz 640
"""

import os
import argparse
import subprocess
import shutil


DATA_YAML = "/dev/shm/estera/weapon_dataset/data.yaml"
SAVE_DIR = os.path.expanduser("~/test/weapon_detection2")


def pick_gpu():
    """Pick the GPU with the most free memory."""
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolov8m.pt",
                        help="Pretrained YOLO model (default: yolov8m.pt)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--gpu", type=int, default=None,
                        help="Force specific GPU index")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from last.pt")
    args = parser.parse_args()

    gpu_id = args.gpu if args.gpu is not None else pick_gpu()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    from ultralytics import YOLO

    os.makedirs(SAVE_DIR, exist_ok=True)

    project_dir = SAVE_DIR
    run_name = "weapon_yolo"

    if args.resume:
        last_pt = os.path.join(project_dir, run_name, "weights", "last.pt")
        if not os.path.exists(last_pt):
            print(f"ERROR: Cannot resume, {last_pt} not found")
            return
        print(f"Resuming from {last_pt}")
        model = YOLO(last_pt)
        model.train(resume=True)
    else:
        print(f"Loading pretrained model: {args.model}")
        model = YOLO(args.model)

        model.train(
            data=DATA_YAML,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            workers=args.workers,
            device=0,  # after CUDA_VISIBLE_DEVICES, this is the selected GPU
            project=project_dir,
            name=run_name,
            exist_ok=True,
            pretrained=True,
            # Augmentation
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=10.0,
            translate=0.1,
            scale=0.5,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.1,
            # Optimization
            optimizer="AdamW",
            lr0=0.001,
            lrf=0.01,
            weight_decay=0.0005,
            warmup_epochs=3,
            cos_lr=True,
            # Misc
            patience=20,
            save=True,
            save_period=10,
            plots=True,
            val=True,
        )

    # Copy best.pt and last.pt to the main save dir for easy access
    weights_dir = os.path.join(project_dir, run_name, "weights")
    for pt_file in ["best.pt", "last.pt"]:
        src = os.path.join(weights_dir, pt_file)
        dst = os.path.join(SAVE_DIR, pt_file)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"Copied {src} -> {dst}")

    print(f"\n=== Training complete ===")
    print(f"Results: {os.path.join(project_dir, run_name)}")
    print(f"Best model: {os.path.join(SAVE_DIR, 'best.pt')}")


if __name__ == "__main__":
    main()