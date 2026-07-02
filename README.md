# A Multi-Task Deep Learning Framework for Real-Time Intelligent Video Surveillance with Temporal Event Validation

This repository contains the implementation used for the paper *"A Multi-Task Deep
Learning Framework for Real-Time Intelligent Video Surveillance with Temporal Event
Validation"*.

The project presents a unified multi-task deep-learning framework that
simultaneously performs face recognition with zone-based authorization, automatic
license plate recognition, weapon detection, fire and smoke detection, and human
action recognition on a shared GPU. The five heterogeneous vision tasks run
concurrently over the same video streams through a multi-threaded pipeline that
keeps per-frame latency below 100 ms on commodity hardware.

The central idea is that a security detector on a continuous video stream is only
as useful as it is *reliable*, and reliability is governed far more by how per-frame
outputs are validated over time than by raw accuracy on isolated images. Every
detector is therefore wrapped in a **temporal event-validation** layer —
multi-frame confirmation, confidence-weighted temporal voting, and cascaded
filtering — that converts noisy frame-level detections into reliable security
events. Two of the models are trained specifically for this work to address
scenarios under-represented in public datasets: a single-class weapon detector and
a SlowFast-R50 action recognizer trained on a purpose-built vandalism dataset.

## Main Contributions

- A dedicated **weapon detector**: a single-class YOLOv8m model fine-tuned on a
  merged, relabeled dataset built from two heterogeneous public sources, with
  non-weapon images retained as negative examples to suppress false positives,
  reaching a mAP@0.5 of 0.947.
- A custom **human action recognizer**: a SlowFast-R50 network fine-tuned in two
  stages for the classes {normal, fight, vandalism}, reaching 94.33% validation
  accuracy.
- A new **vandalism dataset**: a purpose-built corpus of 614 manually validated
  video clips for a category that public datasets under-represent.
- A **temporal event-validation architecture** (multi-frame confirmation,
  confidence-weighted voting, cascaded filtering, IoU tracking) that turns noisy
  detections into reliable events, quantified experimentally per module.
- A **unified real-time multi-task pipeline** running five tasks concurrently on a
  single shared GPU, with a seven-thread "last-available-frame" fusion design that
  keeps per-frame latency below 100 ms.
- A **systematic evaluation** of each module on independent public datasets (LFW,
  D-Fire, FIRESENSE, UCF-Crime), plus integrated end-to-end functional and stress
  testing under maximum concurrent load.

## Repository Structure

The repository is organized into four independent sub-projects, each with its own
`README.md`:

```
ai-surveillance-system/
│
├── surveillance-platform/          The application (FastAPI backend + React frontend)
│   ├── app.py                        Multi-threaded, multi-task inference pipeline
│   ├── facial_recognition_system.py  InsightFace recognition + demographics
│   ├── license_plate_recognition_system.py
│   ├── fire_detection_system.py
│   ├── weapon_detection_system.py
│   ├── har_system.py                 SlowFast action recognition
│   ├── faiss_index.py                FAISS index for face embeddings
│   ├── database_manager.py           PostgreSQL access layer
│   └── frontend/                     React + Tailwind operator interface
│
├── har-model-training/             SlowFast-R50 action recognition training
│   ├── train.py                      Two-phase fine-tuning (frozen head → full)
│   ├── models/slowfast_model.py      SlowFast with custom 3-class head
│   ├── data/dataset.py               Video dataset loader
│   └── configs/config.py
│
├── weapon-detection-training/      YOLOv8m weapon detection training
│   ├── 01_download_datasets.py
│   ├── 02_relabel_and_merge.py
│   ├── 03_train_yolo.py
│   ├── 04_evaluate_test.py
│   └── 05_dataset_distribution.py
│
├── evaluation-and-testing/         Accuracy evaluation, test harness, benchmark
│   ├── evaluation/                   Per-module accuracy scripts (LFW, plate, fire, HAR)
│   ├── testing/                      End-to-end harness (drives the backend over HTTP)
│   └── benchmark_inference.py        Per-detector inference-latency benchmark
│
├── README.md
└── .gitignore
```

## Method Overview

Each frame is dispatched, in parallel, to five detectors that run concurrently on a
single shared GPU (two of them — action recognition and weapon detection — are
trained specifically for this work). Rather than emitting raw per-frame detections,
every detector feeds the temporal event-validation layer before its output becomes
an event.

```
Video streams (webcam / RTSP)
        ↓
Frame capture (per-detector clone)
        ↓
Five detectors, concurrent on a shared GPU:
    - Face & demographics   InsightFace (ArcFace 512-d) + FAISS, double-threshold decision
    - License plate         YOLOv8 + EasyOCR, confidence-weighted voting
    - Fire & smoke          YOLOv10, five-level filtering cascade
    - Action recognition    SlowFast-R50  (custom-trained)
    - Weapon detection      YOLOv8m       (custom-trained)
        ↓
Temporal Event Validation
    (multi-frame confirmation, confidence-weighted voting, cascaded filtering, IoU tracking)
        ↓
Fusion (last-available-frame policy)
        ↓
Alarms + logs (PostgreSQL, deduplicated)   ·   WebSocket (annotated frames)
```

The models integrated in the pipeline are:

- **Face & demographics** — InsightFace `buffalo_s` (detection, ArcFace 512-d
  embeddings, and age/gender in one pass) with a FAISS `IndexFlatL2` index, an
  open-set double-threshold decision rule (L2 < 1.0 **and** cosine ≥ 0.4), and
  per-person spatial zone authorization; emotion via a Mini-Xception (FER) model.
- **License plate** — YOLOv8 detection + EasyOCR reading, with an image-enhancement
  chain (bicubic up-scaling, bilateral filter, CLAHE, unsharp mask), IoU tracking,
  and confidence-weighted temporal voting validated against the Romanian plate format.
- **Fire & smoke** — a pretrained YOLOv10 detector wrapped in a five-level cascade
  (per-class confidence, size plausibility, HSV color, temporal confirmation,
  per-location cooldown).
- **Action recognition (custom-trained)** — SlowFast-R50 fine-tuned for {normal,
  fight, vandalism}; clip-level classification over a 2 s window.
- **Weapon detection (custom-trained)** — single-class YOLOv8m consumed through a
  stricter secondary confidence filter and multi-frame confirmation.

## Datasets

The datasets are **not included** in this repository because of their size and
licensing restrictions.

Model training uses:

- **Dangerous Items** (Zenodo, 8478 images, five weapon classes) and **SOHAS Weapon
  Detection** (Roboflow, 5858 images) — merged and relabeled into a single YOLO
  dataset of 14,336 images and 12,592 weapon bounding boxes for the weapon detector.
- **RWF-2000** (for the `normal` and `fight` classes) plus a **purpose-built
  vandalism dataset** (614 manually validated clips) — 2538 videos in total for the
  SlowFast action recognizer.

Module evaluation uses:

- **LFW** (Labeled Faces in the Wild) for face recognition.
- **D-Fire** (images) and **FIRESENSE** (video) for fire and smoke detection.
- **UCF-Crime** for action-recognition generalization.
- A public Romanian license-plate dataset for plate detection/recognition.

Trained model weights and large datasets are not tracked by Git.

## Installation

Each sub-project has its own `requirements.txt`. Create a Python environment per
sub-project (or a shared one) and install its dependencies, e.g. for the application:

```bash
cd surveillance-platform
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # then set your JWT secret, PostgreSQL and admin credentials
```

The application requires a running PostgreSQL instance and an NVIDIA GPU with CUDA
for real-time performance. The main technologies are FastAPI, React, InsightFace,
Ultralytics YOLO, EasyOCR, FAISS, PyTorch, and PyTorchVideo (SlowFast).

## Training

### Weapon detector (YOLOv8m)

```bash
cd weapon-detection-training
python 01_download_datasets.py --roboflow-api-key YOUR_KEY
python 02_relabel_and_merge.py
python 03_train_yolo.py
```

All weapon classes from both sources are remapped to a single `weapon` class; the
non-weapon SOHAS images are kept with empty labels as background negatives. Training
ran for 100 epochs (early stopping, patience 20) at batch size 16 and 640×640 input,
with the AdamW optimizer, on an NVIDIA A100 GPU.

### Action recognizer (SlowFast-R50)

```bash
cd har-model-training
python prepare_data.py --verify
python train.py --epochs 50 --batch_size 4 --lr 5e-4 --freeze_epochs 10 --device cuda
```

Fine-tuning is two-phase: the Kinetics-400 backbone is frozen for the first epochs
while the new 3-class head is trained, then the whole model is unfrozen with a 10×
lower backbone learning rate.

## Evaluation

The `evaluation-and-testing/` sub-project evaluates each module and the integrated
system. Its scripts expect the application to sit in the sibling
`surveillance-platform/` directory (override with `SURVEILLANCE_APP_DIR`).

```bash
cd evaluation-and-testing
python evaluation/evaluate_facial_lfw.py --all          # face recognition on LFW
python evaluation/plate_02_evaluate.py                  # license plate reading
python evaluation/fire_01_evaluate_images.py            # fire/smoke on images
python evaluation/har_01_evaluate.py                    # action recognition
python benchmark_inference.py --source /path/clip.mp4   # inference latency
python testing/run_stress_test.py /path/clip.mp4        # end-to-end stress test
```

## Results Summary

### Custom-trained models

Weapon detector (test partition, 2191 unseen images):

| Metric | Validation | Test |
|---|---|---|
| Precision | 0.937 | 0.943 |
| Recall | 0.894 | 0.902 |
| mAP@0.5 | 0.947 | 0.943 |
| mAP@0.5:0.95 | 0.700 | 0.696 |

Action recognizer (SlowFast-R50, validation): overall accuracy **94.33%**,
macro-F1 0.9456.

| Class | Precision | Recall | F1 |
|---|---|---|---|
| normal | 0.920 | 0.947 | 0.933 |
| fight | 0.942 | 0.935 | 0.939 |
| vandalism | 0.982 | 0.949 | 0.965 |

On a UCF-Crime subset (150 clips, 50 per class) the action recognizer reaches 86.7%
accuracy and macro-F1 0.87, generalizing to real surveillance footage unseen in
training.

### Temporal event validation

| Module | Metric | Without validation | With validation |
|---|---|---|---|
| Fire & smoke (video) | False-alarm rate | 52% | **4%** |
| License plate (video) | Exact-match accuracy | 66.7% | **81.8%** |
| Face (LFW) | Identification / false acceptance | — | **97% / 0%** |

### Real-time operation

Per-module inference latency (isolated, RTX 4060 Laptop): face 83 ms, fire/smoke
25 ms, weapon 13 ms, action (forward) 8 ms, plate 4 ms. Under maximum concurrent
load (six simultaneous streams) the system processes about 10 fps with a comfortable
GPU/VRAM reserve, keeping per-frame latency below the 100 ms budget.

## Key Observation

Single-frame and static-image accuracy metrics understate the real behavior of a
surveillance detector. The processing steps that complement the AI models —
temporal voting, multi-frame confirmation, and image enhancement — contribute
decisively to real operating performance: they can look overly conservative on
isolated images, yet on continuous video they sharply reduce false alarms without
compromising sensitivity. System effectiveness therefore comes not only from the
neural detectors but equally from the temporal event-validation layer built around
them.

## Reproducibility Notes

To reproduce the main experiments:

1. Prepare the datasets described above (they are not included).
2. Install the dependencies of the relevant sub-project.
3. Train the weapon and action-recognition models, or place their checkpoints
   locally under the expected paths.
4. Configure and start the application (PostgreSQL + `.env`).
5. Run the evaluation and end-to-end testing scripts.

The repository focuses on the final reproducible code and paper-relevant scripts.
Large datasets, model checkpoints, downloaded data, images and videos are not
included.

## Citation

If this repository is used as part of academic work, please cite the related paper:

E. Dumitru and S. Spînu, "A Multi-Task Deep Learning Framework for Real-Time
Intelligent Video Surveillance with Temporal Event Validation."

## Author

Estera Dumitru

Faculty of Information Systems and Cyber Security, Military Technical Academy
"Ferdinand I", Bucharest, Romania.
