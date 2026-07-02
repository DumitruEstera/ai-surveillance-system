# Surveillance Platform

Integration layer of an AI video-surveillance system: a FastAPI backend that
orchestrates several detectors over live camera or video streams, plus a React
frontend for operators. Built as part of a bachelor's thesis.

## Detectors

| Module | Models / techniques |
|---|---|
| Facial recognition | InsightFace (buffalo_s, 512-d embeddings) + FAISS index |
| Demographics (age / gender / emotion) | InsightFace genderage + FER |
| License-plate recognition | YOLOv8 detection + EasyOCR, with temporal voting |
| Fire & smoke detection | YOLOv8 + per-class confidence / size / colour filters + temporal voting |
| Weapon detection | YOLOv8m (single `weapon` class) |
| Human action recognition | SlowFast-R50 (Kinetics-400 pretrained, fine-tuned) |

The weapon and action-recognition models are trained in separate repositories
(see [Related repositories](#related-repositories)).

## Architecture

The backend is **multi-threaded, not async**: a frame-capture thread feeds
per-detector worker threads through `queue.Queue` hand-offs; a merger thread
combines the per-detector results and a broadcaster pushes annotated frames to
each connected WebSocket client. Blocking model calls run on worker threads, never
on the FastAPI event loop.

```
Camera / video ─► capture ─► [face | plate | fire | weapon | HAR] workers ─► merge ─► /ws broadcast
```

- Face embeddings live in a FAISS index that is rebuilt from PostgreSQL on startup.
- Authentication is JWT-based; clients connect over REST and a `/ws` WebSocket.

## Requirements

- Python 3.10+
- PostgreSQL (backing store for people, plates, logs and alarms)
- NVIDIA GPU with CUDA for real-time performance (InsightFace uses the ONNX Runtime
  `CUDAExecutionProvider`; it falls back to CPU but is much slower)
- Node.js 18+ for the frontend

## Setup

### Backend

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# then edit .env with your JWT secret and PostgreSQL credentials
```

On first run the backend creates an `admin` user. Its password comes from the
`ADMIN_PASSWORD` environment variable; if unset, a random one is generated and
printed once in the startup logs. Change it after the first login.

Place the trained model files (not tracked in git — see below) under:

```
models/har/best_model.pth
models/fire_and_smoke/best.pt
models/weapon/best.pt
models/license_plate/best.pt
```

Run the backend (FastAPI, port 8000):

```bash
python app.py
```

### Frontend

```bash
cd frontend
npm install
npm start          # dev server
npm run build      # production build
```

## Models

Trained weights are **not committed** (they exceed GitHub's 100 MB per-file limit)
and are distributed separately. Drop them into `models/` as shown above before
starting the backend.

## Repository layout

```
app.py                          FastAPI backend + multi-threaded pipeline
database_manager.py             PostgreSQL access layer
faiss_index.py                  FAISS index for face embeddings
facial_recognition_system.py    InsightFace recognition + demographics
license_plate_recognition_system.py
fire_detection_system.py
weapon_detection_system.py
har_system.py                   SlowFast action recognition
frontend/                       React + Tailwind operator UI
```

## Related repositories

- **HAR model training** — SlowFast fine-tuning pipeline for human action recognition.
- **Weapon detection training** — YOLOv8m fine-tuning pipeline for the weapon class.
- **Evaluation and testing** — per-module accuracy scripts, an end-to-end test
  harness and an inference benchmark for this application. Expects this app to sit
  in a sibling `surveillance-platform/` directory.
