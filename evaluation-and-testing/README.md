# Evaluation and Testing

Accuracy evaluation, end-to-end testing and inference benchmarking for the
**surveillance-platform** application. These scripts are kept out of the app repo
so the application stays focused on runtime code.

## Contents

```
evaluation/              Per-module accuracy scripts
  evaluate_facial_lfw.py   Facial recognition on LFW
  plate_0{1..4}_*.py       License-plate recognition (images + video)
  fire_0{1,2}_*.py         Fire/smoke on images and video
  har_01_evaluate.py       Human action recognition
testing/                 End-to-end test harness (drives the running backend over HTTP)
  test_harness.py, run_stress_test.py, run_sequential_test.py
benchmark_inference.py   Per-detector inference-latency benchmark
```

## Dependency on the application

`evaluation/` and `benchmark_inference.py` import the application's modules
(`fire_detection_system`, `har_system`, `faiss_index`, …) and use its trained
models under `models/`. They expect the application to live in a **sibling
directory named `surveillance-platform/`**:

```
<parent>/
  surveillance-platform/      # the application
  evaluation-and-testing/     # this folder
```

Override the location with an environment variable if your layout differs:

```bash
export SURVEILLANCE_APP_DIR=/path/to/surveillance-platform
```

Run these scripts with the application's virtual environment (they need the same
dependencies — torch, insightface, ultralytics, easyocr, …) and the trained model
weights in place under `surveillance-platform/models/`.

`testing/` is self-contained: it drives the backend through its REST API and
WebSocket, so it only needs the backend running (`python app.py` on `:8000`) plus
`requests`, `websockets` and `psutil`.

## Usage

```bash
# Accuracy evaluation (examples)
python evaluation/evaluate_facial_lfw.py --all
python evaluation/plate_02_evaluate.py
python evaluation/fire_01_evaluate_images.py

# Inference latency benchmark
python benchmark_inference.py --source /path/clip.mp4 --frames 200

# End-to-end harness (backend must be running)
python testing/run_stress_test.py /path/clip_stress.mp4
python testing/run_sequential_test.py /path/clip_sequential.mp4
```

Generated datasets, caches and result files are written under `evaluation/` and
`testing/results/`.
