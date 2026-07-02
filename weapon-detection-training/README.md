# Weapon Detection — YOLOv8 Fine-Tuning

## Objective

Train a YOLOv8 model to detect weapons (a single class: `weapon`) as part of an
AI-based video-surveillance system.

## Datasets

Two public datasets were used, combined into one:

| Dataset | Source | Images | Original classes |
|---------|--------|--------|------------------|
| Dangerous Items | [Zenodo](https://zenodo.org/records/16422779) | 8,478 | machete, knife, baseball bat, rifle, gun |
| SOHAS Weapon Detection | [Roboflow](https://universe.roboflow.com/aditikulkarni-1710-gmail-com/sohas-weapon-detection) | 5,858 | pistol, knife, smartphone, billete, monedero, tarjeta |

**Combined dataset**: 14,336 images, 12,592 bounding boxes.

## Preprocessing

1. **Relabel**: every weapon class from both datasets is remapped to class `0`
   (weapon). The non-weapon Roboflow classes (smartphone, billete, monedero,
   tarjeta) are dropped.
2. **Merge**: the two datasets are combined into a single YOLO structure
   (train/valid/test), with filename prefixes (`zen_`, `rf_`) to avoid collisions.
3. **Negative samples**: images without annotations (background) are kept as
   negative examples.

## Training

- **Model**: YOLOv8m pretrained on COCO
- **GPU**: NVIDIA A100 80GB
- **Parameters**: 100 epochs, batch 16, 640px, AdamW optimizer, cosine LR schedule
- **Early stopping**: patience 20

## File layout

```
01_download_datasets.py    — download the two datasets into /dev/shm/estera/
02_relabel_and_merge.py    — relabel + merge into a single dataset
03_train_yolo.py           — YOLOv8 training with automatic GPU selection
04_evaluate_test.py        — evaluate the trained model on the test split
05_dataset_distribution.py — plot the class distribution of the raw datasets
```

Trained weights (`best.pt`, `last.pt`) are not committed — they exceed GitHub's
100 MB per-file limit and are distributed separately.

## Usage

```bash
# Download datasets
python 01_download_datasets.py --roboflow-api-key YOUR_KEY

# Relabel + merge
python 02_relabel_and_merge.py

# Train
nohup python 03_train_yolo.py > train_log.txt 2>&1 &
```

## Inference

```python
from ultralytics import YOLO

model = YOLO("best.pt")
results = model("image.jpg")
results[0].show()
```
