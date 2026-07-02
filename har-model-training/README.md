# Human Action Recognition — SlowFast Training

Training pipeline for a **SlowFast-R50** action-recognition model (Kinetics-400
pretrained, fine-tuned) on surveillance clips, for three action classes.

## Action classes

| Class | Source |
|---|---|
| `normal` | RWF-2000 (NonFight) |
| `fight` | RWF-2000 (Fight) |
| `vandalism` | Custom dataset |

## Project structure

```
configs/
  config.py            # Hyperparameters, paths, class definitions
data/
  dataset.py           # Video dataset loader for SlowFast
models/
  slowfast_model.py    # SlowFast with custom head + freeze/unfreeze
utils/
  metrics.py           # Accuracy, F1, confusion matrix, early stopping
train.py               # Fine-tuning script (2-phase: frozen → full)
prepare_data.py        # Dataset verifier / folder-structure helper
kaggle_RWF2000.py      # RWF-2000 download helper (kagglehub)
requirements.txt
README.md
```

## Setup

### 1. Install dependencies

```bash
python -m venv venv
source venv/bin/activate

# Install PyTorch (visit https://pytorch.org for your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt
```

### 2. Prepare the dataset

Download **RWF-2000** (fight vs. non-fight) and add your own **vandalism** clips.
The RWF-2000 helper uses `kagglehub`:

```bash
python kaggle_RWF2000.py
```

Organize the videos into the folder structure below (classes are auto-discovered
from the folder names in `data/train/`):

```
data/
  train/
    normal/     *.avi / *.mp4
    fight/      ...
    vandalism/  ...
  test/
    normal/     ...
    fight/      ...
    vandalism/  ...
```

Create the empty structure and verify your data:

```bash
python prepare_data.py --create      # create empty class folders
python prepare_data.py --verify      # check videos are in place
```

## Training

```bash
python train.py                                    # defaults from configs/config.py
python train.py --epochs 50 --batch_size 4 --lr 5e-4 --freeze_epochs 10 --device cuda
python train.py --resume output/checkpoints/last_model.pth
```

### Training strategy

A **two-phase approach**:

1. **Phase 1 (frozen backbone):** only the new classification head is trained for
   `freeze_backbone_epochs`, so the head learns to map Kinetics-400 features to the
   action classes without corrupting the pretrained backbone.
2. **Phase 2 (full fine-tuning):** the whole model is unfrozen; the backbone gets a
   10× lower learning rate than the head to preserve general features while adapting
   to the surveillance domain.

Additional techniques: class-weighted sampling, label smoothing, gradient clipping,
mixed-precision training (AMP), early stopping.

Training curves and the confusion matrix are written to `output/`.

## Key configuration (configs/config.py)

| Parameter | Default | Description |
|---|---|---|
| `clip_duration_sec` | 2.0 | Duration of each clip fed to SlowFast |
| `num_frames_slow` | 8 | Frames for the SlowFast slow pathway |
| `num_frames_fast` | 32 | Frames for the SlowFast fast pathway |
| `crop_size` | 224 | Spatial resolution |
| `batch_size` | 8 | Training batch size (reduce if OOM) |
| `freeze_backbone_epochs` | 5 | Epochs with frozen backbone |

## Notes

- Trained weights are **not committed** (they exceed GitHub's 100 MB per-file limit)
  and are distributed separately. They are written to `output/checkpoints/`.
- Reduce `batch_size` to 2–4 on limited GPU memory, or run on CPU with
  `--device cpu` (much slower).
