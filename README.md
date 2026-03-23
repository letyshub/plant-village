# Plant Disease Classifier

EfficientNetB0 fine-tuned on the PlantVillage dataset to detect 38 plant diseases from leaf photos.
Includes a Gradio web demo with Grad-CAM heatmaps showing which regions of the image influenced the prediction.

---

## How it works

### The problem

Farmers lose a significant portion of their crops every year to plant diseases. Identifying the right disease early is critical — but it requires expertise that isn't always available locally. This tool lets anyone take a photo of a sick leaf and get an instant diagnosis.

### What you do

1. Open the web app in your browser.
2. Upload a photo of a plant leaf (or drag and drop one).
3. Click **Diagnose**.
4. The app tells you what disease it detected and how confident it is.

### What the app shows you

**Diagnosis** — the name of the detected disease and the confidence level, for example:
> Apple — Cedar Rust, Confidence: 94.2%

**Heatmap** — a colour overlay on your photo highlighting the exact spots the model looked at to make its decision. Red/yellow areas were the most important. This makes the AI's reasoning visible and helps you verify it focused on the right parts of the leaf.

**Top 5 predictions** — a ranked table of the most likely diseases, in case the top result isn't a perfect match.

### How the AI learns

The model was trained on over 54 000 photos of healthy and diseased leaves from the [PlantVillage dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset), covering 14 crop species and 38 disease/healthy categories.

Training happens in two stages:

1. **Learning the basics** — the model starts from a network already trained on millions of everyday photos (ImageNet). It already knows shapes, textures, and edges. In this stage only the final classification layer is trained, so the model learns to map leaf features to disease names.
2. **Fine-tuning** — the deeper layers of the network are gradually unlocked and adjusted on the leaf dataset, allowing the model to pick up on the subtle visual patterns specific to plant diseases.

### Supported crops and diseases

| Crop | Conditions covered |
|------|--------------------|
| Apple | Apple scab, Black rot, Cedar rust, Healthy |
| Corn | Cercospora leaf spot, Common rust, Northern leaf blight, Healthy |
| Grape | Black rot, Esca, Leaf blight, Healthy |
| Tomato | Bacterial spot, Early blight, Late blight, Leaf mold, Septoria, Spider mites, Target spot, Yellow leaf curl virus, Mosaic virus, Healthy |
| Potato | Early blight, Late blight, Healthy |
| … and more | Pepper, Peach, Cherry, Strawberry, Squash, Soybean, Raspberry |

> The model covers 38 classes in total. For the full list see `data/raw/plantvillage/`.

### Limitations

- The model was trained on controlled, close-up leaf photos. Blurry, dark, or distant photos may reduce accuracy.
- It cannot detect diseases not present in the PlantVillage dataset.
- Always confirm a diagnosis with an agronomist before treating crops.

---

## Results

| Model          | Val Accuracy | F1 (macro) | Classes |
|----------------|-------------|------------|---------|
| EfficientNetB0 | —           | —          | 38      |

> Train the model and fill in your results above.

## Quick start

```bash
# 1. Clone and set up environment
git clone <repo-url>
cd plant-village
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
# source .venv/bin/activate
pip install -r requirements.txt

# 2. (Optional) Enable GPU training — ~10-15x faster than CPU
#    See "GPU setup" section below for instructions.

# 3. Download data
python scripts/download_data.py

# 3. Train
python train.py                                           # default config
python train.py configs/train.yaml training.phase1.epochs=5  # override via CLI

# 4. Run the demo
python app.py
python app.py --model outputs/models/plant_disease_model.h5 \
              --classes outputs/models/plant_disease_model_classes.txt
```

## GPU setup

Training on CPU is slow. TensorFlow 2.11+ does not support GPU natively on Windows — use one of the options below.

### Option A — WSL2 (recommended for local GPU)

Installs a Linux environment inside Windows. Your NVIDIA GPU works out of the box via NVIDIA's WSL2 driver.

```powershell
# In PowerShell as Administrator:
wsl --install
```

After reboot, open the Ubuntu terminal and run the project from there:

```bash
cd /mnt/d/Programs/plant-village
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install "tensorflow[and-cuda]==2.20.0"

# Verify GPU is visible:
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
# Expected: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]

python train.py
```

### Option B — Google Colab (zero setup, free)

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. `Runtime → Change runtime type → T4 GPU`
3. Upload the project or clone from GitHub
4. Run training — Colab's T4 GPU is faster than a GTX 1660 Ti

### Option C — train on CPU, but faster

Reduce epochs and batch size in `configs/train.yaml` to shorten training time:

```yaml
training:
  batch_size: 16   # default: 32

  phase1:
    epochs: 5      # default: 10

  phase2:
    epochs: 10     # default: 20
```

---

## Project structure

```
plant-village/
├── configs/
│   └── train.yaml          # all hyperparameters — edit here, not in code
├── src/
│   ├── models/classifier.py   # EfficientNetB0 architecture
│   ├── data/dataset.py        # tf.data pipeline + augmentation
│   ├── training/trainer.py    # two-phase training loop
│   ├── evaluation/evaluate.py # classification report + confusion matrix
│   └── predict.py             # inference + Grad-CAM
├── tests/                  # pytest unit tests
├── outputs/
│   ├── models/             # saved .h5 weights
│   ├── logs/               # train.log
│   └── figures/            # confusion matrix plot
├── data/
│   ├── raw/                # original data — never modified
│   └── processed/          # preprocessed data
├── app.py                  # Gradio demo entry point
└── train.py                # training entry point
```

## Configuration

All hyperparameters live in [configs/train.yaml](configs/train.yaml).
Override any value from the command line without editing the file:

```bash
python train.py configs/train.yaml \
    training.phase1.epochs=15 \
    training.batch_size=64 \
    model.dropout=0.4
```

## Reproducibility

The seed is set in `configs/train.yaml` (`training.seed: 42`).
`set_seed()` seeds `random`, `numpy`, and `tf.random` before any data loading or model initialization.

## Demo

```bash
python app.py --share   # generates a public HuggingFace Spaces link
```

## Development

```bash
pip install -e ".[dev]"      # installs dev extras (pytest, ruff, black, mypy)
pre-commit install           # auto-runs linters before every commit

pytest tests/ -v             # run test suite
black src/ tests/            # format
ruff check src/              # lint
mypy src/                    # type check
```

## Dataset

[PlantVillage](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) —
54 000+ leaf images across 38 disease/healthy classes from 14 crop species.

### Downloading the data

**Option A — kagglehub (recommended)**

```bash
pip install kagglehub
```

```python
import kagglehub
import shutil

path = kagglehub.dataset_download("abdallahalidev/plantvillage-dataset")
print("Downloaded to:", path)

# Copy into the project's expected location
shutil.copytree(path, "data/raw/plantvillage", dirs_exist_ok=True)
```

Requires a free Kaggle account. On first run it will ask for your Kaggle credentials and cache them.

**Option B — manual download**

1. Go to https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset and click **Download**.
2. Unzip into the project:
   ```bash
   unzip plantvillage-dataset.zip -d data/raw/plantvillage
   ```

**Verify the structure**

Class folders must be directly inside `plantvillage/`:
```
data/raw/plantvillage/
├── Apple___Apple_scab/
├── Apple___Black_rot/
├── Apple___Cedar_apple_rust/
├── Apple___healthy/
└── ... (38 folders total)
```
If you see a single subfolder instead of class folders, move its contents up one level:
```bash
mv data/raw/plantvillage/*/* data/raw/plantvillage/
```
