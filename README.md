# 🎵 FMA Genre Classification (CNN + Mel-Spectrograms)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![License](https://img.shields.io/badge/License-CentraleSup%C3%A9lec-blue)

This project implements a complete **Deep Learning** pipeline for music genre classification using the **FMA-small** dataset (8 genres, 8,000 tracks, 30s excerpts).

Two pipelines are provided: a **baseline** on raw Mel-spectrograms and an **augmented version** with Data Augmentation (TimeStretch, PitchShift, Noise, etc.) optimized for **Apple Silicon (M1/M2/M3)** via PyTorch's `MPS` backend.

---

## 📂 Project Structure

```bash
FMA_CNN_AL/
│
├── data/
│   ├── raw/                 # .mp3 files (not versioned)
│   └── metadata/            # tracks.csv
│
├── mels/                    # Baseline spectrograms (.npy, git-ignored)
├── mels_augmented/          # Augmented spectrograms (.npy, git-ignored)
│
├── src/                     # 🟢 BASELINE CODE
│   ├── preprocess.py        # MP3 -> Mel-spectrograms
│   ├── dataset.py           # PyTorch Dataset (baseline)
│   ├── model.py             # Lightweight CNN (baseline)
│   ├── train.py             # Baseline training
│   ├── visualize.py         # Curves & comparisons
│   └── analyze.py           # Model analysis & evaluation
│
├── src_aug/                 # 🟠 AUGMENTED CODE (Data Augmentation)
│   ├── preprocess.py        # Mel generation + augmented versions
│   ├── dataset.py           # Dataset handling augmented files
│   ├── model.py             # Adjusted / regularized CNN
│   ├── train.py             # Training on augmented data
│   └── analyze.py           # Analysis & evaluation (augmented)
│
├── results/                 # Metrics & graphics
│   ├── comparison_curves.png
│   ├── confusion_matrix.png
│   ├── final_metrics.txt
│   └── *.npy                # Loss histories (git-ignored)
│
├── requirements.txt
└── README.md
```

---

## 🚀 Installation

### 2.1. Prerequisites

- Python ≥ 3.10  
- `git` and a virtual environment (recommended, e.g. `venv` or `conda`).

Clone the repository and install dependencies:

```bash
git clone https://github.com/leducantoine/FMA_CNN_AL.git
cd FMA_CNN_AL

# Optional but recommended
python -m venv .venv
source .venv/bin/activate  # on macOS / Linux
# .venv\\Scripts\\activate   # on Windows PowerShell

pip install -r requirements.txt
```

### 2.2. Apple Silicon (MPS) Acceleration

The code automatically detects MPS (`device = "mps"`), otherwise falls back to CPU.
Verify that PyTorch sees MPS:

```bash
python -c "import torch; print(f'MPS Available: {torch.backends.mps.is_available()}')"
```

---

## 💾 Dataset Preparation (FMA-Small)

The **FMA-small** dataset is not versioned in this repository.

1. Download the dataset from Kaggle (e.g. FMA-small derived from original `fma`):  
   https://www.kaggle.com/datasets/aaronyim/fma-small
2. Extract files and organize as follows:

```bash
data/
├── raw/
│   ├── 000/             # .mp3 files
│   ├── 001/             # .mp3 files
│   ├── ...
│   └── 155/             # .mp3 files
└── metadata/
    └── tracks.csv    # FMA metadata file
```

> **Important**: The `data/raw/` folder should not be versioned on GitHub (see `.gitignore`).

---

## ⚙️ Step-by-Step Usage

This section explains exactly what to run and in which order.

### Step A – Baseline (fast)

**1. Pre-compute Mel-spectrograms** (once):

```bash
python src/preprocess.py
```

- Reads `.mp3` files from `data/raw/` (folders 000/ to 155/).
- Saves one `.npy` file per track in `mels/`.

**2. Train the baseline model**:

```bash
python src/train.py
```

This script:
- Loads spectrograms from `mels/`.  
- Performs train/validation/test split.  
- Trains a lightweight CNN on `mps` (if available) or CPU.  
- Saves curves / metrics in `results/`  
  (e.g. loss / accuracy, `final_metrics.txt`, etc.).

### Step B – Augmented pipeline (better performance)

**1. Generate augmented Mel-spectrograms**:

```bash
python src_aug/preprocess.py
```

- Creates multiple versions per track (Original + Noise + TimeStretch/PitchShift).
- Stores spectrograms in `mels_augmented/`.

**2. Train on augmented data**:

```bash
python src_aug/train.py
```

This script:
- Loads `mels_augmented/`.  
- Trains a slightly modified model (CNN with adjustments / regularization).  
- Saves metrics and curves in `results/`  
  (including data needed for baseline vs augmented comparison).

### Step C – Evaluation & model comparison

Once both trainings are completed (baseline + augmented):

**Option 1: Complete evaluation with `analyze.py`** (recommended):

```bash
# For baseline
python src/analyze.py

# For augmented version
python src_aug/analyze.py
```

This script:
- Loads trained models from `results/`.
- Evaluates on test set with detailed metrics (accuracy, F1, confusion matrix).
- Saves results in `results/`.

**Option 2: Simple visual comparison with `visualize.py`**:

```bash
python src/visualize.py
```

- Loads loss/accuracy histories stored in `.npy`.
- Generates comparison figures in `results/`  
  (e.g. `comparison_curves.png`, `confusion_matrix.png`).

---

## 📊 Results & Architecture

- Main output files:  
  - `results/comparison_curves.png`: metric evolution baseline vs augmented.
  - `results/confusion_matrix.png`: confusion matrix of the best model.  

- The model is a **lightweight CNN** composed of 4 blocks Convolution → BatchNorm → ReLU → MaxPool, followed by flatten and a final linear layer to 8 genres.

---

## 🛠 Technical Notes & Extensions

- Spectrograms are computed **offline** to speed up training and limit CPU load.
- The `MPS` backend is used automatically on Mac M1/M2/M3 if available, otherwise falls back to CPU.

Possible extension paths:  
- Add more data augmentation (SpecAugment, mixup).  
- Test deeper architectures (CRNN, attention).  
- Integrate scheduler, early stopping, or advanced logging (Weights & Biases, TensorBoard).

---

## 📝 License & Author

- License: **CentraleSupélec Academic Project**-
- Author: **Antoine Leduc**

Project conducted as a study on the effectiveness of lightweight CNNs for music genre classification on ARM/Apple Silicon architectures.
