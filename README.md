# 🎵 FMA Genre Classification (CNN + Mel-Spectrograms)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

Ce projet implémente un pipeline complet de **Deep Learning** pour la classification de genres musicaux à partir du dataset **FMA-small** (8 genres, 8 000 pistes, extraits de 30s).

Deux pipelines sont fournis : une **baseline** sur Mel-spectrogrammes bruts et une **version augmentée** avec Data Augmentation (TimeStretch, PitchShift, bruit, etc.) optimisée pour **Apple Silicon (M1/M2/M3)** via le backend `MPS` de PyTorch.

---

## 📂 Structure du projet

```bash
FMA_CNN_AL/
│
├── data/
│   ├── raw/                 # Fichiers .mp3 (non versionnés)
│   └── metadata/            # tracks.csv
│
├── mels/                    # Spectrogrammes de base (.npy, ignorés par git)
├── mels_augmented/          # Spectrogrammes augmentés (.npy, ignorés par git)
│
├── src/                     # 🟢 CODE BASELINE
│   ├── preprocess.py        # MP3 -> Mel-spectrogrammes
│   ├── dataset.py           # Dataset PyTorch (baseline)
│   ├── model.py             # CNN léger (baseline)
│   ├── train.py             # Entraînement baseline
│   └── visualize.py         # Courbes & comparaisons
│
├── src_aug/                 # 🟠 CODE AUGMENTÉ (Data Augmentation)
│   ├── preprocess.py        # Génération mels + versions augmentées
│   ├── dataset.py           # Dataset gérant les fichiers augmentés
│   ├── model.py             # CNN ajusté / régularisé
│   └── train.py             # Entraînement sur données augmentées
│
├── results/                 # Métriques & graphiques
│   ├── comparison_curves.png
│   ├── confusion_matrix.png
│   ├── final_metrics.txt
│   └── *.npy                # Historiques de loss (ignorés par git)
│
├── requirements.txt
└── README.md
```

---

## 🚀 Installation

### 2.1. Prérequis

- Python ≥ 3.10  
- `git` et un environnement virtuel (recommandé, ex. `venv` ou `conda`).

Cloner le dépôt puis installer les dépendances :

```bash
git clone https://github.com/leducantoine/FMA_CNN_AL.git
cd FMA_CNN_AL

# Optionnel mais recommandé
python -m venv .venv
source .venv/bin/activate  # sous macOS / Linux
# .venv\Scripts\activate   # sous Windows PowerShell

pip install -r requirements.txt
```

### 2.2. Accélération Apple Silicon (MPS)

Le code détecte automatiquement la présence de MPS (`device = "mps"`), sinon il bascule sur CPU.
Vérifier que PyTorch voit bien MPS :

```bash
python -c "import torch; print(f'MPS Available: {torch.backends.mps.is_available()}')"
```

---

## 💾 Préparation du dataset (FMA-Small)

Le dataset **FMA-small** n'est pas versionné dans le dépôt.

1. Télécharger le dataset depuis Kaggle (ex. FMA-small dérivé de `fma` original) :  
   https://www.kaggle.com/datasets/aaronyim/fma-small
2. Extraire les fichiers et organiser comme suit :

```bash
data/
├── raw/
│   └── fma_small/        # Contient les dossiers 000, 001, ..., 155
└── metadata/
    └── tracks.csv        # Fichier de métadonnées FMA
```

> **Important** : le dossier `data/raw/` ne doit pas être versionné sur GitHub (voir `.gitignore`).

---

## ⚙️ Utilisation pas à pas

Cette section explique exactement quoi lancer et dans quel ordre.

### Étape A – Baseline (rapide)

**1. Pré-calcul des Mel-spectrogrammes** (une seule fois) :

```bash
python src/preprocess.py
```

- Lit les `.mp3` dans `data/raw/fma_small/`.  
- Sauvegarde un `.npy` par piste dans `mels/`.

**2. Entraînement du modèle baseline** :

```bash
python src/train.py
```

Ce script :
- Charge les spectrogrammes depuis `mels/`.  
- Effectue le split train/validation/test.  
- Entraîne un CNN léger sur `mps` (si dispo) ou CPU.  
- Sauvegarde les courbes / métriques dans `results/`  
  (par ex. loss / accuracy, `final_metrics.txt`, etc.).

### Étape B – Pipeline augmenté (performant)

**1. Génération des Mel-spectrogrammes augmentés** :

```bash
python src_aug/preprocess.py
```

- Crée plusieurs versions par piste (Original + Noise + TimeStretch/PitchShift).
- Stocke les spectrogrammes dans `mels_augmented/`.

**2. Entraînement sur données augmentées** :

```bash
python src_aug/train.py
```

Ce script :
- Charge `mels_augmented/`.  
- Entraîne un modèle légèrement modifié (CNN avec ajustements / régularisation).  
- Sauvegarde les métriques et courbes dans `results/`  
  (y compris les données nécessaires pour la comparaison baseline vs augmenté).

### Étape C – Visualisation & comparaison

Une fois les deux entraînements effectués (baseline + augmenté) :

```bash
python src/visualize.py
```

- Charge les historiques de loss/accuracy stockés en `.npy`.
- Génère les figures de comparaison dans `results/`  
  (par ex. `comparison_curves.png`, `confusion_matrix.png`).

---

## 📊 Résultats & architecture

- Les fichiers principaux produits sont :  
  - `results/comparison_curves.png` : évolution des métriques baseline vs augmenté.
  - `results/confusion_matrix.png` : matrice de confusion du meilleur modèle.  

- Le modèle est un **CNN léger** composé de 4 blocs Convolution → BatchNorm → ReLU → MaxPool, suivi d'un flatten et d'une couche linéaire finale vers 8 genres.

---

## 🛠 Notes techniques & extension

- Les spectrogrammes sont calculés **offline** pour accélérer l'entraînement et limiter la charge CPU.
- Le backend `MPS` est utilisé automatiquement sur Mac M1/M2/M3 s'il est disponible, sinon bascule sur CPU.

Pistes d'extension possibles :  
- Ajouter davantage de data augmentation (SpecAugment, mixup).  
- Tester des architectures plus profondes (CRNN, attention).  
- Intégrer un scheduler, de l'early stopping ou du logging avancé (Weights & Biases, TensorBoard).

---

## 📝 Licence & auteur

- Licence : **MIT**  
- Auteur : **Antoine Leduc**  

Projet réalisé dans le cadre d'une étude sur l'efficacité de CNN légers pour la classification de genres musicaux sur architectures ARM/Apple Silicon.
