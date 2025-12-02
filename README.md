# FMA_CNN_AL
Deep Learning model for music recognition
FMA Genre Classification (CNN + Mel-Spectrograms)

Ce projet implémente un pipeline complet de classification musicale sur la base FMA-small.
Le modèle utilise des Mel-spectrogrammes pré-calculés et un CNN léger entraîné sur Apple Silicon via le backend PyTorch MPS.

L’objectif : construire un pipeline audio minimal mais propre, reproductible et suffisamment performant pour servir de base à un rapport ou à un projet plus vaste.

1. Structure du projet
fma-classification/
│
├── data/
│   ├── raw/                 # Fichiers .mp3 (non versionnés)
│   └── metadata/            # tracks.csv
│
├── mels/                    # Mel-spectrogrammes pré-calculés (.npy)
│
├── src/
│   ├── dataset.py
│   ├── preprocess.py
│   ├── model.py
│   ├── train.py
│   └── utils.py (optionnel)
│
├── notebooks/
│   ├── exploration.ipynb
│   └── evaluation.ipynb
│
├── models/
│   └── cnn_epoch10.pth
│
├── results/
│   ├── loss_plot.png
│   └── confusion_matrix.png
│
├── requirements.txt
└── .gitignore

2. Installation
2.1. Installer les dépendances Python

Recommandé : Python ≥ 3.10

pip install -r requirements.txt

2.2. Activer PyTorch pour Apple Silicon (M1/M2/M3)

Si tu n’as pas encore PyTorch version MPS :

pip install torch torchvision torchaudio


Puis vérifie :

python3 -c "import torch; print(torch.backends.mps.is_available())"


Tu dois obtenir :

True

3. Récupération du dataset
Option A — Téléchargement manuel (recommandé)

Télécharger FMA-small depuis Kaggle :
https://www.kaggle.com/datasets/aaronyim/fma-small

Puis place le dossier dans :

data/raw/           → tous les .mp3 dans leurs sous-dossiers
data/metadata/       → tracks.csv


Important : ne versionne jamais data/raw/ sur GitHub.

Option B — Kaggle API

Place ton fichier kaggle.json ici :

~/.kaggle/kaggle.json


Télécharge via le script :

python download_dataset.py


Le script extrait automatiquement les fichiers dans data/.

4. Pré-calcul des Mel-Spectrogrammes

Ce projet est optimisé pour entraîner vite.
On calcule donc les spectrogrammes une seule fois, avant l’entraînement.

Lancer :

python src/preprocess.py


Cela génère un .npy par fichier audio dans :

mels/


Durée typique sur Mac M3 : 10 à 15 minutes.

5. Entraînement du modèle

Lancer :

python src/train.py


Le script :

charge les Mel spectrogrammes

split train/test

entraîne un CNN sur MPS

loggue la loss train+val

sauvegarde le modèle dans models/cnn_epoch10.pth

génère une courbe d’entraînement dans results/loss_plot.png

Durée estimée sur Mac M3 : 8 à 12 minutes.

6. Modèle

Le CNN utilisé est volontairement simple :

4 blocs Conv → BatchNorm → ReLU → MaxPool

Flatten

Linear final (128 × 8 × 80 → n_classes)

Ce modèle est suffisant pour un rapport, un baseline ou une introduction au traitement audio.

7. Notebooks

Deux notebooks sont fournis :

notebooks/exploration.ipynb

Aperçu des fichiers audio

Visualisation de spectrogrammes

Distribution des genres

notebooks/evaluation.ipynb

Chargement du modèle entraîné

Prédictions sur des samples

Matrice de confusion (sauvegardée dans results/)

8. Reproductibilité

Recréer les résultats :

python src/preprocess.py
python src/train.py


Les fichiers suivants seront régénérés :

models/cnn_epoch10.pth

results/loss_plot.png

results/confusion_matrix.png (si tu l’ajoutes dans ton notebook)

9. .gitignore recommandé
data/raw/
mels/
models/*.pth
__pycache__/
.ipynb_checkpoints/

10. Notes techniques

Backend PyTorch MPS accélère énormément l’entraînement sur Mac

Le pre-processing évite de recalculer les spectrogrammes à chaque epoch

Le CNN est volontairement compact pour rester rapide sur CPU/MPS

Le projet peut être étendu avec :

augmentation audio

modèles plus profonds (CRNN)

optimisations (scheduler, early stopping)

augmentation via SpecAugment

11. Licence

MIT License.

12. Contact

Projet réalisé dans le cadre d’un pipeline audio complet :
préparation → extraction → classification.
Pour toute question ou extension, se référer aux notebooks et au code source.