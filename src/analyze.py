"""
Evaluation script for the Baseline vs Modified CNN models on FMA-small.
Generates performance metrics and comparison plots.
"""

import os
import glob
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from dataset import MelDataset
from model import CNN

# --- Configuration ---
METADATA_PATH = "data/metadata/tracks.csv"
RAW_DATA_PATH = "data/raw"
RESULTS_DIR = "results"
MODELS_DIR = "models"
BATCH_SIZE = 32
SEED = 42

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
if torch.cuda.is_available(): DEVICE = "cuda"

def load_test_data():
    """Loads metadata, filters available tracks, and returns test split."""
    tracks = pd.read_csv(METADATA_PATH, index_col=0, header=[0, 1])
    genres = tracks["track"]["genre_top"]
    mp3_files = glob.glob(f"{RAW_DATA_PATH}/**/*.mp3", recursive=True)

    paths, labels = [], []
    for p in mp3_files:
        try:
            tid = int(os.path.splitext(os.path.basename(p))[0])
            g = genres.get(tid)
            if isinstance(g, str) and os.path.exists(os.path.join("mels", os.path.basename(p) + ".npy")):
                paths.append(p)
                labels.append(g)
        except:
            continue

    le = LabelEncoder()
    y_enc = le.fit_transform(labels)
    
    # Consistent split with training
    _, X_test, _, y_test = train_test_split(
        paths, y_enc, test_size=0.2, stratify=y_enc, random_state=SEED
    )
    
    return X_test, y_test, le

def plot_learning_curves():
    """Generates comparison plot for validation loss between Baseline and Modified models."""
    plt.figure(figsize=(10, 6))
    
    baseline_path = os.path.join(RESULTS_DIR, "val_loss_baseline.npy")
    modified_path = os.path.join(RESULTS_DIR, "val_loss_modified_bn.npy")

    if os.path.exists(baseline_path):
        bl_val = np.load(baseline_path)
        plt.plot(bl_val, linestyle='--', color='blue', label='Baseline (Val Loss)')
    
    if os.path.exists(modified_path):
        mod_val = np.load(modified_path)
        plt.plot(mod_val, linestyle='-', color='red', label='Modified BN (Val Loss)')

    plt.title("Validation Loss Comparison")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(RESULTS_DIR, "comparison_curves.png"))
    plt.close()

def evaluate_model(X_test, y_test, le):
    """Evaluates the best modified model on the test set."""
    print("Evaluating Modified Model...")
    
    model = CNN(len(le.classes_), use_batchnorm=True).to(DEVICE)
    model_path = os.path.join(MODELS_DIR, "best_modified_bn.pth")
    
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return

    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    test_dl = DataLoader(MelDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in test_dl:
            x = x.to(DEVICE)
            out = model(x)
            _, preds = torch.max(out, 1)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=le.classes_)
    
    with open(os.path.join(RESULTS_DIR, "final_metrics.txt"), "w") as f:
        f.write(f"Test Accuracy: {acc:.4f}\n\n")
        f.write(report)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title("Confusion Matrix (Modified Model)")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"))
    plt.close()

    print(f"Analysis complete. Test Accuracy: {acc:.4f}")

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    X_test, y_test, le = load_test_data()
    plot_learning_curves()
    evaluate_model(X_test, y_test, le)

if __name__ == "__main__":
    main()