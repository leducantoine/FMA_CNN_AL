import os
import glob
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    top_k_accuracy_score
)
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from dataset import MelDataset
from model import CNN

device = "mps" if torch.backends.mps.is_available() else "cpu"

os.makedirs("results", exist_ok=True)

# ----------------------------------------------------
# RELOAD LABELS AND TEST SET FROM METADATA
# ----------------------------------------------------
tracks = pd.read_csv("data/metadata/tracks.csv", index_col=0, header=[0, 1])
genres = tracks["track"]["genre_top"]

mp3_files = glob.glob("data/raw/**/*.mp3", recursive=True)

xs, ys = [], []
for p in mp3_files:
    try:
        tid = int(os.path.splitext(os.path.basename(p))[0])
        g = genres.get(tid)
        if isinstance(g, str):
            base = os.path.basename(p)
            npy_path = os.path.join("mels", base + ".npy")

            if os.path.exists(npy_path):
                xs.append(p)
                ys.append(g)
    except:
        pass

# Encode labels
le = LabelEncoder()
y_enc = le.fit_transform(ys)

# Same split as before
X_train, X_test, y_train, y_test = train_test_split(
    xs, y_enc, test_size=0.2, stratify=y_enc, random_state=42
)

# ----------------------------------------------------
# LOAD BEST MODEL
# ----------------------------------------------------
model = CNN(len(le.classes_)).to(device)
model.load_state_dict(torch.load("models/best_model.pth", map_location=device))
model.eval()

test_dl = torch.utils.data.DataLoader(
    MelDataset(X_test, y_test),
    batch_size=32,
    shuffle=False
)

y_true = []
y_pred = []
logits_list = []

with torch.no_grad():
    for x, y in test_dl:
        x = x.to(device)
        out = model(x)

        preds = torch.argmax(out, dim=1).cpu().numpy()
        y_pred.extend(preds)
        y_true.extend(y.numpy())
        logits_list.append(out.cpu().numpy())

logits = np.vstack(logits_list)

# ----------------------------------------------------
# METRICS
# ----------------------------------------------------
acc = accuracy_score(y_true, y_pred)
top3 = top_k_accuracy_score(y_true, logits, k=3)

print("Accuracy :", acc)
print("Top-3 Accuracy :", top3)

with open("results/accuracy.txt", "w") as f:
    f.write(f"Accuracy = {acc}\n")
    f.write(f"Top-3 Accuracy = {top3}\n")

# Classification report
report = classification_report(y_true, y_pred, target_names=le.classes_)
with open("results/classification_report.txt", "w") as f:
    f.write(report)

# ----------------------------------------------------
# CONFUSION MATRIX
# ----------------------------------------------------
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=False, cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("results/confusion_matrix.png", dpi=200)
plt.close()

# ----------------------------------------------------
# PREDICTION DISTRIBUTION
# ----------------------------------------------------
counts = np.bincount(y_pred)

plt.figure(figsize=(12, 5))
plt.bar(range(len(counts)), counts)
plt.xticks(range(len(counts)), le.classes_, rotation=90)
plt.title("Prediction Distribution")
plt.tight_layout()
plt.savefig("results/prediction_distribution.png", dpi=200)
plt.close()

# ----------------------------------------------------
# LOSS CURVE (si train.py les a sauvegardées)
# ----------------------------------------------------
if os.path.exists("results/train_hist.npy") and os.path.exists("results/val_hist.npy"):
    train_hist = np.load("results/train_hist.npy")
    val_hist = np.load("results/val_hist.npy")

    plt.figure(figsize=(8, 5))
    plt.plot(train_hist, label="Train")
    plt.plot(val_hist, label="Val")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Learning Curve")
    plt.tight_layout()
    plt.savefig("results/loss_curve.png", dpi=200)
    plt.close()

print("\n✓ Analysis done — results saved in /results/")
