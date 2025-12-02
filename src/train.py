import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from dataset import MelDataset
from model import CNN


# ----------------------------------------------------
# Device
# ----------------------------------------------------
device = "mps" if torch.backends.mps.is_available() else "cpu"

np.random.seed(42)
torch.manual_seed(42)


# ----------------------------------------------------
# Load metadata
# ----------------------------------------------------
tracks = pd.read_csv("data/metadata/tracks.csv", index_col=0, header=[0, 1])
genres = tracks["track"]["genre_top"]


# ----------------------------------------------------
# Gather audio paths and genres
# ----------------------------------------------------
mp3_files = glob.glob("data/raw/**/*.mp3", recursive=True)

xs = []
ys = []

for p in mp3_files:
    try:
        tid = int(os.path.splitext(os.path.basename(p))[0])
        g = genres.get(tid)
        if isinstance(g, str):
            xs.append(p)
            ys.append(g)
    except:
        pass


# ----------------------------------------------------
# Filter only files with corresponding mel spectrogram
# ----------------------------------------------------
xs_f = []
ys_f = []

for p, y in zip(xs, ys):
    base = os.path.basename(p)               # ex: "099134.mp3"
    mel_path = os.path.join("mels", base + ".npy")   # "mels/099134.mp3.npy"
    if os.path.exists(mel_path):
        xs_f.append(p)
        ys_f.append(y)

xs, ys = xs_f, ys_f
print(f"Total usable audio files with spectrogram: {len(xs)}")


# ----------------------------------------------------
# Label encoding
# ----------------------------------------------------
le = LabelEncoder()
y_enc = le.fit_transform(ys)


# ----------------------------------------------------
# Train / Test split
# ----------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    xs, y_enc, test_size=0.2, stratify=y_enc, random_state=42
)


# ----------------------------------------------------
# DataLoaders
# ----------------------------------------------------
train_dl = DataLoader(MelDataset(X_train, y_train), batch_size=32, shuffle=True)
val_dl = DataLoader(MelDataset(X_test, y_test), batch_size=32)


# ----------------------------------------------------
# Model
# ----------------------------------------------------
model = CNN(len(le.classes_)).to(device)
opt = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

best_val = float("inf")

os.makedirs("models", exist_ok=True)


# ----------------------------------------------------
# Training Loop
# ----------------------------------------------------
for epoch in range(10):
    model.train()
    total_loss = 0.0

    for x, y in train_dl:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        opt.step()
        total_loss += loss.item()

    train_loss = total_loss / len(train_dl)

    # Validation
    model.eval()
    val_loss_sum = 0.0
    with torch.no_grad():
        for x, y in val_dl:
            x, y = x.to(device), y.to(device)
            val_loss_sum += criterion(model(x), y).item()

    val_loss = val_loss_sum / len(val_dl)
    print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    # Save checkpoint
    ckpt_path = f"models/epoch_{epoch+1}.pth"
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved checkpoint → {ckpt_path}")

    # Save best
    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), "models/best_model.pth")
        print(f"New BEST model saved (val_loss={best_val:.4f})")

print("Training complete.")
