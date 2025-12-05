"""
Training script for the Augmented CNN model (10s segments).
Implements training loop, validation, and history logging.
"""

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
from tqdm import tqdm

from dataset import AugmentedMelDataset
from model import AugmentedCNN

# --- Configuration ---
CONFIG = {
    "batch_size": 32,
    "epochs": 20,
    "lr": 0.0005,
    "seed": 42,
    "device": "mps" if torch.backends.mps.is_available() else "cpu",
    "metadata_path": "data/metadata/tracks.csv",
    "data_dir": "mels_augmented",
    "save_dir": "models_aug",
    "log_dir": "results_aug"
}

def load_data(config):
    """Loads metadata and augmented segments, returns loaders."""
    print("Loading augmented dataset...")
    tracks = pd.read_csv(config["metadata_path"], index_col=0, header=[0, 1])
    genres = tracks["track"]["genre_top"]
    npy_files = glob.glob(os.path.join(config["data_dir"], "*.npy"))

    paths, labels = [], []
    for p in npy_files:
        try:
            # Filename format: {track_id}_{segment_idx}.npy
            tid = int(os.path.basename(p).split('_')[0])
            g = genres.get(tid)
            if isinstance(g, str):
                paths.append(p)
                labels.append(g)
        except:
            continue

    le = LabelEncoder()
    y_enc = le.fit_transform(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        paths, y_enc, test_size=0.2, stratify=y_enc, random_state=config["seed"]
    )

    train_dl = DataLoader(AugmentedMelDataset(X_train, y_train), 
                          batch_size=config["batch_size"], shuffle=True, num_workers=0)
    val_dl = DataLoader(AugmentedMelDataset(X_test, y_test), 
                        batch_size=config["batch_size"], num_workers=0)
    
    return train_dl, val_dl, len(le.classes_)

def main():
    print(f"--- Training Augmented CNN on {CONFIG['device']} ---")
    
    os.makedirs(CONFIG["save_dir"], exist_ok=True)
    os.makedirs(CONFIG["log_dir"], exist_ok=True)

    # Data
    train_dl, val_dl, num_classes = load_data(CONFIG)
    print(f"Train samples: {len(train_dl.dataset)} | Val samples: {len(val_dl.dataset)}")

    # Model
    model = AugmentedCNN(num_classes).to(CONFIG["device"])
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"])
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_acc = 0.0

    # Training Loop
    for epoch in range(CONFIG["epochs"]):
        model.train()
        running_loss = 0.0
        
        pbar = tqdm(train_dl, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}", unit="batch")
        
        for x, y in pbar:
            x, y = x.to(CONFIG["device"]), y.to(CONFIG["device"])
            
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_train_loss = running_loss / len(train_dl)

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(CONFIG["device"]), y.to(CONFIG["device"])
                out = model(x)
                val_loss += criterion(out, y).item()
                _, preds = torch.max(out, 1)
                total += y.size(0)
                correct += (preds == y).sum().item()
        
        avg_val_loss = val_loss / len(val_dl)
        val_acc = correct / total
        
        # Logging
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_acc"].append(val_acc)
        
        print(f" -> Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(CONFIG["save_dir"], "best_model_aug.pth"))

    np.save(os.path.join(CONFIG["log_dir"], "history.npy"), history)
    print(f"Training finished. Best Accuracy: {best_acc:.4f}")

if __name__ == "__main__":
    main()