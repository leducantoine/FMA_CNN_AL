"""
Training script for the Baseline and Modified CNN models on FMA-small.
Handles data loading, model initialization, training loop, and logging.
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

from dataset import MelDataset
from model import CNN

# --- Configuration ---
CONFIG = {
    "use_batchnorm": True,  # Set to False for Baseline, True for Modified
    "batch_size": 32,
    "epochs": 30,
    "learning_rate": 0.0001,
    "seed": 42,
    "device": "mps" if torch.backends.mps.is_available() else "cpu",
    "metadata_path": "data/metadata/tracks.csv",
    "raw_data_path": "data/raw",
    "mel_dir": "mels"
}

def load_data(config):
    """Loads metadata, filters available tracks, and returns data loaders."""
    print("Loading dataset...")
    tracks = pd.read_csv(config["metadata_path"], index_col=0, header=[0, 1])
    genres = tracks["track"]["genre_top"]
    
    mp3_files = glob.glob(f"{config['raw_data_path']}/**/*.mp3", recursive=True)
    paths, labels = [], []

    for p in mp3_files:
        try:
            tid = int(os.path.splitext(os.path.basename(p))[0])
            g = genres.get(tid)
            # Verify corresponding mel spectrogram exists
            mel_path = os.path.join(config["mel_dir"], os.path.basename(p) + ".npy")
            if isinstance(g, str) and os.path.exists(mel_path):
                paths.append(p)
                labels.append(g)
        except:
            continue

    print(f"Total tracks found: {len(paths)}")
    
    if len(paths) == 0:
        raise FileNotFoundError(f"No valid data found in {config['mel_dir']}. Run preprocess.py first.")

    le = LabelEncoder()
    y_enc = le.fit_transform(labels)
    num_classes = len(le.classes_)

    X_train, X_test, y_train, y_test = train_test_split(
        paths, y_enc, test_size=0.2, stratify=y_enc, random_state=config["seed"]
    )

    train_dl = DataLoader(MelDataset(X_train, y_train), batch_size=config["batch_size"], shuffle=True)
    val_dl = DataLoader(MelDataset(X_test, y_test), batch_size=config["batch_size"], shuffle=False)
    
    return train_dl, val_dl, num_classes

def main():
    # Set reproducibility
    np.random.seed(CONFIG["seed"])
    torch.manual_seed(CONFIG["seed"])
    
    model_name = "modified_bn" if CONFIG["use_batchnorm"] else "baseline"
    print(f"--- Starting training for: {model_name} on {CONFIG['device']} ---")
    
    # 1. Load Data
    train_dl, val_dl, num_classes = load_data(CONFIG)
    
    # 2. Initialize Model
    model = CNN(num_classes, use_batchnorm=CONFIG["use_batchnorm"]).to(CONFIG["device"])
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    criterion = nn.CrossEntropyLoss()
    
    # 3. Setup directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    # 4. Training Loop
    print("Starting training loop...")
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
        val_loss_sum = 0.0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(CONFIG["device"]), y.to(CONFIG["device"])
                out = model(x)
                val_loss_sum += criterion(out, y).item()
                
        avg_val_loss = val_loss_sum / len(val_dl)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        print(f" -> Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # Save best model checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f"models/best_{model_name}.pth")

    # Save training history
    np.save(f"results/train_loss_{model_name}.npy", np.array(train_losses))
    np.save(f"results/val_loss_{model_name}.npy", np.array(val_losses))
    print("Training complete.")

if __name__ == "__main__":
    main()