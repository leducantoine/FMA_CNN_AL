"""
Data preprocessing script for FMA-small.
Implements Data Augmentation by splitting 30s tracks into three 10s segments (Chunking).
"""

import os
import glob
import numpy as np
import librosa
from tqdm import tqdm

# --- Configuration ---
CONFIG = {
    "raw_data_dir": "data/raw",
    "output_dir": "mels_augmented",
    "sample_rate": 22050,
    "duration": 30,
    "n_mels": 128,
    "target_width": 430  # ~10 seconds @ 22050Hz with hop_length=512
}

def process_track(file_path):
    """
    Process a single audio file: load, compute Mel-spectrogram, split into chunks, and save.
    """
    filename = os.path.basename(file_path).replace(".mp3", "")
    
    # Skip if already processed (check if last segment exists)
    if os.path.exists(os.path.join(CONFIG["output_dir"], f"{filename}_2.npy")):
        return

    try:
        # Load audio
        y, _ = librosa.load(file_path, sr=CONFIG["sample_rate"], duration=CONFIG["duration"])
        
        # Pad with zeros if track is shorter than 30s
        target_len = int(CONFIG["sample_rate"] * CONFIG["duration"])
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)))
        else:
            y = y[:target_len]

        # Compute Mel Spectrogram
        mel = librosa.feature.melspectrogram(y=y, sr=CONFIG["sample_rate"], n_mels=CONFIG["n_mels"])
        mel_db = librosa.power_to_db(mel, ref=np.max)
        
        # Normalize (Standardization)
        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)

        # Split into 3 segments (Chunking)
        for i in range(3):
            start = i * CONFIG["target_width"]
            end = start + CONFIG["target_width"]
            
            # Handle padding for the chunk if necessary
            if end > mel_db.shape[1]:
                segment = mel_db[:, start:]
                pad_width = CONFIG["target_width"] - segment.shape[1]
                segment = np.pad(segment, ((0, 0), (0, pad_width)))
            else:
                segment = mel_db[:, start:end]
            
            # Save segment
            save_path = os.path.join(CONFIG["output_dir"], f"{filename}_{i}.npy")
            np.save(save_path, segment)

    except Exception:
        # Skip corrupted files
        pass

def main():
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    
    files = glob.glob(f"{CONFIG['raw_data_dir']}/**/*.mp3", recursive=True)
    print(f"Found {len(files)} tracks. Starting data augmentation...")
    
    for f in tqdm(files, desc="Processing"):
        process_track(f)
        
    print("Preprocessing complete.")

if __name__ == "__main__":
    main()