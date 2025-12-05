"""
Preprocessing script for FMA-small.
Converts MP3 audio files into Mel-spectrograms.
Implements Data Augmentation by splitting tracks into 10s segments.
"""

import os
import glob
import numpy as np
import librosa
from tqdm import tqdm

# Configuration
RAW_DATA_DIR = "data/raw"
OUTPUT_DIR = "mels"  # Output folder for .npy files
SR = 22050
TOTAL_DURATION = 30 
N_MELS = 128
TARGET_WIDTH = 430  # Approx. width for a 10s segment (hop_length=512)

def process_track(file_path):
    """
    Loads an audio file, computes its Mel-spectrogram, and saves 3 segments (10s each).
    """
    filename = os.path.basename(file_path).replace(".mp3", "")
    
    # Skip if already processed (check last segment)
    if os.path.exists(os.path.join(OUTPUT_DIR, f"{filename}_2.npy")):
        return

    try:
        # 1. Load full 30s track
        y, _ = librosa.load(file_path, sr=SR, duration=TOTAL_DURATION)
        
        # 2. Pad to ensure consistent length if track < 30s
        target_len = int(SR * TOTAL_DURATION)
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)))
        else:
            y = y[:target_len]

        # 3. Compute Mel Spectrogram
        mel = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        
        # 4. Normalize (Standardization per track)
        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)

        # 5. Split into 3 segments (Data Augmentation)
        for i in range(3):
            start = i * TARGET_WIDTH
            end = start + TARGET_WIDTH
            
            # Handle edge case where segment is shorter than target width
            if end > mel_db.shape[1]:
                segment = mel_db[:, start:]
                pad_width = TARGET_WIDTH - segment.shape[1]
                segment = np.pad(segment, ((0, 0), (0, pad_width)))
            else:
                segment = mel_db[:, start:end]
            
            # Save segment: {track_id}_{segment_index}.npy
            save_path = os.path.join(OUTPUT_DIR, f"{filename}_{i}.npy")
            np.save(save_path, segment)

    except Exception as e:
        # Silently skip corrupted files to avoid stopping the pipeline
        pass

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    files = glob.glob(f"{RAW_DATA_DIR}/**/*.mp3", recursive=True)
    print(f"Found {len(files)} audio files. Starting preprocessing...")
    
    for f in tqdm(files, desc="Generating Mels"):
        process_track(f)
        
    print("Preprocessing complete.")

if __name__ == "__main__":
    main()