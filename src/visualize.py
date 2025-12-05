"""
Visualization utility script.
Finds a specific track (e.g., 'Rock' genre) and generates comparison spectrograms
(Full 30s vs 10s Chunk) for the report.
"""

import os
import glob
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt

# --- Configuration ---
CONFIG = {
    "metadata_path": "data/metadata/tracks.csv",
    "raw_data_dir": "data/raw",
    "image_dir": "images",
    "target_genre": "Rock",
    "sample_rate": 22050
}

def get_track_by_genre(genre):
    """
    Scans metadata and raw files to find a valid audio file matching the target genre.
    Returns: (file_path, track_id) or (None, None)
    """
    print(f"Searching for a '{genre}' track...")
    
    try:
        # Load metadata
        tracks = pd.read_csv(CONFIG["metadata_path"], index_col=0, header=[0, 1])
        # Filter by genre
        target_tracks = tracks[tracks[('track', 'genre_top')] == genre]
        target_ids = set(target_tracks.index.tolist())
        
        print(f"Found {len(target_ids)} candidates in metadata.")
    except Exception as e:
        print(f"Error reading metadata: {e}")
        return None, None

    # Scan raw files
    files = glob.glob(f"{CONFIG['raw_data_dir']}/**/*.mp3", recursive=True)
    
    for path in files:
        try:
            # Extract ID from filename (e.g., '000135.mp3' -> 135)
            filename = os.path.basename(path)
            tid = int(filename.split('.')[0])
            
            if tid in target_ids:
                return path, tid
        except:
            continue
            
    return None, None

def generate_comparison_plots(file_path, track_id):
    """
    Generates and saves 30s and 10s Mel-spectrograms.
    """
    print(f"Processing Track ID: {track_id}")
    os.makedirs(CONFIG["image_dir"], exist_ok=True)

    # Load Full Audio (30s)
    y, sr = librosa.load(file_path, sr=CONFIG["sample_rate"], duration=30)
    
    # 1. Baseline Input (30s)
    mel_30 = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_30_db = librosa.power_to_db(mel_30, ref=np.max)
    
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_30_db, sr=sr, x_axis='time', y_axis='mel', cmap='magma')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Baseline Input (30s) - {CONFIG["target_genre"]} Track', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG["image_dir"], "mel_spectro_30s.png"), dpi=150)
    plt.close()

    # 2. Augmented Input (10s Chunk)
    # Slice first 10 seconds
    y_10 = y[:int(sr * 10)]
    mel_10 = librosa.feature.melspectrogram(y=y_10, sr=sr, n_mels=128)
    mel_10_db = librosa.power_to_db(mel_10, ref=np.max)

    plt.figure(figsize=(5, 4))
    librosa.display.specshow(mel_10_db, sr=sr, x_axis='time', y_axis='mel', cmap='magma')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Augmented Input (10s Chunk)', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG["image_dir"], "mel_spectro_10s.png"), dpi=150)
    plt.close()
    
    print(f"Figures saved to {CONFIG['image_dir']}/")

def main():
    path, tid = get_track_by_genre(CONFIG["target_genre"])
    
    if path:
        generate_comparison_plots(path, tid)
    else:
        print(f"No audio file found for genre '{CONFIG['target_genre']}'.")

if __name__ == "__main__":
    main()