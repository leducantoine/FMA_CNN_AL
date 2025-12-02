import os
import glob
import numpy as np
import librosa
from tqdm import tqdm

DATA_RAW = "data/raw"
SAVE_DIR = "mels"

os.makedirs(SAVE_DIR, exist_ok=True)

sr = 22050
duration = 30
n_mels = 128
target_len = sr * duration

mp3_files = glob.glob(f"{DATA_RAW}/**/*.mp3", recursive=True)

def process(path):
    name = os.path.basename(path)
    out = os.path.join(SAVE_DIR, name + ".npy")

    if os.path.exists(out):
        return  # déjà traité

    try:
        y, _ = librosa.load(path, sr=sr, duration=duration)

        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)))
        else:
            y = y[:target_len]

        mel = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_mels=n_mels
        )

        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)

        np.save(out, mel_db)

    except Exception:
        pass  # skip fichier corrompu

for p in tqdm(mp3_files, desc="Génération des mels"):
    process(p)


