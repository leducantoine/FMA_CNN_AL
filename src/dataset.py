import os
import numpy as np
import torch
from torch.utils.data import Dataset

class MelDataset(Dataset):
    def __init__(self, paths, labels, mel_dir="mels"):
        self.paths = paths
        self.labels = labels
        self.mel_dir = mel_dir

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        name = os.path.basename(self.paths[idx])
        mel = np.load(os.path.join(self.mel_dir, name + ".npy"))
        mel = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return mel, label
