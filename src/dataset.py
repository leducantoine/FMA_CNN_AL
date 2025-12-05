"""
Dataset module for loading pre-processed audio features.
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset

class MelDataset(Dataset):
    """
    PyTorch Dataset for loading Mel-spectrograms stored as .npy files.
    """
    def __init__(self, file_paths, labels, mel_dir="mels"):
        """
        Args:
            file_paths (list): List of file paths or IDs.
            labels (list): List of corresponding integer labels.
            mel_dir (str): Directory containing the .npy files.
        """
        self.file_paths = file_paths
        self.labels = labels
        self.mel_dir = mel_dir

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # Resolve file path
        filename = os.path.basename(self.file_paths[idx])
        mel_path = os.path.join(self.mel_dir, filename + ".npy")
        
        # Load spectrogram
        mel_spec = np.load(mel_path)
        
        # Prepare tensor: Add channel dimension (1, n_mels, time)
        x = torch.tensor(mel_spec, dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return x, y