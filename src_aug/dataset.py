"""
Dataset module for loading augmented audio features.
"""

import torch
import numpy as np
from torch.utils.data import Dataset

class AugmentedMelDataset(Dataset):
    """
    PyTorch Dataset for loading pre-computed Mel-spectrogram chunks (.npy).
    """
    def __init__(self, file_paths, labels):
        """
        Args:
            file_paths (list): List of paths to .npy files.
            labels (list): List of integer labels corresponding to files.
        """
        self.file_paths = file_paths
        self.labels = labels

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # Load spectrogram chunk (Shape: [n_mels, time_steps])
        mel_spec = np.load(self.file_paths[idx])
        
        # Add channel dimension: (1, n_mels, time_steps)
        x = torch.tensor(mel_spec, dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return x, y