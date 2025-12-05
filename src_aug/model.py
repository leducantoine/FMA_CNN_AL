"""
CNN Architecture definition for the Augmented model.
Includes Batch Normalization layers and is optimized for 10s audio segments.
"""

import torch
import torch.nn as nn

class AugmentedCNN(nn.Module):
    """
    Convolutional Neural Network with Batch Normalization.
    Designed for input shapes of (1, 128, 430).
    
    Architecture:
    - 4 Convolutional Blocks (Conv -> BN -> ReLU -> MaxPool -> Dropout)
    - Flatten
    - Dense Layers (256 -> Output)
    """
    def __init__(self, num_classes):
        super(AugmentedCNN, self).__init__()
        
        def _conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
                nn.Dropout(0.25)
            )

        # Feature Extractor
        self.features = nn.Sequential(
            _conv_block(1, 16),
            _conv_block(16, 32),
            _conv_block(32, 64),
            _conv_block(64, 128)
        )
        
        self.flatten = nn.Flatten()
        
        # Dimension Calculation:
        # Input: (1, 128, 430)
        # After 4 MaxPool layers (div by 16):
        # - Height: 128 / 16 = 8
        # - Width:  430 / 16 ≈ 26
        self.fc_input_dim = 128 * 8 * 26
        
        self.classifier = nn.Sequential(
            nn.Linear(self.fc_input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x