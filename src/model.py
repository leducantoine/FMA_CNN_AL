"""
CNN Architecture definition.
Standard 4-layer Convolutional Neural Network with optional Batch Normalization.
"""

import torch
import torch.nn as nn

class CNN(nn.Module):
    """
    2D CNN for audio classification.
    
    Architecture:
        [Conv2D -> (BatchNorm) -> ReLU -> MaxPool -> Dropout] x 4
        Flatten
        Dense (256) -> ReLU -> Dropout
        Dense (num_classes)
    """
    def __init__(self, num_classes, use_batchnorm=False):
        super(CNN, self).__init__()
        
        def _conv_block(in_channels, out_channels):
            layers = [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            ]
            
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(out_channels))
                
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(kernel_size=2))
            layers.append(nn.Dropout(0.25))
            
            return nn.Sequential(*layers)

        # Feature Extractor (4 blocks)
        self.features = nn.Sequential(
            _conv_block(1, 16),
            _conv_block(16, 32),
            _conv_block(32, 64),
            _conv_block(64, 128)
        )
        
        self.flatten = nn.Flatten()
        
        # Dimension Calculation for 10s segments (430 frames width):
        # Height: 128 -> 64 -> 32 -> 16 -> 8
        # Width:  430 -> 215 -> 107 -> 53 -> 26
        # Total: 128 filters * 8 height * 26 width
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