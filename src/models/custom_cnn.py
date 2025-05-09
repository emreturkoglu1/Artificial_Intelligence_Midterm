"""
Custom CNN Model

This module provides a custom CNN architecture for image classification.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomCNN(nn.Module):
    """
    Custom CNN architecture for flower classification.
    
    Architecture:
        - 5 Convolutional blocks (Conv2d + BatchNorm + ReLU + MaxPool)
        - 3 Fully connected layers with dropout
        - Output layer with 5 classes
    """

    def __init__(self, num_classes=5):
        """
        Initialize the Custom CNN model.

        Args:
            num_classes (int): Number of output classes (default: 5 for flower dataset)
        """
        super(CustomCNN, self).__init__()
        
        # First Convolutional Block
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Second Convolutional Block
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Third Convolutional Block
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Fourth Convolutional Block
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Fifth Convolutional Block
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Fully Connected Layers
        self.fc = nn.Sequential(
            nn.Dropout(0.7),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.7),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, height, width)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes)
        """
        # Apply convolutional blocks
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        
        # Flatten the output for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Apply fully connected layers
        x = self.fc(x)
        
        return x

    def get_feature_maps(self, x):
        """
        Get intermediate feature maps for visualization.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, height, width)

        Returns:
            dict: Dictionary containing feature maps from each convolutional block
        """
        feature_maps = {}
        
        x = self.conv1(x)
        feature_maps['conv1'] = x
        
        x = self.conv2(x)
        feature_maps['conv2'] = x
        
        x = self.conv3(x)
        feature_maps['conv3'] = x
        
        x = self.conv4(x)
        feature_maps['conv4'] = x
        
        x = self.conv5(x)
        feature_maps['conv5'] = x
        
        return feature_maps 