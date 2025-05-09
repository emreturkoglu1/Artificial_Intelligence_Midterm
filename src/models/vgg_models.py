"""
VGG16-based Models

This module contains VGG16-based model implementations, including a feature
extractor model and a fine-tuned model.

"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import vgg16, VGG16_Weights


class VGG16FeatureExtractor(nn.Module):
    """
    VGG16 model used as a feature extractor.
    The convolutional layers are frozen, and only the classifier is trained.
    """

    def __init__(self, num_classes=5):
        """
        Initialize the VGG16 feature extractor model.

        Args:
            num_classes (int): Number of output classes (default: 5 for flower dataset)
        """
        super(VGG16FeatureExtractor, self).__init__()
        
        # Load pretrained VGG16
        vgg = vgg16(weights=VGG16_Weights.DEFAULT)
        
        # Freeze all parameters
        for param in vgg.parameters():
            param.requires_grad = False
            
        # Features part (convolutional layers)
        self.features = vgg.features
        
        # Classifier part (fully connected layers)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(0.7),
            nn.Linear(4096, 1024),
            nn.ReLU(True),
            nn.Dropout(0.7),
            nn.Linear(1024, num_classes)
        )
        
        # Initialize the new classifier weights
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, height, width)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes)
        """
        # Extract features
        x = self.features(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Apply classifier
        x = self.classifier(x)
        
        return x

    def get_feature_maps(self, x):
        """
        Get intermediate feature maps for visualization.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, height, width)

        Returns:
            dict: Dictionary containing feature maps from selected layers
        """
        feature_maps = {}
        
        # Get feature maps from specific layers
        for i, layer in enumerate(self.features):
            x = layer(x)
            if isinstance(layer, nn.MaxPool2d):
                feature_maps[f'pool{len(feature_maps)+1}'] = x
                
        return feature_maps


class VGG16FineTuned(nn.Module):
    """
    Fine-tuned VGG16 model.
    The first few convolutional layers are frozen, while later layers are fine-tuned.
    """

    def __init__(self, num_classes=5):
        """
        Initialize the fine-tuned VGG16 model.

        Args:
            num_classes (int): Number of output classes (default: 5 for flower dataset)
        """
        super(VGG16FineTuned, self).__init__()
        
        # Load pretrained VGG16
        vgg = vgg16(weights=VGG16_Weights.DEFAULT)
        
        # Features part (convolutional layers)
        self.features = vgg.features
        
        # Freeze only the first few layers (up to conv2_2)
        for i, param in enumerate(self.features.parameters()):
            if i < 10:  # First 10 parameters (4 conv layers + BN)
                param.requires_grad = False
                
        # Classifier part (fully connected layers)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(0.7),
            nn.Linear(4096, 1024),
            nn.ReLU(True),
            nn.Dropout(0.7),
            nn.Linear(1024, num_classes)
        )
        
        # Initialize the new classifier weights
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, height, width)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes)
        """
        # Extract features
        x = self.features(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Apply classifier
        x = self.classifier(x)
        
        return x

    def get_feature_maps(self, x):
        """
        Get intermediate feature maps for visualization.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, height, width)

        Returns:
            dict: Dictionary containing feature maps from selected layers
        """
        feature_maps = {}
        
        # Get feature maps from specific layers
        for i, layer in enumerate(self.features):
            x = layer(x)
            if isinstance(layer, nn.MaxPool2d):
                feature_maps[f'pool{len(feature_maps)+1}'] = x
                
        return feature_maps

    def get_feature_map(self, x, layer_name):
        """
        Returns the feature map of a specific layer
        
        Args:
            x (tensor): Input tensor
            layer_name (str): Name of the layer to get feature map from
            
        Returns:
            tensor: Feature map
        """
        # Layer indices in VGG16
        layer_indices = {
            'conv1_1': 0,   # First convolution layer
            'conv3_3': 16,  # Middle convolution layer
            'conv5_3': 30   # Last convolution layer
        }
        
        if layer_name not in layer_indices:
            return None
            
        layer_idx = layer_indices[layer_name]
        
        # Get the feature map by doing forward pass until the requested layer
        for i in range(layer_idx + 1):
            x = self.features[i](x)
            
        return x 