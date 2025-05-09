"""
Feature Visualization Utilities

This module provides utilities for visualizing CNN feature maps and filters,
implementing various visualization techniques including Grad-CAM and
feature map visualization.


"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from torch.nn import functional as F


class FeatureVisualizer:
    """
    A class for visualizing CNN features and attention maps.
    
    Attributes:
        model (nn.Module): The model to visualize
        device (torch.device): Device to run computations on
        transform (transforms.Compose): Image preprocessing transforms
    """

    def __init__(self, model, device):
        """
        Initialize the visualizer.

        Args:
            model (nn.Module): The model to visualize
            device (torch.device): Device to run computations on
        """
        self.model = model
        self.device = device
        self.model.eval()
        
        # Define image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                              [0.229, 0.224, 0.225])
        ])

    def preprocess_image(self, image_path):
        """
        Preprocess an image for the model.

        Args:
            image_path (str): Path to the image file

        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image)
        return image_tensor.unsqueeze(0).to(self.device)

    def visualize_feature_maps(self, image_tensor, layer_name=None):
        """
        Visualize feature maps from a specific layer.

        Args:
            image_tensor (torch.Tensor): Input image tensor
            layer_name (str, optional): Name of the layer to visualize

        Returns:
            numpy.ndarray: Feature map visualization
        """
        # Get feature maps
        if hasattr(self.model, 'get_feature_maps'):
            feature_maps = self.model.get_feature_maps(image_tensor)
        else:
            raise NotImplementedError("Model doesn't support feature map extraction")
        
        if layer_name and layer_name in feature_maps:
            features = feature_maps[layer_name]
        else:
            # Use the last layer if none specified
            features = list(feature_maps.values())[-1]
        
        # Convert to numpy and normalize
        feature_maps = features.detach().cpu().numpy()
        feature_maps = np.squeeze(feature_maps)
        
        return self._create_feature_grid(feature_maps)

    def _create_feature_grid(self, feature_maps, max_features=16):
        """
        Create a grid visualization of feature maps.

        Args:
            feature_maps (numpy.ndarray): Feature maps to visualize
            max_features (int): Maximum number of features to show

        Returns:
            numpy.ndarray: Grid of feature maps
        """
        # Select subset of features if too many
        if feature_maps.shape[0] > max_features:
            indices = np.linspace(0, feature_maps.shape[0]-1, max_features, dtype=int)
            feature_maps = feature_maps[indices]
        
        # Calculate grid dimensions
        n_features = len(feature_maps)
        grid_size = int(np.ceil(np.sqrt(n_features)))
        
        # Calculate the size of the grid
        height = feature_maps.shape[1]
        width = feature_maps.shape[2]
        grid = np.zeros((grid_size * height, grid_size * width))
        
        # Fill the grid
        for i in range(n_features):
            row = i // grid_size
            col = i % grid_size
            feature_map = feature_maps[i]
            # Normalize feature map
            feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() + 1e-8)
            grid[row*height:(row+1)*height,
                 col*width:(col+1)*width] = feature_map
        
        return grid

    def compute_gradcam(self, image_tensor, target_class):
        """
        Compute Grad-CAM visualization.

        Args:
            image_tensor (torch.Tensor): Input image tensor
            target_class (int): Target class index

        Returns:
            numpy.ndarray: Grad-CAM heatmap
        """
        # Register hooks for the last convolutional layer
        self.gradients = None
        self.activations = None
        
        def save_gradient(grad):
            self.gradients = grad
        
        # Get the last convolutional layer
        for module in self.model.modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module
        
        # Register hooks
        h = last_conv.register_forward_hook(
            lambda module, input, output: setattr(self, 'activations', output))
        h_grad = last_conv.register_backward_hook(
            lambda module, grad_in, grad_out: save_gradient(grad_out[0]))
        
        # Forward pass
        logits = self.model(image_tensor)
        
        # Clear existing gradients
        self.model.zero_grad()
        
        # Backward pass for target class
        one_hot = torch.zeros_like(logits)
        one_hot[0, target_class] = 1
        logits.backward(gradient=one_hot)
        
        # Remove hooks
        h.remove()
        h_grad.remove()
        
        # Compute Grad-CAM
        pooled_gradients = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        activations = self.activations.detach()
        
        for i in range(activations.size(1)):
            activations[:, i, :, :] *= pooled_gradients[:, i, :, :]
        
        heatmap = torch.mean(activations, dim=1).squeeze()
        heatmap = F.relu(heatmap)
        heatmap = heatmap / (torch.max(heatmap) + 1e-8)
        
        return heatmap.cpu().numpy()

    def visualize_filters(self, layer_idx=0):
        """
        Visualize convolutional filters from a specific layer.

        Args:
            layer_idx (int): Index of the convolutional layer to visualize

        Returns:
            numpy.ndarray: Filter visualization grid
        """
        # Get the first convolutional layer
        conv_layers = [module for module in self.model.modules() 
                      if isinstance(module, nn.Conv2d)]
        
        if layer_idx >= len(conv_layers):
            raise ValueError(f"Layer index {layer_idx} is out of range")
        
        # Get filters
        filters = conv_layers[layer_idx].weight.data.cpu().numpy()
        
        return self._create_filter_grid(filters)

    def _create_filter_grid(self, filters, max_filters=16):
        """
        Create a grid visualization of convolutional filters.

        Args:
            filters (numpy.ndarray): Filters to visualize
            max_filters (int): Maximum number of filters to show

        Returns:
            numpy.ndarray: Grid of filters
        """
        # Select subset of filters if too many
        if filters.shape[0] > max_filters:
            filters = filters[:max_filters]
        
        # Calculate grid dimensions
        n_filters = len(filters)
        grid_size = int(np.ceil(np.sqrt(n_filters)))
        
        # Create grid
        grid = np.zeros((grid_size * filters.shape[2], 
                        grid_size * filters.shape[3], 
                        filters.shape[1]))
        
        for i in range(n_filters):
            row = i // grid_size
            col = i % grid_size
            filter_img = np.transpose(filters[i], (1, 2, 0))
            # Normalize filter
            filter_img = (filter_img - filter_img.min()) / (filter_img.max() - filter_img.min() + 1e-8)
            grid[row*filters.shape[2]:(row+1)*filters.shape[2],
                 col*filters.shape[3]:(col+1)*filters.shape[3], :] = filter_img
        
        return grid

def visualize_features(model, image_tensor, layer_name, num_features=16, figsize=(20, 10), save_path=None):
    """
    Visualizes feature maps of a specific convolutional layer
    
    Args:
        model: Model to visualize
        image_tensor (tensor): Input image tensor [1, 3, H, W]
        layer_name (str): Name of the layer to visualize
        num_features (int): Number of features to visualize
        figsize (tuple): Figure size
        save_path (str): File path to save the figure (optional)
        
    Returns:
        None
    """
    # Set model to evaluation mode
    model.eval()
    
    # Move image tensor to GPU (if available)
    device = next(model.parameters()).device
    image_tensor = image_tensor.to(device)
    
    # Get feature maps of the relevant layer
    if hasattr(model, 'get_feature_map'):
        features = model.get_feature_map(image_tensor, layer_name)
    else:
        # Appropriate error message for models without feature map extraction method
        print(f"Model doesn't have get_feature_map method")
        return
    
    # Move feature maps to CPU and convert to numpy array
    features = features.detach().cpu().squeeze(0)
    
    # How many feature maps to visualize?
    num_features = min(num_features, features.size(0))
    
    # Prepare original image for visualization
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    original_img = inv_normalize(image_tensor.squeeze(0).cpu())
    original_img = original_img.permute(1, 2, 0).numpy()
    original_img = np.clip(original_img, 0, 1)
    
    # Equalize image dimensions
    img_height, img_width = original_img.shape[:2]
    
    # Creating fixed grid
    n_cols = 9  # 8 features + original image
    n_rows = (num_features + n_cols - 1) // n_cols  # Round up
    
    # Create figure (with fixed size axes)
    fig = plt.figure(figsize=figsize)
    
    # Place original image in first cell
    ax = fig.add_subplot(n_rows, n_cols, 1)
    ax.imshow(original_img)
    ax.set_title('Original Image')
    ax.set_aspect('equal')  # For preserving aspect ratio
    ax.axis('off')
    
    # Visualize feature maps
    for i in range(num_features):
        ax = fig.add_subplot(n_rows, n_cols, i + 2)  # +2 because 1 is for original image
        
        feature_map = features[i].numpy()
            
        # Min-max normalization
        feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() + 1e-8)
            
        # Show image
        ax.imshow(feature_map, cmap='viridis')
        ax.set_title(f'Feature {i+1}')
        ax.set_aspect('equal')  # For preserving aspect ratio
        ax.axis('off')
    
    plt.suptitle(f'Feature Maps from {layer_name} Layer', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()

def visualize_zeiler_fergus(model, image_tensor, layer_name, figsize=(20, 10), patch_size=50, stride=20, save_path=None):
    """
    Visualization according to Zeiler and Fergus (2014) method
    Observes how the model responds by masking part of the image
    
    Args:
        model: Model to visualize
        image_tensor (tensor): Input image tensor [1, 3, H, W]
        layer_name (str): Name of the layer to visualize
        figsize (tuple): Figure size
        patch_size (int): Size of the patch to be masked
        stride (int): Masking step
        save_path (str): File path to save the figure (optional)
        
    Returns:
        None
    """
    # Set model to evaluation mode
    model.eval()
    
    # Move image tensor to GPU (if available)
    device = next(model.parameters()).device
    image_tensor = image_tensor.to(device)
    
    # Original prediction
    with torch.no_grad():
        original_output = model(image_tensor)
        _, original_pred = torch.max(original_output, 1)
        original_pred = original_pred.item()
    
    # Image dimensions
    _, _, height, width = image_tensor.shape
    
    # Sliding window parameters for masking
    heatmap = torch.zeros((height, width)).to(device)
    
    # Mask different regions of the image and measure activation change
    num_patches = 0
    
    with torch.no_grad():
        for h in range(0, height - patch_size + 1, stride):
            for w in range(0, width - patch_size + 1, stride):
                # Create a copy of the original image
                masked_img = image_tensor.clone()
                
                # Mask a specific region (replace with gray value)
                masked_img[:, :, h:h+patch_size, w:w+patch_size] = 0.5
                
                # Make prediction for masked image
                masked_output = model(masked_img)
                
                # Get activation value for original class
                class_activation = masked_output[0, original_pred].item()
                
                # Assign activation value to masked region
                heatmap[h:h+patch_size, w:w+patch_size] += class_activation
                
                num_patches += 1
    
    # Normalize heatmap
    heatmap = heatmap.cpu().numpy()
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    
    # Prepare original image for visualization
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    original_img = inv_normalize(image_tensor.squeeze(0).cpu())
    original_img = original_img.permute(1, 2, 0).numpy()
    original_img = np.clip(original_img, 0, 1)
    
    # Visualization - fixed version
    fig = plt.figure(figsize=figsize)
    
    # Original image
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(original_img)
    ax1.set_title('Original Image')
    ax1.set_aspect('equal')  # For preserving aspect ratio
    ax1.axis('off')
    
    # Heatmap
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(heatmap, cmap='viridis')
    ax2.set_title(f'Zeiler & Fergus Visualization\n{layer_name} Layer')
    ax2.set_aspect('equal')  # For preserving aspect ratio
    ax2.axis('off')
    
    # Image combined with heatmap
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.imshow(original_img)
    ax3.imshow(heatmap, cmap='viridis', alpha=0.5)
    ax3.set_title('Overlay')
    ax3.set_aspect('equal')  # For preserving aspect ratio
    ax3.axis('off')
    
    plt.suptitle(f'Zeiler & Fergus Visualization - {layer_name} Layer', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show() 