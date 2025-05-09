"""
Data Loading and Preprocessing Utilities

This module provides utilities for loading and preprocessing the flower dataset,
including data augmentation, normalization, and DataLoader creation.


"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
import warnings

# Suppress pandas version warnings
warnings.filterwarnings("ignore", category=FutureWarning)

class FlowerDataset(Dataset):
    """
    Custom Dataset class for loading flower images.
    
    Attributes:
        root_dir (str): Root directory of the dataset
        transform (callable): Optional transform to be applied to samples
        class_to_idx (dict): Mapping from class names to indices
        samples (list): List of (image_path, class_index) tuples
    """

    def __init__(self, root_dir, transform=None, split='train'):
        """
        Initialize the dataset.

        Args:
            root_dir (str): Root directory containing the dataset
            transform (callable, optional): Transform to apply to the images
            split (str): 'train' or 'test' split
        """
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        self.samples = self._load_samples(split)
        
    def _load_samples(self, split):
        samples = []
        
        if split == 'train':
            # For training set - folders are already separated by class labels
            train_dir = os.path.join(self.root_dir, 'train')
            
            # If train folder doesn't exist, check root_dir directly
            if not os.path.exists(train_dir):
                print(f"Warning: {train_dir} not found, checking {self.root_dir} directly")
                train_dir = self.root_dir
            
            for class_name in self.classes:
                class_dir = os.path.join(train_dir, class_name)
                if not os.path.isdir(class_dir):
                    print(f"Warning: {class_dir} not found, skipping class")
                    continue
                    
                try:
                    files = sorted([f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                    
                    if not files:
                        print(f"Warning: No image files found in {class_dir}")
                        continue
                        
                    for file_name in files:
                        img_path = os.path.join(class_dir, file_name)
                        # Check if file can be read
                        try:
                            with Image.open(img_path) as img:
                                pass  # Just checking if it can be opened
                            samples.append((
                                img_path,
                                self.class_to_idx[class_name]
                            ))
                        except Exception as e:
                            print(f"Warning: Could not read file {img_path}: {e}")
                except Exception as e:
                    print(f"Error reading directory {class_dir}: {e}")
        
        elif split == 'test':
            # For test set - all images in a single folder
            test_dir = os.path.join(self.root_dir, 'test')
            
            # If test folder doesn't exist, warn and return empty list
            if not os.path.exists(test_dir):
                print(f"Warning: {test_dir} not found, test dataset will be empty")
                return []
            
            try:
                # Get all images in test folder
                test_images = sorted([f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                
                if not test_images:
                    print(f"Warning: No image files found in {test_dir}")
                    return []
                
                # Check if there's a CSV file containing test image labels
                csv_files = ['Testing_set_flower.csv', 'sample_submission.csv', 'test_labels.csv']
                csv_path = None
                
                for csv_file in csv_files:
                    potential_path = os.path.join(self.root_dir, csv_file)
                    if os.path.exists(potential_path):
                        csv_path = potential_path
                        break
                
                if csv_path:
                    try:
                        # Read labels from CSV
                        df = pd.read_csv(csv_path)
                        
                        # If CSV has label/prediction/class column, use it
                        label_columns = ['label', 'prediction', 'class']
                        existing_label_col = None
                        
                        for col in label_columns:
                            if col in df.columns:
                                existing_label_col = col
                                break
                        
                        # Check different names for filename column
                        filename_columns = ['filename', 'file', 'image', 'name']
                        filename_col = None
                        
                        for col in filename_columns:
                            if col in df.columns:
                                filename_col = col
                                break
                                
                        if filename_col is None:
                            print(f"Warning: No filename column found in CSV file")
                            
                        if existing_label_col and filename_col:
                            for index, row in df.iterrows():
                                image_file = row[filename_col]
                                
                                # Filename might be given as full path
                                base_image_file = os.path.basename(image_file)
                                
                                if base_image_file in test_images or image_file in test_images:
                                    # The filename to use
                                    actual_file = base_image_file if base_image_file in test_images else image_file
                                    
                                    # Check label
                                    label_name = str(row[existing_label_col]).lower()
                                    
                                    # If label is class name, use class index
                                    if label_name in self.class_to_idx:
                                        img_path = os.path.join(test_dir, actual_file)
                                        # Check if file can be read
                                        try:
                                            with Image.open(img_path) as img:
                                                pass  # Just checking if it can be opened
                                            samples.append((
                                                img_path,
                                                self.class_to_idx[label_name]
                                            ))
                                        except Exception as e:
                                            print(f"Warning: Could not read file {img_path}: {e}")
                                    # If label is numeric, use it directly
                                    elif label_name.isdigit() and int(label_name) < len(self.classes):
                                        img_path = os.path.join(test_dir, actual_file)
                                        # Check if file can be read
                                        try:
                                            with Image.open(img_path) as img:
                                                pass  # Just checking if it can be opened
                                            samples.append((
                                                img_path,
                                                int(label_name)
                                            ))
                                        except Exception as e:
                                            print(f"Warning: Could not read file {img_path}: {e}")
                                    else:
                                        # Invalid label, use 0 as default
                                        img_path = os.path.join(test_dir, actual_file)
                                        # Check if file can be read
                                        try:
                                            with Image.open(img_path) as img:
                                                pass  # Just checking if it can be opened
                                            samples.append((
                                                img_path,
                                                0  # Temporary label
                                            ))
                                        except Exception as e:
                                            print(f"Warning: Could not read file {img_path}: {e}")
                        else:
                            # If no label column, use test images with temporary labels
                            for image_file in test_images:
                                img_path = os.path.join(test_dir, image_file)
                                # Check if file can be read
                                try:
                                    with Image.open(img_path) as img:
                                        pass  # Just checking if it can be opened
                                    samples.append((
                                        img_path,
                                        0  # Temporary label
                                    ))
                                except Exception as e:
                                    print(f"Warning: Could not read file {img_path}: {e}")
                    except Exception as e:
                        print(f"Error reading CSV: {e}")
                        # In case of error, use test images with temporary labels
                        for image_file in test_images:
                            img_path = os.path.join(test_dir, image_file)
                            # Check if file can be read
                            try:
                                with Image.open(img_path) as img:
                                    pass  # Just checking if it can be opened
                                samples.append((
                                    img_path,
                                    0  # Temporary label
                                ))
                            except Exception as e:
                                print(f"Warning: Could not read file {img_path}: {e}")
                else:
                    # If no CSV, use test images for evaluation but set labels to 0
                    for image_file in test_images:
                        img_path = os.path.join(test_dir, image_file)
                        # Check if file can be read
                        try:
                            with Image.open(img_path) as img:
                                pass  # Just checking if it can be opened
                            samples.append((
                                img_path,
                                0  # Temporary label
                            ))
                        except Exception as e:
                            print(f"Warning: Could not read file {img_path}: {e}")
            except Exception as e:
                print(f"Error processing test directory: {e}")
                
        # If dataset is empty, warn
        if not samples:
            print(f"Warning: {split} dataset is empty!")
            
        return samples
    
    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Args:
            idx (int): Index of the sample to get

        Returns:
            tuple: (image, class_index) where image is the transformed image
                  and class_index is the class label
        """
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
                
            return image, label
        except Exception as e:
            print(f"Image loading error: {img_path} - {e}")
            # Return a default image in case of error
            dummy_img = torch.zeros((3, 224, 224))
            return dummy_img, label

def get_data_transforms(input_size=224):
    """
    Get the data transformations for training and validation.

    Args:
        input_size (int): Size to resize the images to (default: 224)

    Returns:
        dict: Dictionary containing 'train' and 'val' transforms
    """
    # Define normalization parameters
    mean = [0.485, 0.456, 0.406]  # ImageNet mean
    std = [0.229, 0.224, 0.225]   # ImageNet std
    
    # Training transforms with data augmentation
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    # Validation transforms
    val_transforms = transforms.Compose([
        transforms.Resize(int(input_size * 1.14)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    return {
        'train': train_transforms,
        'val': val_transforms
    }

def create_dataloaders(data_dir, batch_size=32, num_workers=4):
    """
    Create training and validation DataLoaders.

    Args:
        data_dir (str): Root directory of the dataset
        batch_size (int): Batch size for training and validation
        num_workers (int): Number of worker processes for data loading

    Returns:
        tuple: (train_loader, val_loader, class_names)
    """
    # Get transforms
    transforms_dict = get_data_transforms()
    
    # Create datasets
    full_dataset = FlowerDataset(data_dir, transform=transforms_dict['train'])
    
    # Calculate split sizes
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)  # 80% for training
    val_size = total_size - train_size   # 20% for validation
    
    # Split the dataset
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # Override transforms for validation dataset
    val_dataset.dataset.transform = transforms_dict['val']
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, full_dataset.classes

def get_mean_std(data_dir, num_samples=1000):
    """
    Calculate mean and standard deviation of the dataset.

    Args:
        data_dir (str): Root directory of the dataset
        num_samples (int): Number of samples to use for calculation

    Returns:
        tuple: (mean, std) arrays for the dataset
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    
    dataset = FlowerDataset(data_dir, transform=transform)
    
    # Randomly sample images
    indices = torch.randperm(len(dataset))[:num_samples]
    
    # Calculate mean and std
    mean = torch.zeros(3)
    std = torch.zeros(3)
    
    for idx in indices:
        img, _ = dataset[idx]
        mean += img.mean([1, 2])
        std += img.std([1, 2])
    
    mean /= num_samples
    std /= num_samples
    
    return mean.numpy(), std.numpy() 