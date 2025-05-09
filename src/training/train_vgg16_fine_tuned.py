"""
VGG16 Fine-Tuned Model Training Script

This script trains a VGG16-based model using fine-tuning. The early layers are
frozen while later layers are fine-tuned for the specific task.

"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import matplotlib.pyplot as plt

from ..models.vgg_models import VGG16FineTuned
from ..utils.data_loader import create_dataloaders
from ..utils.trainer import ModelTrainer
from ..visualizations.feature_visualization import FeatureVisualizer


def save_metrics(metrics, output_dir):
    """
    Save training metrics to a file.

    Args:
        metrics (dict): Dictionary containing training metrics
        output_dir (str): Directory to save the metrics file
    """
    # Ensure metrics directory exists
    metrics_dir = os.path.join(output_dir, 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)
    
    metrics_file = os.path.join(metrics_dir, 'vgg16_fine_tuned_metrics.txt')
    
    with open(metrics_file, 'w') as f:
        f.write('Model: VGG16 Fine-Tuned\n')
        f.write(f'Training Time: {metrics["training_time"]:.2f} seconds\n')
        f.write(f'Best Validation Accuracy: {metrics["best_val_acc"]:.4f}\n')
        f.write(f'Final Training Loss: {metrics["train_history"]["loss"][-1]:.4f}\n')
        f.write(f'Final Training Accuracy: {metrics["train_history"]["acc"][-1]:.4f}\n')
        f.write(f'Final Validation Loss: {metrics["val_history"]["loss"][-1]:.4f}\n')
        f.write(f'Final Validation Accuracy: {metrics["val_history"]["acc"][-1]:.4f}\n')


def plot_training_history(metrics, output_dir):
    """
    Plot training and validation metrics.

    Args:
        metrics (dict): Dictionary containing training metrics
        output_dir (str): Directory to save the plots
    """
    # Ensure visualizations directory exists
    vis_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(metrics['train_history']['loss'], label='Training Loss')
    plt.plot(metrics['val_history']['loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(metrics['train_history']['acc'], label='Training Accuracy')
    plt.plot(metrics['val_history']['acc'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'vgg16_fine_tuned_history.png'))
    plt.close()


def main(args):
    """
    Main training function.

    Args:
        args: Parsed command line arguments
    """
    # Create output directories if they don't exist
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'metrics'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'visualizations'), exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    
    # Create data loaders
    train_loader, val_loader, class_names = create_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Create model
    model = VGG16FineTuned(num_classes=len(class_names))
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    # Only optimize parameters that require gradients
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Create trainer
    trainer = ModelTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=args.epochs
    )
    
    # Model save path
    model_save_path = os.path.join(args.output_dir, 'models')
    
    # Train model
    print(f"Starting training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    metrics = trainer.train(train_loader, val_loader, model_save_path, model_name="vgg16_fine_tuned_best.pth")
    
    # Save metrics and plots
    save_metrics(metrics, args.output_dir)
    plot_training_history(metrics, args.output_dir)
    
    # Create feature visualizations
    visualizer = FeatureVisualizer(model, device)
    
    # Ensure visualizations directory exists
    vis_dir = os.path.join(args.output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Save feature map visualizations
    if len(train_loader) > 0:
        # Get a sample image
        images, _ = next(iter(train_loader))
        image = images[0:1].to(device)
        
        # Get feature maps
        feature_maps = visualizer.visualize_feature_maps(image)
        
        # Save visualization
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(feature_maps, cmap='viridis')
        ax.set_aspect('equal')  # For preserving aspect ratio
        ax.axis('off')
        ax.set_title('VGG16 Fine-Tuned Feature Maps')
        plt.savefig(os.path.join(vis_dir, 'vgg16_fine_tuned_maps.png'))
        plt.close()
        
        # Compute and save Grad-CAM visualization
        if hasattr(visualizer, 'compute_gradcam'):
            gradcam = visualizer.compute_gradcam(image, target_class=0)
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(1, 1, 1)
            ax.imshow(gradcam, cmap='jet')
            ax.set_aspect('equal')  # For preserving aspect ratio
            ax.axis('off')
            ax.set_title('VGG16 Fine-Tuned Grad-CAM')
            plt.savefig(os.path.join(vis_dir, 'vgg16_fine_tuned_gradcam.png'))
            plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train VGG16 Fine-Tuned Model')
    
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Directory containing the dataset')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Directory to save outputs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Training batch size')
    parser.add_argument('--epochs', type=int, default=25,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.00001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-3,
                        help='Weight decay for optimizer')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Disable CUDA training')
    
    args = parser.parse_args()
    main(args) 