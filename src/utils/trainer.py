"""
Model Training and Evaluation Utilities

This module provides utilities for training and evaluating deep learning models,
including training loops, validation, and metric calculation.


"""

import time
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os

def train_model(model, train_loader, valid_loader, criterion, optimizer, 
                num_epochs=25, device='cuda', scheduler=None, model_name='model'):
    """
    Function used for model training
    
    Args:
        model: Model to be trained
        train_loader: Training data loader
        valid_loader: Validation data loader
        criterion: Loss function
        optimizer: Optimization algorithm
        num_epochs (int): Number of epochs
        device (str): Device used for training ('cuda' or 'cpu')
        scheduler: Learning rate scheduler (optional)
        model_name (str): Filename to save the model
        
    Returns:
        model: Trained model
        history (dict): Training history
        training_time (float): Training duration (seconds)
    """
    start_time = time.time()
    
    # Create scaler for mixed precision if using GPU
    scaler = GradScaler() if device == 'cuda' else None
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # Best model parameters
    best_model_params = None
    best_val_acc = 0.0
    
    # Set model to training mode
    model = model.to(device)
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Loop over training data loader
        train_progress = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for inputs, labels in train_progress:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Use mixed precision for GPU
            if device == 'cuda':
                with autocast():
                    # Forward pass
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                
                # Backpropagation (mixed precision)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Regular training for CPU
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            # Update statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            train_progress.set_postfix({'loss': loss.item(), 'acc': correct/total})
            
        # End of epoch training statistics
        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = correct / total
        
        # Update scheduler
        if scheduler is not None:
            scheduler.step()
            
        # Validation phase
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Validation without gradient calculation
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                # Let's use autocast for mixed precision
                if device == 'cuda':
                    with autocast():
                        # Forward pass
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                
                # Update statistics
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        # End of epoch validation statistics
        epoch_val_loss = running_loss / len(valid_loader.dataset)
        epoch_val_acc = correct / total
        
        # Update training history
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        
        # Print training progress
        print(f'Epoch {epoch+1}/{num_epochs} - '
              f'Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}, '
              f'Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}')
        
        # Save the best model
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            best_model_params = model.state_dict().copy()
            # Save the model
            model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'output', 'models')
            os.makedirs(model_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(model_dir, f'{model_name}_best.pth'))
            print(f'Model saved with val_acc: {best_val_acc:.4f}')
    
    # Calculate training time
    training_time = time.time() - start_time
    print(f'Training completed in {training_time:.2f} s')
    
    # Load the best model
    if best_model_params is not None:
        model.load_state_dict(best_model_params)
    
    return model, history, training_time

def evaluate_model(model, test_loader, device='cuda'):
    """
    Model evaluation function
    
    Args:
        model: Model to evaluate
        test_loader: Test data loader
        device (str): Device used for evaluation ('cuda' or 'cpu')
        
    Returns:
        metrics (dict): Evaluation metrics (accuracy, precision, recall, f1)
    """
    model.eval()
    model = model.to(device)
    
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    
    # Calculate classification metrics
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    
    metrics = {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    print(f'Model Evaluation:')
    print(f'Accuracy: {acc:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    
    return metrics

class ModelTrainer:
    """
    A trainer class to handle model training and evaluation.
    
    Attributes:
        model (nn.Module): The neural network model to train
        criterion (nn.Module): Loss function
        optimizer (optim.Optimizer): Optimizer for training
        scheduler (optim.lr_scheduler): Learning rate scheduler
        device (torch.device): Device to train on (CPU/GPU)
        num_epochs (int): Number of training epochs
        best_val_acc (float): Best validation accuracy achieved
    """

    def __init__(self, model, criterion, optimizer, device, num_epochs=25):
        """
        Initialize the trainer.

        Args:
            model (nn.Module): The model to train
            criterion (nn.Module): Loss function
            optimizer (optim.Optimizer): Optimizer
            device (torch.device): Device to train on
            num_epochs (int): Number of epochs to train for
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.num_epochs = num_epochs
        
        # Initialize learning rate scheduler
        self.scheduler = ReduceLROnPlateau(optimizer, mode='max', 
                                         factor=0.1, patience=5, 
                                         verbose=True)
        
        self.best_val_acc = 0.0
        self.best_val_loss = float('inf')
        self.train_history = {'loss': [], 'acc': []}
        self.val_history = {'loss': [], 'acc': []}

    def train_epoch(self, train_loader):
        """
        Train the model for one epoch.

        Args:
            train_loader (DataLoader): Training data loader

        Returns:
            tuple: (epoch_loss, epoch_acc) Average loss and accuracy for the epoch
        """
        self.model.train()
        running_loss = 0.0
        running_corrects = 0
        
        # Use tqdm for progress bar
        pbar = tqdm(train_loader, desc='Training')
        
        for inputs, labels in pbar:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            # Zero the parameter gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            with torch.set_grad_enabled(True):
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = self.criterion(outputs, labels)
                
                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        
        return epoch_loss, epoch_acc.item()

    def validate(self, val_loader):
        """
        Validate the model.

        Args:
            val_loader (DataLoader): Validation data loader

        Returns:
            tuple: (val_loss, val_acc) Validation loss and accuracy
        """
        self.model.eval()
        running_loss = 0.0
        running_corrects = 0
        
        # No gradient computation for validation
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = self.criterion(outputs, labels)
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
        
        val_loss = running_loss / len(val_loader.dataset)
        val_acc = running_corrects.double() / len(val_loader.dataset)
        
        return val_loss, val_acc.item()

    def train(self, train_loader, val_loader, output_dir, model_name):
        """
        Train the model for the specified number of epochs.

        Args:
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader): Validation data loader
            output_dir (str): Directory to save model checkpoints
            model_name (str): Name of the saved model file

        Returns:
            dict: Training history containing loss and accuracy metrics
        """
        print(f"Training on {self.device}")
        start_time = time.time()
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Early stopping parameters
        patience = 10  # Number of epochs to wait for improvement
        patience_counter = 0
        best_val_loss = float('inf')

        for epoch in range(self.num_epochs):
            print(f'\nEpoch {epoch+1}/{self.num_epochs}')
            print('-' * 10)
            
            # Training phase
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validation phase
            val_loss, val_acc = self.validate(val_loader)
            
            # Update learning rate
            self.scheduler.step(val_acc)
            
            # Save history
            self.train_history['loss'].append(train_loss)
            self.train_history['acc'].append(train_acc)
            self.val_history['loss'].append(val_loss)
            self.val_history['acc'].append(val_acc)
            
            print(f'Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}')
            print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                torch.save(self.model.state_dict(), 
                         os.path.join(output_dir, model_name))
                
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            # If no improvement in validation loss for 'patience' epochs, stop training
            if patience_counter >= patience:
                print(f'\nEarly stopping triggered after {epoch+1} epochs!')
                print(f'No improvement in validation loss for {patience} consecutive epochs.')
                break
        
        time_elapsed = time.time() - start_time
        print(f'\nTraining completed in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s')
        print(f'Best val Acc: {self.best_val_acc:.4f}')
        print(f'Best val Loss: {self.best_val_loss:.4f}')
        
        return {
            'train_history': self.train_history,
            'val_history': self.val_history,
            'best_val_acc': self.best_val_acc,
            'best_val_loss': self.best_val_loss,
            'training_time': time_elapsed
        }

    def evaluate(self, test_loader):
        """
        Evaluate the model on test data.

        Args:
            test_loader (DataLoader): Test data loader

        Returns:
            dict: Dictionary containing evaluation metrics
        """
        self.model.eval()
        
        all_preds = []
        all_labels = []
        test_loss = 0.0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = self.criterion(outputs, labels)
                
                test_loss += loss.item() * inputs.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        test_loss = test_loss / len(test_loader.dataset)
        accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
        
        # Calculate per-class metrics
        classes = len(set(all_labels))
        per_class_acc = []
        
        for i in range(classes):
            mask = np.array(all_labels) == i
            class_acc = np.mean(np.array(all_preds)[mask] == np.array(all_labels)[mask])
            per_class_acc.append(class_acc)
        
        return {
            'test_loss': test_loss,
            'accuracy': accuracy,
            'per_class_accuracy': per_class_acc
        } 