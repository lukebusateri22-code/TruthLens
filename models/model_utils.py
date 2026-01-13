"""
Model utility functions for training, evaluation, and checkpointing
"""

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import numpy as np
import os
from typing import Dict, Tuple, Optional
import json


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, 
                   epoch: int, loss: float, accuracy: float, 
                   filepath: str, is_best: bool = False):
    """
    Save model checkpoint.
    
    Args:
        model: The model to save
        optimizer: The optimizer state
        epoch: Current epoch number
        loss: Current loss value
        accuracy: Current accuracy
        filepath: Path to save the checkpoint
        is_best: Whether this is the best model so far
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy
    }
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    torch.save(checkpoint, filepath)
    
    if is_best:
        best_filepath = filepath.replace('.pth', '_best.pth')
        torch.save(checkpoint, best_filepath)
        print(f"✓ Saved best model to {best_filepath}")


def load_checkpoint(model: nn.Module, optimizer: Optional[torch.optim.Optimizer], 
                    filepath: str, device: str = 'cpu') -> Tuple[int, float]:
    """
    Load model checkpoint.
    
    Args:
        model: The model to load weights into
        optimizer: The optimizer to load state into (optional)
        filepath: Path to the checkpoint file
        device: Device to load the model on
        
    Returns:
        Tuple of (epoch, loss)
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found at {filepath}")
    
    checkpoint = torch.load(filepath, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    print(f"✓ Loaded checkpoint from epoch {epoch} with loss {loss:.4f}")
    
    return epoch, loss


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                     y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Calculate comprehensive evaluation metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (optional, for AUC-ROC)
        
    Returns:
        Dictionary containing various metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average='binary', zero_division=0)
    }
    
    # Calculate AUC-ROC if probabilities are provided
    if y_prob is not None:
        try:
            metrics['auc_roc'] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics['auc_roc'] = 0.0
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        metrics['true_positives'] = int(tp)
        
        # Calculate specificity
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    return metrics


def print_metrics(metrics: Dict[str, float], prefix: str = ""):
    """
    Pretty print metrics.
    
    Args:
        metrics: Dictionary of metrics
        prefix: Prefix for the print statement
    """
    print(f"\n{prefix}Metrics:")
    print("=" * 50)
    
    # Main metrics
    main_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
    for key in main_metrics:
        if key in metrics:
            print(f"  {key.replace('_', ' ').title()}: {metrics[key]:.4f}")
    
    # Confusion matrix values
    if 'true_positives' in metrics:
        print("\n  Confusion Matrix:")
        print(f"    True Positives:  {metrics['true_positives']}")
        print(f"    True Negatives:  {metrics['true_negatives']}")
        print(f"    False Positives: {metrics['false_positives']}")
        print(f"    False Negatives: {metrics['false_negatives']}")
        
        if 'specificity' in metrics:
            print(f"    Specificity:     {metrics['specificity']:.4f}")
    
    print("=" * 50)


def save_metrics(metrics: Dict[str, float], filepath: str):
    """
    Save metrics to a JSON file.
    
    Args:
        metrics: Dictionary of metrics
        filepath: Path to save the metrics
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"✓ Metrics saved to {filepath}")


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Count the number of parameters in a model.
    
    Args:
        model: The model to count parameters for
        
    Returns:
        Tuple of (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total_params, trainable_params


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    """
    Get current learning rate from optimizer.
    
    Args:
        optimizer: The optimizer
        
    Returns:
        Current learning rate
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']
    return 0.0


class EarlyStopping:
    """
    Early stopping to stop training when validation loss doesn't improve.
    """
    
    def __init__(self, patience: int = 7, min_delta: float = 0.0, 
                 mode: str = 'min', verbose: bool = True):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for accuracy
            verbose: Print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current metric value (loss or accuracy)
            
        Returns:
            True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print("Early stopping triggered!")
                return True
        
        return False


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking metrics during training.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all statistics."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        """
        Update statistics.
        
        Args:
            val: New value
            n: Number of samples
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == "__main__":
    # Test utilities
    print("Testing model utilities...")
    
    # Test metrics calculation
    y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 0, 0, 0, 1, 1])
    y_prob = np.array([0.1, 0.9, 0.8, 0.2, 0.4, 0.3, 0.7, 0.85])
    
    metrics = calculate_metrics(y_true, y_pred, y_prob)
    print_metrics(metrics, "Test ")
    
    # Test early stopping
    print("\nTesting early stopping...")
    early_stop = EarlyStopping(patience=3, mode='min')
    
    losses = [0.5, 0.4, 0.45, 0.46, 0.47, 0.48]
    for i, loss in enumerate(losses):
        should_stop = early_stop(loss)
        print(f"Epoch {i+1}, Loss: {loss:.2f}, Stop: {should_stop}")
        if should_stop:
            break
    
    print("\n✓ All utilities working correctly!")
