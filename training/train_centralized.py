"""
Centralized training script for deepfake detection model
Train the model using standard supervised learning
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from tqdm import tqdm
import numpy as np
import json
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.deepfake_detector import get_model
from models.model_utils import (
    save_checkpoint, calculate_metrics, print_metrics, 
    save_metrics, EarlyStopping, AverageMeter, count_parameters
)
from data.data_loader import create_data_loaders


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """
    Train for one epoch.
    
    Args:
        model: The model to train
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
        
    Returns:
        Average loss for the epoch
    """
    model.train()
    
    losses = AverageMeter()
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
    
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update metrics
        losses.update(loss.item(), images.size(0))
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{losses.avg:.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    accuracy = 100. * correct / total
    return losses.avg, accuracy


def validate(model, val_loader, criterion, device, epoch=0):
    """
    Validate the model.
    
    Args:
        model: The model to validate
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to validate on
        epoch: Current epoch number
        
    Returns:
        Tuple of (average_loss, metrics_dict)
    """
    model.eval()
    
    losses = AverageMeter()
    all_labels = []
    all_predictions = []
    all_probabilities = []
    
    pbar = tqdm(val_loader, desc=f'Epoch {epoch} [Val]')
    
    with torch.no_grad():
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Get predictions and probabilities
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            # Store for metrics calculation
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_probabilities.extend(probabilities[:, 1].cpu().numpy())  # Probability of fake class
            
            # Update metrics
            losses.update(loss.item(), images.size(0))
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{losses.avg:.4f}'})
    
    # Calculate comprehensive metrics
    metrics = calculate_metrics(
        np.array(all_labels),
        np.array(all_predictions),
        np.array(all_probabilities)
    )
    
    return losses.avg, metrics


def main(args):
    """Main training function."""
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"Training Deepfake Detection Model (Centralized)")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"{'='*60}\n")
    
    # Create data loaders
    print("Loading data...")
    try:
        train_loader, val_loader, test_loader = create_data_loaders(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            image_size=args.image_size,
            use_videos=args.use_videos
        )
    except Exception as e:
        print(f"Error loading data: {e}")
        print("\nPlease ensure your data is organized as:")
        print("  data_dir/")
        print("    train/")
        print("      real/  (real images)")
        print("      fake/  (fake images)")
        print("    val/")
        print("      real/")
        print("      fake/")
        print("    test/")
        print("      real/")
        print("      fake/")
        return
    
    # Create model
    print("\nCreating model...")
    model = get_model(
        model_name=args.model,
        num_classes=2,
        pretrained=args.pretrained,
        dropout_rate=args.dropout
    )
    model = model.to(device)
    
    # Print model info
    total_params, trainable_params = count_parameters(model)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")
    
    # Learning rate scheduler
    if args.scheduler == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    elif args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    else:
        scheduler = None
    
    # Early stopping
    early_stopping = EarlyStopping(patience=args.patience, mode='max', verbose=True)
    
    # Create directories for saving
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Training loop
    print("\nStarting training...\n")
    best_accuracy = 0.0
    training_history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_metrics': []
    }
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 60)
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        val_loss, val_metrics = validate(
            model, val_loader, criterion, device, epoch
        )
        
        val_acc = val_metrics['accuracy'] * 100
        
        # Update learning rate
        if scheduler is not None:
            if args.scheduler == 'plateau':
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # Print epoch summary
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"  Val Precision: {val_metrics['precision']:.4f}")
        print(f"  Val Recall: {val_metrics['recall']:.4f}")
        print(f"  Val F1-Score: {val_metrics['f1_score']:.4f}")
        
        # Save history
        training_history['train_loss'].append(train_loss)
        training_history['train_acc'].append(train_acc)
        training_history['val_loss'].append(val_loss)
        training_history['val_acc'].append(val_acc)
        training_history['val_metrics'].append(val_metrics)
        
        # Save checkpoint
        is_best = val_acc > best_accuracy
        if is_best:
            best_accuracy = val_acc
        
        checkpoint_path = os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
        save_checkpoint(
            model, optimizer, epoch, val_loss, val_acc,
            checkpoint_path, is_best=is_best
        )
        
        # Early stopping check
        if early_stopping(val_acc):
            print(f"\nEarly stopping triggered at epoch {epoch}")
            break
    
    # Save training history
    history_path = os.path.join(args.results_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=4)
    print(f"\n✓ Training history saved to {history_path}")
    
    # Final evaluation on test set
    print("\n" + "="*60)
    print("Final Evaluation on Test Set")
    print("="*60)
    
    test_loss, test_metrics = validate(model, test_loader, criterion, device)
    print_metrics(test_metrics, "Test ")
    
    # Save test metrics
    metrics_path = os.path.join(args.results_dir, 'test_metrics.json')
    save_metrics(test_metrics, metrics_path)
    
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"Best Validation Accuracy: {best_accuracy:.2f}%")
    print(f"Test Accuracy: {test_metrics['accuracy']*100:.2f}%")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train deepfake detection model (centralized)')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='Path to data directory')
    parser.add_argument('--image_size', type=int, default=224,
                       help='Input image size')
    parser.add_argument('--use_videos', action='store_true',
                       help='Use videos instead of images')
    
    # Model parameters
    parser.add_argument('--model', type=str, default='efficientnet',
                       choices=['efficientnet', 'hybrid', 'resnet'],
                       help='Model architecture')
    parser.add_argument('--pretrained', type=bool, default=True,
                       help='Use pretrained weights')
    parser.add_argument('--dropout', type=float, default=0.5,
                       help='Dropout rate')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='adam',
                       choices=['adam', 'sgd', 'adamw'],
                       help='Optimizer')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--scheduler', type=str, default='plateau',
                       choices=['plateau', 'cosine', 'none'],
                       help='Learning rate scheduler')
    parser.add_argument('--patience', type=int, default=7,
                       help='Early stopping patience')
    
    # System parameters
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    # Save parameters
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/centralized',
                       help='Directory to save checkpoints')
    parser.add_argument('--results_dir', type=str, default='./results/centralized',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    main(args)
