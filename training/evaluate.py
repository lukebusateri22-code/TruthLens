"""
Evaluation script for deepfake detection model
Load a trained model and evaluate on test data
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.deepfake_detector import get_model
from models.model_utils import load_checkpoint, calculate_metrics, print_metrics, save_metrics
from data.data_loader import create_data_loaders


def evaluate_model(model, test_loader, device):
    """
    Evaluate model on test data.
    
    Args:
        model: The model to evaluate
        test_loader: Test data loader
        device: Device to evaluate on
        
    Returns:
        Dictionary containing predictions, labels, and probabilities
    """
    model.eval()
    
    all_labels = []
    all_predictions = []
    all_probabilities = []
    
    print("\nEvaluating model...")
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Evaluating'):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            # Store results
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    results = {
        'labels': np.array(all_labels),
        'predictions': np.array(all_predictions),
        'probabilities': np.array(all_probabilities)
    }
    
    return results


def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path: Path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Real', 'Fake'],
                yticklabels=['Real', 'Fake'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrix saved to {save_path}")
    
    plt.close()


def plot_roc_curve(y_true, y_prob, save_path=None):
    """
    Plot ROC curve.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities for positive class
        save_path: Path to save the plot
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ ROC curve saved to {save_path}")
    
    plt.close()


def plot_prediction_distribution(y_true, y_prob, save_path=None):
    """
    Plot distribution of prediction probabilities.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities for positive class
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    # Separate probabilities by true label
    real_probs = y_prob[y_true == 0]
    fake_probs = y_prob[y_true == 1]
    
    plt.hist(real_probs, bins=50, alpha=0.5, label='Real', color='blue')
    plt.hist(fake_probs, bins=50, alpha=0.5, label='Fake', color='red')
    
    plt.xlabel('Predicted Probability (Fake)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Probabilities')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Prediction distribution saved to {save_path}")
    
    plt.close()


def main(args):
    """Main evaluation function."""
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"Evaluating Deepfake Detection Model")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"{'='*60}\n")
    
    # Create data loader
    print("Loading test data...")
    try:
        _, _, test_loader = create_data_loaders(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            image_size=args.image_size,
            use_videos=args.use_videos
        )
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Create model
    print("\nCreating model...")
    model = get_model(
        model_name=args.model,
        num_classes=2,
        pretrained=False,  # We'll load trained weights
        dropout_rate=args.dropout
    )
    model = model.to(device)
    
    # Load checkpoint
    print(f"\nLoading checkpoint from {args.checkpoint}...")
    try:
        load_checkpoint(model, None, args.checkpoint, device)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return
    
    # Evaluate
    results = evaluate_model(model, test_loader, device)
    
    # Calculate metrics
    metrics = calculate_metrics(
        results['labels'],
        results['predictions'],
        results['probabilities'][:, 1]  # Probability of fake class
    )
    
    # Print metrics
    print_metrics(metrics, "Test ")
    
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Save metrics
    metrics_path = os.path.join(args.results_dir, 'evaluation_metrics.json')
    save_metrics(metrics, metrics_path)
    
    # Generate visualizations
    if args.plot:
        print("\nGenerating visualizations...")
        
        # Confusion matrix
        cm_path = os.path.join(args.results_dir, 'confusion_matrix.png')
        plot_confusion_matrix(results['labels'], results['predictions'], cm_path)
        
        # ROC curve
        roc_path = os.path.join(args.results_dir, 'roc_curve.png')
        plot_roc_curve(results['labels'], results['probabilities'][:, 1], roc_path)
        
        # Prediction distribution
        dist_path = os.path.join(args.results_dir, 'prediction_distribution.png')
        plot_prediction_distribution(results['labels'], results['probabilities'][:, 1], dist_path)
    
    print(f"\n{'='*60}")
    print(f"Evaluation Complete!")
    print(f"Results saved to: {args.results_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate deepfake detection model')
    
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
    parser.add_argument('--dropout', type=float, default=0.5,
                       help='Dropout rate')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    
    # Evaluation parameters
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--plot', action='store_true', default=True,
                       help='Generate visualization plots')
    
    # Save parameters
    parser.add_argument('--results_dir', type=str, default='./results/evaluation',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    main(args)
