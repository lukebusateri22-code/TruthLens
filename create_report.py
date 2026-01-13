"""
Comprehensive Evaluation Report Generator
Creates a detailed PDF/HTML report with all metrics and visualizations
"""

import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve
)
import json
from datetime import datetime

class ReportGenerator:
    """Generate comprehensive evaluation reports."""
    
    def __init__(self, model, test_loader, device='cpu'):
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.results = {}
    
    def evaluate(self):
        """Run full evaluation."""
        print("Running comprehensive evaluation...")
        
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(self.device)
                outputs = self.model(images)
                probs = torch.softmax(outputs, dim=1)
                preds = outputs.argmax(dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of fake class
        
        self.results['predictions'] = np.array(all_preds)
        self.results['labels'] = np.array(all_labels)
        self.results['probabilities'] = np.array(all_probs)
        
        # Calculate metrics
        self._calculate_metrics()
        
        return self.results
    
    def _calculate_metrics(self):
        """Calculate all evaluation metrics."""
        y_true = self.results['labels']
        y_pred = self.results['predictions']
        y_prob = self.results['probabilities']
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        self.results['confusion_matrix'] = cm
        
        # Classification report
        report = classification_report(y_true, y_pred, 
                                      target_names=['Real', 'Fake'],
                                      output_dict=True)
        self.results['classification_report'] = report
        
        # ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        self.results['roc'] = {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc}
        
        # Precision-Recall curve
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        self.results['pr_curve'] = {'precision': precision, 'recall': recall}
        
        # Overall metrics
        accuracy = (y_true == y_pred).mean()
        self.results['accuracy'] = accuracy
    
    def generate_visualizations(self, output_dir='./results'):
        """Generate all visualization plots."""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # 1. Confusion Matrix
        self._plot_confusion_matrix(output_dir / 'confusion_matrix.png')
        
        # 2. ROC Curve
        self._plot_roc_curve(output_dir / 'roc_curve.png')
        
        # 3. Precision-Recall Curve
        self._plot_pr_curve(output_dir / 'pr_curve.png')
        
        # 4. Prediction Distribution
        self._plot_prediction_distribution(output_dir / 'prediction_dist.png')
        
        print(f"Visualizations saved to {output_dir}")
    
    def _plot_confusion_matrix(self, save_path):
        """Plot confusion matrix."""
        cm = self.results['confusion_matrix']
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Real', 'Fake'],
                   yticklabels=['Real', 'Fake'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
    
    def _plot_roc_curve(self, save_path):
        """Plot ROC curve."""
        roc = self.results['roc']
        
        plt.figure(figsize=(8, 6))
        plt.plot(roc['fpr'], roc['tpr'], 
                label=f"AUC = {roc['auc']:.4f}", linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
    
    def _plot_pr_curve(self, save_path):
        """Plot Precision-Recall curve."""
        pr = self.results['pr_curve']
        
        plt.figure(figsize=(8, 6))
        plt.plot(pr['recall'], pr['precision'], linewidth=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
    
    def _plot_prediction_distribution(self, save_path):
        """Plot prediction probability distribution."""
        y_true = self.results['labels']
        y_prob = self.results['probabilities']
        
        plt.figure(figsize=(10, 6))
        
        # Real images
        real_probs = y_prob[y_true == 0]
        plt.hist(real_probs, bins=50, alpha=0.5, label='Real Images', color='blue')
        
        # Fake images
        fake_probs = y_prob[y_true == 1]
        plt.hist(fake_probs, bins=50, alpha=0.5, label='Fake Images', color='red')
        
        plt.xlabel('Predicted Probability (Fake)')
        plt.ylabel('Count')
        plt.title('Prediction Distribution')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
    
    def save_report(self, output_path='./results/report.json'):
        """Save report as JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(exist_ok=True)
        
        # Prepare serializable results
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'accuracy': float(self.results['accuracy']),
            'roc_auc': float(self.results['roc']['auc']),
            'classification_report': self.results['classification_report'],
            'confusion_matrix': self.results['confusion_matrix'].tolist()
        }
        
        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"Report saved to {output_path}")
    
    def print_summary(self):
        """Print summary to console."""
        print("\n" + "="*70)
        print("EVALUATION SUMMARY")
        print("="*70)
        
        print(f"\nOverall Accuracy: {self.results['accuracy']:.4f}")
        print(f"ROC AUC: {self.results['roc']['auc']:.4f}")
        
        print("\nPer-Class Metrics:")
        report = self.results['classification_report']
        for class_name in ['Real', 'Fake']:
            metrics = report[class_name]
            print(f"\n{class_name}:")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1-Score: {metrics['f1-score']:.4f}")
        
        print("\nConfusion Matrix:")
        cm = self.results['confusion_matrix']
        print(f"  True Negatives:  {cm[0,0]:5d}")
        print(f"  False Positives: {cm[0,1]:5d}")
        print(f"  False Negatives: {cm[1,0]:5d}")
        print(f"  True Positives:  {cm[1,1]:5d}")
        
        print("="*70 + "\n")


def generate_full_report(model, test_loader, device='cpu', output_dir='./results'):
    """
    Generate complete evaluation report.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device to run on
        output_dir: Output directory for results
    """
    generator = ReportGenerator(model, test_loader, device)
    
    # Run evaluation
    generator.evaluate()
    
    # Generate visualizations
    generator.generate_visualizations(output_dir)
    
    # Save report
    generator.save_report(Path(output_dir) / 'report.json')
    
    # Print summary
    generator.print_summary()
    
    return generator.results


# Example usage
if __name__ == "__main__":
    print("Comprehensive Report Generator")
    print("=" * 50)
    print("\nGenerates:")
    print("  ✓ Confusion Matrix")
    print("  ✓ ROC Curve")
    print("  ✓ Precision-Recall Curve")
    print("  ✓ Prediction Distribution")
    print("  ✓ Detailed Metrics (JSON)")
    print("\nUsage:")
    print("  from create_report import generate_full_report")
    print("  results = generate_full_report(model, test_loader)")
