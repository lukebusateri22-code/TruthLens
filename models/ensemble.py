"""
Ensemble Model - Combines multiple models for better accuracy
Uses voting or averaging to make final predictions
"""

import torch
import torch.nn as nn
from typing import List
import numpy as np

class EnsembleModel(nn.Module):
    """
    Ensemble of multiple deepfake detection models.
    Combines predictions using voting or averaging.
    """
    
    def __init__(self, models: List[nn.Module], method='vote'):
        """
        Args:
            models: List of trained models
            method: 'vote' (majority voting) or 'average' (average probabilities)
        """
        super().__init__()
        self.models = nn.ModuleList(models)
        self.method = method
        self.num_models = len(models)
    
    def forward(self, x):
        """
        Forward pass through all models.
        
        Args:
            x: Input tensor
        
        Returns:
            Ensemble prediction
        """
        # Get predictions from all models
        predictions = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(x)
                predictions.append(pred)
        
        # Stack predictions
        predictions = torch.stack(predictions)  # [num_models, batch_size, num_classes]
        
        if self.method == 'vote':
            # Majority voting
            votes = torch.argmax(predictions, dim=2)  # [num_models, batch_size]
            # Count votes for each class
            final_pred = torch.mode(votes, dim=0)[0]  # [batch_size]
            # Convert to one-hot for compatibility
            final_pred = torch.nn.functional.one_hot(final_pred, num_classes=predictions.shape[2])
            return final_pred.float()
        
        elif self.method == 'average':
            # Average probabilities
            probs = torch.softmax(predictions, dim=2)  # [num_models, batch_size, num_classes]
            avg_probs = torch.mean(probs, dim=0)  # [batch_size, num_classes]
            return torch.log(avg_probs + 1e-10)  # Return log probabilities
        
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def predict(self, x):
        """
        Make prediction with ensemble.
        
        Args:
            x: Input tensor
        
        Returns:
            Class predictions
        """
        output = self.forward(x)
        return torch.argmax(output, dim=1)


class WeightedEnsemble(nn.Module):
    """
    Weighted ensemble - learns optimal weights for each model.
    """
    
    def __init__(self, models: List[nn.Module], num_classes=2):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.num_models = len(models)
        self.num_classes = num_classes
        
        # Learnable weights for each model
        self.weights = nn.Parameter(torch.ones(self.num_models) / self.num_models)
    
    def forward(self, x):
        """Forward pass with learned weights."""
        predictions = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(x)
                predictions.append(pred)
        
        predictions = torch.stack(predictions)  # [num_models, batch_size, num_classes]
        
        # Apply softmax to weights
        weights = torch.softmax(self.weights, dim=0)
        
        # Weighted average
        weighted_pred = torch.zeros_like(predictions[0])
        for i, pred in enumerate(predictions):
            weighted_pred += weights[i] * torch.softmax(pred, dim=1)
        
        return torch.log(weighted_pred + 1e-10)


def create_ensemble(model_paths: List[str], model_class, method='average', device='cpu'):
    """
    Create ensemble from saved model checkpoints.
    
    Args:
        model_paths: List of paths to saved models
        model_class: Model class to instantiate
        method: Ensemble method ('vote', 'average', or 'weighted')
        device: Device to load models on
    
    Returns:
        Ensemble model
    """
    models = []
    
    for path in model_paths:
        model = model_class()
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
        model.eval()
        models.append(model)
    
    if method == 'weighted':
        ensemble = WeightedEnsemble(models)
    else:
        ensemble = EnsembleModel(models, method=method)
    
    return ensemble.to(device)


# Example usage
if __name__ == "__main__":
    print("Ensemble Model Module")
    print("=" * 50)
    print("\nEnsemble Methods:")
    print("  1. Majority Voting - Each model votes, majority wins")
    print("  2. Average - Average probabilities from all models")
    print("  3. Weighted - Learn optimal weights for each model")
    print("\nBenefits:")
    print("  ✓ Higher accuracy (typically 2-5% improvement)")
    print("  ✓ More robust predictions")
    print("  ✓ Reduces individual model errors")
    print("\nUsage:")
    print("  ensemble = create_ensemble(")
    print("      ['model1.pth', 'model2.pth', 'model3.pth'],")
    print("      model_class=SimpleDeepfakeDetector,")
    print("      method='average'")
    print("  )")
