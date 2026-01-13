"""
Federated Learning Server Implementation
Aggregates model updates from clients and manages the global model
"""

import flwr as fl
from flwr.common import Metrics
from typing import List, Tuple, Dict, Optional
import numpy as np


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """
    Aggregate metrics from multiple clients using weighted average.
    
    Args:
        metrics: List of (num_examples, metrics_dict) tuples
        
    Returns:
        Aggregated metrics dictionary
    """
    # Get total number of examples
    total_examples = sum([num_examples for num_examples, _ in metrics])
    
    # Initialize aggregated metrics
    aggregated = {}
    
    # Get all metric keys from first client
    if len(metrics) > 0:
        metric_keys = metrics[0][1].keys()
        
        # Calculate weighted average for each metric
        for key in metric_keys:
            weighted_sum = sum([
                num_examples * m[key] 
                for num_examples, m in metrics
            ])
            aggregated[key] = weighted_sum / total_examples
    
    return aggregated


def get_evaluate_fn(model_fn, test_loader, device):
    """
    Create evaluation function for the server.
    
    Args:
        model_fn: Function that returns a new model instance
        test_loader: Test data loader
        device: Device to evaluate on
        
    Returns:
        Evaluation function
    """
    def evaluate(server_round: int, parameters, config):
        """
        Evaluate global model on centralized test set.
        
        Args:
            server_round: Current round number
            parameters: Model parameters
            config: Configuration dictionary
            
        Returns:
            Tuple of (loss, metrics_dict)
        """
        import torch
        import torch.nn as nn
        from collections import OrderedDict
        
        # Create model and load parameters
        model = model_fn()
        model = model.to(device)
        
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        
        # Evaluate
        model.eval()
        criterion = nn.CrossEntropyLoss()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        avg_loss = total_loss / len(test_loader.dataset)
        accuracy = correct / total
        
        print(f"\nRound {server_round} - Server Evaluation:")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  Accuracy: {accuracy:.4f}")
        
        return avg_loss, {"accuracy": accuracy}
    
    return evaluate


class FederatedServer:
    """
    Custom federated server with additional functionality.
    """
    
    def __init__(self, model_fn, test_loader, device, 
                 num_rounds: int = 10, fraction_fit: float = 1.0,
                 fraction_evaluate: float = 1.0, min_fit_clients: int = 2,
                 min_evaluate_clients: int = 2, min_available_clients: int = 2):
        """
        Initialize federated server.
        
        Args:
            model_fn: Function that returns a new model instance
            test_loader: Test data loader for server-side evaluation
            device: Device to use
            num_rounds: Number of federated rounds
            fraction_fit: Fraction of clients to sample for training
            fraction_evaluate: Fraction of clients to sample for evaluation
            min_fit_clients: Minimum number of clients for training
            min_evaluate_clients: Minimum number of clients for evaluation
            min_available_clients: Minimum number of available clients
        """
        self.model_fn = model_fn
        self.test_loader = test_loader
        self.device = device
        self.num_rounds = num_rounds
        
        # Create strategy
        self.strategy = fl.server.strategy.FedAvg(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=get_evaluate_fn(model_fn, test_loader, device),
            fit_metrics_aggregation_fn=weighted_average,
            evaluate_metrics_aggregation_fn=weighted_average,
        )
        
        print(f"Federated Server initialized:")
        print(f"  Rounds: {num_rounds}")
        print(f"  Strategy: FedAvg")
        print(f"  Min clients: {min_available_clients}")
    
    def get_strategy(self):
        """Get the federated learning strategy."""
        return self.strategy


def create_strategy(model_fn, test_loader, device,
                   fraction_fit: float = 1.0,
                   fraction_evaluate: float = 1.0,
                   min_fit_clients: int = 2,
                   min_evaluate_clients: int = 2,
                   min_available_clients: int = 2) -> fl.server.strategy.Strategy:
    """
    Create a federated learning strategy.
    
    Args:
        model_fn: Function that returns a new model instance
        test_loader: Test data loader
        device: Device to use
        fraction_fit: Fraction of clients to sample for training
        fraction_evaluate: Fraction of clients to sample for evaluation
        min_fit_clients: Minimum number of clients for training
        min_evaluate_clients: Minimum number of clients for evaluation
        min_available_clients: Minimum number of available clients
        
    Returns:
        Flower strategy
    """
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_evaluate_clients,
        min_available_clients=min_available_clients,
        evaluate_fn=get_evaluate_fn(model_fn, test_loader, device),
        fit_metrics_aggregation_fn=weighted_average,
        evaluate_metrics_aggregation_fn=weighted_average,
    )
    
    return strategy


class CustomFedAvg(fl.server.strategy.FedAvg):
    """
    Custom FedAvg strategy with additional features.
    Can be extended to implement differential privacy, secure aggregation, etc.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.round_losses = []
        self.round_accuracies = []
    
    def aggregate_fit(self, server_round, results, failures):
        """Aggregate training results with custom logic."""
        # Call parent aggregation
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )
        
        # Store metrics
        if aggregated_metrics is not None:
            if 'loss' in aggregated_metrics:
                self.round_losses.append(aggregated_metrics['loss'])
            if 'accuracy' in aggregated_metrics:
                self.round_accuracies.append(aggregated_metrics['accuracy'])
        
        # Print round summary
        print(f"\n{'='*60}")
        print(f"Round {server_round} Summary:")
        if aggregated_metrics:
            for key, value in aggregated_metrics.items():
                print(f"  {key.capitalize()}: {value:.4f}")
        print(f"{'='*60}\n")
        
        return aggregated_parameters, aggregated_metrics
    
    def aggregate_evaluate(self, server_round, results, failures):
        """Aggregate evaluation results with custom logic."""
        # Call parent aggregation
        loss, aggregated_metrics = super().aggregate_evaluate(
            server_round, results, failures
        )
        
        return loss, aggregated_metrics


if __name__ == "__main__":
    print("Federated Learning Server Module")
    print("This module defines the server-side logic for federated learning.")
    print("\nKey features:")
    print("  - FedAvg strategy for model aggregation")
    print("  - Weighted averaging of client updates")
    print("  - Centralized evaluation on test set")
    print("  - Customizable aggregation strategies")
    print("\n✓ Server module ready for use!")
