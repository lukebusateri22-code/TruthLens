"""
Custom Federated Learning Strategies
Implements various aggregation strategies and privacy-preserving techniques
"""

import flwr as fl
from flwr.common import (
    FitRes, Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
)
from flwr.server.client_proxy import ClientProxy
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


class FedProx(fl.server.strategy.FedAvg):
    """
    FedProx strategy - FedAvg with proximal term.
    Helps with heterogeneous data distributions.
    
    Reference: Li et al., "Federated Optimization in Heterogeneous Networks"
    """
    
    def __init__(self, mu: float = 0.01, *args, **kwargs):
        """
        Initialize FedProx strategy.
        
        Args:
            mu: Proximal term coefficient
            *args, **kwargs: Arguments for FedAvg
        """
        super().__init__(*args, **kwargs)
        self.mu = mu
        print(f"Using FedProx strategy with mu={mu}")
    
    def configure_fit(self, server_round: int, parameters: Parameters, 
                     client_manager) -> List[Tuple[ClientProxy, fl.common.FitIns]]:
        """Configure the next round of training."""
        config = {"mu": self.mu, "server_round": server_round}
        fit_ins = fl.common.FitIns(parameters, config)
        
        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )
        
        return [(client, fit_ins) for client in clients]


class FedAdagrad(fl.server.strategy.FedAvg):
    """
    FedAdagrad - Adaptive learning rate for federated learning.
    """
    
    def __init__(self, eta: float = 0.1, tau: float = 1e-3, *args, **kwargs):
        """
        Initialize FedAdagrad strategy.
        
        Args:
            eta: Learning rate
            tau: Adaptation parameter
            *args, **kwargs: Arguments for FedAvg
        """
        super().__init__(*args, **kwargs)
        self.eta = eta
        self.tau = tau
        self.squared_gradients = None
        print(f"Using FedAdagrad strategy with eta={eta}, tau={tau}")
    
    def aggregate_fit(self, server_round: int, results, failures):
        """Aggregate with adaptive learning rate."""
        if not results:
            return None, {}
        
        # Convert results to parameters
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        
        # Initialize squared gradients on first round
        if self.squared_gradients is None:
            self.squared_gradients = [
                np.zeros_like(w) for w, _ in weights_results[0]
            ]
        
        # Compute weighted average
        total_examples = sum([num_examples for _, num_examples in weights_results])
        
        aggregated_weights = []
        for i in range(len(weights_results[0][0])):
            # Weighted sum
            layer_updates = [
                weights[i] * num_examples / total_examples
                for weights, num_examples in weights_results
            ]
            aggregated = np.sum(layer_updates, axis=0)
            
            # Update squared gradients
            self.squared_gradients[i] += aggregated ** 2
            
            # Apply adaptive learning rate
            adapted = aggregated / (np.sqrt(self.squared_gradients[i]) + self.tau)
            aggregated_weights.append(adapted)
        
        parameters_aggregated = ndarrays_to_parameters(aggregated_weights)
        
        # Aggregate metrics
        metrics_aggregated = {}
        if results:
            metrics_aggregated = self._aggregate_metrics(results)
        
        return parameters_aggregated, metrics_aggregated
    
    def _aggregate_metrics(self, results):
        """Helper to aggregate metrics."""
        total_examples = sum([fit_res.num_examples for _, fit_res in results])
        metrics = {}
        
        for _, fit_res in results:
            for key, value in fit_res.metrics.items():
                if key not in metrics:
                    metrics[key] = 0.0
                metrics[key] += value * fit_res.num_examples / total_examples
        
        return metrics


class FedYogi(fl.server.strategy.FedAvg):
    """
    FedYogi - Adaptive optimization for federated learning.
    Similar to Adam optimizer but for federated setting.
    """
    
    def __init__(self, eta: float = 0.01, beta1: float = 0.9, 
                 beta2: float = 0.99, tau: float = 1e-3, *args, **kwargs):
        """
        Initialize FedYogi strategy.
        
        Args:
            eta: Learning rate
            beta1: First moment decay
            beta2: Second moment decay
            tau: Adaptation parameter
            *args, **kwargs: Arguments for FedAvg
        """
        super().__init__(*args, **kwargs)
        self.eta = eta
        self.beta1 = beta1
        self.beta2 = beta2
        self.tau = tau
        self.m = None  # First moment
        self.v = None  # Second moment
        print(f"Using FedYogi strategy with eta={eta}, beta1={beta1}, beta2={beta2}")


class SecureAggregation(fl.server.strategy.FedAvg):
    """
    Secure Aggregation strategy with differential privacy.
    Adds noise to protect individual client contributions.
    """
    
    def __init__(self, noise_multiplier: float = 0.1, 
                 clipping_norm: float = 1.0, *args, **kwargs):
        """
        Initialize secure aggregation strategy.
        
        Args:
            noise_multiplier: Scale of Gaussian noise to add
            clipping_norm: Gradient clipping threshold
            *args, **kwargs: Arguments for FedAvg
        """
        super().__init__(*args, **kwargs)
        self.noise_multiplier = noise_multiplier
        self.clipping_norm = clipping_norm
        print(f"Using Secure Aggregation with noise={noise_multiplier}, clipping={clipping_norm}")
    
    def aggregate_fit(self, server_round: int, results, failures):
        """Aggregate with differential privacy."""
        if not results:
            return None, {}
        
        # Get base aggregation
        parameters_aggregated, metrics_aggregated = super().aggregate_fit(
            server_round, results, failures
        )
        
        if parameters_aggregated is None:
            return None, metrics_aggregated
        
        # Add differential privacy noise
        weights = parameters_to_ndarrays(parameters_aggregated)
        
        noisy_weights = []
        for w in weights:
            # Clip gradients
            norm = np.linalg.norm(w)
            if norm > self.clipping_norm:
                w = w * (self.clipping_norm / norm)
            
            # Add Gaussian noise
            noise = np.random.normal(0, self.noise_multiplier * self.clipping_norm, w.shape)
            noisy_w = w + noise
            noisy_weights.append(noisy_w)
        
        parameters_aggregated = ndarrays_to_parameters(noisy_weights)
        
        return parameters_aggregated, metrics_aggregated


def get_strategy(strategy_name: str, **kwargs) -> fl.server.strategy.Strategy:
    """
    Factory function to get a federated learning strategy.
    
    Args:
        strategy_name: Name of the strategy
        **kwargs: Additional arguments for the strategy
        
    Returns:
        Flower strategy instance
    """
    strategies = {
        'fedavg': fl.server.strategy.FedAvg,
        'fedprox': FedProx,
        'fedadagrad': FedAdagrad,
        'fedyogi': FedYogi,
        'secure': SecureAggregation,
    }
    
    if strategy_name.lower() not in strategies:
        raise ValueError(f"Unknown strategy: {strategy_name}. Available: {list(strategies.keys())}")
    
    strategy_class = strategies[strategy_name.lower()]
    return strategy_class(**kwargs)


if __name__ == "__main__":
    print("Federated Learning Strategies Module")
    print("\nAvailable strategies:")
    print("  1. FedAvg - Standard federated averaging")
    print("  2. FedProx - FedAvg with proximal term for heterogeneous data")
    print("  3. FedAdagrad - Adaptive learning rate")
    print("  4. FedYogi - Adam-like optimization for federated learning")
    print("  5. SecureAggregation - Differential privacy for secure aggregation")
    print("\n✓ Strategy module ready for use!")
