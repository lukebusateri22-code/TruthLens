"""
Federated Learning Client Implementation
Each client trains locally on its own data and shares model updates
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import flwr as fl
from collections import OrderedDict
from typing import Dict, List, Tuple
import numpy as np


class DeepfakeClient(fl.client.NumPyClient):
    """
    Flower client for federated deepfake detection.
    """
    
    def __init__(self, cid: str, model: nn.Module, train_loader: DataLoader,
                 val_loader: DataLoader, device: str, epochs: int = 1, lr: float = 0.001):
        """
        Initialize federated client.
        
        Args:
            cid: Client ID
            model: PyTorch model
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to train on
            epochs: Number of local epochs per round
            lr: Learning rate
        """
        self.cid = cid
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = epochs
        self.lr = lr
        
        # Loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        print(f"Client {self.cid} initialized with {len(train_loader.dataset)} training samples")
    
    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """
        Get model parameters as numpy arrays.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            List of numpy arrays containing model parameters
        """
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """
        Set model parameters from numpy arrays.
        
        Args:
            parameters: List of numpy arrays containing model parameters
        """
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        """
        Train the model on local data.
        
        Args:
            parameters: Global model parameters
            config: Configuration dictionary
            
        Returns:
            Tuple of (updated_parameters, num_examples, metrics)
        """
        # Set model parameters
        self.set_parameters(parameters)
        
        # Train
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                # Track metrics
                epoch_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            
            total_loss += epoch_loss
        
        # Calculate average loss and accuracy
        avg_loss = total_loss / (len(self.train_loader.dataset) * self.epochs)
        accuracy = correct / total
        
        print(f"Client {self.cid} - Training Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        
        # Return updated parameters and metrics
        return (
            self.get_parameters(config={}),
            len(self.train_loader.dataset),
            {"loss": avg_loss, "accuracy": accuracy}
        )
    
    def evaluate(self, parameters: List[np.ndarray], config: Dict) -> Tuple[float, int, Dict]:
        """
        Evaluate the model on local validation data.
        
        Args:
            parameters: Model parameters to evaluate
            config: Configuration dictionary
            
        Returns:
            Tuple of (loss, num_examples, metrics)
        """
        # Set model parameters
        self.set_parameters(parameters)
        
        # Evaluate
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Track metrics
                total_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        # Calculate metrics
        avg_loss = total_loss / len(self.val_loader.dataset)
        accuracy = correct / total
        
        print(f"Client {self.cid} - Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        
        return avg_loss, len(self.val_loader.dataset), {"accuracy": accuracy}


def get_client_fn(model_fn, train_loaders: List[DataLoader], 
                  val_loaders: List[DataLoader], device: str,
                  epochs: int = 1, lr: float = 0.001):
    """
    Factory function to create client instances.
    
    Args:
        model_fn: Function that returns a new model instance
        train_loaders: List of training data loaders for each client
        val_loaders: List of validation data loaders for each client
        device: Device to train on
        epochs: Number of local epochs
        lr: Learning rate
        
    Returns:
        Function that creates a client given a client ID
    """
    def client_fn(cid: str) -> DeepfakeClient:
        """Create a client instance."""
        # Get client index
        client_idx = int(cid)
        
        # Create new model instance for this client
        model = model_fn()
        model = model.to(device)
        
        # Get data loaders for this client
        train_loader = train_loaders[client_idx]
        val_loader = val_loaders[client_idx]
        
        # Create and return client
        return DeepfakeClient(
            cid=cid,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=epochs,
            lr=lr
        )
    
    return client_fn


class SimpleClient(fl.client.NumPyClient):
    """
    Simplified client for quick testing.
    """
    
    def __init__(self, model, train_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
    
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        
        for images, labels in self.train_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
        
        return self.get_parameters(config={}), len(self.train_loader.dataset), {}
    
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        
        loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss += self.criterion(outputs, labels).item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        accuracy = correct / total if total > 0 else 0.0
        return loss, len(self.train_loader.dataset), {"accuracy": accuracy}


if __name__ == "__main__":
    print("Federated Learning Client Module")
    print("This module defines the client-side logic for federated learning.")
    print("\nKey features:")
    print("  - Local training on client data")
    print("  - Model parameter exchange (not raw data)")
    print("  - Privacy-preserving by design")
    print("\n✓ Client module ready for use!")
