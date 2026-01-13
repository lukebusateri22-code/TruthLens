"""Federated learning module for deepfake detection."""

from .client import DeepfakeClient, get_client_fn, SimpleClient
from .server import (
    weighted_average,
    get_evaluate_fn,
    FederatedServer,
    create_strategy,
    CustomFedAvg
)
from .strategy import (
    FedProx,
    FedAdagrad,
    FedYogi,
    SecureAggregation,
    get_strategy
)

__all__ = [
    'DeepfakeClient',
    'get_client_fn',
    'SimpleClient',
    'weighted_average',
    'get_evaluate_fn',
    'FederatedServer',
    'create_strategy',
    'CustomFedAvg',
    'FedProx',
    'FedAdagrad',
    'FedYogi',
    'SecureAggregation',
    'get_strategy'
]
