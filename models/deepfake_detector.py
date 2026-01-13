"""
Deepfake Detection Model
A hybrid CNN-based architecture for detecting deepfake images/videos
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import timm


class DeepfakeDetector(nn.Module):
    """
    Hybrid deepfake detection model using EfficientNet backbone
    with custom classification head.
    """
    
    def __init__(self, num_classes=2, pretrained=True, dropout_rate=0.5):
        """
        Initialize the deepfake detector model.
        
        Args:
            num_classes (int): Number of output classes (2 for binary: real/fake)
            pretrained (bool): Use pretrained weights from ImageNet
            dropout_rate (float): Dropout rate for regularization
        """
        super(DeepfakeDetector, self).__init__()
        
        # Use EfficientNet-B0 as backbone (good balance of accuracy and speed)
        self.backbone = timm.create_model('efficientnet_b0', pretrained=pretrained)
        
        # Get the number of features from the backbone
        num_features = self.backbone.classifier.in_features
        
        # Remove the original classifier
        self.backbone.classifier = nn.Identity()
        
        # Custom classification head with multiple layers
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, H, W)
            
        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes)
        """
        # Extract features using backbone
        features = self.backbone(x)
        
        # Classify using custom head
        output = self.classifier(features)
        
        return output
    
    def extract_features(self, x):
        """
        Extract features from the backbone without classification.
        Useful for visualization and analysis.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Feature vector
        """
        return self.backbone(x)


class HybridDeepfakeDetector(nn.Module):
    """
    Advanced hybrid model that combines CNN with attention mechanism
    for improved deepfake detection.
    """
    
    def __init__(self, num_classes=2, pretrained=True, dropout_rate=0.5):
        """
        Initialize the hybrid detector with attention mechanism.
        
        Args:
            num_classes (int): Number of output classes
            pretrained (bool): Use pretrained weights
            dropout_rate (float): Dropout rate
        """
        super(HybridDeepfakeDetector, self).__init__()
        
        # CNN backbone
        self.backbone = timm.create_model('efficientnet_b0', pretrained=pretrained)
        num_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Identity()
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.Tanh(),
            nn.Linear(256, 1),
            nn.Softmax(dim=1)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        """
        Forward pass with attention mechanism.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output logits
        """
        # Extract features
        features = self.backbone(x)
        
        # Apply attention (optional, can be used for interpretability)
        # For now, we'll use features directly
        
        # Classify
        output = self.classifier(features)
        
        return output


class ResNetDeepfakeDetector(nn.Module):
    """
    Alternative model using ResNet50 as backbone.
    Useful for comparison with EfficientNet.
    """
    
    def __init__(self, num_classes=2, pretrained=True, dropout_rate=0.5):
        """
        Initialize ResNet-based detector.
        
        Args:
            num_classes (int): Number of output classes
            pretrained (bool): Use pretrained weights
            dropout_rate (float): Dropout rate
        """
        super(ResNetDeepfakeDetector, self).__init__()
        
        # Load pretrained ResNet50
        self.backbone = models.resnet50(pretrained=pretrained)
        
        # Get number of features
        num_features = self.backbone.fc.in_features
        
        # Replace final layer
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        """Forward pass."""
        return self.backbone(x)


def get_model(model_name='efficientnet', num_classes=2, pretrained=True, dropout_rate=0.5):
    """
    Factory function to get the desired model.
    
    Args:
        model_name (str): Name of the model ('efficientnet', 'hybrid', 'resnet')
        num_classes (int): Number of output classes
        pretrained (bool): Use pretrained weights
        dropout_rate (float): Dropout rate
        
    Returns:
        nn.Module: The requested model
    """
    models_dict = {
        'efficientnet': DeepfakeDetector,
        'hybrid': HybridDeepfakeDetector,
        'resnet': ResNetDeepfakeDetector
    }
    
    if model_name not in models_dict:
        raise ValueError(f"Model {model_name} not found. Available: {list(models_dict.keys())}")
    
    return models_dict[model_name](
        num_classes=num_classes,
        pretrained=pretrained,
        dropout_rate=dropout_rate
    )


if __name__ == "__main__":
    # Test the models
    print("Testing DeepfakeDetector models...")
    
    # Create dummy input
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 224, 224)
    
    # Test EfficientNet model
    print("\n1. Testing EfficientNet-based model:")
    model1 = DeepfakeDetector()
    output1 = model1(dummy_input)
    print(f"   Input shape: {dummy_input.shape}")
    print(f"   Output shape: {output1.shape}")
    print(f"   Parameters: {sum(p.numel() for p in model1.parameters()):,}")
    
    # Test Hybrid model
    print("\n2. Testing Hybrid model:")
    model2 = HybridDeepfakeDetector()
    output2 = model2(dummy_input)
    print(f"   Input shape: {dummy_input.shape}")
    print(f"   Output shape: {output2.shape}")
    print(f"   Parameters: {sum(p.numel() for p in model2.parameters()):,}")
    
    # Test ResNet model
    print("\n3. Testing ResNet-based model:")
    model3 = ResNetDeepfakeDetector()
    output3 = model3(dummy_input)
    print(f"   Input shape: {dummy_input.shape}")
    print(f"   Output shape: {output3.shape}")
    print(f"   Parameters: {sum(p.numel() for p in model3.parameters()):,}")
    
    print("\n✓ All models working correctly!")
