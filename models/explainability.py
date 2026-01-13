"""
Model Explainability - Grad-CAM Visualization
Shows which parts of the image the model focuses on
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM)
    Visualizes which regions of an image are important for the model's decision.
    """
    
    def __init__(self, model, target_layer):
        """
        Args:
            model: The neural network model
            target_layer: The layer to visualize (usually last conv layer)
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        """Hook to save forward pass activations."""
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        """Hook to save backward pass gradients."""
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_image, target_class=None):
        """
        Generate Class Activation Map.
        
        Args:
            input_image: Input tensor [1, C, H, W]
            target_class: Target class index (if None, uses predicted class)
        
        Returns:
            CAM heatmap
        """
        # Forward pass
        self.model.eval()
        output = self.model(input_image)
        
        # Get target class
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        class_loss = output[0, target_class]
        class_loss.backward()
        
        # Get gradients and activations
        gradients = self.gradients[0]  # [C, H, W]
        activations = self.activations[0]  # [C, H, W]
        
        # Calculate weights (global average pooling of gradients)
        weights = gradients.mean(dim=(1, 2))  # [C]
        
        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # Apply ReLU (only positive contributions)
        cam = F.relu(cam)
        
        # Normalize to [0, 1]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.cpu().numpy()
    
    def visualize(self, input_image, original_image, target_class=None, alpha=0.5):
        """
        Create visualization overlay.
        
        Args:
            input_image: Preprocessed input tensor
            original_image: Original PIL image
            target_class: Target class (if None, uses prediction)
            alpha: Overlay transparency
        
        Returns:
            Visualization image
        """
        # Generate CAM
        cam = self.generate_cam(input_image, target_class)
        
        # Resize CAM to match original image
        cam = cv2.resize(cam, (original_image.width, original_image.height))
        
        # Convert to heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Convert original image to numpy
        original_np = np.array(original_image)
        
        # Overlay heatmap on original image
        overlay = (alpha * heatmap + (1 - alpha) * original_np).astype(np.uint8)
        
        return Image.fromarray(overlay), cam


def visualize_predictions(model, image, transform, device='cpu', save_path=None):
    """
    Visualize model predictions with Grad-CAM.
    
    Args:
        model: Trained model
        image: PIL Image
        transform: Preprocessing transform
        device: Device to run on
        save_path: Path to save visualization (optional)
    
    Returns:
        Visualization figure
    """
    # Preprocess image
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Get prediction
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1)
        pred_class = output.argmax(dim=1).item()
        confidence = probs[0, pred_class].item()
    
    # Find last convolutional layer
    last_conv = None
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv = module
    
    if last_conv is None:
        print("No convolutional layer found!")
        return None
    
    # Generate Grad-CAM
    grad_cam = GradCAM(model, last_conv)
    overlay, cam = grad_cam.visualize(input_tensor, image)
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Heatmap
    axes[1].imshow(cam, cmap='jet')
    axes[1].set_title('Attention Heatmap')
    axes[1].axis('off')
    
    # Overlay
    axes[2].imshow(overlay)
    class_name = 'FAKE' if pred_class == 1 else 'REAL'
    axes[2].set_title(f'Prediction: {class_name} ({confidence:.2%})')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    
    return fig


# Example usage
if __name__ == "__main__":
    print("Model Explainability Module")
    print("=" * 50)
    print("\nGrad-CAM Visualization:")
    print("  ✓ Shows which image regions influence the prediction")
    print("  ✓ Helps understand model decision-making")
    print("  ✓ Identifies potential biases or artifacts")
    print("\nUsage:")
    print("  from models.explainability import visualize_predictions")
    print("  fig = visualize_predictions(model, image, transform)")
    print("  plt.show()")
