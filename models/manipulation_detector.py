"""
Image Manipulation Detection
Detects photoshop, splicing, and composites
"""

import torch
import torch.nn as nn
import numpy as np
from scipy import ndimage
import cv2

class ManipulationDetector:
    """
    Detects image manipulation through:
    1. Noise inconsistency analysis
    2. JPEG compression artifacts
    3. Lighting inconsistency
    4. Edge detection anomalies
    """
    
    def __init__(self):
        self.threshold = 0.6
    
    def detect_noise_inconsistency(self, image):
        """
        Detect inconsistent noise patterns (sign of splicing)
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Divide image into blocks
        h, w = gray.shape
        block_size = 64
        noise_levels = []
        
        for i in range(0, h - block_size, block_size):
            for j in range(0, w - block_size, block_size):
                block = gray[i:i+block_size, j:j+block_size]
                
                # Estimate noise level using high-pass filter
                high_pass = block - cv2.GaussianBlur(block, (5, 5), 0)
                noise_level = np.std(high_pass)
                noise_levels.append(noise_level)
        
        # Check variance in noise levels
        noise_variance = np.var(noise_levels)
        
        # High variance suggests splicing
        return noise_variance > 100  # Threshold
    
    def detect_jpeg_inconsistency(self, image):
        """
        Detect inconsistent JPEG compression (sign of editing)
        """
        # Convert to YCrCb
        ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        y_channel = ycrcb[:, :, 0]
        
        # Detect 8x8 block boundaries (JPEG artifacts)
        dct_blocks = []
        for i in range(0, y_channel.shape[0] - 8, 8):
            for j in range(0, y_channel.shape[1] - 8, 8):
                block = y_channel[i:i+8, j:j+8].astype(float)
                dct = cv2.dct(block)
                dct_blocks.append(np.std(dct))
        
        # Inconsistent DCT coefficients suggest manipulation
        return np.var(dct_blocks) > 500
    
    def detect_lighting_inconsistency(self, image):
        """
        Detect inconsistent lighting (sign of composite)
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l_channel = lab[:, :, 0]
        
        # Divide into regions
        h, w = l_channel.shape
        regions = [
            l_channel[:h//2, :w//2],      # Top-left
            l_channel[:h//2, w//2:],      # Top-right
            l_channel[h//2:, :w//2],      # Bottom-left
            l_channel[h//2:, w//2:]       # Bottom-right
        ]
        
        # Check brightness consistency
        mean_brightness = [np.mean(r) for r in regions]
        brightness_variance = np.var(mean_brightness)
        
        # High variance suggests different lighting sources
        return brightness_variance > 200
    
    def detect_edge_inconsistency(self, image):
        """
        Detect unnatural edges (sign of splicing)
        """
        # Edge detection
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Check for abrupt transitions
        edge_density = np.sum(edges) / edges.size
        
        # Analyze edge continuity
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=1)
        
        # Count disconnected edge regions
        num_labels, labels = cv2.connectedComponents(dilated)
        
        # Too many disconnected regions suggests splicing
        return num_labels > 50
    
    def predict(self, image):
        """
        Comprehensive manipulation detection
        """
        # Ensure numpy array
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        
        # Ensure uint8
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        # Ensure RGB
        if len(image.shape) == 4:
            image = image[0]  # Remove batch dimension
        if image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))
        
        # Run all tests
        tests = {
            'noise_inconsistency': self.detect_noise_inconsistency(image),
            'jpeg_inconsistency': self.detect_jpeg_inconsistency(image),
            'lighting_inconsistency': self.detect_lighting_inconsistency(image),
            'edge_inconsistency': self.detect_edge_inconsistency(image)
        }
        
        # Count positive tests
        manipulation_score = sum(tests.values()) / len(tests)
        
        is_manipulated = manipulation_score >= self.threshold
        
        return {
            'is_fake': is_manipulated,
            'confidence': manipulation_score,
            'tests': tests,
            'type': 'manipulation' if is_manipulated else 'authentic'
        }

# Example usage
if __name__ == "__main__":
    detector = ManipulationDetector()
    
    # Test on image
    import sys
    from PIL import Image
    
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        image = Image.open(img_path).convert('RGB')
        image_np = np.array(image)
        
        result = detector.predict(image_np)
        
        print(f"\n{'='*60}")
        print(f"Manipulation Detection Results")
        print(f"{'='*60}")
        print(f"Is Manipulated: {result['is_fake']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"\nTest Results:")
        for test, passed in result['tests'].items():
            status = "✓ DETECTED" if passed else "✗ Not detected"
            print(f"  {test}: {status}")
        print(f"{'='*60}\n")
