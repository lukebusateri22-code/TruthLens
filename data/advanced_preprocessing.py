"""
Advanced Preprocessing from Kaggle Notebook
Combines YOLO face detection + CS-LBP + CLAHE
"""

import cv2
import numpy as np
from pathlib import Path
import torch
from PIL import Image

class AdvancedPreprocessor:
    """
    Advanced preprocessing pipeline:
    1. YOLO face detection (extract face ROI)
    2. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    3. CS-LBP (Center-Symmetric Local Binary Patterns)
    """
    
    def __init__(self, use_yolo=False, use_clahe=True, use_cslbp=False):
        self.use_yolo = use_yolo
        self.use_clahe = use_clahe
        self.use_cslbp = use_cslbp
        
        # CLAHE parameters
        self.clip_limit = 2.0
        self.grid_size = (8, 8)
        
        if use_yolo:
            # Load YOLO model (would need weights)
            print("⚠️  YOLO requires model weights - skipping for now")
            self.use_yolo = False
    
    def apply_clahe(self, image):
        """Apply CLAHE to enhance contrast."""
        if len(image.shape) == 3:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit,
            tileGridSize=self.grid_size
        )
        enhanced = clahe.apply(gray)
        
        # Convert back to RGB
        if len(image.shape) == 3:
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
        
        return enhanced
    
    def get_pixel(self, img, x1, y1, x, y):
        """Helper for CS-LBP calculation."""
        try:
            return 1 if img[x1][y1] >= img[x][y] else 0
        except IndexError:
            return 0
    
    def cs_lbp_pixel(self, img, x, y):
        """Calculate CS-LBP for a pixel."""
        val_ar = [
            self.get_pixel(img, x, y+1, x, y-1),
            self.get_pixel(img, x+1, y+1, x-1, y-1),
            self.get_pixel(img, x+1, y, x-1, y),
            self.get_pixel(img, x+1, y-1, x-1, y+1)
        ]
        
        power_val = [1, 2, 4, 8]
        return sum(v * p for v, p in zip(val_ar, power_val))
    
    def apply_cslbp(self, image):
        """Apply CS-LBP texture analysis."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        height, width = gray.shape
        result = np.zeros((height, width), np.uint16)
        
        for i in range(height):
            for j in range(width):
                result[i, j] = self.cs_lbp_pixel(gray, i, j)
        
        return result
    
    def __call__(self, image):
        """
        Apply preprocessing pipeline.
        
        Args:
            image: PIL Image or numpy array
        
        Returns:
            Preprocessed image
        """
        # Convert to numpy if PIL
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Apply CLAHE
        if self.use_clahe:
            image = self.apply_clahe(image)
        
        # Apply CS-LBP
        if self.use_cslbp:
            image = self.apply_cslbp(image)
            # Normalize to 0-255 range
            image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
            # Convert back to RGB
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        return image


class FaceDetector:
    """
    Face detection using OpenCV's Haar Cascades
    (Alternative to YOLO, lighter weight)
    """
    
    def __init__(self):
        # Load Haar Cascade for face detection
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
    
    def detect_and_crop(self, image, padding=20):
        """
        Detect face and crop to face region.
        
        Args:
            image: Input image (numpy array or PIL)
            padding: Pixels to add around detected face
        
        Returns:
            Cropped face image or original if no face detected
        """
        # Convert to numpy if PIL
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Convert to grayscale for detection
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        
        if len(faces) == 0:
            return image  # No face detected, return original
        
        # Get largest face
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        
        # Add padding
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(image.shape[1] - x, w + 2 * padding)
        h = min(image.shape[0] - y, h + 2 * padding)
        
        # Crop to face
        face = image[y:y+h, x:x+w]
        
        return face


def create_advanced_transform(image_size=224, use_clahe=True, use_face_detection=False):
    """
    Create advanced preprocessing transform.
    
    Args:
        image_size: Target image size
        use_clahe: Whether to use CLAHE enhancement
        use_face_detection: Whether to detect and crop faces
    
    Returns:
        Transform function
    """
    preprocessor = AdvancedPreprocessor(use_clahe=use_clahe)
    face_detector = FaceDetector() if use_face_detection else None
    
    def transform(image):
        # Detect and crop face if enabled
        if face_detector:
            image = face_detector.detect_and_crop(image)
        
        # Apply advanced preprocessing
        image = preprocessor(image)
        
        # Resize
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        image = image.resize((image_size, image_size))
        
        # Convert to tensor
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)
        
        return image
    
    return transform


# Example usage
if __name__ == "__main__":
    print("Advanced Preprocessing Module")
    print("=" * 50)
    print("\nFeatures:")
    print("  ✓ CLAHE (Contrast enhancement)")
    print("  ✓ CS-LBP (Texture analysis)")
    print("  ✓ Face detection (Haar Cascades)")
    print("\nUsage:")
    print("  from data.advanced_preprocessing import create_advanced_transform")
    print("  transform = create_advanced_transform(use_clahe=True)")
    print("  enhanced_image = transform(image)")
