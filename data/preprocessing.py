"""
Data preprocessing utilities for deepfake detection
Handles image/video preprocessing, augmentation, and normalization
"""

import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Tuple, Optional
import os


class VideoFrameExtractor:
    """
    Extract frames from video files for deepfake detection.
    """
    
    def __init__(self, num_frames: int = 10, frame_size: Tuple[int, int] = (224, 224)):
        """
        Initialize frame extractor.
        
        Args:
            num_frames: Number of frames to extract per video
            frame_size: Target size for extracted frames (width, height)
        """
        self.num_frames = num_frames
        self.frame_size = frame_size
    
    def extract_frames(self, video_path: str, method: str = 'uniform') -> np.ndarray:
        """
        Extract frames from a video file.
        
        Args:
            video_path: Path to the video file
            method: Extraction method ('uniform', 'random', 'first')
            
        Returns:
            numpy array of shape (num_frames, height, width, channels)
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            raise ValueError(f"Video has no frames: {video_path}")
        
        # Determine which frames to extract
        if method == 'uniform':
            frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        elif method == 'random':
            frame_indices = np.random.choice(total_frames, self.num_frames, replace=False)
            frame_indices = np.sort(frame_indices)
        elif method == 'first':
            frame_indices = np.arange(min(self.num_frames, total_frames))
        else:
            raise ValueError(f"Unknown extraction method: {method}")
        
        frames = []
        current_frame = 0
        
        for target_frame in frame_indices:
            # Seek to target frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            ret, frame = cap.read()
            
            if ret:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Resize frame
                frame = cv2.resize(frame, self.frame_size)
                frames.append(frame)
            else:
                # If frame read fails, use last successful frame or black frame
                if len(frames) > 0:
                    frames.append(frames[-1])
                else:
                    frames.append(np.zeros((*self.frame_size, 3), dtype=np.uint8))
        
        cap.release()
        
        return np.array(frames)
    
    def extract_single_frame(self, video_path: str, frame_idx: int = 0) -> np.ndarray:
        """
        Extract a single frame from video.
        
        Args:
            video_path: Path to video file
            frame_idx: Index of frame to extract
            
        Returns:
            Single frame as numpy array
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            raise ValueError(f"Cannot read frame {frame_idx} from {video_path}")
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, self.frame_size)
        
        cap.release()
        
        return frame


def get_train_transforms(image_size: int = 224) -> A.Compose:
    """
    Get training data augmentation transforms.
    
    Args:
        image_size: Target image size
        
    Returns:
        Albumentations composition of transforms
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.3),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1),
        ], p=0.5),
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=1),
            A.GaussianBlur(blur_limit=(3, 7), p=1),
        ], p=0.3),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=15, p=0.5),
        A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


def get_val_transforms(image_size: int = 224) -> A.Compose:
    """
    Get validation/test data transforms (no augmentation).
    
    Args:
        image_size: Target image size
        
    Returns:
        Albumentations composition of transforms
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


def get_torchvision_transforms(image_size: int = 224, is_training: bool = True) -> transforms.Compose:
    """
    Get torchvision transforms (alternative to albumentations).
    
    Args:
        image_size: Target image size
        is_training: Whether this is for training (includes augmentation)
        
    Returns:
        Torchvision composition of transforms
    """
    if is_training:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


class FaceDetector:
    """
    Detect and extract faces from images using OpenCV.
    Useful for focusing on facial regions in deepfake detection.
    """
    
    def __init__(self, scale_factor: float = 1.1, min_neighbors: int = 5):
        """
        Initialize face detector.
        
        Args:
            scale_factor: Parameter specifying how much the image size is reduced
            min_neighbors: Parameter specifying how many neighbors each candidate rectangle should have
        """
        # Load Haar Cascade for face detection
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
    
    def detect_faces(self, image: np.ndarray) -> list:
        """
        Detect faces in an image.
        
        Args:
            image: Input image as numpy array (RGB)
            
        Returns:
            List of face bounding boxes [(x, y, w, h), ...]
        """
        # Convert to grayscale for detection
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=(30, 30)
        )
        
        return faces
    
    def extract_largest_face(self, image: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> Optional[np.ndarray]:
        """
        Extract the largest face from an image.
        
        Args:
            image: Input image as numpy array (RGB)
            target_size: Size to resize the extracted face
            
        Returns:
            Extracted and resized face, or None if no face found
        """
        faces = self.detect_faces(image)
        
        if len(faces) == 0:
            return None
        
        # Find largest face
        largest_face = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = largest_face
        
        # Extract face region
        face = image[y:y+h, x:x+w]
        
        # Resize to target size
        face = cv2.resize(face, target_size)
        
        return face


def preprocess_image(image_path: str, transform: Optional[A.Compose] = None, 
                    image_size: int = 224) -> torch.Tensor:
    """
    Load and preprocess a single image.
    
    Args:
        image_path: Path to the image file
        transform: Albumentations transform to apply
        image_size: Target image size if no transform provided
        
    Returns:
        Preprocessed image as torch tensor
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    image = np.array(image)
    
    # Apply transforms
    if transform is None:
        transform = get_val_transforms(image_size)
    
    transformed = transform(image=image)
    image_tensor = transformed['image']
    
    return image_tensor


def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    """
    Denormalize an image tensor for visualization.
    
    Args:
        tensor: Normalized image tensor
        
    Returns:
        Denormalized image as numpy array (0-255)
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    # Denormalize
    tensor = tensor * std + mean
    
    # Clip to valid range
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to numpy and scale to 0-255
    image = tensor.permute(1, 2, 0).cpu().numpy()
    image = (image * 255).astype(np.uint8)
    
    return image


if __name__ == "__main__":
    print("Testing preprocessing utilities...")
    
    # Test transforms
    print("\n1. Testing transforms:")
    train_transform = get_train_transforms(224)
    val_transform = get_val_transforms(224)
    print("   ✓ Train and validation transforms created")
    
    # Test with dummy image
    dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    transformed = train_transform(image=dummy_image)
    print(f"   ✓ Transformed image shape: {transformed['image'].shape}")
    
    # Test face detector
    print("\n2. Testing face detector:")
    face_detector = FaceDetector()
    faces = face_detector.detect_faces(dummy_image)
    print(f"   ✓ Detected {len(faces)} faces in dummy image")
    
    # Test denormalization
    print("\n3. Testing denormalization:")
    tensor = torch.randn(3, 224, 224)
    denorm = denormalize_image(tensor)
    print(f"   ✓ Denormalized image shape: {denorm.shape}")
    
    print("\n✓ All preprocessing utilities working correctly!")
