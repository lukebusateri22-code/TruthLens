"""
Data loading utilities for deepfake detection
Handles dataset creation, loading, and splitting for federated learning
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets
from PIL import Image
import numpy as np
from typing import Tuple, List, Optional, Dict
import json
from pathlib import Path
import albumentations as A

from .preprocessing import get_train_transforms, get_val_transforms, VideoFrameExtractor


class DeepfakeDataset(Dataset):
    """
    Custom dataset for deepfake detection.
    Supports both image and video data.
    """
    
    def __init__(self, data_dir: str, split: str = 'train', 
                 transform: Optional[A.Compose] = None,
                 image_size: int = 224,
                 use_videos: bool = False,
                 num_frames: int = 10):
        """
        Initialize deepfake dataset.
        
        Args:
            data_dir: Root directory containing 'real' and 'fake' subdirectories
            split: Dataset split ('train', 'val', 'test')
            transform: Albumentations transform to apply
            image_size: Target image size
            use_videos: Whether to load videos instead of images
            num_frames: Number of frames to extract from videos
        """
        self.data_dir = Path(data_dir) / split
        self.transform = transform
        self.image_size = image_size
        self.use_videos = use_videos
        self.num_frames = num_frames
        
        if self.transform is None:
            if split == 'train':
                self.transform = get_train_transforms(image_size)
            else:
                self.transform = get_val_transforms(image_size)
        
        # Initialize video frame extractor if needed
        if use_videos:
            self.frame_extractor = VideoFrameExtractor(num_frames, (image_size, image_size))
        
        # Load file paths and labels
        self.samples = []
        self.labels = []
        
        self._load_samples()
        
        print(f"Loaded {len(self.samples)} samples from {split} split")
        print(f"  Real: {sum(1 for l in self.labels if l == 0)}")
        print(f"  Fake: {sum(1 for l in self.labels if l == 1)}")
    
    def _load_samples(self):
        """Load all sample paths and labels."""
        # Check if data directory exists
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        # Define class directories
        class_dirs = {
            'real': 0,  # Real images/videos are labeled as 0
            'fake': 1   # Fake images/videos are labeled as 1
        }
        
        # Supported extensions
        if self.use_videos:
            extensions = ('.mp4', '.avi', '.mov', '.mkv')
        else:
            extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        
        # Load samples from each class
        for class_name, label in class_dirs.items():
            class_dir = self.data_dir / class_name
            
            if not class_dir.exists():
                print(f"Warning: {class_dir} does not exist, skipping...")
                continue
            
            # Get all files with valid extensions
            for ext in extensions:
                files = list(class_dir.glob(f'*{ext}'))
                self.samples.extend(files)
                self.labels.extend([label] * len(files))
    
    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample and its label.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (image_tensor, label)
        """
        file_path = self.samples[idx]
        label = self.labels[idx]
        
        try:
            if self.use_videos:
                # Extract frame from video
                frames = self.frame_extractor.extract_frames(str(file_path))
                # Use middle frame
                image = frames[len(frames) // 2]
            else:
                # Load image
                image = Image.open(file_path).convert('RGB')
                image = np.array(image)
            
            # Apply transforms
            if self.transform:
                transformed = self.transform(image=image)
                image = transformed['image']
            
            return image, label
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            # Return a black image if loading fails
            image = torch.zeros(3, self.image_size, self.image_size)
            return image, label


class ImageFolderDataset(Dataset):
    """
    Simple dataset using torchvision's ImageFolder structure.
    Expects directory structure: data_dir/class_name/image.jpg
    """
    
    def __init__(self, data_dir: str, transform: Optional[A.Compose] = None, image_size: int = 224):
        """
        Initialize image folder dataset.
        
        Args:
            data_dir: Root directory with class subdirectories
            transform: Transform to apply
            image_size: Target image size
        """
        self.data_dir = data_dir
        self.transform = transform or get_val_transforms(image_size)
        
        # Use torchvision's ImageFolder to get samples
        self.dataset = datasets.ImageFolder(data_dir)
        
        print(f"Loaded {len(self.dataset)} samples from {data_dir}")
        print(f"Classes: {self.dataset.classes}")
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image, label = self.dataset[idx]
        image = np.array(image)
        
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        return image, label


def create_data_loaders(data_dir: str, batch_size: int = 32, 
                       num_workers: int = 4, image_size: int = 224,
                       use_videos: bool = False) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders.
    
    Args:
        data_dir: Root data directory
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes
        image_size: Target image size
        use_videos: Whether to load videos
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = DeepfakeDataset(
        data_dir, 
        split='train', 
        image_size=image_size,
        use_videos=use_videos
    )
    
    val_dataset = DeepfakeDataset(
        data_dir, 
        split='val', 
        image_size=image_size,
        use_videos=use_videos
    )
    
    test_dataset = DeepfakeDataset(
        data_dir, 
        split='test', 
        image_size=image_size,
        use_videos=use_videos
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def partition_data_for_federated_learning(data_dir: str, num_clients: int = 5,
                                         split: str = 'train',
                                         partition_method: str = 'iid',
                                         alpha: float = 0.5) -> List[List[int]]:
    """
    Partition data indices for federated learning clients.
    
    Args:
        data_dir: Root data directory
        num_clients: Number of federated clients
        split: Dataset split to partition
        partition_method: 'iid' for independent and identically distributed,
                         'non_iid' for non-IID distribution
        alpha: Dirichlet distribution parameter for non-IID (lower = more skewed)
        
    Returns:
        List of lists, where each inner list contains indices for one client
    """
    # Load dataset to get total number of samples
    dataset = DeepfakeDataset(data_dir, split=split)
    num_samples = len(dataset)
    labels = np.array(dataset.labels)
    
    if partition_method == 'iid':
        # IID partition: randomly shuffle and split evenly
        indices = np.random.permutation(num_samples)
        client_indices = np.array_split(indices, num_clients)
        client_indices = [idx.tolist() for idx in client_indices]
        
    elif partition_method == 'non_iid':
        # Non-IID partition using Dirichlet distribution
        num_classes = len(np.unique(labels))
        client_indices = [[] for _ in range(num_clients)]
        
        for class_idx in range(num_classes):
            # Get indices for this class
            class_indices = np.where(labels == class_idx)[0]
            np.random.shuffle(class_indices)
            
            # Sample proportions from Dirichlet distribution
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            proportions = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]
            
            # Split class indices according to proportions
            split_indices = np.split(class_indices, proportions)
            
            # Assign to clients
            for client_idx, indices in enumerate(split_indices):
                client_indices[client_idx].extend(indices.tolist())
        
        # Shuffle each client's indices
        for i in range(num_clients):
            np.random.shuffle(client_indices[i])
    
    else:
        raise ValueError(f"Unknown partition method: {partition_method}")
    
    # Print statistics
    print(f"\nData partitioning ({partition_method}):")
    print(f"Total samples: {num_samples}")
    print(f"Number of clients: {num_clients}")
    for i, indices in enumerate(client_indices):
        client_labels = labels[indices]
        real_count = np.sum(client_labels == 0)
        fake_count = np.sum(client_labels == 1)
        print(f"  Client {i}: {len(indices)} samples (Real: {real_count}, Fake: {fake_count})")
    
    return client_indices


def create_client_dataloaders(data_dir: str, client_indices: List[List[int]],
                              batch_size: int = 32, num_workers: int = 2,
                              image_size: int = 224) -> List[DataLoader]:
    """
    Create data loaders for federated learning clients.
    
    Args:
        data_dir: Root data directory
        client_indices: List of index lists for each client
        batch_size: Batch size
        num_workers: Number of workers
        image_size: Image size
        
    Returns:
        List of DataLoader objects, one per client
    """
    # Create full dataset
    full_dataset = DeepfakeDataset(data_dir, split='train', image_size=image_size)
    
    # Create subset datasets for each client
    client_loaders = []
    for indices in client_indices:
        subset = torch.utils.data.Subset(full_dataset, indices)
        loader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        client_loaders.append(loader)
    
    return client_loaders


if __name__ == "__main__":
    print("Testing data loading utilities...")
    
    # Note: This test assumes you have data in the correct structure
    # For demonstration, we'll just test the partitioning logic
    
    print("\n1. Testing data partitioning:")
    # Create dummy labels for testing
    dummy_labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1] * 10)
    
    print("\n   IID partitioning:")
    num_clients = 5
    indices = np.arange(len(dummy_labels))
    client_indices_iid = np.array_split(indices, num_clients)
    
    for i, idx in enumerate(client_indices_iid):
        labels = dummy_labels[idx]
        print(f"   Client {i}: {len(idx)} samples (Real: {np.sum(labels==0)}, Fake: {np.sum(labels==1)})")
    
    print("\n✓ Data loading utilities ready!")
    print("\nNote: To fully test, organize your data as:")
    print("  data/")
    print("    train/")
    print("      real/  (real images)")
    print("      fake/  (fake images)")
    print("    val/")
    print("      real/")
    print("      fake/")
    print("    test/")
    print("      real/")
    print("      fake/")
