"""
Training with Advanced Augmentation
Includes compression artifacts, color jittering, and more
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

sys.path.append(str(Path(__file__).parent))

from train_simple import SimpleDeepfakeDetector, train_epoch, validate
from data.data_loader import DeepfakeDataset

def get_advanced_train_transforms(image_size=224):
    """
    Advanced augmentation pipeline with compression artifacts and more.
    """
    return A.Compose([
        # Geometric transforms
        A.RandomResizedCrop(image_size, image_size, scale=(0.8, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.5),
        
        # Color augmentation
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
        A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.5),
        
        # Compression artifacts (simulates JPEG compression)
        A.ImageCompression(quality_lower=60, quality_upper=100, p=0.5),
        
        # Blur and noise
        A.OneOf([
            A.MotionBlur(blur_limit=5, p=1.0),
            A.MedianBlur(blur_limit=5, p=1.0),
            A.GaussianBlur(blur_limit=5, p=1.0),
        ], p=0.3),
        
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
        ], p=0.3),
        
        # Pixel-level transforms
        A.OneOf([
            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1.0),
            A.Emboss(alpha=(0.2, 0.5), strength=(0.2, 0.7), p=1.0),
        ], p=0.2),
        
        # Cutout/Dropout
        A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3),
        
        # Normalization
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

def get_val_transforms(image_size=224):
    """Validation transforms (no augmentation)."""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

class AdvancedDataset(DeepfakeDataset):
    """Dataset with advanced augmentation."""
    
    def __init__(self, *args, use_advanced_aug=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_advanced_aug = use_advanced_aug
        
        if use_advanced_aug and self.split == 'train':
            self.transform = get_advanced_train_transforms(self.image_size)
        else:
            self.transform = get_val_transforms(self.image_size)

def main():
    print("\n" + "="*70)
    print("Training with Advanced Augmentation")
    print("="*70 + "\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create datasets with advanced augmentation
    print("\nLoading data with advanced augmentation...")
    train_dataset = AdvancedDataset('./data', split='train', image_size=224, use_advanced_aug=True)
    val_dataset = AdvancedDataset('./data', split='val', image_size=224, use_advanced_aug=False)
    test_dataset = AdvancedDataset('./data', split='test', image_size=224, use_advanced_aug=False)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)
    
    print(f"Train: {len(train_dataset)} images")
    print(f"Val: {len(val_dataset)} images")
    print(f"Test: {len(test_dataset)} images")
    
    # Create model
    print("\nCreating model...")
    model = SimpleDeepfakeDetector().to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    
    # Training loop
    print("\nStarting training with advanced augmentation...\n")
    epochs = 10
    best_acc = 0
    
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        print("-" * 70)
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Step scheduler
        scheduler.step()
        
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"  Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model_advanced_aug.pth')
            print(f"  ✓ New best accuracy: {best_acc:.2f}% (model saved)")
    
    # Final test
    print("\n" + "="*70)
    print("Final Evaluation on Test Set")
    print("="*70)
    
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")
    
    print("\n" + "="*70)
    print("Training Complete!")
    print(f"Best Validation Accuracy: {best_acc:.2f}%")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print("="*70 + "\n")
    
    print("💡 Advanced augmentation includes:")
    print("  ✓ Compression artifacts (JPEG simulation)")
    print("  ✓ Color jittering")
    print("  ✓ Motion/Gaussian/Median blur")
    print("  ✓ Gaussian/ISO noise")
    print("  ✓ Sharpen/Emboss")
    print("  ✓ Coarse dropout")
    print("  ✓ Geometric transforms")
    print("\nExpected improvement: +2-5% accuracy")

if __name__ == "__main__":
    main()
