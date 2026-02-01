"""
Final training script - focused on 85%+ accuracy
Uses proven techniques and fixes all issues
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Use EfficientNet-style architecture but simplified
class EfficientManipulationCNN(nn.Module):
    """
    Efficient CNN optimized for manipulation detection.
    Uses depthwise separable convolutions and attention.
    """
    
    def __init__(self, num_classes=2):
        super().__init__()
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
        )
        
        # MBConv blocks (simplified EfficientNet)
        self.blocks = nn.Sequential(
            # Block 1: 32->64
            self._make_mbconv(32, 64, stride=1, expansion=1),
            # Block 2: 64->128
            self._make_mbconv(64, 128, stride=2, expansion=2),
            # Block 3: 128->256
            self._make_mbconv(128, 256, stride=2, expansion=4),
            # Block 4: 256->512
            self._make_mbconv(256, 512, stride=2, expansion=4),
        )
        
        # Head
        self.head = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU6(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU6(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
        self._initialize_weights()
    
    def _make_mbconv(self, in_channels, out_channels, stride, expansion):
        """Create MBConv block."""
        expanded = in_channels * expansion
        return nn.Sequential(
            # Expand
            nn.Conv2d(in_channels, expanded, kernel_size=1, bias=False),
            nn.BatchNorm2d(expanded),
            nn.ReLU6(inplace=True),
            # Depthwise
            nn.Conv2d(expanded, expanded, kernel_size=3, stride=stride, 
                     padding=1, groups=expanded, bias=False),
            nn.BatchNorm2d(expanded),
            nn.ReLU6(inplace=True),
            # Project
            nn.Conv2d(expanded, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x

class CleanManipulationDataset(torch.utils.data.Dataset):
    """Clean dataset with proper preprocessing."""
    
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples = []
        
        # Load and validate images
        for class_dir, label in [('authentic', 0), ('manipulated', 1)]:
            class_path = self.data_dir / class_dir
            for img_path in class_path.glob('*.*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    # Validate image
                    try:
                        with Image.open(img_path) as img:
                            img = img.convert('RGB')
                            # Only use images with reasonable size
                            if img.size[0] >= 100 and img.size[1] >= 100:
                                self.samples.append((str(img_path), label))
                    except:
                        continue
        
        print(f"Loaded {len(self.samples)} valid samples")
        authentic = sum(1 for _, label in self.samples if label == 0)
        manipulated = sum(1 for _, label in self.samples if label == 1)
        print(f"  Authentic: {authentic}, Manipulated: {manipulated}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_optimal_transforms(image_size=224):
    """Optimized transforms for manipulation detection."""
    
    train_transform = transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),  # Slightly larger
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        transforms.RandomRotation(5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225]),
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def train_epoch(model, loader, criterion, optimizer, scheduler, device):
    """Training with one-cycle policy."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(tqdm(loader, desc='Training')):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(loader), 100. * correct / total

def validate(model, loader, criterion, device):
    """Validation with detailed metrics."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Validating'):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(loader), 100. * correct / total

def main():
    print("="*60)
    print("FINAL MANIPULATION DETECTOR TRAINING")
    print("Target: 85%+ Accuracy")
    print("="*60)
    
    # Configuration
    DATA_DIR = 'manipulation_data_combined'
    BATCH_SIZE = 64  # Larger batch for stable training
    EPOCHS = 30
    MAX_LR = 0.01
    IMAGE_SIZE = 224
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data loading
    train_transform, val_transform = get_optimal_transforms(IMAGE_SIZE)
    
    print("\nLoading clean data...")
    train_dataset = CleanManipulationDataset(
        Path(DATA_DIR) / 'train',
        transform=train_transform
    )
    
    val_dataset = CleanManipulationDataset(
        Path(DATA_DIR) / 'val',
        transform=val_transform
    )
    
    # Balanced sampling
    labels = [label for _, label in train_dataset.samples]
    class_counts = np.bincount(labels)
    class_weights = 1. / class_counts
    sample_weights = [class_weights[label] for label in labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Model
    print("\nInitializing Efficient CNN...")
    model = EfficientManipulationCNN().to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer with one-cycle policy
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=0.01)
    
    # One-cycle learning rate scheduler
    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=MAX_LR,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.3,
        anneal_strategy='cos'
    )
    
    # Training
    best_val_acc = 0
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    
    print(f"\nStarting training with One-Cycle LR...")
    print("="*60)
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print("-" * 60)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, device
        )
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"  Learning Rate: {current_lr:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_manipulation_model_final.pth')
            print(f"  ✓ New best model saved! (Val Acc: {val_acc:.2f}%)")
        
        # Early stopping if reached target
        if val_acc >= 85.0:
            print(f"\n🎉 TARGET REACHED! {val_acc:.2f}% accuracy!")
            break
    
    print("\n" + "="*60)
    print("Training complete!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Model saved as: best_manipulation_model_final.pth")
    
    # Plot results
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('manipulation_final_training_curves.png', dpi=150, bbox_inches='tight')
    print(f"\nTraining curves saved as: manipulation_final_training_curves.png")

if __name__ == "__main__":
    main()
