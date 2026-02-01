"""
Train CNN for Image Manipulation Detection
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from models.manipulation_cnn import LightManipulationCNN

class ManipulationDataset(Dataset):
    """Dataset for manipulation detection."""
    
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir: Root directory with 'authentic' and 'manipulated' subdirs
            transform: Optional transforms
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        
        # Get all images
        self.samples = []
        
        # Authentic images (label 0)
        authentic_dir = self.data_dir / 'authentic'
        if authentic_dir.exists():
            for img_path in authentic_dir.glob('*.*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    self.samples.append((str(img_path), 0))
        
        # Manipulated images (label 1)
        manipulated_dir = self.data_dir / 'manipulated'
        if manipulated_dir.exists():
            for img_path in manipulated_dir.glob('*.*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    self.samples.append((str(img_path), 1))
        
        print(f"Loaded {len(self.samples)} samples")
        print(f"  Authentic: {sum(1 for _, label in self.samples if label == 0)}")
        print(f"  Manipulated: {sum(1 for _, label in self.samples if label == 1)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_transforms(image_size=224):
    """Get training and validation transforms."""
    
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{total_loss/len(loader):.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    return total_loss / len(loader), 100. * correct / total

def validate(model, loader, criterion, device):
    """Validate the model."""
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
    # Configuration
    DATA_DIR = 'manipulation_data_combined'  # Combined CG1050 + CASIA
    BATCH_SIZE = 32
    EPOCHS = 10
    LEARNING_RATE = 0.001
    IMAGE_SIZE = 224
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Check if data directory exists
    if not Path(DATA_DIR).exists():
        print(f"\n⚠️  Data directory '{DATA_DIR}' not found!")
        print("Please provide the manipulation detection dataset.")
        print("Expected structure:")
        print("  manipulation_data/")
        print("    ├── train/")
        print("    │   ├── authentic/")
        print("    │   └── manipulated/")
        print("    └── val/")
        print("        ├── authentic/")
        print("        └── manipulated/")
        return
    
    # Transforms
    train_transform, val_transform = get_transforms(IMAGE_SIZE)
    
    # Datasets
    print("\nLoading training data...")
    train_dataset = ManipulationDataset(
        Path(DATA_DIR) / 'train',
        transform=train_transform
    )
    
    print("\nLoading validation data...")
    val_dataset = ManipulationDataset(
        Path(DATA_DIR) / 'val',
        transform=val_transform
    )
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
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
    
    # Model (use lightweight version for faster training)
    print("\nInitializing model...")
    model = LightManipulationCNN().to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2
    )
    
    # Training loop
    best_val_acc = 0
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    
    print(f"\nStarting training for {EPOCHS} epochs...")
    print("="*60)
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print("-" * 60)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validate
        val_loss, val_acc = validate(
            model, val_loader, criterion, device
        )
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Learning rate scheduling
        scheduler.step(val_acc)
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_manipulation_model.pth')
            print(f"  ✓ New best model saved! (Val Acc: {val_acc:.2f}%)")
    
    print("\n" + "="*60)
    print("Training complete!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Model saved as: best_manipulation_model.pth")
    
    # Plot training curves
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
    plt.savefig('manipulation_training_curves.png', dpi=150, bbox_inches='tight')
    print(f"\nTraining curves saved as: manipulation_training_curves.png")

if __name__ == "__main__":
    main()
