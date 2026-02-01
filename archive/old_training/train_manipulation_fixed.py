"""
Fixed training script for manipulation detection
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

from models.manipulation_cnn import LightManipulationCNN

class ManipulationDataset(torch.utils.data.Dataset):
    """Dataset with class balancing."""
    
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples = []
        
        # Load images
        authentic_dir = self.data_dir / 'authentic'
        manipulated_dir = self.data_dir / 'manipulated'
        
        for img_path in authentic_dir.glob('*.*'):
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                self.samples.append((str(img_path), 0))
        
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
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_transforms(image_size=224):
    """Better transforms for manipulation detection."""
    
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(5),  # Less rotation
        transforms.ColorJitter(brightness=0.1, contrast=0.1),  # Less color jitter
        transforms.RandomResizedCrop(image_size, scale=(0.9, 1.0)),  # Less aggressive
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def train_epoch(model, loader, criterion, optimizer, device):
    """Train with better logging."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc='Training')
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar every 10 batches
        if batch_idx % 10 == 0:
            current_acc = 100. * correct / total
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{current_acc:.2f}%'
            })
    
    return total_loss / len(loader), 100. * correct / total

def validate(model, loader, criterion, device):
    """Validate with detailed metrics."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    # Track per-class accuracy
    class_correct = [0, 0]
    class_total = [0, 0]
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Validating'):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Per-class accuracy
            for i in range(labels.size(0)):
                label = labels[i]
                class_correct[label] += (predicted[i] == label).item()
                class_total[label] += 1
    
    class_acc = [100. * class_correct[i] / max(class_total[i], 1) for i in range(2)]
    
    return total_loss / len(loader), 100. * correct / total, class_acc

def main():
    print("="*60)
    print("FIXED MANIPULATION DETECTOR TRAINING")
    print("="*60)
    
    # Configuration
    DATA_DIR = 'manipulation_data_combined'
    BATCH_SIZE = 64  # Larger batch size
    EPOCHS = 20
    LEARNING_RATE = 0.0001  # Lower learning rate
    IMAGE_SIZE = 224
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Transforms
    train_transform, val_transform = get_transforms(IMAGE_SIZE)
    
    # Datasets
    print("\nLoading data...")
    train_dataset = ManipulationDataset(
        Path(DATA_DIR) / 'train',
        transform=train_transform
    )
    
    val_dataset = ManipulationDataset(
        Path(DATA_DIR) / 'val',
        transform=val_transform
    )
    
    # Create balanced sampler for training
    labels = [label for _, label in train_dataset.samples]
    class_counts = np.bincount(labels)
    class_weights = 1. / class_counts
    sample_weights = [class_weights[label] for label in labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,  # Use balanced sampler
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
    print("\nInitializing model...")
    model = LightManipulationCNN().to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Better optimizer with weight decay
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
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
        val_loss, val_acc, class_acc = validate(
            model, val_loader, criterion, device
        )
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Learning rate scheduling
        scheduler.step()
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"  Class Acc - Authentic: {class_acc[0]:.2f}%, Manipulated: {class_acc[1]:.2f}%")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_manipulation_model.pth')
            print(f"  ✓ New best model saved! (Val Acc: {val_acc:.2f}%)")
        
        # Early stopping if no improvement for 5 epochs
        if epoch > 5 and val_acc < max(val_accs[-5:]):
            print(f"\nEarly stopping - no improvement for 5 epochs")
            break
    
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
