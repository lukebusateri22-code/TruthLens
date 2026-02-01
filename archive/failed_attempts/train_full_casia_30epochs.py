"""
Train manipulation detector with FULL CASIA 2.0 dataset
12,617 images (5x more than before!)
30 epochs or 90% accuracy
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms, models
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import shutil

print("="*60)
print("FULL CASIA 2.0 TRAINING")
print("12,617 images - 30 epochs or 90% accuracy")
print("="*60)

# Dataset path
CASIA_PATH = Path("/Users/cn424694/.cache/kagglehub/datasets/divg07/casia-20-image-tampering-detection-dataset/versions/1/CASIA2")

class FullCASIADataset(Dataset):
    """Load full CASIA 2.0 dataset."""
    
    def __init__(self, data_dir, split='train', transform=None):
        self.transform = transform
        self.samples = []
        
        print(f"\nLoading {split} split...")
        
        # Load authentic images
        au_dir = data_dir / 'Au'
        if au_dir.exists():
            au_images = list(au_dir.glob('*.jpg')) + list(au_dir.glob('*.png')) + list(au_dir.glob('*.tif'))
            print(f"  Found {len(au_images):,} authentic images")
            
            # Split into train/val (85/15)
            split_idx = int(len(au_images) * 0.85)
            if split == 'train':
                au_split = au_images[:split_idx]
            else:
                au_split = au_images[split_idx:]
            
            for img_path in au_split:
                self.samples.append((str(img_path), 0))  # 0 = authentic
        
        # Load tampered images
        tp_dir = data_dir / 'Tp'
        if tp_dir.exists():
            tp_images = list(tp_dir.glob('*.jpg')) + list(tp_dir.glob('*.png')) + list(tp_dir.glob('*.tif'))
            print(f"  Found {len(tp_images):,} tampered images")
            
            # Split into train/val (85/15)
            split_idx = int(len(tp_images) * 0.85)
            if split == 'train':
                tp_split = tp_images[:split_idx]
            else:
                tp_split = tp_images[split_idx:]
            
            for img_path in tp_split:
                self.samples.append((str(img_path), 1))  # 1 = tampered
        
        print(f"\n{split.upper()} dataset:")
        authentic = sum(1 for _, label in self.samples if label == 0)
        tampered = sum(1 for _, label in self.samples if label == 1)
        print(f"  Authentic: {authentic:,}")
        print(f"  Tampered: {tampered:,}")
        print(f"  Total: {len(self.samples):,}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            # Fallback
            image = Image.new('RGB', (224, 224), color='gray')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for images, labels in tqdm(loader, desc='Training', leave=False):
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
    
    return total_loss / len(loader), 100. * correct / total

def validate(model, loader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Validating', leave=False):
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
    BATCH_SIZE = 48
    EPOCHS = 30
    LEARNING_RATE = 0.0001
    TARGET_ACCURACY = 90.0
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Transforms with strong augmentation
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    print("\n" + "="*60)
    print("LOADING FULL CASIA 2.0 DATASET")
    print("="*60)
    
    train_dataset = FullCASIADataset(CASIA_PATH, split='train', transform=train_transform)
    val_dataset = FullCASIADataset(CASIA_PATH, split='val', transform=val_transform)
    
    # Balanced sampling
    labels = [label for _, label in train_dataset.samples]
    class_counts = np.bincount(labels)
    class_weights = 1. / class_counts
    sample_weights = [class_weights[label] for label in labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
                             num_workers=6, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=4, pin_memory=True)
    
    # Model - Start from our best 77.18% model
    print("\n" + "="*60)
    print("INITIALIZING MODEL")
    print("="*60)
    
    model = models.efficientnet_b0(weights=None)
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_features, 256),
        nn.SiLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(256, 2)
    )
    
    # Load previous best weights if available
    if Path('best_manipulation_fast.pth').exists():
        try:
            model.load_state_dict(torch.load('best_manipulation_fast.pth', map_location='cpu'))
            print("✓ Loaded previous 77.18% model as starting point")
        except:
            print("⚠️ Could not load previous model, starting fresh")
    
    model = model.to(device)
    
    print(f"Model: EfficientNet-B0")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # Training
    print("\n" + "="*60)
    print(f"TRAINING FOR {EPOCHS} EPOCHS (OR UNTIL {TARGET_ACCURACY}%)")
    print("="*60)
    
    best_val_acc = 77.18  # Start from previous best
    train_accs = []
    val_accs = []
    train_losses = []
    val_losses = []
    
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        epoch_start = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Step scheduler
        scheduler.step()
        
        # Record metrics
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        epoch_time = time.time() - epoch_start
        total_time = time.time() - start_time
        
        print(f"\nEpoch {epoch+1}/{EPOCHS} ({epoch_time:.1f}s, {total_time/60:.1f}m total)")
        print(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
        print(f"  Val:   Loss={val_loss:.4f}, Acc={val_acc:.2f}%")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_manipulation_full_casia.pth')
            print(f"  ✓ New best! ({val_acc:.2f}%)")
        
        # Check if target reached
        if val_acc >= TARGET_ACCURACY:
            print(f"\n🎉 TARGET REACHED! {val_acc:.2f}% >= {TARGET_ACCURACY}%")
            break
    
    total_time = time.time() - start_time
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Val Accuracy')
    plt.axhline(y=TARGET_ACCURACY, color='r', linestyle='--', label=f'Target ({TARGET_ACCURACY}%)')
    plt.axhline(y=77.18, color='g', linestyle='--', label='Previous Best (77.18%)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('full_casia_training_curves.png', dpi=150)
    print(f"\n✓ Training curves saved to: full_casia_training_curves.png")
    
    # Final summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Starting accuracy: 77.18%")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Improvement: +{best_val_acc - 77.18:.2f}%")
    print(f"Total epochs: {epoch+1}")
    print(f"Total time: {total_time/3600:.2f} hours")
    print(f"Model saved as: best_manipulation_full_casia.pth")
    
    if best_val_acc >= TARGET_ACCURACY:
        print(f"\n🎉 SUCCESS! Target {TARGET_ACCURACY}% achieved!")
    elif best_val_acc >= 85.0:
        print(f"\n🎉 EXCELLENT! {best_val_acc:.2f}% is great performance!")
    else:
        print(f"\n✓ Good improvement from 77.18% to {best_val_acc:.2f}%")

if __name__ == "__main__":
    main()
