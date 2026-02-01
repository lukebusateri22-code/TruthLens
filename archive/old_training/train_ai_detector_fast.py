"""
FAST AI-Generated Image Detector Training
Optimized for speed with smaller model and smart sampling
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, models
from PIL import Image
import os
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import random

class FastAIDetector(nn.Module):
    """Lightweight but effective AI detector using MobileNetV2."""
    
    def __init__(self):
        super(FastAIDetector, self).__init__()
        
        # Use MobileNetV2 - much faster than EfficientNet
        self.backbone = models.mobilenet_v2(pretrained=True)
        
        # Replace classifier
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 2)  # Real vs AI
        )
    
    def forward(self, x):
        return self.backbone(x)


class FastAIDataset(Dataset):
    """Fast dataset with smart sampling."""
    
    def __init__(self, root_dir, transform=None, max_samples_per_class=25000):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        
        # Find all images
        for label_dir in self.root_dir.iterdir():
            if not label_dir.is_dir():
                continue
            
            dir_name = label_dir.name.lower()
            if 'real' in dir_name or 'authentic' in dir_name:
                label = 0
            elif 'fake' in dir_name or 'ai' in dir_name or 'generated' in dir_name or 'synthetic' in dir_name:
                label = 1
            else:
                continue
            
            # Collect all images
            class_samples = []
            for img_path in label_dir.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    class_samples.append((str(img_path), label))
            
            # Smart sampling - limit per class to balance and speed up
            if len(class_samples) > max_samples_per_class:
                class_samples = random.sample(class_samples, max_samples_per_class)
            
            self.samples.extend(class_samples)
        
        print(f"  Loaded {len(self.samples)} images from {root_dir}")
        real_count = sum(1 for _, label in self.samples if label == 0)
        ai_count = len(self.samples) - real_count
        print(f"    Real: {real_count}, AI: {ai_count}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except:
            return self.__getitem__((idx + 1) % len(self))


def train_fast():
    """Fast training with optimizations."""
    
    print("="*70)
    print("⚡ FAST AI-GENERATED IMAGE DETECTOR TRAINING")
    print("="*70)
    print()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cpu':
        print("⚠️  Training on CPU - will be slower but works!")
    print()
    
    # Paths
    cifake_path = Path.home() / ".cache/kagglehub/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images/versions/3/train"
    dalle_path = Path.home() / ".cache/kagglehub/datasets/superpotato9/dalle-recognition-dataset/versions/7"
    
    # Fast transforms - smaller size for speed
    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Smaller = faster
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load datasets with sampling
    print("📦 Loading datasets (with smart sampling for speed)...")
    datasets = []
    
    if cifake_path.exists():
        print(f"\n1. CIFAKE Dataset (sampling 25K per class)")
        cifake_dataset = FastAIDataset(cifake_path, transform=train_transform, max_samples_per_class=25000)
        datasets.append(cifake_dataset)
    
    if dalle_path.exists():
        print(f"\n2. DALL-E Dataset")
        dalle_dataset = FastAIDataset(dalle_path, transform=train_transform, max_samples_per_class=10000)
        datasets.append(dalle_dataset)
    
    if not datasets:
        print("❌ No datasets found!")
        return
    
    # Combine
    from torch.utils.data import ConcatDataset
    combined = ConcatDataset(datasets)
    print(f"\n✓ Total samples: {len(combined)}")
    
    # Split
    train_size = int(0.85 * len(combined))
    val_size = len(combined) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(combined, [train_size, val_size])
    
    print(f"  Training: {len(train_dataset)}")
    print(f"  Validation: {len(val_dataset)}")
    
    # DataLoaders - larger batch size for speed
    batch_size = 64 if device.type == 'cuda' else 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"\n  Batch size: {batch_size}")
    print(f"  Batches per epoch: {len(train_loader)}")
    
    # Model - lightweight
    print(f"\n🧠 Creating lightweight model...")
    model = FastAIDetector().to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {params:,} (much smaller = faster!)")
    
    # Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5)
    
    # Training
    print(f"\n🚀 Starting fast training...")
    print("="*70)
    
    num_epochs = 15  # Fewer epochs
    best_val_acc = 0.0
    train_losses = []
    val_accs = []
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        train_acc = 100 * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        # Validate
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        scheduler.step(val_acc)
        
        train_losses.append(avg_train_loss)
        val_accs.append(val_acc)
        
        epoch_time = time.time() - epoch_start
        
        print(f"\nEpoch {epoch+1}/{num_epochs} ({epoch_time:.1f}s)")
        print(f"  Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Acc: {val_acc:.2f}%")
        
        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_ai_detector_fast.pth')
            print(f"  ✓ New best model saved! (Val Acc: {val_acc:.2f}%)")
        
        # Early stopping
        if val_acc >= 90.0:
            print(f"\n🎯 Target accuracy reached! ({val_acc:.2f}%)")
            break
    
    total_time = time.time() - start_time
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Total training time: {total_time/60:.1f} minutes")
    print(f"Model saved as: best_ai_detector_fast.pth")
    
    # Plot
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accs)
    plt.axhline(y=90, color='r', linestyle='--', label='Target (90%)')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('ai_detector_fast_training.png', dpi=150, bbox_inches='tight')
    print(f"\n📊 Training curves saved to: ai_detector_fast_training.png")
    print("="*70)


if __name__ == "__main__":
    train_fast()
