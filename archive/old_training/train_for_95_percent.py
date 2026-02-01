"""
TRAIN FOR 95%+ ACCURACY
Enhanced training with data augmentation and longer training
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import random

class EnhancedPreloadedDataset(Dataset):
    """Preloaded dataset with augmentation for 95%+ accuracy."""
    
    def __init__(self, root_dir, max_samples=25000, augment=True):
        print(f"  Preloading from {root_dir}...")
        self.data = []
        self.labels = []
        self.augment = augment
        
        # Base transform
        base_transform = transforms.Compose([
            transforms.Resize((112, 112)),  # Slightly larger for better features
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Augmentation transform
        self.aug_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        ]) if augment else None
        
        # Find images
        samples = []
        for label_dir in root_dir.iterdir():
            if not label_dir.is_dir():
                continue
            
            dir_name = label_dir.name.lower()
            if 'real' in dir_name or 'authentic' in dir_name:
                label = 0
            elif 'fake' in dir_name or 'ai' in dir_name or 'generated' in dir_name or 'synthetic' in dir_name:
                label = 1
            else:
                continue
            
            for img_path in label_dir.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    samples.append((str(img_path), label))
        
        # Sample
        if len(samples) > max_samples:
            samples = random.sample(samples, max_samples)
        
        # Preload
        print(f"  Loading {len(samples)} images into RAM...")
        for img_path, label in tqdm(samples, desc="  Preloading"):
            try:
                image = Image.open(img_path).convert('RGB')
                tensor = base_transform(image)
                self.data.append(tensor)
                self.labels.append(label)
            except:
                continue
        
        self.data = torch.stack(self.data)
        self.labels = torch.tensor(self.labels)
        
        print(f"  ✓ Loaded {len(self.data)} images")
        print(f"    Real: {(self.labels == 0).sum().item()}, Fake: {(self.labels == 1).sum().item()}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx]
        
        # Apply augmentation during training
        if self.augment and random.random() > 0.5:
            # Convert back to PIL for augmentation
            img_pil = transforms.ToPILImage()(img)
            img_pil = self.aug_transform(img_pil)
            img = transforms.ToTensor()(img_pil)
            img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
        
        return img, label


class ImprovedDetector(nn.Module):
    """Improved detector for 95%+ accuracy."""
    
    def __init__(self):
        super(ImprovedDetector, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),
            
            # Block 2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
            
            # Block 3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
            
            # Block 4
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, 2)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def train_for_95():
    """Train for 95%+ accuracy."""
    
    print("="*70)
    print("🎯 TRAINING FOR 95%+ ACCURACY")
    print("="*70)
    print()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print()
    
    # Path
    cifake_path = Path.home() / ".cache/kagglehub/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images/versions/3/train"
    
    # Load dataset with augmentation
    print("📦 Loading dataset with augmentation...")
    dataset = EnhancedPreloadedDataset(cifake_path, max_samples=30000, augment=True)
    
    print(f"\n✓ Total samples: {len(dataset)}")
    
    # Split
    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    print(f"  Training: {len(train_dataset)}")
    print(f"  Validation: {len(val_dataset)}")
    
    # DataLoaders
    batch_size = 64  # Smaller batch for better gradients
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"\n  Batch size: {batch_size}")
    print(f"  Batches per epoch: {len(train_loader)}")
    
    # Model - larger for 95%
    print(f"\n🧠 Creating improved model...")
    model = ImprovedDetector().to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {params:,} (larger for better accuracy)")
    
    # Optimizer with cosine annealing
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-6)
    
    # Training
    print(f"\n🚀 Starting training for 95%+...")
    print("="*70)
    
    num_epochs = 30  # More epochs for 95%
    best_val_acc = 0.0
    train_losses = []
    val_accs = []
    patience = 5
    no_improve = 0
    
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
        scheduler.step()
        
        train_losses.append(avg_train_loss)
        val_accs.append(val_acc)
        
        epoch_time = time.time() - epoch_start
        
        print(f"\nEpoch {epoch+1}/{num_epochs} ({epoch_time:.1f}s)")
        print(f"  Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Acc: {val_acc:.2f}%")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_detector_95_percent.pth')
            print(f"  ✓ New best model saved! (Val Acc: {val_acc:.2f}%)")
            no_improve = 0
        else:
            no_improve += 1
        
        # Early stopping at 95%
        if val_acc >= 95.0:
            print(f"\n🎯 95% TARGET REACHED! ({val_acc:.2f}%)")
            break
        
        # Patience check
        if no_improve >= patience and epoch > 10:
            print(f"\n⏸️  No improvement for {patience} epochs, stopping early")
            break
    
    total_time = time.time() - start_time
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Total training time: {total_time/60:.1f} minutes")
    print(f"Model saved as: best_detector_95_percent.pth")
    
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
    plt.axhline(y=95, color='r', linestyle='--', label='Target (95%)')
    plt.axhline(y=90, color='orange', linestyle='--', label='Previous (90%)')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_95_percent.png', dpi=150, bbox_inches='tight')
    print(f"\n📊 Training curves saved to: training_95_percent.png")
    print("="*70)


if __name__ == "__main__":
    train_for_95()
