"""
ULTRA-FAST Manipulation Detector Training
RAM-cached for maximum speed
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

class PreloadedManipulationDataset(Dataset):
    """Preloads all images into RAM."""
    
    def __init__(self, root_dir, max_samples=15000):
        print(f"  Preloading from {root_dir}...")
        self.data = []
        self.labels = []
        
        transform = transforms.Compose([
            transforms.Resize((96, 96)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        label_map = {
            'authentic': 0, 'real': 0,
            'copy-move': 1, 'copymove': 1, 'splicing': 1, 'splice': 1,
            'retouching': 1, 'retouch': 1, 'manipulated': 1, 'fake': 1, 'tampered': 1
        }
        
        # Collect samples
        samples = []
        for label_dir in root_dir.rglob('*'):
            if not label_dir.is_dir():
                continue
            
            dir_name = label_dir.name.lower()
            label = None
            for key, val in label_map.items():
                if key in dir_name:
                    label = val
                    break
            
            if label is None:
                continue
            
            for img_path in label_dir.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tif']:
                    samples.append((str(img_path), label))
        
        # Sample
        if len(samples) > max_samples:
            samples = random.sample(samples, max_samples)
        
        # Preload
        print(f"  Loading {len(samples)} images into RAM...")
        for img_path, label in tqdm(samples, desc="  Preloading"):
            try:
                image = Image.open(img_path).convert('RGB')
                tensor = transform(image)
                self.data.append(tensor)
                self.labels.append(label)
            except:
                continue
        
        self.data = torch.stack(self.data)
        self.labels = torch.tensor(self.labels)
        
        print(f"  ✓ Loaded {len(self.data)} images")
        print(f"    Authentic: {(self.labels == 0).sum().item()}, Manipulated: {(self.labels == 1).sum().item()}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class TinyManipulationDetector(nn.Module):
    """Ultra-lightweight manipulation detector."""
    
    def __init__(self):
        super(TinyManipulationDetector, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def train_ultra_fast():
    """Ultra-fast training."""
    
    print("="*70)
    print("⚡⚡⚡ ULTRA-FAST MANIPULATION DETECTOR")
    print("="*70)
    print()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print()
    
    # Paths
    artifact_path = Path.home() / ".cache/kagglehub/datasets/awsaf49/artifact-dataset/versions/1"
    manipulation_path = Path("manipulation_data_combined/train")
    
    # Load datasets
    print("📦 Loading datasets into RAM...")
    datasets = []
    
    if artifact_path.exists():
        print(f"\n1. Artifact Dataset")
        artifact_dataset = PreloadedManipulationDataset(artifact_path, max_samples=10000)
        if len(artifact_dataset) > 0:
            datasets.append(artifact_dataset)
    
    if manipulation_path.exists():
        print(f"\n2. Manipulation Dataset")
        manip_dataset = PreloadedManipulationDataset(manipulation_path, max_samples=5000)
        if len(manip_dataset) > 0:
            datasets.append(manip_dataset)
    
    if not datasets:
        print("❌ No datasets found!")
        return
    
    # Combine
    from torch.utils.data import ConcatDataset
    combined = ConcatDataset(datasets)
    print(f"\n✓ Total samples in RAM: {len(combined)}")
    
    # Split
    train_size = int(0.85 * len(combined))
    val_size = len(combined) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(combined, [train_size, val_size])
    
    print(f"  Training: {len(train_dataset)}")
    print(f"  Validation: {len(val_dataset)}")
    
    # DataLoaders
    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"\n  Batch size: {batch_size}")
    print(f"  Batches per epoch: {len(train_loader)}")
    
    # Model
    print(f"\n🧠 Creating model...")
    model = TinyManipulationDetector().to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {params:,}")
    
    # Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5)
    
    # Training
    print(f"\n🚀 Starting training...")
    print("="*70)
    
    num_epochs = 20
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
            torch.save(model.state_dict(), 'best_manipulation_ultra_fast.pth')
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
    print(f"Total training time: {total_time/60:.1f} minutes ({total_time:.0f} seconds)")
    print(f"Model saved as: best_manipulation_ultra_fast.pth")
    
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
    plt.savefig('manipulation_ultra_fast_training.png', dpi=150, bbox_inches='tight')
    print(f"\n📊 Training curves saved to: manipulation_ultra_fast_training.png")
    print("="*70)


if __name__ == "__main__":
    train_ultra_fast()
