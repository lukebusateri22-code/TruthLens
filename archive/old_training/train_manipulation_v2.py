"""
Train Advanced Manipulation Detector V2
Uses Artifact dataset with ELA and multi-scale analysis
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

from models.manipulation_detector_v2 import AdvancedManipulationDetector, get_manipulation_transforms

class ManipulationDataset(Dataset):
    """Dataset for manipulation detection."""
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        
        # Map directory names to labels
        label_map = {
            'authentic': 0,
            'real': 0,
            'copy-move': 1,
            'copymove': 1,
            'splicing': 2,
            'splice': 2,
            'retouching': 3,
            'retouch': 3,
            'manipulated': 4,
            'fake': 4,
            'tampered': 4
        }
        
        # Find all images
        for label_dir in self.root_dir.rglob('*'):
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
            
            # Add all images
            for img_path in label_dir.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tif']:
                    self.samples.append((str(img_path), label))
        
        print(f"  Found {len(self.samples)} images")
        
        # Print class distribution
        from collections import Counter
        label_counts = Counter([label for _, label in self.samples])
        class_names = ['authentic', 'copy-move', 'splicing', 'retouching', 'other']
        for i, name in enumerate(class_names):
            print(f"    {name}: {label_counts.get(i, 0)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            return self.__getitem__((idx + 1) % len(self))


def train_manipulation_detector():
    """Train the advanced manipulation detector."""
    
    print("="*70)
    print("🔍 TRAINING ADVANCED MANIPULATION DETECTOR V2")
    print("="*70)
    print()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print()
    
    # Dataset paths
    artifact_path = Path.home() / ".cache/kagglehub/datasets/awsaf49/artifact-dataset/versions/1"
    manipulation_path = Path("manipulation_data_combined")
    
    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = get_manipulation_transforms()
    
    # Load datasets
    print("📦 Loading datasets...")
    datasets = []
    
    if artifact_path.exists():
        print(f"\n1. Artifact Dataset")
        artifact_dataset = ManipulationDataset(artifact_path, transform=train_transform)
        if len(artifact_dataset) > 0:
            datasets.append(artifact_dataset)
    
    if manipulation_path.exists():
        print(f"\n2. Manipulation Dataset")
        manip_train = ManipulationDataset(manipulation_path / "train", transform=train_transform)
        if len(manip_train) > 0:
            datasets.append(manip_train)
    
    if not datasets:
        print("❌ No datasets found! Please check paths.")
        return
    
    # Combine datasets
    from torch.utils.data import ConcatDataset
    combined_dataset = ConcatDataset(datasets)
    print(f"\n✓ Total training samples: {len(combined_dataset)}")
    
    # Split
    train_size = int(0.85 * len(combined_dataset))
    val_size = len(combined_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        combined_dataset, [train_size, val_size]
    )
    
    print(f"  Training: {len(train_dataset)}")
    print(f"  Validation: {len(val_dataset)}")
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=24, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=24, shuffle=False, num_workers=4)
    
    # Model
    print(f"\n🧠 Creating model...")
    model = AdvancedManipulationDetector(pretrained=True).to(device)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)
    
    # Training
    print(f"\n🚀 Starting training...")
    print("="*70)
    
    num_epochs = 25
    best_val_acc = 0.0
    train_losses = []
    val_accs = []
    
    for epoch in range(num_epochs):
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
        
        # Update scheduler
        scheduler.step(val_acc)
        
        # Save metrics
        train_losses.append(avg_train_loss)
        val_accs.append(val_acc)
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_manipulation_v2.pth')
            print(f"  ✓ New best model saved! (Val Acc: {val_acc:.2f}%)")
        
        # Early stopping if target reached
        if val_acc >= 85.0:
            print(f"\n🎯 Target accuracy reached! ({val_acc:.2f}%)")
            break
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Model saved as: best_manipulation_v2.pth")
    
    # Plot results
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accs)
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('manipulation_v2_training_curves.png', dpi=150, bbox_inches='tight')
    print(f"\n📊 Training curves saved to: manipulation_v2_training_curves.png")
    print("="*70)


if __name__ == "__main__":
    train_manipulation_detector()
