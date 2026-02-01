"""
Train AI-Generated Image Detector
Uses CIFAKE, DALL-E, and Real vs AI Art datasets
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from PIL import Image
import os
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

from models.ai_generated_detector import AIGeneratedDetector, get_ai_detector_transforms

class AIImageDataset(Dataset):
    """Dataset for AI-generated vs real images."""
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        
        # Find all images and their labels
        for label_dir in self.root_dir.iterdir():
            if not label_dir.is_dir():
                continue
                
            # Determine label (0=real, 1=AI-generated)
            dir_name = label_dir.name.lower()
            if 'real' in dir_name or 'authentic' in dir_name:
                label = 0
            elif 'fake' in dir_name or 'ai' in dir_name or 'generated' in dir_name or 'synthetic' in dir_name:
                label = 1
            else:
                continue
            
            # Add all images from this directory
            for img_path in label_dir.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    self.samples.append((str(img_path), label))
        
        print(f"  Found {len(self.samples)} images in {root_dir}")
        real_count = sum(1 for _, label in self.samples if label == 0)
        ai_count = sum(1 for _, label in self.samples if label == 1)
        print(f"    Real: {real_count}, AI-generated: {ai_count}")
    
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
            # Return a random other sample if this one fails
            return self.__getitem__((idx + 1) % len(self))


def train_ai_detector():
    """Train the AI-generated image detector."""
    
    print("="*70)
    print("🤖 TRAINING AI-GENERATED IMAGE DETECTOR")
    print("="*70)
    print()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print()
    
    # Dataset paths
    cifake_path = Path.home() / ".cache/kagglehub/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images/versions/3"
    dalle_path = Path.home() / ".cache/kagglehub/datasets/superpotato9/dalle-recognition-dataset/versions/7"
    real_ai_art_path = Path.home() / ".cache/kagglehub/datasets/ravidussilva/real-ai-art/versions/5"
    
    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = get_ai_detector_transforms()
    
    # Load datasets
    print("📦 Loading datasets...")
    datasets = []
    
    if cifake_path.exists():
        print(f"\n1. CIFAKE Dataset")
        cifake_train = AIImageDataset(cifake_path / "train", transform=train_transform)
        datasets.append(cifake_train)
    
    if dalle_path.exists():
        print(f"\n2. DALL-E Dataset")
        dalle_dataset = AIImageDataset(dalle_path, transform=train_transform)
        datasets.append(dalle_dataset)
    
    if real_ai_art_path.exists():
        print(f"\n3. Real vs AI Art Dataset")
        real_ai_dataset = AIImageDataset(real_ai_art_path, transform=train_transform)
        datasets.append(real_ai_dataset)
    
    if not datasets:
        print("❌ No datasets found! Please check paths.")
        return
    
    # Combine datasets
    combined_dataset = ConcatDataset(datasets)
    print(f"\n✓ Total training samples: {len(combined_dataset)}")
    
    # Split into train/val
    train_size = int(0.8 * len(combined_dataset))
    val_size = len(combined_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        combined_dataset, [train_size, val_size]
    )
    
    print(f"  Training: {len(train_dataset)}")
    print(f"  Validation: {len(val_dataset)}")
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Model
    print(f"\n🧠 Creating model...")
    model = AIGeneratedDetector(pretrained=True).to(device)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)
    
    # Training
    print(f"\n🚀 Starting training...")
    print("="*70)
    
    num_epochs = 20
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
            torch.save(model.state_dict(), 'best_ai_detector.pth')
            print(f"  ✓ New best model saved! (Val Acc: {val_acc:.2f}%)")
        
        # Early stopping if target reached
        if val_acc >= 90.0:
            print(f"\n🎯 Target accuracy reached! ({val_acc:.2f}%)")
            break
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Model saved as: best_ai_detector.pth")
    
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
    plt.savefig('ai_detector_training_curves.png', dpi=150, bbox_inches='tight')
    print(f"\n📊 Training curves saved to: ai_detector_training_curves.png")
    print("="*70)


if __name__ == "__main__":
    train_ai_detector()
