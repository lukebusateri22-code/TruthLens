"""
Simple incremental training - FIXED for Python 3.13 multiprocessing
Original data + 2,000 new CASIA images, 10 epochs
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
import random
import time
import shutil

def prepare_dataset():
    """Prepare the incremental dataset."""
    ORIGINAL_DATA = Path("manipulation_data_combined")
    CASIA_FULL = Path("/Users/cn424694/.cache/kagglehub/datasets/divg07/casia-20-image-tampering-detection-dataset/versions/1/CASIA2")
    NEW_DATA_DIR = Path("manipulation_data_incremental")
    
    print("="*60)
    print("PREPARING DATASET")
    print("="*60)
    
    if NEW_DATA_DIR.exists():
        shutil.rmtree(NEW_DATA_DIR)
    
    for split in ['train', 'val']:
        for category in ['authentic', 'manipulated']:
            (NEW_DATA_DIR / split / category).mkdir(parents=True, exist_ok=True)
    
    # Copy original data
    print("\nCopying original data...")
    for split in ['train', 'val']:
        for category in ['authentic', 'manipulated']:
            src_dir = ORIGINAL_DATA / split / category
            dst_dir = NEW_DATA_DIR / split / category
            
            if src_dir.exists():
                files = list(src_dir.glob('*.*'))
                for f in tqdm(files, desc=f'{split}/{category}'):
                    shutil.copy2(f, dst_dir / f.name)
    
    # Add 2,000 new CASIA images
    print("\nAdding 2,000 new CASIA images...")
    casia_auth = list((CASIA_FULL / 'Au').glob('*.jpg')) + list((CASIA_FULL / 'Au').glob('*.png'))
    casia_tamper = list((CASIA_FULL / 'Tp').glob('*.jpg')) + list((CASIA_FULL / 'Tp').glob('*.png'))
    
    new_auth = random.sample(casia_auth, min(1000, len(casia_auth)))
    new_tamper = random.sample(casia_tamper, min(1000, len(casia_tamper)))
    
    train_split = int(len(new_auth) * 0.85)
    
    for i, img_path in enumerate(tqdm(new_auth, desc="Adding authentic")):
        try:
            img = Image.open(img_path).convert('RGB')
            if i < train_split:
                dest = NEW_DATA_DIR / 'train' / 'authentic' / f'casia_auth_{i:04d}.jpg'
            else:
                dest = NEW_DATA_DIR / 'val' / 'authentic' / f'casia_auth_{i:04d}.jpg'
            img.save(dest, quality=95)
        except:
            continue
    
    for i, img_path in enumerate(tqdm(new_tamper, desc="Adding tampered")):
        try:
            img = Image.open(img_path).convert('RGB')
            if i < train_split:
                dest = NEW_DATA_DIR / 'train' / 'manipulated' / f'casia_tamper_{i:04d}.jpg'
            else:
                dest = NEW_DATA_DIR / 'val' / 'manipulated' / f'casia_tamper_{i:04d}.jpg'
            img.save(dest, quality=95)
        except:
            continue
    
    # Count final
    final_train = len(list((NEW_DATA_DIR / 'train' / 'authentic').glob('*.*'))) + len(list((NEW_DATA_DIR / 'train' / 'manipulated').glob('*.*')))
    final_val = len(list((NEW_DATA_DIR / 'val' / 'authentic').glob('*.*'))) + len(list((NEW_DATA_DIR / 'val' / 'manipulated').glob('*.*')))
    
    print(f"\n✓ Dataset ready: {final_train} train, {final_val} val")
    return NEW_DATA_DIR

class ImageDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.samples = []
        self.transform = transform
        
        data_path = Path(data_dir)
        for class_dir, label in [('authentic', 0), ('manipulated', 1)]:
            class_path = data_path / class_dir
            if class_path.exists():
                for img_path in class_path.glob('*.*'):
                    self.samples.append((str(img_path), label))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            image = Image.new('RGB', (224, 224), color='gray')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def train():
    """Train the model."""
    print("\n" + "="*60)
    print("TRAINING MODEL")
    print("="*60)
    
    NEW_DATA_DIR = Path("manipulation_data_incremental")
    
    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    train_dataset = ImageDataset(NEW_DATA_DIR / 'train', transform=train_transform)
    val_dataset = ImageDataset(NEW_DATA_DIR / 'val', transform=val_transform)
    
    print(f"Dataset: {len(train_dataset)} train, {len(val_dataset)} val")
    
    # Balanced sampling
    labels = [label for _, label in train_dataset.samples]
    class_counts = np.bincount(labels)
    class_weights = 1. / class_counts
    sample_weights = [class_weights[label] for label in labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    # DataLoaders - num_workers=0 to avoid multiprocessing issues
    BATCH_SIZE = 48
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_features, 256),
        nn.SiLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(256, 2)
    )
    model = model.to(device)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    
    # Train
    EPOCHS = 10
    best_val_acc = 0
    start_time = time.time()
    
    print(f"\nTraining for {EPOCHS} epochs...")
    print("="*60)
    
    for epoch in range(EPOCHS):
        # Train
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS}'):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_acc = 100. * correct / total
        
        # Validate
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc='Validating'):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * correct / total
        scheduler.step()
        
        elapsed = time.time() - start_time
        print(f"\nEpoch {epoch+1}/{EPOCHS} ({elapsed/60:.1f}m)")
        print(f"  Train: {train_acc:.2f}%")
        print(f"  Val:   {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_manipulation_incremental.pth')
            print(f"  ✓ NEW BEST!")
        
        if val_acc >= 85.0:
            print(f"\n🎉 TARGET REACHED!")
            break
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Best accuracy: {best_val_acc:.2f}%")
    print(f"Time: {(time.time() - start_time)/60:.1f} minutes")
    print(f"Model saved: best_manipulation_incremental.pth")
    
    if best_val_acc >= 85.0:
        print("\n🎉 SUCCESS! 85%+ achieved!")
    elif best_val_acc >= 80.0:
        print("\n✓ EXCELLENT! 80%+ is great!")
    elif best_val_acc >= 75.0:
        print("\n✓ GOOD! 75%+ is solid!")
    else:
        print(f"\n⚠️ {best_val_acc:.2f}% - may need more data")

def main():
    print("="*60)
    print("INCREMENTAL TRAINING - FIXED")
    print("Original + 2,000 new images, 10 epochs")
    print("="*60)
    
    # Prepare dataset
    prepare_dataset()
    
    # Train
    train()

if __name__ == '__main__':
    main()
