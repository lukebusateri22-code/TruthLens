"""
Monitored training - 5 epochs with progress logging
Smaller batch size to avoid freezing
Only save if better than 74%
Target: 87% validation accuracy
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
import sys

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

def main():
    print("="*60)
    print("MONITORED TRAINING - 5 EPOCHS")
    print("Starting from 74% checkpoint")
    print("Target: 87% validation accuracy")
    print("Smaller batch size + progress logging")
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
    print("\nLoading datasets...")
    train_dataset = ImageDataset(NEW_DATA_DIR / 'train', transform=train_transform)
    val_dataset = ImageDataset(NEW_DATA_DIR / 'val', transform=val_transform)
    
    print(f"✓ Dataset: {len(train_dataset)} train, {len(val_dataset)} val")
    
    # Balanced sampling
    labels = [label for _, label in train_dataset.samples]
    class_counts = np.bincount(labels)
    class_weights = 1. / class_counts
    sample_weights = [class_weights[label] for label in labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    # DataLoaders - SMALLER batch size to avoid freezing
    BATCH_SIZE = 32  # Reduced from 48
    print(f"✓ Batch size: {BATCH_SIZE}")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✓ Device: {device}")
    
    model = models.efficientnet_b0(weights=None)
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_features, 256),
        nn.SiLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(256, 2)
    )
    
    # Load checkpoint
    print("\nLoading 74% checkpoint...")
    model.load_state_dict(torch.load('best_manipulation_incremental.pth', map_location='cpu'))
    print("✓ Checkpoint loaded successfully")
    
    model = model.to(device)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
    
    # Train
    EPOCHS = 5
    best_val_acc = 74.00  # Only save if better than this
    TARGET_ACC = 87.00
    start_time = time.time()
    
    print(f"\n{'='*60}")
    print(f"TRAINING FOR {EPOCHS} EPOCHS")
    print(f"Current best: {best_val_acc:.2f}%")
    print(f"Target: {TARGET_ACC:.2f}%")
    print(f"{'='*60}\n")
    
    for epoch in range(EPOCHS):
        epoch_start = time.time()
        
        # Train
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        print(f"\n[Epoch {epoch+1}/{EPOCHS}] Training...")
        sys.stdout.flush()
        
        batch_count = 0
        for images, labels in tqdm(train_loader, desc=f'Training', ncols=80):
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
            
            batch_count += 1
            # Log every 25 batches
            if batch_count % 25 == 0:
                current_acc = 100. * correct / total
                print(f"  Batch {batch_count}: Train Acc = {current_acc:.2f}%", flush=True)
        
        train_acc = 100. * correct / total
        train_loss = train_loss / len(train_loader)
        
        # Validate
        print(f"\n[Epoch {epoch+1}/{EPOCHS}] Validating...")
        sys.stdout.flush()
        
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc='Validating', ncols=80):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * correct / total
        val_loss = val_loss / len(val_loader)
        scheduler.step()
        
        epoch_time = time.time() - epoch_start
        total_time = time.time() - start_time
        
        # Print results
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch+1}/{EPOCHS} COMPLETE ({epoch_time/60:.1f}m, {total_time/60:.1f}m total)")
        print(f"{'='*60}")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save if improved
        if val_acc > best_val_acc:
            improvement = val_acc - best_val_acc
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_manipulation_incremental.pth')
            print(f"✓ NEW BEST! {val_acc:.2f}% (+{improvement:.2f}%)")
            print(f"✓ Model saved!")
        else:
            print(f"⚠️ No improvement (best: {best_val_acc:.2f}%)")
        
        # Check if target reached
        if val_acc >= TARGET_ACC:
            print(f"\n{'='*60}")
            print(f"🎉 TARGET REACHED! {val_acc:.2f}% >= {TARGET_ACC:.2f}%")
            print(f"{'='*60}")
            break
        
        print(f"{'='*60}\n")
        sys.stdout.flush()
    
    # Final summary
    total_time = time.time() - start_time
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE!")
    print(f"{'='*60}")
    print(f"Starting accuracy: 74.00%")
    print(f"Final best accuracy: {best_val_acc:.2f}%")
    print(f"Improvement: +{best_val_acc - 74.00:.2f}%")
    print(f"Epochs completed: {epoch+1}/{EPOCHS}")
    print(f"Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    print(f"Model saved: best_manipulation_incremental.pth")
    
    if best_val_acc >= TARGET_ACC:
        print(f"\n🎉 SUCCESS! Target {TARGET_ACC:.2f}% achieved!")
    elif best_val_acc >= 80.0:
        print(f"\n✓ EXCELLENT! {best_val_acc:.2f}% is great progress!")
    elif best_val_acc > 74.0:
        print(f"\n✓ IMPROVED! +{best_val_acc - 74.00:.2f}% gain")
    else:
        print(f"\n⚠️ No improvement from 74.00%")
    
    print(f"{'='*60}\n")

if __name__ == '__main__':
    main()
