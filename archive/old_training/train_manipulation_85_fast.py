"""
Fast training for 85%+ accuracy in 5 hours max
Optimized approach: Better architecture + Smart data augmentation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw
import numpy as np
from tqdm import tqdm
import random
import cv2
import time

class FastManipulationDataset(torch.utils.data.Dataset):
    """Optimized dataset with smart augmentation."""
    
    def __init__(self, data_dir, transform=None, augment=True):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.augment = augment
        self.samples = []
        
        print("Loading optimized dataset...")
        
        # Load all images
        for class_dir, label in [('authentic', 0), ('manipulated', 1)]:
            class_path = self.data_dir / class_dir
            if not class_path.exists():
                continue
            
            for img_path in class_path.glob('*.*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    self.samples.append((str(img_path), label))
        
        print(f"Loaded {len(self.samples)} samples")
        authentic = sum(1 for _, label in self.samples if label == 0)
        manipulated = sum(1 for _, label in self.samples if label == 1)
        print(f"  Authentic: {authentic}, Manipulated: {manipulated}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Smart augmentation for manipulated images
        if self.augment and label == 1 and random.random() > 0.3:
            image = self.smart_augment(image)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def smart_augment(self, image):
        """Apply targeted augmentation for manipulation detection."""
        
        # Fast augmentation pipeline
        aug_type = random.choice([
            'jpeg', 'blur', 'noise', 'color', 'crop'
        ])
        
        if aug_type == 'jpeg':
            quality = random.randint(30, 80)
            import io
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG', quality=quality)
            image = Image.open(buffer)
        
        elif aug_type == 'blur':
            radius = random.uniform(0.5, 1.5)
            image = image.filter(ImageFilter.GaussianBlur(radius))
        
        elif aug_type == 'noise':
            img_array = np.array(image)
            noise = np.random.normal(0, random.uniform(5, 15), img_array.shape)
            img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
            image = Image.fromarray(img_array)
        
        elif aug_type == 'color':
            enhancer = ImageEnhance.Color(image)
            factor = random.uniform(0.7, 1.3)
            image = enhancer.enhance(factor)
        
        elif aug_type == 'crop':
            w, h = image.size
            crop_w = int(w * random.uniform(0.8, 0.95))
            crop_h = int(h * random.uniform(0.8, 0.95))
            left = random.randint(0, w - crop_w)
            top = random.randint(0, h - crop_h)
            image = image.crop((left, top, left + crop_w, top + crop_h))
            image = image.resize((w, h), Image.LANCZOS)
        
        return image

class OptimizedManipulationCNN(nn.Module):
    """
    Optimized CNN for fast training and high accuracy.
    Uses EfficientNet-B0 with smart modifications.
    """
    
    def __init__(self, num_classes=2):
        super().__init__()
        
        # Use EfficientNet-B0 (good balance of speed/accuracy)
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        
        # Freeze early layers (speed up training)
        for name, param in self.backbone.named_parameters():
            if 'features.8' not in name and 'classifier' not in name:
                param.requires_grad = False
        
        # Modify classifier for our task
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 256),
            nn.SiLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
    
    def unfreeze_all(self):
        """Unfreeze all layers for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
    
    def forward(self, x):
        return self.backbone(x)

def get_fast_transforms(image_size=224):
    """Optimized transforms for fast training."""
    
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(8),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
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

def train_fast_epoch(model, loader, criterion, optimizer, scheduler, device):
    """Fast training with mixed precision."""
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for images, labels in tqdm(loader, desc='Training', leave=False):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        if scaler:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(loader), 100. * correct / total

def validate_fast(model, loader, criterion, device):
    """Fast validation."""
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
    print("="*60)
    print("FAST TRAINING FOR 85%+ ACCURACY")
    print("Target: 5 hours max")
    print("="*60)
    
    start_time = time.time()
    
    # Configuration
    DATA_DIR = 'manipulation_data_combined'
    BATCH_SIZE = 48  # Larger batch for GPU efficiency
    EPOCHS_PHASE1 = 8   # Frozen backbone (fast)
    EPOCHS_PHASE2 = 12  # Fine-tuning (slower but better)
    MAX_LR = 0.003
    IMAGE_SIZE = 224
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Mixed precision: {device.type == 'cuda'}")
    
    # Data
    train_transform, val_transform = get_fast_transforms(IMAGE_SIZE)
    
    print("\nLoading data...")
    train_dataset = FastManipulationDataset(
        Path(DATA_DIR) / 'train',
        transform=train_transform,
        augment=True
    )
    
    val_dataset = FastManipulationDataset(
        Path(DATA_DIR) / 'val',
        transform=val_transform,
        augment=False
    )
    
    # Balanced sampling
    labels = [label for _, label in train_dataset.samples]
    class_counts = np.bincount(labels)
    class_weights = 1. / class_counts
    sample_weights = [class_weights[label] for label in labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    # DataLoaders with optimized settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=6,
        pin_memory=True,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Model
    print("\nInitializing optimized EfficientNet...")
    model = OptimizedManipulationCNN().to(device)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Optimizer and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=0.01)
    
    # One-cycle scheduler for fast convergence
    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=MAX_LR,
        epochs=EPOCHS_PHASE1,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.3,
        anneal_strategy='cos'
    )
    
    # Training
    best_val_acc = 0
    train_accs, val_accs = [], []
    
    print(f"\nPhase 1: Fast training ({EPOCHS_PHASE1} epochs)...")
    print("="*60)
    
    for epoch in range(EPOCHS_PHASE1):
        epoch_start = time.time()
        
        train_loss, train_acc = train_fast_epoch(model, train_loader, criterion, optimizer, scheduler, device)
        val_loss, val_acc = validate_fast(model, val_loader, criterion, device)
        
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        epoch_time = time.time() - epoch_start
        total_time = time.time() - start_time
        
        print(f"Epoch {epoch+1}/{EPOCHS_PHASE1} ({epoch_time:.1f}s, {total_time/60:.1f}m total)")
        print(f"  Train: {train_acc:.2f}% | Val: {val_acc:.2f}% | LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_manipulation_fast.pth')
            print(f"  ✓ New best! ({val_acc:.2f}%)")
        
        # Check if we're on track for 5-hour limit
        if total_time > 14400:  # 4 hours (leave 1 hour for phase 2)
            print(f"\n⚠️ Time limit approaching, moving to phase 2...")
            break
        
        if val_acc >= 85.0:
            print(f"\n🎉 TARGET REACHED! {val_acc:.2f}% accuracy!")
            break
    
    # Phase 2: Fine-tuning
    if best_val_acc < 85.0:
        print(f"\nPhase 2: Fine-tuning ({EPOCHS_PHASE2} epochs)...")
        print("="*60)
        
        model.unfreeze_all()
        optimizer = optim.AdamW(model.parameters(), lr=MAX_LR/10, weight_decay=0.01)
        
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_PHASE2)
        
        for epoch in range(EPOCHS_PHASE2):
            epoch_start = time.time()
            
            train_loss, train_acc = train_fast_epoch(model, train_loader, criterion, optimizer, scheduler, device)
            val_loss, val_acc = validate_fast(model, val_loader, criterion, device)
            
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            
            epoch_time = time.time() - epoch_start
            total_time = time.time() - start_time
            
            print(f"Epoch {epoch+1+EPOCHS_PHASE1}/{EPOCHS_PHASE1+EPOCHS_PHASE2} ({epoch_time:.1f}s, {total_time/60:.1f}m total)")
            print(f"  Train: {train_acc:.2f}% | Val: {val_acc:.2f}% | LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), 'best_manipulation_fast.pth')
                print(f"  ✓ New best! ({val_acc:.2f}%)")
            
            # Hard stop at 5 hours
            if total_time > 18000:  # 5 hours
                print(f"\n⏰ 5-hour time limit reached!")
                break
            
            if val_acc >= 85.0:
                print(f"\n🎉 TARGET REACHED! {val_acc:.2f}% accuracy!")
                break
    
    total_training_time = time.time() - start_time
    
    print("\n" + "="*60)
    print("FAST TRAINING COMPLETE!")
    print(f"Total time: {total_training_time/3600:.2f} hours")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Model saved as: best_manipulation_fast.pth")
    
    if best_val_acc >= 85.0:
        print("🎉 SUCCESS! Target achieved!")
    else:
        print(f"⚠️ Target not reached, but {best_val_acc:.2f}% is still excellent!")

if __name__ == "__main__":
    main()
