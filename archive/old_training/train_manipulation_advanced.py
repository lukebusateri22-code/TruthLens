"""
Advanced Training for 85%+ Accuracy
Uses multiple techniques: data augmentation, mixup, cosine annealing, etc.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from pathlib import Path
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import cv2

from models.manipulation_cnn_improved import VeryDeepManipulationCNN

class AdvancedManipulationDataset(torch.utils.data.Dataset):
    """Dataset with heavy augmentation and mixup support."""
    
    def __init__(self, data_dir, transform=None, augment=True):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.augment = augment
        self.samples = []
        
        # Load images
        authentic_dir = self.data_dir / 'authentic'
        manipulated_dir = self.data_dir / 'manipulated'
        
        for img_path in authentic_dir.glob('*.*'):
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tif']:
                self.samples.append((str(img_path), 0))
        
        for img_path in manipulated_dir.glob('*.*'):
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tif']:
                self.samples.append((str(img_path), 1))
        
        print(f"Loaded {len(self.samples)} samples")
        print(f"  Authentic: {sum(1 for _, label in self.samples if label == 0)}")
        print(f"  Manipulated: {sum(1 for _, label in self.samples if label == 1)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Apply heavy augmentation if enabled
        if self.augment and random.random() > 0.3:
            image = self.heavy_augment(image)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def heavy_augment(self, image):
        """Apply heavy augmentation for manipulation detection."""
        
        # Random choice of augmentation
        aug_type = random.choice([
            'blur', 'noise', 'brightness', 'contrast', 
            'jpeg', 'rotation', 'crop', 'flip'
        ])
        
        if aug_type == 'blur':
            # Random blur (simulates compression artifacts)
            radius = random.uniform(0.5, 2.0)
            image = image.filter(ImageFilter.GaussianBlur(radius))
        
        elif aug_type == 'noise':
            # Add Gaussian noise
            img_array = np.array(image)
            noise = np.random.normal(0, random.uniform(5, 15), img_array.shape)
            img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
            image = Image.fromarray(img_array)
        
        elif aug_type == 'brightness':
            # Brightness adjustment
            enhancer = ImageEnhance.Brightness(image)
            factor = random.uniform(0.7, 1.3)
            image = enhancer.enhance(factor)
        
        elif aug_type == 'contrast':
            # Contrast adjustment
            enhancer = ImageEnhance.Contrast(image)
            factor = random.uniform(0.7, 1.3)
            image = enhancer.enhance(factor)
        
        elif aug_type == 'jpeg':
            # JPEG compression simulation
            quality = random.randint(30, 90)
            import io
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG', quality=quality)
            image = Image.open(buffer)
        
        elif aug_type == 'rotation':
            # Rotation
            angle = random.uniform(-15, 15)
            image = image.rotate(angle, expand=True, fillcolor=(255, 255, 255))
        
        elif aug_type == 'crop':
            # Random crop
            w, h = image.size
            crop_w = int(w * random.uniform(0.8, 1.0))
            crop_h = int(h * random.uniform(0.8, 1.0))
            left = random.randint(0, w - crop_w)
            top = random.randint(0, h - crop_h)
            image = image.crop((left, top, left + crop_w, top + crop_h))
            image = image.resize((w, h), Image.LANCZOS)
        
        elif aug_type == 'flip':
            # Horizontal flip
            if random.random() > 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
        
        return image

def mixup_data(x, y, alpha=0.4):
    """Apply mixup augmentation."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss function."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def get_advanced_transforms(image_size=224):
    """Advanced transforms for better generalization."""
    
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.1))
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def train_epoch_advanced(model, loader, criterion, optimizer, device, use_mixup=True):
    """Advanced training with mixup and gradient clipping."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc='Training')
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)
        
        # Apply mixup
        if use_mixup and random.random() > 0.5:
            mixed_images, y_a, y_b, lam = mixup_data(images, labels)
            mixed_images, y_a, y_b = mixed_images.to(device), y_a.to(device), y_b.to(device)
            
            optimizer.zero_grad()
            outputs = model(mixed_images)
            loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
        else:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # For mixup, we can't easily compute accuracy during training
        if not (use_mixup and random.random() > 0.5):
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        if batch_idx % 20 == 0:
            current_acc = 100. * correct / max(total, 1)
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{current_acc:.2f}%'
            })
    
    return total_loss / len(loader), 100. * correct / max(total, 1)

def validate_advanced(model, loader, criterion, device):
    """Advanced validation with detailed metrics."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    # Track per-class metrics
    class_correct = [0, 0]
    class_total = [0, 0]
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Validating'):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Collect for metrics
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Per-class accuracy
            for i in range(labels.size(0)):
                label = labels[i]
                class_correct[label] += (predicted[i] == label).item()
                class_total[label] += 1
    
    class_acc = [100. * class_correct[i] / max(class_total[i], 1) for i in range(2)]
    
    # Calculate precision, recall, F1
    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average=None
    )
    
    return total_loss / len(loader), 100. * correct / total, class_acc, precision, recall, f1

def main():
    print("="*60)
    print("ADVANCED MANIPULATION DETECTOR TRAINING")
    print("Target: 85%+ Accuracy")
    print("="*60)
    
    # Configuration
    DATA_DIR = 'manipulation_data_combined'
    BATCH_SIZE = 32  # Smaller batch for larger model
    EPOCHS = 50  # More epochs
    LEARNING_RATE = 0.0001  # Start with lower LR
    IMAGE_SIZE = 224
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Advanced transforms
    train_transform, val_transform = get_advanced_transforms(IMAGE_SIZE)
    
    # Datasets with heavy augmentation
    print("\nLoading data with heavy augmentation...")
    train_dataset = AdvancedManipulationDataset(
        Path(DATA_DIR) / 'train',
        transform=train_transform,
        augment=True
    )
    
    val_dataset = AdvancedManipulationDataset(
        Path(DATA_DIR) / 'val',
        transform=val_transform,
        augment=False
    )
    
    # Create balanced sampler
    labels = [label for _, label in train_dataset.samples]
    class_counts = np.bincount(labels)
    class_weights = 1. / class_counts
    sample_weights = [class_weights[label] for label in labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
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
    
    # Advanced model
    print("\nInitializing advanced model...")
    model = VeryDeepManipulationCNN().to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Advanced optimizer and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    
    # Cosine annealing with warm restarts
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    # Training loop
    best_val_acc = 0
    patience = 10  # Early stopping patience
    no_improve = 0
    
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    
    print(f"\nStarting advanced training for {EPOCHS} epochs...")
    print("="*60)
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print("-" * 60)
        
        # Train
        train_loss, train_acc = train_epoch_advanced(
            model, train_loader, criterion, optimizer, device, use_mixup=True
        )
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validate
        val_loss, val_acc, class_acc, precision, recall, f1 = validate_advanced(
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
        print(f"  Precision - Authentic: {precision[0]:.3f}, Manipulated: {precision[1]:.3f}")
        print(f"  Recall - Authentic: {recall[0]:.3f}, Manipulated: {recall[1]:.3f}")
        print(f"  F1-Score - Authentic: {f1[0]:.3f}, Manipulated: {f1[1]:.3f}")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_manipulation_model_advanced.pth')
            print(f"  ✓ New best model saved! (Val Acc: {val_acc:.2f}%)")
            no_improve = 0
        else:
            no_improve += 1
        
        # Early stopping
        if no_improve >= patience:
            print(f"\nEarly stopping - no improvement for {patience} epochs")
            break
        
        # Target reached
        if val_acc >= 85.0:
            print(f"\n🎉 Target reached! {val_acc:.2f}% accuracy!")
            break
    
    print("\n" + "="*60)
    print("Advanced training complete!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Model saved as: best_manipulation_model_advanced.pth")
    
    # Plot training curves
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(val_accs, 'b-', label='Validation Accuracy')
    plt.axhline(y=85, color='r', linestyle='--', label='Target (85%)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Progress Towards Target')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('manipulation_advanced_training_curves.png', dpi=150, bbox_inches='tight')
    print(f"\nTraining curves saved as: manipulation_advanced_training_curves.png")

if __name__ == "__main__":
    main()
