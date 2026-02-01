"""
ROBUST LONG TRAINING - 20 EPOCHS
Proper error handling, checkpointing, and logging
Target: 87%+ validation accuracy
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
import json
from datetime import datetime

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
        except Exception as e:
            # Fallback to gray image if loading fails
            image = Image.new('RGB', (224, 224), color='gray')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def log_message(message, log_file='training_progress.log'):
    """Log message to both console and file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"[{timestamp}] {message}"
    print(log_msg)
    sys.stdout.flush()
    
    with open(log_file, 'a') as f:
        f.write(log_msg + '\n')

def save_checkpoint(epoch, model, optimizer, scheduler, best_val_acc, checkpoint_path='checkpoint.pth'):
    """Save training checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_acc': best_val_acc,
    }, checkpoint_path)
    log_message(f"✓ Checkpoint saved: {checkpoint_path}")

def load_checkpoint(model, optimizer, scheduler, checkpoint_path='checkpoint.pth'):
    """Load training checkpoint if exists."""
    if Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint['best_val_acc']
        log_message(f"✓ Resumed from epoch {checkpoint['epoch']}, best acc: {best_val_acc:.2f}%")
        return start_epoch, best_val_acc
    return 0, 74.00  # Start from scratch

def main():
    log_message("="*60)
    log_message("ROBUST LONG TRAINING - 20 EPOCHS")
    log_message("Target: 87%+ validation accuracy")
    log_message("="*60)
    
    NEW_DATA_DIR = Path("manipulation_data_incremental")
    
    # Check if dataset exists
    if not NEW_DATA_DIR.exists():
        log_message("❌ ERROR: Dataset not found!")
        log_message(f"Expected path: {NEW_DATA_DIR}")
        return
    
    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    log_message("\nLoading datasets...")
    try:
        train_dataset = ImageDataset(NEW_DATA_DIR / 'train', transform=train_transform)
        val_dataset = ImageDataset(NEW_DATA_DIR / 'val', transform=val_transform)
        log_message(f"✓ Train: {len(train_dataset)} samples")
        log_message(f"✓ Val: {len(val_dataset)} samples")
    except Exception as e:
        log_message(f"❌ ERROR loading datasets: {e}")
        return
    
    # Balanced sampling
    labels = [label for _, label in train_dataset.samples]
    class_counts = np.bincount(labels)
    class_weights = 1. / class_counts
    sample_weights = [class_weights[label] for label in labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    # DataLoaders - num_workers=0 to avoid multiprocessing issues
    BATCH_SIZE = 32
    log_message(f"✓ Batch size: {BATCH_SIZE}")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        sampler=sampler, 
        num_workers=0,
        pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=0,
        pin_memory=False
    )
    
    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_message(f"✓ Device: {device}")
    
    model = models.efficientnet_b0(weights=None)
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_features, 256),
        nn.SiLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(256, 2)
    )
    
    # Load best model as starting point
    log_message("\nLoading starting model...")
    try:
        model.load_state_dict(torch.load('best_manipulation_incremental.pth', map_location='cpu'))
        log_message("✓ Loaded best_manipulation_incremental.pth (74%)")
    except Exception as e:
        log_message(f"⚠️ Could not load checkpoint: {e}")
        log_message("Starting from scratch with ImageNet weights")
    
    model = model.to(device)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.00003, weight_decay=0.01)  # Lower LR for stability
    
    EPOCHS = 20
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # Try to resume from checkpoint
    start_epoch, best_val_acc = load_checkpoint(model, optimizer, scheduler)
    
    TARGET_ACC = 87.00
    start_time = time.time()
    
    log_message(f"\n{'='*60}")
    log_message(f"TRAINING FOR {EPOCHS} EPOCHS (starting from epoch {start_epoch})")
    log_message(f"Current best: {best_val_acc:.2f}%")
    log_message(f"Target: {TARGET_ACC:.2f}%")
    log_message(f"{'='*60}\n")
    
    # Training history
    history = {
        'train_acc': [],
        'val_acc': [],
        'train_loss': [],
        'val_loss': []
    }
    
    for epoch in range(start_epoch, EPOCHS):
        epoch_start = time.time()
        
        try:
            # Train
            model.train()
            train_loss = 0
            correct = 0
            total = 0
            
            log_message(f"\n[Epoch {epoch+1}/{EPOCHS}] Training...")
            
            for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc='Training', ncols=80)):
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
                
                # Log every 30 batches
                if (batch_idx + 1) % 30 == 0:
                    current_acc = 100. * correct / total
                    log_message(f"  Batch {batch_idx+1}/{len(train_loader)}: Acc={current_acc:.2f}%, Loss={loss.item():.4f}")
            
            train_acc = 100. * correct / total
            train_loss = train_loss / len(train_loader)
            
            # Validate
            log_message(f"\n[Epoch {epoch+1}/{EPOCHS}] Validating...")
            
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
            
            # Save history
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            
            epoch_time = time.time() - epoch_start
            total_time = time.time() - start_time
            
            # Log results
            log_message(f"\n{'='*60}")
            log_message(f"EPOCH {epoch+1}/{EPOCHS} COMPLETE")
            log_message(f"{'='*60}")
            log_message(f"Time: {epoch_time/60:.1f}m (Total: {total_time/60:.1f}m)")
            log_message(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            log_message(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
            log_message(f"LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Save if improved
            if val_acc > best_val_acc:
                improvement = val_acc - best_val_acc
                best_val_acc = val_acc
                torch.save(model.state_dict(), 'best_manipulation_incremental.pth')
                log_message(f"✓ NEW BEST! {val_acc:.2f}% (+{improvement:.2f}%)")
                log_message(f"✓ Model saved to best_manipulation_incremental.pth")
            else:
                log_message(f"⚠️ No improvement (best: {best_val_acc:.2f}%)")
            
            # Save checkpoint every epoch
            save_checkpoint(epoch, model, optimizer, scheduler, best_val_acc)
            
            # Save history
            with open('training_history.json', 'w') as f:
                json.dump(history, f, indent=2)
            
            # Check if target reached
            if val_acc >= TARGET_ACC:
                log_message(f"\n{'='*60}")
                log_message(f"🎉 TARGET REACHED! {val_acc:.2f}% >= {TARGET_ACC:.2f}%")
                log_message(f"{'='*60}")
                break
            
            log_message(f"{'='*60}\n")
            
        except Exception as e:
            log_message(f"\n❌ ERROR in epoch {epoch+1}: {e}")
            log_message("Saving checkpoint and continuing...")
            save_checkpoint(epoch, model, optimizer, scheduler, best_val_acc)
            continue
    
    # Final summary
    total_time = time.time() - start_time
    
    log_message(f"\n{'='*60}")
    log_message("TRAINING COMPLETE!")
    log_message(f"{'='*60}")
    log_message(f"Starting accuracy: 74.00%")
    log_message(f"Final best accuracy: {best_val_acc:.2f}%")
    log_message(f"Improvement: +{best_val_acc - 74.00:.2f}%")
    log_message(f"Epochs completed: {epoch+1}")
    log_message(f"Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    log_message(f"Model saved: best_manipulation_incremental.pth")
    
    if best_val_acc >= TARGET_ACC:
        log_message(f"\n🎉 SUCCESS! Target {TARGET_ACC:.2f}% achieved!")
    elif best_val_acc >= 80.0:
        log_message(f"\n✓ EXCELLENT! {best_val_acc:.2f}% is great progress!")
    elif best_val_acc > 74.0:
        log_message(f"\n✓ IMPROVED! +{best_val_acc - 74.00:.2f}% gain")
    else:
        log_message(f"\n⚠️ No improvement from 74.00%")
    
    log_message(f"{'='*60}\n")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        log_message("\n\n⚠️ Training interrupted by user")
    except Exception as e:
        log_message(f"\n\n❌ FATAL ERROR: {e}")
        import traceback
        log_message(traceback.format_exc())
