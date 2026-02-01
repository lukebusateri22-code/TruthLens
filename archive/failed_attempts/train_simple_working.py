"""
SIMPLE WORKING TRAINING - Uses the original 77.18% model architecture
No complex imports, just direct training
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

# Simple CNN model that matches the frontend
class SimpleManipulationCNN(nn.Module):
    def __init__(self):
        super(SimpleManipulationCNN, self).__init__()
        self.backbone = models.efficientnet_b0(weights=None)
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 256),
            nn.SiLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 2)
        )
    
    def forward(self, x):
        return self.backbone(x)

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

def log_message(message, log_file='training_simple.log'):
    """Log message to both console and file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"[{timestamp}] {message}"
    print(log_msg)
    sys.stdout.flush()
    
    with open(log_file, 'a') as f:
        f.write(log_msg + '\n')

def main():
    log_message("="*60)
    log_message("SIMPLE WORKING TRAINING - Frontend Compatible")
    log_message("Using EfficientNet-B0 (matches webapp)")
    log_message("Target: 87%+ validation accuracy")
    log_message("="*60)
    
    NEW_DATA_DIR = Path("manipulation_data_incremental")
    
    # Check if dataset exists
    if not NEW_DATA_DIR.exists():
        log_message("❌ ERROR: Dataset not found!")
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
    
    # DataLoaders
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
    
    model = SimpleManipulationCNN()
    
    # Load best available weights
    log_message("\nLoading model weights...")
    try:
        if Path('best_manipulation_model_final.pth').exists():
            model.load_state_dict(torch.load('best_manipulation_model_final.pth', map_location='cpu'))
            log_message("✓ Loaded best_manipulation_model_final.pth (77.18%)")
        else:
            log_message("✓ Starting from scratch")
    except Exception as e:
        log_message(f"⚠️ Could not load weights: {e}")
        log_message("✓ Starting from scratch")
    
    model = model.to(device)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.00003, weight_decay=0.01)
    
    EPOCHS = 20
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    TARGET_ACC = 87.00
    best_val_acc = 77.18  # Starting from 77.18%
    start_time = time.time()
    
    log_message(f"\n{'='*60}")
    log_message(f"TRAINING FOR {EPOCHS} EPOCHS")
    log_message(f"Starting accuracy: {best_val_acc:.2f}%")
    log_message(f"Target: {TARGET_ACC:.2f}%")
    log_message(f"{'='*60}\n")
    
    # Training history
    history = {
        'train_acc': [],
        'val_acc': [],
        'train_loss': [],
        'val_loss': []
    }
    
    for epoch in range(EPOCHS):
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
                torch.save(model.state_dict(), 'best_manipulation_fast.pth')  # Frontend expects this
                log_message(f"✓ NEW BEST! {val_acc:.2f}% (+{improvement:.2f}%)")
                log_message(f"✓ Model saved to best_manipulation_fast.pth")
            else:
                log_message(f"⚠️ No improvement (best: {best_val_acc:.2f}%)")
            
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
            import traceback
            log_message(traceback.format_exc())
            continue
    
    # Final summary
    total_time = time.time() - start_time
    
    log_message(f"\n{'='*60}")
    log_message("TRAINING COMPLETE!")
    log_message(f"{'='*60}")
    log_message(f"Starting accuracy: 77.18%")
    log_message(f"Final best accuracy: {best_val_acc:.2f}%")
    log_message(f"Improvement: +{best_val_acc - 77.18:.2f}%")
    log_message(f"Epochs completed: {epoch+1}")
    log_message(f"Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    log_message(f"Model saved: best_manipulation_fast.pth (frontend compatible)")
    
    if best_val_acc >= TARGET_ACC:
        log_message(f"\n🎉 SUCCESS! Target {TARGET_ACC:.2f}% achieved!")
    elif best_val_acc >= 80.0:
        log_message(f"\n✓ EXCELLENT! {best_val_acc:.2f}% is great progress!")
    elif best_val_acc > 77.18:
        log_message(f"\n✓ IMPROVED! +{best_val_acc - 77.18:.2f}% gain")
    else:
        log_message(f"\n⚠️ No improvement from 77.18%")
    
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
