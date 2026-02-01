"""
TRAIN ON FULL CASIA2 DATASET - 14,678 images
This should get us to 85-90%
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import numpy as np
import time
from sklearn.model_selection import train_test_split

class CASIA2Dataset(Dataset):
    """Load CASIA2 dataset."""
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            image = Image.new('RGB', (224, 224), color='gray')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

print("="*70)
print(" TRAINING ON FULL CASIA2 DATASET")
print(" 14,678 images - Target: 85-90%")
print("="*70)

# Collect all CASIA2 images
print("\nScanning CASIA2 dataset...")
casia_path = Path('data/CASIA2')

authentic_images = []
manipulated_images = []

# CASIA2 structure: Au (authentic) and Tp (tampered)
for subdir in casia_path.rglob('*'):
    if subdir.is_dir():
        dir_name = subdir.name.lower()
        
        if 'au' in dir_name or 'authentic' in dir_name:
            for img in subdir.glob('*'):
                if img.suffix.lower() in ['.jpg', '.png', '.bmp', '.tif']:
                    authentic_images.append(str(img))
        
        elif 'tp' in dir_name or 'tamper' in dir_name or 'fake' in dir_name:
            for img in subdir.glob('*'):
                if img.suffix.lower() in ['.jpg', '.png', '.bmp', '.tif']:
                    manipulated_images.append(str(img))

print(f"Found {len(authentic_images)} authentic images")
print(f"Found {len(manipulated_images)} manipulated images")
print(f"Total: {len(authentic_images) + len(manipulated_images)} images")

# Create balanced dataset
all_images = authentic_images + manipulated_images
all_labels = [0] * len(authentic_images) + [1] * len(manipulated_images)

# Split 80/20
train_imgs, val_imgs, train_labels, val_labels = train_test_split(
    all_images, all_labels, test_size=0.2, random_state=42, stratify=all_labels
)

print(f"\nTrain: {len(train_imgs)} images")
print(f"Val: {len(val_imgs)} images")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Transforms
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Datasets
train_dataset = CASIA2Dataset(train_imgs, train_labels, transform=train_transform)
val_dataset = CASIA2Dataset(val_imgs, val_labels, transform=val_transform)

# Balanced sampling
class_counts = np.bincount(train_labels)
class_weights = 1.0 / class_counts
sample_weights = [class_weights[label] for label in train_labels]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

# DataLoaders
BATCH_SIZE = 64
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"Batch size: {BATCH_SIZE}")
print(f"Batches per epoch: {len(train_loader)}")

# Model
print("\nInitializing model...")
model = models.efficientnet_b0(weights='IMAGENET1K_V1')
num_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(num_features, 256),
    nn.SiLU(),
    nn.Dropout(0.2),
    nn.Linear(256, 2)
)

# Try to load existing weights
best_val_acc = 0.0
if Path('best_manipulation_fast.pth').exists():
    try:
        model.load_state_dict(torch.load('best_manipulation_fast.pth', map_location='cpu'))
        print("✓ Loaded best_manipulation_fast.pth (78% baseline)")
        best_val_acc = 78.0
    except:
        print("✓ Starting from ImageNet weights")
else:
    print("✓ Starting from ImageNet weights")

model = model.to(device)

# Training setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25, eta_min=0.00001)

EPOCHS = 25
start_time = time.time()

print(f"\n{'='*70}")
print(f"TRAINING FOR {EPOCHS} EPOCHS ON FULL CASIA2")
print(f"{'='*70}\n")

for epoch in range(EPOCHS):
    epoch_start = time.time()
    
    # TRAIN
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS} [TRAIN]', ncols=100)
    for images, labels in pbar:
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
        
        pbar.set_postfix({'loss': f'{loss.item():.3f}', 'acc': f'{100.*correct/total:.1f}%'})
    
    train_acc = 100. * correct / total
    scheduler.step()
    
    # VALIDATE
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{EPOCHS} [VAL]  ', ncols=100)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'loss': f'{loss.item():.3f}', 'acc': f'{100.*correct/total:.1f}%'})
    
    val_acc = 100. * correct / total
    
    epoch_time = time.time() - epoch_start
    total_time = time.time() - start_time
    
    # Print results
    print(f"\n{'='*70}")
    print(f"EPOCH {epoch+1}/{EPOCHS} - {epoch_time/60:.1f}min ({total_time/60:.1f}min total)")
    print(f"Train: {train_acc:.2f}% | Val: {val_acc:.2f}%")
    
    # Save best model
    if val_acc > best_val_acc:
        improvement = val_acc - best_val_acc
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_manipulation_fast.pth')
        print(f"✓ NEW BEST! {val_acc:.2f}% (+{improvement:.2f}%) - SAVED")
    else:
        print(f"  Best: {best_val_acc:.2f}%")
    
    print(f"{'='*70}\n")
    
    # Check target
    if val_acc >= 90.0:
        print(f"🎉 TARGET REACHED! {val_acc:.2f}% >= 90%\n")
        break

# Final summary
total_time = time.time() - start_time
print(f"{'='*70}")
print(" TRAINING COMPLETE!")
print(f"{'='*70}")
print(f"Best Accuracy: {best_val_acc:.2f}%")
print(f"Training Time: {total_time/3600:.2f} hours")
print(f"Dataset: {len(train_imgs) + len(val_imgs)} images (CASIA2)")

if best_val_acc >= 90:
    print(f"\n🎉 SUCCESS! 90%+ ACHIEVED!")
elif best_val_acc >= 85:
    print(f"\n🎉 EXCELLENT! 85%+ achieved!")
elif best_val_acc >= 80:
    print(f"\n✓ GOOD! 80%+ achieved!")

print(f"{'='*70}\n")
