"""
Quick test - 3 epochs to see if this dataset can work
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

class SimpleDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform
        root_path = Path(root_dir)
        
        for class_name, label in [('authentic', 0), ('manipulated', 1)]:
            class_dir = root_path / class_name
            if class_dir.exists():
                for img_path in class_dir.glob('*.*'):
                    if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                        self.samples.append((str(img_path), label))
        
        print(f"  Loaded {len(self.samples)} images")
    
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

print("="*60)
print("QUICK DATASET TEST - 3 EPOCHS")
print("="*60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}")

# Simple transforms
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load data
print("\nLoading datasets...")
train_dataset = SimpleDataset('manipulation_data_incremental/train', transform=train_transform)
val_dataset = SimpleDataset('manipulation_data_incremental/val', transform=val_transform)

# Balanced sampling
labels = [label for _, label in train_dataset.samples]
class_counts = np.bincount(labels)
print(f"Train class distribution: Authentic={class_counts[0]}, Manipulated={class_counts[1]}")
class_weights = 1.0 / class_counts
sample_weights = [class_weights[label] for label in labels]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

# Simple model
print("\nCreating model...")
model = models.efficientnet_b0(weights='IMAGENET1K_V1')
num_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(num_features, 256),
    nn.SiLU(),
    nn.Dropout(0.2),
    nn.Linear(256, 2)
)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0001)

print("\nTraining for 3 epochs to test dataset...")
print("="*60)

for epoch in range(3):
    # Train
    model.train()
    correct = 0
    total = 0
    
    for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/3 [TRAIN]'):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    train_acc = 100. * correct / total
    
    # Validate
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f'Epoch {epoch+1}/3 [VAL]  '):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    val_acc = 100. * correct / total
    
    print(f"\nEpoch {epoch+1}/3: Train={train_acc:.2f}% | Val={val_acc:.2f}%")

print("\n" + "="*60)
print("TEST COMPLETE!")
print("="*60)
print(f"Final Validation Accuracy: {val_acc:.2f}%")

if val_acc >= 75:
    print("\n✓ GOOD! Dataset looks promising - can proceed with full training")
elif val_acc >= 70:
    print("\n⚠️ OKAY - Dataset might work with more epochs")
else:
    print("\n❌ POOR - Dataset may have issues, need to investigate")

print("="*60)
