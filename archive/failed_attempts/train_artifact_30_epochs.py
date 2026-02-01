"""
Train manipulation detector with Artifact Dataset
30 epochs or until 90% accuracy is reached
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
import matplotlib.pyplot as plt

print("="*60)
print("ARTIFACT DATASET TRAINING")
print("Target: 30 epochs OR 90% accuracy")
print("="*60)

# Dataset path
ARTIFACT_PATH = Path("/Users/cn424694/.cache/kagglehub/datasets/awsaf49/artifact-dataset/versions/1")

class ArtifactDataset(Dataset):
    """
    Load images from the artifact dataset.
    Real images: original datasets (ffhq, celebahq, etc.)
    Fake images: generated/manipulated (big_gan, cycle_gan, etc.)
    """
    
    def __init__(self, artifact_path, num_samples=50000, split='train', transform=None):
        self.transform = transform
        self.samples = []
        
        # Define which subdirectories are real vs fake
        real_sources = ['ffhq', 'celebahq', 'coco', 'imagenet', 'lsun', 'landscape']
        fake_sources = ['big_gan', 'cycle_gan', 'ddpm', 'diffusion_gan', 'gansformer',
                       'gau_gan', 'generative_inpainting', 'glide', 'latent_diffusion',
                       'progan', 'projected_gan', 'stargan', 'style_gan', 'style_gan2',
                       'style_gan3', 'taming_transformers', 'vq_diffusion', 'vq_gan']
        
        print(f"\nCollecting images for {split} split...")
        
        # Collect all images
        all_real = []
        all_fake = []
        
        for source in real_sources:
            source_path = artifact_path / source
            if source_path.exists():
                images = list(source_path.rglob('*.png')) + list(source_path.rglob('*.jpg'))
                all_real.extend(images)
                print(f"  Real - {source}: {len(images):,} images")
        
        for source in fake_sources:
            source_path = artifact_path / source
            if source_path.exists():
                images = list(source_path.rglob('*.png')) + list(source_path.rglob('*.jpg'))
                all_fake.extend(images)
                print(f"  Fake - {source}: {len(images):,} images")
        
        print(f"\nTotal available:")
        print(f"  Real: {len(all_real):,}")
        print(f"  Fake: {len(all_fake):,}")
        
        # Sample balanced dataset
        samples_per_class = num_samples // 2
        
        if len(all_real) > samples_per_class:
            sampled_real = random.sample(all_real, samples_per_class)
        else:
            sampled_real = all_real
        
        if len(all_fake) > samples_per_class:
            sampled_fake = random.sample(all_fake, samples_per_class)
        else:
            sampled_fake = all_fake
        
        # Split into train/val (85/15)
        if split == 'train':
            real_split = sampled_real[:int(len(sampled_real) * 0.85)]
            fake_split = sampled_fake[:int(len(sampled_fake) * 0.85)]
        else:  # val
            real_split = sampled_real[int(len(sampled_real) * 0.85):]
            fake_split = sampled_fake[int(len(sampled_fake) * 0.85):]
        
        # Add to samples
        for img_path in real_split:
            self.samples.append((str(img_path), 0))  # 0 = real
        
        for img_path in fake_split:
            self.samples.append((str(img_path), 1))  # 1 = fake
        
        print(f"\n{split.upper()} dataset:")
        real_count = sum(1 for _, label in self.samples if label == 0)
        fake_count = sum(1 for _, label in self.samples if label == 1)
        print(f"  Real: {real_count:,}")
        print(f"  Fake: {fake_count:,}")
        print(f"  Total: {len(self.samples):,}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            # Fallback to black image if loading fails
            image = Image.new('RGB', (224, 224), color='black')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for images, labels in tqdm(loader, desc='Training', leave=False):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(loader), 100. * correct / total

def validate(model, loader, criterion, device):
    """Validate the model."""
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
    # Configuration
    NUM_SAMPLES = 50000
    BATCH_SIZE = 64
    EPOCHS = 30
    LEARNING_RATE = 0.0001
    TARGET_ACCURACY = 90.0
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
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
    print("\n" + "="*60)
    print("LOADING DATASETS")
    print("="*60)
    
    train_dataset = ArtifactDataset(ARTIFACT_PATH, num_samples=NUM_SAMPLES, 
                                    split='train', transform=train_transform)
    val_dataset = ArtifactDataset(ARTIFACT_PATH, num_samples=NUM_SAMPLES,
                                  split='val', transform=val_transform)
    
    # Balanced sampling
    labels = [label for _, label in train_dataset.samples]
    class_counts = np.bincount(labels)
    class_weights = 1. / class_counts
    sample_weights = [class_weights[label] for label in labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, 
                             num_workers=6, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=4, pin_memory=True)
    
    # Model
    print("\n" + "="*60)
    print("INITIALIZING MODEL")
    print("="*60)
    
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
    
    print(f"Model: EfficientNet-B0")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # Training
    print("\n" + "="*60)
    print(f"TRAINING FOR {EPOCHS} EPOCHS (OR UNTIL {TARGET_ACCURACY}%)")
    print("="*60)
    
    best_val_acc = 0
    train_accs = []
    val_accs = []
    train_losses = []
    val_losses = []
    
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        epoch_start = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Step scheduler
        scheduler.step()
        
        # Record metrics
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        epoch_time = time.time() - epoch_start
        total_time = time.time() - start_time
        
        print(f"\nEpoch {epoch+1}/{EPOCHS} ({epoch_time:.1f}s, {total_time/60:.1f}m total)")
        print(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
        print(f"  Val:   Loss={val_loss:.4f}, Acc={val_acc:.2f}%")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_manipulation_artifact_final.pth')
            print(f"  ✓ New best! ({val_acc:.2f}%)")
        
        # Check if target reached
        if val_acc >= TARGET_ACCURACY:
            print(f"\n🎉 TARGET REACHED! {val_acc:.2f}% >= {TARGET_ACCURACY}%")
            break
    
    total_time = time.time() - start_time
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Val Accuracy')
    plt.axhline(y=TARGET_ACCURACY, color='r', linestyle='--', label=f'Target ({TARGET_ACCURACY}%)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('artifact_training_curves.png', dpi=150)
    print(f"\n✓ Training curves saved to: artifact_training_curves.png")
    
    # Final summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Total epochs: {epoch+1}")
    print(f"Total time: {total_time/3600:.2f} hours")
    print(f"Model saved as: best_manipulation_artifact_final.pth")
    
    if best_val_acc >= TARGET_ACCURACY:
        print(f"\n🎉 SUCCESS! Target {TARGET_ACCURACY}% achieved!")
    else:
        print(f"\n⚠️ Target not reached, but {best_val_acc:.2f}% is excellent!")

if __name__ == "__main__":
    main()
