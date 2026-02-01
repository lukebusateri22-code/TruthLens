"""
Feature-based manipulation detection
Instead of binary classification, detect specific manipulation features
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
from PIL import Image, ImageFilter
import numpy as np
from tqdm import tqdm
import cv2

class FeatureManipulationCNN(nn.Module):
    """
    CNN that detects specific manipulation features:
    - Edge inconsistencies
    - Noise patterns
    - Compression artifacts
    - Color inconsistencies
    """
    
    def __init__(self):
        super().__init__()
        
        # Feature extraction backbone
        self.feature_extractor = nn.Sequential(
            # Input: 3 channels (RGB)
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        
        # Multi-task heads for different features
        self.edge_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        self.noise_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        self.compression_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        self.color_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Final manipulation score
        self.manipulation_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        features = self.feature_extractor(x)
        
        edge_score = self.edge_head(features)
        noise_score = self.noise_head(features)
        compression_score = self.compression_head(features)
        color_score = self.color_head(features)
        manipulation_score = self.manipulation_head(features)
        
        return {
            'edge': edge_score,
            'noise': noise_score,
            'compression': compression_score,
            'color': color_score,
            'manipulation': manipulation_score
        }

class FeatureDataset(torch.utils.data.Dataset):
    """Dataset that generates feature labels."""
    
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples = []
        
        # Load all images and compute feature labels
        for class_dir, label in [('authentic', 0), ('manipulated', 1)]:
            class_path = self.data_dir / class_dir
            if not class_path.exists():
                continue
            
            for img_path in class_path.glob('*.*'):
                if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                    continue
                
                try:
                    # Compute feature labels
                    feature_labels = self.compute_features(img_path, label)
                    self.samples.append((str(img_path), feature_labels))
                except:
                    continue
        
        print(f"Loaded {len(self.samples)} samples with features")
    
    def compute_features(self, img_path, base_label):
        """Compute feature-specific labels."""
        image = cv2.imread(str(img_path))
        if image is None:
            return {'edge': 0, 'noise': 0, 'compression': 0, 'color': 0, 'manipulation': base_label}
        
        # Convert to different color spaces for analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Edge inconsistency detection
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        edge_inconsistent = 1 if edge_density > 0.05 else 0
        
        # Noise inconsistency detection
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        noise_inconsistent = 1 if laplacian_var < 100 or laplacian_var > 500 else 0
        
        # Compression artifact detection
        # Check for JPEG blocking artifacts
        block_size = 8
        h, w = gray.shape
        blocks_h = h // block_size
        blocks_w = w // block_size
        
        block_variances = []
        for i in range(blocks_h):
            for j in range(blocks_w):
                block = gray[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
                block_variances.append(np.var(block))
        
        compression_artifacts = 1 if np.std(block_variances) > 50 else 0
        
        # Color inconsistency detection
        # Check HSV channel variations
        h_std = np.std(hsv[:,:,0])
        s_std = np.std(hsv[:,:,1])
        v_std = np.std(hsv[:,:,2])
        color_inconsistent = 1 if (h_std + s_std + v_std) / 3 > 30 else 0
        
        # If image is labeled authentic, set all features to 0
        if base_label == 0:
            return {'edge': 0, 'noise': 0, 'compression': 0, 'color': 0, 'manipulation': 0}
        
        return {
            'edge': edge_inconsistent,
            'noise': noise_inconsistent,
            'compression': compression_artifacts,
            'color': color_inconsistent,
            'manipulation': base_label
        }
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, labels = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, labels

def get_feature_transforms(image_size=224):
    """Transforms for feature-based training."""
    
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
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

def train_feature_epoch(model, loader, optimizer, device):
    """Train with multi-task loss."""
    model.train()
    total_loss = 0
    
    # Track individual feature accuracies
    feature_correct = {'edge': 0, 'noise': 0, 'compression': 0, 'color': 0, 'manipulation': 0}
    feature_total = {'edge': 0, 'noise': 0, 'compression': 0, 'color': 0, 'manipulation': 0}
    
    for images, labels in tqdm(loader, desc='Training'):
        images = images.to(device)
        
        # Convert labels to tensors
        label_tensors = {}
        batch_size = len(images)
        for feature, value in labels.items():
            if isinstance(value, list):
                label_tensors[feature] = torch.tensor(value, dtype=torch.float32).to(device)
            else:
                # Create tensor of right size for batch
                label_tensors[feature] = torch.full((batch_size,), float(value), dtype=torch.float32).to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        
        # Multi-task loss
        loss = 0
        for feature in outputs.keys():
            feature_loss = nn.BCELoss()(outputs[feature].squeeze(), label_tensors[feature])
            loss += feature_loss
            
            # Track accuracy
            predicted = (outputs[feature].squeeze() > 0.5).float()
            correct = (predicted == label_tensors[feature]).sum().item()
            feature_correct[feature] += correct
            feature_total[feature] += len(label_tensors[feature])
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    # Calculate feature accuracies
    feature_accs = {}
    for feature in feature_correct.keys():
        if feature_total[feature] > 0:
            feature_accs[feature] = 100. * feature_correct[feature] / feature_total[feature]
        else:
            feature_accs[feature] = 0
    
    return total_loss / len(loader), feature_accs

def validate_features(model, loader, device):
    """Validate multi-task model."""
    model.eval()
    total_loss = 0
    
    feature_correct = {'edge': 0, 'noise': 0, 'compression': 0, 'color': 0, 'manipulation': 0}
    feature_total = {'edge': 0, 'noise': 0, 'compression': 0, 'color': 0, 'manipulation': 0}
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Validating'):
            images = images.to(device)
            
            label_tensors = {}
            batch_size = len(images)
            for feature, value in labels.items():
                if isinstance(value, list):
                    label_tensors[feature] = torch.tensor(value, dtype=torch.float32).to(device)
                else:
                    label_tensors[feature] = torch.full((batch_size,), float(value), dtype=torch.float32).to(device)
            
            outputs = model(images)
            
            loss = 0
            for feature in outputs.keys():
                feature_loss = nn.BCELoss()(outputs[feature].squeeze(), label_tensors[feature])
                loss += feature_loss
                
                predicted = (outputs[feature].squeeze() > 0.5).float()
                correct = (predicted == label_tensors[feature]).sum().item()
                feature_correct[feature] += correct
                feature_total[feature] += len(label_tensors[feature])
            
            total_loss += loss.item()
    
    feature_accs = {}
    for feature in feature_correct.keys():
        if feature_total[feature] > 0:
            feature_accs[feature] = 100. * feature_correct[feature] / feature_total[feature]
        else:
            feature_accs[feature] = 0
    
    return total_loss / len(loader), feature_accs

def main():
    print("="*60)
    print("FEATURE-BASED MANIPULATION DETECTION")
    print("Training model to detect specific manipulation features")
    print("="*60)
    
    # Configuration
    DATA_DIR = 'manipulation_data_combined'
    BATCH_SIZE = 32
    EPOCHS = 25
    LEARNING_RATE = 0.001
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data
    train_transform, val_transform = get_feature_transforms()
    
    print("\nLoading data with feature labels...")
    train_dataset = FeatureDataset(
        Path(DATA_DIR) / 'train',
        transform=train_transform
    )
    
    val_dataset = FeatureDataset(
        Path(DATA_DIR) / 'val',
        transform=val_transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Model
    print("\nInitializing feature-based model...")
    model = FeatureManipulationCNN().to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training
    best_manip_acc = 0
    
    print(f"\nTraining for {EPOCHS} epochs...")
    print("="*60)
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print("-" * 60)
        
        # Train
        train_loss, train_feature_accs = train_feature_epoch(model, train_loader, optimizer, device)
        
        # Validate
        val_loss, val_feature_accs = validate_features(model, val_loader, device)
        
        print(f"\nEpoch {epoch+1} Results:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"\nFeature Accuracies:")
        for feature in train_feature_accs.keys():
            print(f"  {feature}: Train {train_feature_accs[feature]:.1f}% | Val {val_feature_accs[feature]:.1f}%")
        
        # Track best manipulation accuracy
        manip_acc = val_feature_accs['manipulation']
        if manip_acc > best_manip_acc:
            best_manip_acc = manip_acc
            torch.save(model.state_dict(), 'best_manipulation_feature_model.pth')
            print(f"  ✓ New best model! Manipulation Acc: {manip_acc:.1f}%")
        
        if manip_acc >= 85.0:
            print(f"\n🎉 Target reached! Manipulation accuracy: {manip_acc:.1f}%")
            break
    
    print("\n" + "="*60)
    print("Feature-based training complete!")
    print(f"Best manipulation accuracy: {best_manip_acc:.1f}%")
    print(f"Model saved as: best_manipulation_feature_model.pth")

if __name__ == "__main__":
    main()
