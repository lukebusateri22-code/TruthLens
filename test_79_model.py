"""
Test the 79.87% manipulation detection model on 100 images
Balanced: 50 authentic + 50 manipulated
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
import random
from tqdm import tqdm

print("="*70)
print(" TESTING 79.87% MANIPULATION DETECTION MODEL")
print(" Testing on 100 images (50 authentic + 50 manipulated)")
print("="*70)

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}")

# Load model - EfficientNet-B0 with custom classifier
print("\nLoading model...")
model = models.efficientnet_b0(weights=None)
num_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(num_features, 256),
    nn.SiLU(),
    nn.Dropout(0.2),
    nn.Linear(256, 2)
)

checkpoint = torch.load('best_manipulation_fast.pth', map_location=device)
model.load_state_dict(checkpoint)
model.to(device)
model.eval()
print("✓ Model loaded: best_manipulation_fast.pth (79.87%)")

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

# Collect images from CASIA2
print("\nCollecting test images from CASIA2...")
casia_path = Path('data/CASIA2/CASIA2')

authentic_images = []
manipulated_images = []

# Collect authentic images
au_path = casia_path / 'Au'
if au_path.exists():
    authentic_images = list(au_path.glob('*.jpg')) + list(au_path.glob('*.png'))

# Collect manipulated images  
tp_path = casia_path / 'Tp'
if tp_path.exists():
    manipulated_images = list(tp_path.glob('*.jpg')) + list(tp_path.glob('*.tif'))

print(f"Found {len(authentic_images)} authentic images")
print(f"Found {len(manipulated_images)} manipulated images")

# Sample 50 of each
random.seed(42)
test_authentic = random.sample(authentic_images, min(50, len(authentic_images)))
test_manipulated = random.sample(manipulated_images, min(50, len(manipulated_images)))

print(f"\nTesting on:")
print(f"  - {len(test_authentic)} authentic images")
print(f"  - {len(test_manipulated)} manipulated images")
print(f"  - {len(test_authentic) + len(test_manipulated)} total images")

# Test function
def test_image(img_path):
    try:
        image = Image.open(img_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(image_tensor)
            probs = torch.softmax(output, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred].item()
        
        return pred, confidence
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None, None

# Test authentic images
print("\n" + "="*70)
print("Testing AUTHENTIC images...")
print("="*70)

authentic_correct = 0
authentic_confidences = []

for img_path in tqdm(test_authentic, desc="Authentic"):
    pred, conf = test_image(img_path)
    if pred is not None:
        if pred == 0:  # 0 = authentic
            authentic_correct += 1
        authentic_confidences.append(conf)

authentic_acc = (authentic_correct / len(test_authentic)) * 100
avg_auth_conf = sum(authentic_confidences) / len(authentic_confidences) * 100

print(f"\nAuthentic Results:")
print(f"  Correct: {authentic_correct}/{len(test_authentic)}")
print(f"  Accuracy: {authentic_acc:.2f}%")
print(f"  Avg Confidence: {avg_auth_conf:.2f}%")

# Test manipulated images
print("\n" + "="*70)
print("Testing MANIPULATED images...")
print("="*70)

manipulated_correct = 0
manipulated_confidences = []

for img_path in tqdm(test_manipulated, desc="Manipulated"):
    pred, conf = test_image(img_path)
    if pred is not None:
        if pred == 1:  # 1 = manipulated
            manipulated_correct += 1
        manipulated_confidences.append(conf)

manipulated_acc = (manipulated_correct / len(test_manipulated)) * 100
avg_manip_conf = sum(manipulated_confidences) / len(manipulated_confidences) * 100

print(f"\nManipulated Results:")
print(f"  Correct: {manipulated_correct}/{len(test_manipulated)}")
print(f"  Accuracy: {manipulated_acc:.2f}%")
print(f"  Avg Confidence: {avg_manip_conf:.2f}%")

# Overall results
print("\n" + "="*70)
print(" FINAL RESULTS")
print("="*70)

total_correct = authentic_correct + manipulated_correct
total_images = len(test_authentic) + len(test_manipulated)
overall_acc = (total_correct / total_images) * 100

print(f"\nOverall Accuracy: {overall_acc:.2f}%")
print(f"  - Authentic: {authentic_acc:.2f}%")
print(f"  - Manipulated: {manipulated_acc:.2f}%")
print(f"\nAverage Confidence:")
print(f"  - Authentic: {avg_auth_conf:.2f}%")
print(f"  - Manipulated: {avg_manip_conf:.2f}%")

# Balance check
balance_gap = abs(authentic_acc - manipulated_acc)
print(f"\nBalance Gap: {balance_gap:.2f}%")

if balance_gap < 5:
    print("✓ Excellent balance!")
elif balance_gap < 10:
    print("✓ Good balance")
else:
    print("⚠ Imbalanced - may need adjustment")

print("\n" + "="*70)
print(" TEST COMPLETE")
print("="*70)
