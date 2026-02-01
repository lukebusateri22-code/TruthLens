"""
Test current 77% model on 100 images from incremental dataset
See actual performance before training
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
import random

# Model architecture (same as frontend)
class ManipulationDetector(nn.Module):
    def __init__(self):
        super(ManipulationDetector, self).__init__()
        self.backbone = models.efficientnet_b0(weights=None)
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 256),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2)
        )
    
    def forward(self, x):
        return self.backbone(x)

print("="*60)
print("TESTING CURRENT 77% MODEL ON 100 IMAGES")
print("="*60)

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}")

model = ManipulationDetector()

# Try to load the 77% model
model_loaded = False
for model_path in ['best_manipulation_model_final.pth', 'best_manipulation_fast.pth']:
    if Path(model_path).exists():
        try:
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            print(f"✓ Loaded {model_path}")
            model_loaded = True
            break
        except Exception as e:
            print(f"⚠️ Could not load {model_path}: {e}")

if not model_loaded:
    print("❌ No model found! Cannot test.")
    exit(1)

model = model.to(device)
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Collect 100 random images (50 authentic, 50 manipulated)
print("\nCollecting 100 test images...")
data_dir = Path('manipulation_data_incremental/val')

authentic_images = list((data_dir / 'authentic').glob('*.*'))
manipulated_images = list((data_dir / 'manipulated').glob('*.*'))

print(f"Available: {len(authentic_images)} authentic, {len(manipulated_images)} manipulated")

# Sample 50 of each
test_images = []
test_images.extend([(img, 0) for img in random.sample(authentic_images, min(50, len(authentic_images)))])
test_images.extend([(img, 1) for img in random.sample(manipulated_images, min(50, len(manipulated_images)))])

print(f"Testing on {len(test_images)} images...")

# Test
correct = 0
total = 0
authentic_correct = 0
authentic_total = 0
manipulated_correct = 0
manipulated_total = 0

errors = []

print("\nTesting...")
for img_path, true_label in test_images:
    try:
        image = Image.open(img_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(image_tensor)
            _, predicted = output.max(1)
            pred_label = predicted.item()
        
        total += 1
        if pred_label == true_label:
            correct += 1
        else:
            errors.append((img_path.name, true_label, pred_label))
        
        if true_label == 0:
            authentic_total += 1
            if pred_label == 0:
                authentic_correct += 1
        else:
            manipulated_total += 1
            if pred_label == 1:
                manipulated_correct += 1
                
    except Exception as e:
        print(f"Error processing {img_path.name}: {e}")

# Results
print("\n" + "="*60)
print("RESULTS")
print("="*60)
print(f"Total Tested: {total} images")
print(f"Overall Accuracy: {100.*correct/total:.2f}% ({correct}/{total})")
print(f"\nAuthentic: {100.*authentic_correct/authentic_total:.2f}% ({authentic_correct}/{authentic_total})")
print(f"Manipulated: {100.*manipulated_correct/manipulated_total:.2f}% ({manipulated_correct}/{manipulated_total})")

print(f"\n❌ Errors: {len(errors)}")
if errors and len(errors) <= 10:
    print("\nFirst errors:")
    for name, true_label, pred_label in errors[:10]:
        true_str = "authentic" if true_label == 0 else "manipulated"
        pred_str = "authentic" if pred_label == 0 else "manipulated"
        print(f"  {name}: TRUE={true_str}, PREDICTED={pred_str}")

print("\n" + "="*60)
print("ANALYSIS")
print("="*60)

overall_acc = 100.*correct/total

if overall_acc >= 75:
    print(f"✓ GOOD! {overall_acc:.2f}% - Model works well on this dataset")
    print("  Can proceed with training to improve further")
elif overall_acc >= 70:
    print(f"⚠️ OKAY - {overall_acc:.2f}% - Model works but has room for improvement")
    print("  Training should help")
elif overall_acc >= 60:
    print(f"⚠️ POOR - {overall_acc:.2f}% - Model struggles with this dataset")
    print("  May need different approach or more data")
else:
    print(f"❌ VERY POOR - {overall_acc:.2f}% - Dataset may be incompatible")
    print("  Need to investigate dataset quality")

print("="*60)
