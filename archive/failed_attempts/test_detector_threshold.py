"""
Test the manipulation detector with lowered threshold (0.30)
Test on 50 real + 50 manipulated images from CASIA dataset
"""

import torch
from pathlib import Path
from PIL import Image
from models.manipulation_detector_final import FinalManipulationDetector
from tqdm import tqdm
import random

print("="*60)
print("TESTING MANIPULATION DETECTOR")
print("Threshold: 0.30 (lowered from 0.50)")
print("Testing on 50 real + 50 manipulated images")
print("="*60)

# Load detector
print("\nLoading detector...")
detector = FinalManipulationDetector()

if not detector.model_loaded:
    print("❌ Model not loaded!")
    exit(1)

print(f"✓ Model loaded: EfficientNet-B0")
print(f"✓ Accuracy: 77.18%")

# Get test images from CASIA dataset
CASIA_PATH = Path("/Users/cn424694/.cache/kagglehub/datasets/divg07/casia-20-image-tampering-detection-dataset/versions/1/CASIA2")

print("\n" + "="*60)
print("COLLECTING TEST IMAGES")
print("="*60)

# Get authentic images
authentic_dir = CASIA_PATH / 'Au'
authentic_images = list(authentic_dir.glob('*.jpg')) + list(authentic_dir.glob('*.png')) + list(authentic_dir.glob('*.tif'))
print(f"Found {len(authentic_images):,} authentic images")

# Get tampered images
tampered_dir = CASIA_PATH / 'Tp'
tampered_images = list(tampered_dir.glob('*.jpg')) + list(tampered_dir.glob('*.png')) + list(tampered_dir.glob('*.tif'))
print(f"Found {len(tampered_images):,} tampered images")

# Sample 50 of each
if len(authentic_images) >= 50:
    test_authentic = random.sample(authentic_images, 50)
else:
    test_authentic = authentic_images
    print(f"⚠️ Only {len(authentic_images)} authentic images available")

if len(tampered_images) >= 50:
    test_tampered = random.sample(tampered_images, 50)
else:
    test_tampered = tampered_images
    print(f"⚠️ Only {len(tampered_images)} tampered images available")

print(f"\nTesting on:")
print(f"  Authentic: {len(test_authentic)} images")
print(f"  Tampered: {len(test_tampered)} images")

# Test on authentic images
print("\n" + "="*60)
print("TESTING ON AUTHENTIC IMAGES")
print("="*60)

authentic_correct = 0
authentic_total = 0
authentic_confidences = []

for img_path in tqdm(test_authentic, desc="Testing authentic"):
    try:
        image = Image.open(img_path).convert('RGB')
        result = detector.predict(image)
        
        # Correct if predicted as real (not fake)
        if not result['is_fake']:
            authentic_correct += 1
        
        authentic_confidences.append(result['confidence'])
        authentic_total += 1
        
    except Exception as e:
        print(f"Error with {img_path.name}: {e}")
        continue

authentic_acc = 100. * authentic_correct / authentic_total if authentic_total > 0 else 0
avg_authentic_conf = sum(authentic_confidences) / len(authentic_confidences) if authentic_confidences else 0

print(f"\nAuthentic Results:")
print(f"  Correct: {authentic_correct}/{authentic_total}")
print(f"  Accuracy: {authentic_acc:.2f}%")
print(f"  Avg Confidence: {avg_authentic_conf:.2%}")

# Test on tampered images
print("\n" + "="*60)
print("TESTING ON TAMPERED IMAGES")
print("="*60)

tampered_correct = 0
tampered_total = 0
tampered_confidences = []

for img_path in tqdm(test_tampered, desc="Testing tampered"):
    try:
        image = Image.open(img_path).convert('RGB')
        result = detector.predict(image)
        
        # Correct if predicted as fake
        if result['is_fake']:
            tampered_correct += 1
        
        tampered_confidences.append(result['confidence'])
        tampered_total += 1
        
    except Exception as e:
        print(f"Error with {img_path.name}: {e}")
        continue

tampered_acc = 100. * tampered_correct / tampered_total if tampered_total > 0 else 0
avg_tampered_conf = sum(tampered_confidences) / len(tampered_confidences) if tampered_confidences else 0

print(f"\nTampered Results:")
print(f"  Correct: {tampered_correct}/{tampered_total}")
print(f"  Accuracy: {tampered_acc:.2f}%")
print(f"  Avg Confidence: {avg_tampered_conf:.2%}")

# Overall results
print("\n" + "="*60)
print("OVERALL RESULTS")
print("="*60)

total_correct = authentic_correct + tampered_correct
total_images = authentic_total + tampered_total
overall_acc = 100. * total_correct / total_images if total_images > 0 else 0

print(f"\nWith Threshold = 0.30:")
print(f"  Total Correct: {total_correct}/{total_images}")
print(f"  Overall Accuracy: {overall_acc:.2f}%")
print(f"  Authentic Accuracy: {authentic_acc:.2f}%")
print(f"  Tampered Accuracy: {tampered_acc:.2f}%")
print(f"  Avg Confidence (Authentic): {avg_authentic_conf:.2%}")
print(f"  Avg Confidence (Tampered): {avg_tampered_conf:.2%}")

# Compare to previous threshold
print("\n" + "="*60)
print("COMPARISON")
print("="*60)
print(f"Previous threshold (0.50): 77.18% validation accuracy")
print(f"New threshold (0.30): {overall_acc:.2f}% on this test set")

if overall_acc > 77.18:
    print(f"\n✅ IMPROVEMENT: +{overall_acc - 77.18:.2f}%")
elif overall_acc < 77.18:
    print(f"\n⚠️ DECREASE: {overall_acc - 77.18:.2f}%")
else:
    print(f"\n➡️ Same performance")

print("\n" + "="*60)
print("ANALYSIS")
print("="*60)

if tampered_acc > authentic_acc:
    print("✓ Better at detecting manipulations than authenticating real images")
    print("  This is GOOD - we want to catch manipulations!")
elif authentic_acc > tampered_acc:
    print("⚠️ Better at authenticating real images than detecting manipulations")
    print("  May need to lower threshold further or retrain")
else:
    print("➡️ Balanced performance on both classes")

# Sensitivity analysis
false_positives = authentic_total - authentic_correct
false_negatives = tampered_total - tampered_correct

print(f"\nError Analysis:")
print(f"  False Positives (real flagged as fake): {false_positives}")
print(f"  False Negatives (fake flagged as real): {false_negatives}")

if false_negatives > false_positives:
    print(f"  ⚠️ Missing {false_negatives} manipulations - threshold may still be too high")
elif false_positives > false_negatives:
    print(f"  ⚠️ {false_positives} false alarms - threshold may be too low")
else:
    print(f"  ✓ Balanced error distribution")

print("\n✓ Testing complete!")
