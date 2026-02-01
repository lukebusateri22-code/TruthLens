"""
Convert the trained model to match frontend architecture
The training saved with 'backbone.' prefix, frontend expects without it
"""

import torch
import torch.nn as nn
from torchvision import models

print("="*70)
print(" CONVERTING MODEL FOR FRONTEND COMPATIBILITY")
print("="*70)

# Load the trained model
print("\nLoading trained model...")
checkpoint = torch.load('best_manipulation_fast.pth', map_location='cpu')

print(f"Model has {len(checkpoint)} parameters")
print("\nSample keys:")
for i, key in enumerate(list(checkpoint.keys())[:5]):
    print(f"  {key}")

# Create the frontend model structure
print("\nCreating frontend-compatible model...")
model = models.efficientnet_b0(weights=None)
num_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(num_features, 256),
    nn.SiLU(),
    nn.Dropout(0.2),
    nn.Linear(256, 2)
)

# Try to load directly first
try:
    model.load_state_dict(checkpoint)
    print("✓ Model loaded directly (no conversion needed)")
    needs_conversion = False
except RuntimeError as e:
    print("✗ Direct loading failed, conversion needed")
    needs_conversion = True

if needs_conversion:
    # Check if keys have 'backbone.' prefix
    if any(key.startswith('backbone.') for key in checkpoint.keys()):
        print("\nRemoving 'backbone.' prefix from keys...")
        new_checkpoint = {}
        for key, value in checkpoint.items():
            if key.startswith('backbone.'):
                new_key = key.replace('backbone.', '')
                new_checkpoint[new_key] = value
            else:
                new_checkpoint[key] = value
        
        print(f"Converted {len(new_checkpoint)} parameters")
        print("\nSample converted keys:")
        for i, key in enumerate(list(new_checkpoint.keys())[:5]):
            print(f"  {key}")
        
        # Try loading converted model
        try:
            model.load_state_dict(new_checkpoint)
            print("\n✓ Conversion successful!")
            
            # Save the converted model
            print("\nSaving frontend-compatible model...")
            torch.save(model.state_dict(), 'best_manipulation_fast.pth')
            print("✓ Saved as: best_manipulation_fast.pth")
            
            # Also save a backup
            torch.save(checkpoint, 'best_manipulation_fast_BACKUP.pth')
            print("✓ Original backed up as: best_manipulation_fast_BACKUP.pth")
            
        except RuntimeError as e:
            print(f"\n✗ Conversion failed: {e}")
            print("\nThis means the model architecture is fundamentally different.")
            print("You may need to retrain with the correct architecture.")
    else:
        print("\nKeys don't have 'backbone.' prefix")
        print("The architecture mismatch is more complex.")
        print("\nChecking key patterns...")
        print("\nExpected keys (frontend):")
        for i, key in enumerate(list(model.state_dict().keys())[:10]):
            print(f"  {key}")
        print("\nActual keys (saved model):")
        for i, key in enumerate(list(checkpoint.keys())[:10]):
            print(f"  {key}")

print("\n" + "="*70)
print(" CONVERSION COMPLETE")
print("="*70)
