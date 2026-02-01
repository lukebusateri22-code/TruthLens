"""
Update frontend to test the current 74.49% model
"""

import shutil
from pathlib import Path

def update_frontend():
    print("🔧 Updating frontend to test current model...")
    
    # Copy the best model to the expected location
    if Path('best_manipulation_incremental.pth').exists():
        shutil.copy2('best_manipulation_incremental.pth', 'best_manipulation_fast.pth')
        print("✓ Updated best_manipulation_fast.pth with 74.49% model")
    else:
        print("❌ Model file not found!")
        return False
    
    # Update the model info in the detector
    detector_file = Path('models/manipulation_detector_final.py')
    if detector_file.exists():
        with open(detector_file, 'r') as f:
            content = f.read()
        
        # Update the accuracy info
        content = content.replace(
            "'accuracy': '77.18%'",
            "'accuracy': '74.49%'"
        )
        content = content.replace(
            "'training_time': '3.18 hours'",
            "'training_time': '4 hours (incremental)'"
        )
        content = content.replace(
            "'dataset': 'CG1050 + CASIA v2.0'",
            "'dataset': 'CG1050 + CASIA v2.0 + 2K new images'"
        )
        
        with open(detector_file, 'w') as f:
            f.write(content)
        
        print("✓ Updated model info in detector")
    
    print("\n✅ Frontend ready for testing!")
    print("📊 Model: 74.49% validation accuracy")
    print("🎯 Can start webapp on port 8506 to test")
    return True

if __name__ == '__main__':
    update_frontend()
