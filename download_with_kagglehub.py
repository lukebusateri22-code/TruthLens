"""
Download Real Deepfake Dataset using kagglehub
This is the easiest method - no SSL issues!
"""

import kagglehub
import shutil
from pathlib import Path

def download_and_organize():
    print("\n" + "="*70)
    print("🎬 Downloading Real Deepfake Dataset with kagglehub")
    print("="*70 + "\n")
    
    print("📥 Downloading dataset (this may take a few minutes)...")
    print("Dataset: manjilkarki/deepfake-and-real-images")
    print("Size: ~1.09 GB\n")
    
    try:
        # Download latest version
        path = kagglehub.dataset_download("manjilkarki/deepfake-and-real-images")
        
        print(f"\n✓ Download complete!")
        print(f"📂 Path to dataset files: {path}\n")
        
        # Organize the data
        print("📁 Organizing data for training...")
        organize_data(path)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you're logged into Kaggle")
        print("2. Check your internet connection")
        print("3. Try manual download from:")
        print("   https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images")
        return False

def organize_data(source_path):
    """Organize downloaded data into train/val/test splits."""
    
    source = Path(source_path) / 'Dataset'
    target = Path('./data')
    
    if not source.exists():
        # Try without 'Dataset' subdirectory
        source = Path(source_path)
    
    # Check what's in the downloaded path
    print(f"\nContents of {source}:")
    if source.exists():
        for item in source.iterdir():
            print(f"  - {item.name}")
    else:
        print(f"  ❌ Directory not found")
        return
    
    # Map Kaggle structure to our structure
    mapping = {
        'Train': 'train',
        'Validation': 'val',
        'Test': 'test'
    }
    
    total_images = 0
    
    for kaggle_split, our_split in mapping.items():
        kaggle_dir = source / kaggle_split
        
        if not kaggle_dir.exists():
            print(f"  ⚠️  {kaggle_split} directory not found, skipping...")
            continue
        
        for class_name in ['Real', 'Fake']:
            src = kaggle_dir / class_name
            dst = target / our_split / class_name.lower()
            
            if src.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                
                # Remove existing if present
                if dst.exists():
                    shutil.rmtree(dst)
                
                # Copy files
                shutil.copytree(src, dst)
                
                # Count files
                files = list(dst.glob('*'))
                count = len(files)
                total_images += count
                
                print(f"  ✓ {our_split}/{class_name.lower()}: {count} images")
    
    print(f"\n✅ Data organized successfully!")
    print(f"📊 Total images: {total_images}")
    print(f"📂 Location: {target.absolute()}\n")
    
    # Print summary
    print("="*70)
    print("Dataset Summary:")
    print("="*70)
    
    for split in ['train', 'val', 'test']:
        split_dir = target / split
        if split_dir.exists():
            real_count = len(list((split_dir / 'real').glob('*'))) if (split_dir / 'real').exists() else 0
            fake_count = len(list((split_dir / 'fake').glob('*'))) if (split_dir / 'fake').exists() else 0
            total = real_count + fake_count
            print(f"{split.capitalize():12} {real_count:4} real + {fake_count:4} fake = {total:4} total")
    
    print("="*70 + "\n")

def main():
    """Main function."""
    
    success = download_and_organize()
    
    if success:
        print("🚀 Ready to train with real deepfakes!")
        print("="*70)
        print("\nNext steps:")
        print("  1. Train centralized model:")
        print("     python train_simple.py")
        print("")
        print("  2. Train with federated learning:")
        print("     python federated_simple.py")
        print("")
        print("  3. Expected accuracy: 85-95% ✨")
        print("="*70 + "\n")
    else:
        print("\n⚠️  Download failed. Please try manual download.")
        print("See: GET_REAL_DATA.md for instructions\n")

if __name__ == "__main__":
    main()
