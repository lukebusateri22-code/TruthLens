"""
Quick Implementation of High-Priority Enhancements
Run this after training completes
"""

import torch
import numpy as np
from pathlib import Path
import sys
import time
import json
from datetime import datetime

sys.path.append(str(Path(__file__).parent))

from train_simple import SimpleDeepfakeDetector
from data.data_loader import DeepfakeDataset
from torch.utils.data import DataLoader

def benchmark_inference_speed(model, test_loader, device, num_samples=100):
    """Benchmark inference speed."""
    
    print("\n" + "="*70)
    print("Inference Speed Benchmark")
    print("="*70 + "\n")
    
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for i, (images, _) in enumerate(test_loader):
            if i >= 5:
                break
            _ = model(images.to(device))
    
    # Actual benchmark
    times = []
    with torch.no_grad():
        for i, (images, _) in enumerate(test_loader):
            if i >= num_samples:
                break
            
            start = time.time()
            _ = model(images.to(device))
            end = time.time()
            
            times.append(end - start)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    fps = 1.0 / avg_time if avg_time > 0 else 0
    
    print(f"Average Inference Time: {avg_time*1000:.2f} ms")
    print(f"Standard Deviation: {std_time*1000:.2f} ms")
    print(f"Throughput: {fps:.2f} FPS")
    print(f"Samples Tested: {len(times)}")
    
    results = {
        'avg_time_ms': avg_time * 1000,
        'std_time_ms': std_time * 1000,
        'fps': fps,
        'device': str(device)
    }
    
    # Save results
    with open('inference_benchmark.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to inference_benchmark.json")
    
    return results

def analyze_failure_cases(model, test_loader, device, num_failures=20):
    """Find and analyze failure cases."""
    
    print("\n" + "="*70)
    print("Failure Case Analysis")
    print("="*70 + "\n")
    
    model.eval()
    
    failures = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            # Find misclassifications
            incorrect = predicted != labels
            
            for i in range(len(labels)):
                if incorrect[i] and len(failures) < num_failures:
                    probs = torch.softmax(outputs[i], dim=0)
                    failures.append({
                        'true_label': 'Real' if labels[i].item() == 0 else 'Fake',
                        'predicted': 'Real' if predicted[i].item() == 0 else 'Fake',
                        'confidence': probs[predicted[i]].item(),
                        'real_prob': probs[0].item(),
                        'fake_prob': probs[1].item()
                    })
            
            if len(failures) >= num_failures:
                break
    
    print(f"Found {len(failures)} failure cases:\n")
    
    # Analyze patterns
    false_positives = [f for f in failures if f['true_label'] == 'Real' and f['predicted'] == 'Fake']
    false_negatives = [f for f in failures if f['true_label'] == 'Fake' and f['predicted'] == 'Real']
    
    print(f"False Positives (Real → Fake): {len(false_positives)}")
    print(f"False Negatives (Fake → Real): {len(false_negatives)}")
    print()
    
    # Average confidence in failures
    avg_confidence = np.mean([f['confidence'] for f in failures])
    print(f"Average Confidence in Failures: {avg_confidence:.2%}")
    print()
    
    print("Sample Failures:")
    for i, f in enumerate(failures[:5]):
        print(f"\n{i+1}. True: {f['true_label']}, Predicted: {f['predicted']}")
        print(f"   Confidence: {f['confidence']:.2%}")
        print(f"   Real Prob: {f['real_prob']:.2%}, Fake Prob: {f['fake_prob']:.2%}")
    
    # Save results
    with open('failure_analysis.json', 'w') as f:
        json.dump({
            'total_failures': len(failures),
            'false_positives': len(false_positives),
            'false_negatives': len(false_negatives),
            'avg_confidence': avg_confidence,
            'failures': failures
        }, f, indent=2)
    
    print(f"\n✓ Results saved to failure_analysis.json")
    
    return failures

def measure_model_size(model):
    """Measure model size and parameters."""
    
    print("\n" + "="*70)
    print("Model Size Analysis")
    print("="*70 + "\n")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Estimate size
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    total_size_mb = (param_size + buffer_size) / 1024 / 1024
    
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Model Size: {total_size_mb:.2f} MB")
    
    results = {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'size_mb': total_size_mb
    }
    
    with open('model_size.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to model_size.json")
    
    return results

def main():
    """Run all enhancements."""
    
    print("\n" + "="*70)
    print("Running High-Priority Enhancements")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Load model
    print("Loading model...")
    model = SimpleDeepfakeDetector().to(device)
    
    model_path = Path('best_model_subset.pth')
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("✓ Loaded trained model\n")
    else:
        print("⚠ Using untrained model (for testing)\n")
    
    model.eval()
    
    # Load test data
    print("Loading test data...")
    test_dataset = DeepfakeDataset('./data', split='test', image_size=224)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)
    print(f"✓ Loaded {len(test_dataset)} test samples\n")
    
    # Run enhancements
    results = {}
    
    # 1. Model size analysis
    results['model_size'] = measure_model_size(model)
    
    # 2. Inference speed benchmark
    results['inference_speed'] = benchmark_inference_speed(model, test_loader, device)
    
    # 3. Failure case analysis
    results['failure_analysis'] = analyze_failure_cases(model, test_loader, device)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70 + "\n")
    
    print("✅ Model Size Analysis Complete")
    print(f"   - Parameters: {results['model_size']['total_params']:,}")
    print(f"   - Size: {results['model_size']['size_mb']:.2f} MB")
    
    print("\n✅ Inference Speed Benchmark Complete")
    print(f"   - Average Time: {results['inference_speed']['avg_time_ms']:.2f} ms")
    print(f"   - Throughput: {results['inference_speed']['fps']:.2f} FPS")
    
    print("\n✅ Failure Case Analysis Complete")
    print(f"   - Failures Found: {results['failure_analysis']['total_failures']}")
    print(f"   - False Positives: {results['failure_analysis']['false_positives']}")
    print(f"   - False Negatives: {results['failure_analysis']['false_negatives']}")
    
    print("\n" + "="*70)
    print("All enhancements completed successfully!")
    print("="*70)
    print("\nGenerated files:")
    print("  - model_size.json")
    print("  - inference_benchmark.json")
    print("  - failure_analysis.json")
    print("\nNext steps:")
    print("  1. Review failure cases")
    print("  2. Run FL with differential privacy")
    print("  3. Test on recent deepfakes")
    print("  4. Create ablation studies")
    print()

if __name__ == "__main__":
    main()
