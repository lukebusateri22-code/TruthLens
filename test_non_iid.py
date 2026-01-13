"""
Test Non-IID Data Distribution in Federated Learning
Compare IID vs Non-IID performance
"""

import torch
import numpy as np
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(str(Path(__file__).parent))

from data.data_loader import partition_data_for_federated_learning, DeepfakeDataset
from torch.utils.data import DataLoader, Subset

def analyze_data_distribution(client_indices, dataset, num_clients):
    """Analyze and visualize data distribution across clients."""
    
    print("\n" + "="*70)
    print("Data Distribution Analysis")
    print("="*70 + "\n")
    
    client_stats = []
    
    for i in range(num_clients):
        indices = client_indices[i]
        labels = [dataset[idx][1] for idx in indices]
        
        real_count = sum(1 for l in labels if l == 0)
        fake_count = len(labels) - real_count
        
        client_stats.append({
            'client': i,
            'total': len(indices),
            'real': real_count,
            'fake': fake_count,
            'real_ratio': real_count / len(indices) if len(indices) > 0 else 0
        })
        
        print(f"Client {i}:")
        print(f"  Total: {len(indices)} samples")
        print(f"  Real: {real_count} ({real_count/len(indices)*100:.1f}%)")
        print(f"  Fake: {fake_count} ({fake_count/len(indices)*100:.1f}%)")
        print()
    
    return client_stats

def plot_distribution(client_stats, title, output_path):
    """Plot data distribution across clients."""
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Bar chart of samples per client
    clients = [s['client'] for s in client_stats]
    real_counts = [s['real'] for s in client_stats]
    fake_counts = [s['fake'] for s in client_stats]
    
    x = np.arange(len(clients))
    width = 0.35
    
    axes[0].bar(x - width/2, real_counts, width, label='Real', color='#10b981')
    axes[0].bar(x + width/2, fake_counts, width, label='Fake', color='#ef4444')
    axes[0].set_xlabel('Client ID')
    axes[0].set_ylabel('Number of Samples')
    axes[0].set_title(f'{title} - Sample Distribution')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f'Client {c}' for c in clients])
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Pie chart of real/fake ratio per client
    real_ratios = [s['real_ratio'] * 100 for s in client_stats]
    
    axes[1].bar(clients, real_ratios, color='#667eea')
    axes[1].axhline(y=50, color='r', linestyle='--', label='Balanced (50%)')
    axes[1].set_xlabel('Client ID')
    axes[1].set_ylabel('Real Image Percentage (%)')
    axes[1].set_title(f'{title} - Class Balance')
    axes[1].set_xticks(clients)
    axes[1].set_xticklabels([f'Client {c}' for c in clients])
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    axes[1].set_ylim([0, 100])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Distribution plot saved to {output_path}")
    plt.close()

def compare_iid_vs_noniid():
    """Compare IID and Non-IID data distributions."""
    
    print("\n" + "="*70)
    print("IID vs Non-IID Comparison")
    print("="*70 + "\n")
    
    # Load dataset
    dataset = DeepfakeDataset('./data', split='train', image_size=224)
    num_clients = 5
    
    # Test IID
    print("Testing IID Distribution...")
    iid_indices = partition_data_for_federated_learning(
        data_dir='./data',
        num_clients=num_clients,
        split='train',
        partition_method='iid'
    )
    
    iid_stats = analyze_data_distribution(iid_indices, dataset, num_clients)
    plot_distribution(iid_stats, 'IID Distribution', './iid_distribution.png')
    
    # Test Non-IID with different alpha values
    alpha_values = [0.1, 0.5, 1.0]
    
    for alpha in alpha_values:
        print(f"\nTesting Non-IID Distribution (alpha={alpha})...")
        noniid_indices = partition_data_for_federated_learning(
            data_dir='./data',
            num_clients=num_clients,
            split='train',
            partition_method='non_iid',
            alpha=alpha
        )
        
        noniid_stats = analyze_data_distribution(noniid_indices, dataset, num_clients)
        plot_distribution(noniid_stats, f'Non-IID (α={alpha})', f'./noniid_alpha{alpha}_distribution.png')
    
    # Summary
    print("\n" + "="*70)
    print("Summary")
    print("="*70 + "\n")
    
    print("IID (Identically and Independently Distributed):")
    print("  - Data is randomly shuffled and evenly distributed")
    print("  - Each client has similar class distribution")
    print("  - Easier to train, faster convergence")
    print("  - Less realistic for real-world scenarios")
    print()
    
    print("Non-IID (Non-Identically Distributed):")
    print("  - Data distribution varies across clients")
    print("  - Some clients may have more of one class")
    print("  - More challenging to train, slower convergence")
    print("  - More realistic for real-world federated learning")
    print()
    
    print("Alpha Parameter (for Non-IID):")
    print("  - Lower alpha (e.g., 0.1) = More heterogeneous")
    print("  - Higher alpha (e.g., 1.0) = More homogeneous")
    print("  - Alpha → ∞ approaches IID distribution")
    print()
    
    print("Recommendations:")
    print("  1. Start with IID to verify FL setup works")
    print("  2. Test Non-IID with alpha=0.5 (moderate heterogeneity)")
    print("  3. Try alpha=0.1 for challenging realistic scenario")
    print("  4. Compare convergence and final accuracy")
    print()
    
    print("Expected Results:")
    print("  - IID: Faster convergence, higher accuracy")
    print("  - Non-IID (α=0.5): Moderate convergence, good accuracy")
    print("  - Non-IID (α=0.1): Slower convergence, lower accuracy")
    print()

def test_federated_with_noniid():
    """Quick test of federated learning with Non-IID data."""
    
    print("\n" + "="*70)
    print("Quick Non-IID Federated Learning Test")
    print("="*70 + "\n")
    
    print("To run full FL with Non-IID data:")
    print()
    print("  # IID (baseline)")
    print("  python federated_with_monitoring.py")
    print()
    print("  # Non-IID (moderate)")
    print("  python federated_with_monitoring.py --partition non_iid --alpha 0.5")
    print()
    print("  # Non-IID (challenging)")
    print("  python federated_with_monitoring.py --partition non_iid --alpha 0.1")
    print()
    
    print("Compare results:")
    print("  - Convergence speed")
    print("  - Final accuracy")
    print("  - Client performance variance")
    print("  - Training stability")

def main():
    """Main function."""
    
    print("\n" + "="*70)
    print("Non-IID Data Distribution Testing")
    print("="*70)
    
    # Compare distributions
    compare_iid_vs_noniid()
    
    # Show how to test
    test_federated_with_noniid()
    
    print("\n" + "="*70)
    print("✅ Analysis Complete!")
    print("="*70)
    print("\nGenerated files:")
    print("  - iid_distribution.png")
    print("  - noniid_alpha0.1_distribution.png")
    print("  - noniid_alpha0.5_distribution.png")
    print("  - noniid_alpha1.0_distribution.png")
    print("\nNext steps:")
    print("  1. Review distribution plots")
    print("  2. Run FL with different distributions")
    print("  3. Compare results")
    print("  4. Document findings")
    print()

if __name__ == "__main__":
    main()
