"""
Federated Learning with Real-Time Monitoring Dashboard
Tracks convergence, client participation, and performance
"""

import torch
import numpy as np
from pathlib import Path
import sys
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(str(Path(__file__).parent))

from train_simple import SimpleDeepfakeDetector
from data.data_loader import partition_data_for_federated_learning, create_client_dataloaders
import flwr as fl
from collections import OrderedDict
import time

class FLMonitor:
    """Monitor and log federated learning progress."""
    
    def __init__(self, output_dir='./fl_results'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.history = {
            'rounds': [],
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'client_metrics': [],
            'timestamps': []
        }
        
        self.start_time = time.time()
    
    def log_round(self, round_num, metrics):
        """Log metrics for a round."""
        self.history['rounds'].append(round_num)
        self.history['train_loss'].append(metrics.get('train_loss', 0))
        self.history['train_accuracy'].append(metrics.get('train_accuracy', 0))
        self.history['val_loss'].append(metrics.get('val_loss', 0))
        self.history['val_accuracy'].append(metrics.get('val_accuracy', 0))
        self.history['timestamps'].append(time.time() - self.start_time)
        
        # Print progress
        print(f"\n{'='*70}")
        print(f"Round {round_num} Summary:")
        print(f"{'='*70}")
        print(f"Train Loss: {metrics.get('train_loss', 0):.4f} | Train Acc: {metrics.get('train_accuracy', 0):.2%}")
        print(f"Val Loss: {metrics.get('val_loss', 0):.4f} | Val Acc: {metrics.get('val_accuracy', 0):.2%}")
        print(f"Time Elapsed: {self.history['timestamps'][-1]:.1f}s")
        print(f"{'='*70}\n")
    
    def log_client_metrics(self, round_num, client_id, metrics):
        """Log individual client metrics."""
        self.history['client_metrics'].append({
            'round': round_num,
            'client_id': client_id,
            'loss': metrics.get('loss', 0),
            'accuracy': metrics.get('accuracy', 0),
            'num_samples': metrics.get('num_samples', 0)
        })
    
    def save_history(self):
        """Save history to JSON."""
        output_file = self.output_dir / 'fl_history.json'
        with open(output_file, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"✓ History saved to {output_file}")
    
    def plot_convergence(self):
        """Plot training convergence."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        rounds = self.history['rounds']
        
        # Loss curves
        axes[0, 0].plot(rounds, self.history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        axes[0, 0].plot(rounds, self.history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        axes[0, 0].set_xlabel('Round')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Loss Convergence')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        # Accuracy curves
        axes[0, 1].plot(rounds, [a*100 for a in self.history['train_accuracy']], 'b-', label='Train Acc', linewidth=2)
        axes[0, 1].plot(rounds, [a*100 for a in self.history['val_accuracy']], 'r-', label='Val Acc', linewidth=2)
        axes[0, 1].set_xlabel('Round')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].set_title('Accuracy Convergence')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
        
        # Training time
        axes[1, 0].plot(rounds, self.history['timestamps'], 'g-', linewidth=2)
        axes[1, 0].set_xlabel('Round')
        axes[1, 0].set_ylabel('Time (seconds)')
        axes[1, 0].set_title('Cumulative Training Time')
        axes[1, 0].grid(alpha=0.3)
        
        # Client participation
        if self.history['client_metrics']:
            client_data = {}
            for metric in self.history['client_metrics']:
                cid = metric['client_id']
                if cid not in client_data:
                    client_data[cid] = []
                client_data[cid].append(metric['accuracy'])
            
            for cid, accs in client_data.items():
                axes[1, 1].plot(range(1, len(accs)+1), [a*100 for a in accs], 
                              marker='o', label=f'Client {cid}', linewidth=2)
            
            axes[1, 1].set_xlabel('Round')
            axes[1, 1].set_ylabel('Accuracy (%)')
            axes[1, 1].set_title('Client Performance')
            axes[1, 1].legend()
            axes[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        
        output_file = self.output_dir / 'fl_convergence.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✓ Convergence plot saved to {output_file}")
        plt.close()
    
    def generate_report(self):
        """Generate comprehensive FL report."""
        report = []
        report.append("="*70)
        report.append("FEDERATED LEARNING TRAINING REPORT")
        report.append("="*70)
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"\nTotal Rounds: {len(self.history['rounds'])}")
        report.append(f"Total Time: {self.history['timestamps'][-1]:.1f} seconds")
        report.append(f"Avg Time per Round: {np.mean(np.diff([0] + self.history['timestamps'])):.1f} seconds")
        
        report.append("\n" + "="*70)
        report.append("FINAL PERFORMANCE")
        report.append("="*70)
        report.append(f"Final Train Loss: {self.history['train_loss'][-1]:.4f}")
        report.append(f"Final Train Accuracy: {self.history['train_accuracy'][-1]:.2%}")
        report.append(f"Final Val Loss: {self.history['val_loss'][-1]:.4f}")
        report.append(f"Final Val Accuracy: {self.history['val_accuracy'][-1]:.2%}")
        
        report.append("\n" + "="*70)
        report.append("CONVERGENCE ANALYSIS")
        report.append("="*70)
        
        # Improvement
        initial_acc = self.history['val_accuracy'][0]
        final_acc = self.history['val_accuracy'][-1]
        improvement = (final_acc - initial_acc) * 100
        report.append(f"Initial Accuracy: {initial_acc:.2%}")
        report.append(f"Final Accuracy: {final_acc:.2%}")
        report.append(f"Improvement: {improvement:+.2f}%")
        
        # Best round
        best_round = np.argmax(self.history['val_accuracy']) + 1
        best_acc = max(self.history['val_accuracy'])
        report.append(f"\nBest Round: {best_round}")
        report.append(f"Best Accuracy: {best_acc:.2%}")
        
        report.append("\n" + "="*70)
        report.append("CLIENT STATISTICS")
        report.append("="*70)
        
        if self.history['client_metrics']:
            client_stats = {}
            for metric in self.history['client_metrics']:
                cid = metric['client_id']
                if cid not in client_stats:
                    client_stats[cid] = {'accuracies': [], 'samples': []}
                client_stats[cid]['accuracies'].append(metric['accuracy'])
                client_stats[cid]['samples'].append(metric['num_samples'])
            
            for cid, stats in client_stats.items():
                avg_acc = np.mean(stats['accuracies'])
                total_samples = stats['samples'][0] if stats['samples'] else 0
                report.append(f"\nClient {cid}:")
                report.append(f"  Average Accuracy: {avg_acc:.2%}")
                report.append(f"  Training Samples: {total_samples}")
        
        report.append("\n" + "="*70)
        
        # Save report
        output_file = self.output_dir / 'fl_report.txt'
        with open(output_file, 'w') as f:
            f.write('\n'.join(report))
        
        print('\n'.join(report))
        print(f"\n✓ Report saved to {output_file}")


class MonitoredClient(fl.client.NumPyClient):
    """FL Client with monitoring."""
    
    def __init__(self, cid, train_loader, val_loader, device, monitor):
        self.cid = cid
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.monitor = monitor
        self.model = SimpleDeepfakeDetector().to(device)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.current_round = 0
    
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.current_round = config.get('round', 0)
        
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for images, labels in self.train_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        avg_loss = total_loss / total
        accuracy = correct / total
        
        # Log to monitor
        self.monitor.log_client_metrics(self.current_round, self.cid, {
            'loss': avg_loss,
            'accuracy': accuracy,
            'num_samples': total
        })
        
        print(f"  Client {self.cid} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2%}")
        
        return self.get_parameters(config={}), len(self.train_loader.dataset), {
            "loss": avg_loss,
            "accuracy": accuracy
        }
    
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        avg_loss = total_loss / total
        accuracy = correct / total
        
        return avg_loss, len(self.val_loader.dataset), {"accuracy": accuracy}


def main():
    print("\n" + "="*70)
    print("FEDERATED LEARNING WITH MONITORING")
    print("="*70 + "\n")
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Initialize monitor
    monitor = FLMonitor()
    
    # Partition data
    print("\nPartitioning data...")
    num_clients = 5
    num_rounds = 10
    
    client_indices = partition_data_for_federated_learning(
        data_dir='./data',
        num_clients=num_clients,
        split='train',
        partition_method='iid'
    )
    
    # Create client data loaders
    train_loaders = create_client_dataloaders(
        data_dir='./data',
        client_indices=client_indices,
        batch_size=32,
        num_workers=2,
        image_size=224
    )
    
    # Create validation loaders (same for all clients for comparison)
    from data.data_loader import DeepfakeDataset
    from torch.utils.data import DataLoader
    
    val_dataset = DeepfakeDataset('./data', split='val', image_size=224)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
    
    # Create client function
    def client_fn(cid: str):
        client_idx = int(cid)
        return MonitoredClient(
            cid,
            train_loaders[client_idx],
            val_loader,
            device,
            monitor
        )
    
    # Custom strategy with monitoring
    class MonitoredFedAvg(fl.server.strategy.FedAvg):
        def __init__(self, monitor, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.monitor = monitor
            self.current_round = 0
        
        def aggregate_fit(self, server_round, results, failures):
            self.current_round = server_round
            aggregated = super().aggregate_fit(server_round, results, failures)
            
            # Calculate metrics
            if results:
                losses = [r.metrics['loss'] for _, r in results if 'loss' in r.metrics]
                accuracies = [r.metrics['accuracy'] for _, r in results if 'accuracy' in r.metrics]
                
                avg_loss = np.mean(losses) if losses else 0
                avg_acc = np.mean(accuracies) if accuracies else 0
                
                self.monitor.log_round(server_round, {
                    'train_loss': avg_loss,
                    'train_accuracy': avg_acc,
                    'val_loss': 0,  # Will be updated in aggregate_evaluate
                    'val_accuracy': 0
                })
            
            return aggregated
        
        def aggregate_evaluate(self, server_round, results, failures):
            aggregated = super().aggregate_evaluate(server_round, results, failures)
            
            # Update validation metrics
            if results:
                losses = [r.loss for _, r in results]
                accuracies = [r.metrics['accuracy'] for _, r in results if 'accuracy' in r.metrics]
                
                avg_val_loss = np.mean(losses) if losses else 0
                avg_val_acc = np.mean(accuracies) if accuracies else 0
                
                # Update last round's metrics
                if self.monitor.history['rounds']:
                    self.monitor.history['val_loss'][-1] = avg_val_loss
                    self.monitor.history['val_accuracy'][-1] = avg_val_acc
            
            return aggregated
    
    strategy = MonitoredFedAvg(
        monitor=monitor,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
    )
    
    # Start simulation
    print(f"\n{'='*70}")
    print(f"Starting Federated Learning")
    print(f"Clients: {num_clients} | Rounds: {num_rounds}")
    print(f"{'='*70}\n")
    
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0},
    )
    
    # Generate reports
    print("\n" + "="*70)
    print("Generating Reports...")
    print("="*70 + "\n")
    
    monitor.save_history()
    monitor.plot_convergence()
    monitor.generate_report()
    
    print("\n" + "="*70)
    print("✅ FEDERATED LEARNING COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: {monitor.output_dir}")
    print("  - fl_history.json (raw data)")
    print("  - fl_convergence.png (plots)")
    print("  - fl_report.txt (summary)")
    print("\n")

if __name__ == "__main__":
    main()
