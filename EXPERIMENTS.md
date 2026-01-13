# Experiment Guide

This guide provides various experiments you can run to explore deepfake detection and federated learning.

## 🧪 Experiment 1: Model Architecture Comparison

Compare different model architectures to find the best performer.

### EfficientNet-B0
```bash
python training/train_centralized.py \
    --data_dir ./data \
    --model efficientnet \
    --epochs 20 \
    --batch_size 32 \
    --results_dir ./results/exp1_efficientnet
```

### ResNet-50
```bash
python training/train_centralized.py \
    --data_dir ./data \
    --model resnet \
    --epochs 20 \
    --batch_size 32 \
    --results_dir ./results/exp1_resnet
```

### Hybrid Model
```bash
python training/train_centralized.py \
    --data_dir ./data \
    --model hybrid \
    --epochs 20 \
    --batch_size 32 \
    --results_dir ./results/exp1_hybrid
```

**Analysis**: Compare accuracy, training time, and model size.

---

## 🧪 Experiment 2: Hyperparameter Tuning

### Learning Rate Comparison
```bash
# Low learning rate
python training/train_centralized.py --lr 0.0001 --results_dir ./results/exp2_lr_0001

# Medium learning rate
python training/train_centralized.py --lr 0.001 --results_dir ./results/exp2_lr_001

# High learning rate
python training/train_centralized.py --lr 0.01 --results_dir ./results/exp2_lr_01
```

### Batch Size Comparison
```bash
# Small batch
python training/train_centralized.py --batch_size 16 --results_dir ./results/exp2_bs_16

# Medium batch
python training/train_centralized.py --batch_size 32 --results_dir ./results/exp2_bs_32

# Large batch
python training/train_centralized.py --batch_size 64 --results_dir ./results/exp2_bs_64
```

**Analysis**: Plot learning curves and compare convergence speed.

---

## 🧪 Experiment 3: Data Distribution Effects (Federated Learning)

### IID Data Distribution
```bash
python federated_train.py \
    --data_dir ./data \
    --num_clients 5 \
    --num_rounds 15 \
    --partition_method iid \
    --results_dir ./results/exp3_iid
```

### Non-IID Data Distribution (Mild)
```bash
python federated_train.py \
    --data_dir ./data \
    --num_clients 5 \
    --num_rounds 15 \
    --partition_method non_iid \
    --alpha 1.0 \
    --results_dir ./results/exp3_non_iid_mild
```

### Non-IID Data Distribution (Severe)
```bash
python federated_train.py \
    --data_dir ./data \
    --num_clients 5 \
    --num_rounds 15 \
    --partition_method non_iid \
    --alpha 0.1 \
    --results_dir ./results/exp3_non_iid_severe
```

**Analysis**: Compare how data heterogeneity affects model performance.

---

## 🧪 Experiment 4: Federated Learning Strategies

### FedAvg (Baseline)
```bash
python federated_train.py \
    --strategy fedavg \
    --num_clients 5 \
    --num_rounds 15 \
    --results_dir ./results/exp4_fedavg
```

### FedProx (Better for heterogeneous data)
```bash
python federated_train.py \
    --strategy fedprox \
    --num_clients 5 \
    --num_rounds 15 \
    --results_dir ./results/exp4_fedprox
```

### FedAdagrad (Adaptive learning)
```bash
python federated_train.py \
    --strategy fedadagrad \
    --num_clients 5 \
    --num_rounds 15 \
    --results_dir ./results/exp4_fedadagrad
```

### Secure Aggregation (With differential privacy)
```bash
python federated_train.py \
    --strategy secure \
    --num_clients 5 \
    --num_rounds 15 \
    --results_dir ./results/exp4_secure
```

**Analysis**: Compare convergence speed and final accuracy across strategies.

---

## 🧪 Experiment 5: Number of Clients Impact

### 3 Clients
```bash
python federated_train.py \
    --num_clients 3 \
    --num_rounds 20 \
    --results_dir ./results/exp5_clients_3
```

### 5 Clients
```bash
python federated_train.py \
    --num_clients 5 \
    --num_rounds 20 \
    --results_dir ./results/exp5_clients_5
```

### 10 Clients
```bash
python federated_train.py \
    --num_clients 10 \
    --num_rounds 20 \
    --results_dir ./results/exp5_clients_10
```

**Analysis**: How does the number of clients affect training dynamics?

---

## 🧪 Experiment 6: Local Training Epochs

### 1 Local Epoch
```bash
python federated_train.py \
    --local_epochs 1 \
    --num_rounds 20 \
    --results_dir ./results/exp6_epochs_1
```

### 3 Local Epochs
```bash
python federated_train.py \
    --local_epochs 3 \
    --num_rounds 20 \
    --results_dir ./results/exp6_epochs_3
```

### 5 Local Epochs
```bash
python federated_train.py \
    --local_epochs 5 \
    --num_rounds 20 \
    --results_dir ./results/exp6_epochs_5
```

**Analysis**: Trade-off between communication rounds and local computation.

---

## 🧪 Experiment 7: Centralized vs Federated

### Centralized Training
```bash
python training/train_centralized.py \
    --data_dir ./data \
    --epochs 20 \
    --batch_size 32 \
    --results_dir ./results/exp7_centralized
```

### Federated Training (Equivalent computation)
```bash
python federated_train.py \
    --data_dir ./data \
    --num_clients 5 \
    --num_rounds 20 \
    --local_epochs 1 \
    --results_dir ./results/exp7_federated
```

**Analysis**: Compare final accuracy, training time, and convergence patterns.

---

## 🧪 Experiment 8: Dropout Regularization

### No Dropout
```bash
python training/train_centralized.py \
    --dropout 0.0 \
    --results_dir ./results/exp8_dropout_0
```

### Light Dropout
```bash
python training/train_centralized.py \
    --dropout 0.3 \
    --results_dir ./results/exp8_dropout_03
```

### Heavy Dropout
```bash
python training/train_centralized.py \
    --dropout 0.7 \
    --results_dir ./results/exp8_dropout_07
```

**Analysis**: Effect of regularization on overfitting.

---

## 🧪 Experiment 9: Client Sampling

### All Clients Participate
```bash
python federated_train.py \
    --fraction_fit 1.0 \
    --num_clients 10 \
    --results_dir ./results/exp9_fraction_1
```

### 50% Client Sampling
```bash
python federated_train.py \
    --fraction_fit 0.5 \
    --num_clients 10 \
    --results_dir ./results/exp9_fraction_05
```

### 30% Client Sampling
```bash
python federated_train.py \
    --fraction_fit 0.3 \
    --num_clients 10 \
    --results_dir ./results/exp9_fraction_03
```

**Analysis**: Impact of partial client participation on convergence.

---

## 🧪 Experiment 10: Transfer Learning

### With Pretrained Weights
```bash
python training/train_centralized.py \
    --pretrained True \
    --epochs 15 \
    --results_dir ./results/exp10_pretrained
```

### Without Pretrained Weights (Train from scratch)
```bash
python training/train_centralized.py \
    --pretrained False \
    --epochs 30 \
    --results_dir ./results/exp10_scratch
```

**Analysis**: Benefits of transfer learning from ImageNet.

---

## 📊 Analysis Scripts

### Compare Results
Create a simple Python script to compare experiment results:

```python
import json
import matplotlib.pyplot as plt
from pathlib import Path

def compare_experiments(exp_dirs):
    """Compare results from multiple experiments."""
    results = {}
    
    for exp_dir in exp_dirs:
        metrics_file = Path(exp_dir) / 'test_metrics.json'
        if metrics_file.exists():
            with open(metrics_file) as f:
                results[exp_dir] = json.load(f)
    
    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        values = [results[exp][metric] for exp in results]
        ax.bar(range(len(results)), values)
        ax.set_title(metric.replace('_', ' ').title())
        ax.set_xticks(range(len(results)))
        ax.set_xticklabels([Path(exp).name for exp in results], rotation=45)
    
    plt.tight_layout()
    plt.savefig('experiment_comparison.png')
    print("✓ Comparison plot saved to experiment_comparison.png")

# Usage
compare_experiments([
    './results/exp1_efficientnet',
    './results/exp1_resnet',
    './results/exp1_hybrid'
])
```

---

## 📝 Experiment Template

Use this template for documenting your experiments:

```markdown
## Experiment: [Name]

**Hypothesis**: [What you expect to happen]

**Setup**:
- Model: [model name]
- Data: [dataset details]
- Parameters: [key parameters]

**Command**:
```bash
[command to run]
```

**Results**:
- Accuracy: [value]
- Precision: [value]
- Recall: [value]
- F1-Score: [value]

**Observations**:
[What you learned]

**Conclusion**:
[Summary and next steps]
```

---

## 🎯 Research Questions to Explore

1. **How does data heterogeneity affect federated learning performance?**
   - Run Experiment 3 with different alpha values

2. **What is the optimal number of local epochs?**
   - Run Experiment 6 and analyze communication efficiency

3. **Can differential privacy maintain good accuracy?**
   - Run Experiment 4 with secure aggregation

4. **Which model architecture is best for deepfake detection?**
   - Run Experiment 1 and compare metrics

5. **How does federated learning compare to centralized training?**
   - Run Experiment 7 and analyze trade-offs

---

## 💡 Tips for Running Experiments

1. **Use consistent random seeds** for reproducibility
2. **Save all configurations** in JSON files
3. **Monitor GPU/CPU usage** during training
4. **Keep detailed logs** of all experiments
5. **Create visualizations** to compare results
6. **Document unexpected findings**

Happy experimenting! 🚀
