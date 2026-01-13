# Quick Start Guide

Get started with deepfake detection and federated learning in minutes!

## 📋 Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-capable GPU for faster training

## 🚀 Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- PyTorch and torchvision
- Flower (federated learning framework)
- OpenCV and image processing libraries
- Data science tools (numpy, pandas, scikit-learn)
- Visualization tools (matplotlib, seaborn)

### 2. Prepare Your Data

#### Option A: Use Your Own Dataset

Organize your data in the following structure:

```
data/
├── train/
│   ├── real/    # Real images/videos
│   └── fake/    # Fake images/videos
├── val/
│   ├── real/
│   └── fake/
└── test/
    ├── real/
    └── fake/
```

**Download Kaggle Dataset:**
1. Go to https://www.kaggle.com/c/deepfake-detection-challenge/data
2. Download the dataset
3. Extract and organize into the structure above

**Helper Script:**
```bash
# Create directory structure
python setup_data.py --mode create_structure --dest_dir ./data

# Split existing data
python setup_data.py --mode split --source_dir /path/to/your/data --dest_dir ./data

# Verify dataset
python setup_data.py --mode verify --dest_dir ./data
```

#### Option B: Create Sample Dataset (for testing)

```bash
python setup_data.py --mode sample --dest_dir ./data --num_samples 100
```

This creates a small synthetic dataset for testing the pipeline.

## 🎯 Part 1: Train Centralized Model

Train a standard deepfake detection model:

```bash
python training/train_centralized.py \
    --data_dir ./data \
    --model efficientnet \
    --epochs 20 \
    --batch_size 32 \
    --lr 0.001
```

**Key Arguments:**
- `--model`: Choose from `efficientnet`, `hybrid`, or `resnet`
- `--epochs`: Number of training epochs
- `--batch_size`: Batch size (reduce if out of memory)
- `--lr`: Learning rate
- `--pretrained`: Use ImageNet pretrained weights (default: True)

**Monitor Training:**
- Progress bars show loss and accuracy
- Checkpoints saved in `./checkpoints/centralized/`
- Results saved in `./results/centralized/`

## 🌐 Part 2: Federated Learning

Train the model using federated learning across multiple simulated clients:

```bash
python federated_train.py \
    --data_dir ./data \
    --num_clients 5 \
    --num_rounds 10 \
    --local_epochs 1 \
    --batch_size 32 \
    --strategy fedavg
```

**Key Arguments:**
- `--num_clients`: Number of federated clients (e.g., 5-10)
- `--num_rounds`: Number of federated rounds
- `--local_epochs`: Local training epochs per round
- `--partition_method`: `iid` or `non_iid` data distribution
- `--strategy`: Aggregation strategy (`fedavg`, `fedprox`, `secure`)

**Federated Learning Strategies:**
- `fedavg`: Standard federated averaging
- `fedprox`: Better for heterogeneous data
- `fedadagrad`: Adaptive learning rate
- `secure`: Differential privacy enabled

## 📊 Evaluate Model

Evaluate a trained model on the test set:

```bash
python training/evaluate.py \
    --data_dir ./data \
    --model efficientnet \
    --checkpoint ./checkpoints/centralized/checkpoint_epoch_20_best.pth \
    --plot
```

This generates:
- Accuracy, precision, recall, F1-score
- Confusion matrix
- ROC curve
- Prediction distribution plots

## 💡 Quick Examples

### Example 1: Quick Test with Sample Data

```bash
# Create sample dataset
python setup_data.py --mode sample --dest_dir ./data --num_samples 50

# Train for 5 epochs (quick test)
python training/train_centralized.py --data_dir ./data --epochs 5 --batch_size 16

# Run federated learning (3 clients, 5 rounds)
python federated_train.py --data_dir ./data --num_clients 3 --num_rounds 5
```

### Example 2: Full Training Pipeline

```bash
# 1. Verify your data
python setup_data.py --mode verify --dest_dir ./data

# 2. Train centralized model
python training/train_centralized.py \
    --data_dir ./data \
    --model efficientnet \
    --epochs 20 \
    --batch_size 32

# 3. Evaluate
python training/evaluate.py \
    --data_dir ./data \
    --checkpoint ./checkpoints/centralized/checkpoint_epoch_20_best.pth

# 4. Run federated learning
python federated_train.py \
    --data_dir ./data \
    --num_clients 5 \
    --num_rounds 10 \
    --partition_method non_iid
```

### Example 3: Compare Different Models

```bash
# Train EfficientNet
python training/train_centralized.py --model efficientnet --epochs 20

# Train ResNet
python training/train_centralized.py --model resnet --epochs 20

# Train Hybrid model
python training/train_centralized.py --model hybrid --epochs 20
```

## 🔧 Troubleshooting

### Out of Memory Error
```bash
# Reduce batch size
python training/train_centralized.py --batch_size 16

# Reduce number of workers
python training/train_centralized.py --num_workers 2
```

### Data Not Found Error
```bash
# Verify your data structure
python setup_data.py --mode verify --dest_dir ./data

# Check that you have the correct structure:
# data/train/real/, data/train/fake/, etc.
```

### Slow Training
```bash
# Use fewer workers if CPU-bound
python training/train_centralized.py --num_workers 2

# Use smaller image size
python training/train_centralized.py --image_size 128

# Use smaller model
python training/train_centralized.py --model efficientnet
```

## 📈 Understanding Results

### Centralized Training
- **Checkpoints**: Saved in `./checkpoints/centralized/`
- **Metrics**: Saved in `./results/centralized/`
- **Best model**: `checkpoint_epoch_X_best.pth`

### Federated Learning
- **History**: `./results/federated/federated_history.json`
- **Config**: `./results/federated/config.json`
- **Metrics**: Accuracy per round for each client

### Evaluation Metrics
- **Accuracy**: Overall correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under ROC curve (higher is better)

## 🎓 Next Steps

1. **Experiment with hyperparameters**: Try different learning rates, batch sizes, epochs
2. **Try different models**: Compare efficientnet, resnet, and hybrid
3. **Test federated strategies**: Compare fedavg, fedprox, and secure aggregation
4. **Analyze results**: Look at confusion matrices and ROC curves
5. **Add data augmentation**: Modify preprocessing.py for custom augmentations
6. **Implement differential privacy**: Use the secure strategy with custom noise levels

## 📚 Additional Resources

- **Full README**: See `README.md` for detailed documentation
- **Model Architecture**: Check `models/deepfake_detector.py`
- **Data Processing**: See `data/preprocessing.py` and `data/data_loader.py`
- **Federated Learning**: Explore `federated/` directory

## 🆘 Getting Help

If you encounter issues:
1. Check the error message carefully
2. Verify your data structure with `setup_data.py --mode verify`
3. Try with sample data first: `setup_data.py --mode sample`
4. Reduce batch size if out of memory
5. Check that all dependencies are installed: `pip install -r requirements.txt`

Happy training! 🚀
