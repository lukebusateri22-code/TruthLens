# Project Architecture

Detailed technical documentation of the deepfake detection system architecture.

## 📐 System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Deepfake Detection System                 │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐         ┌──────────────┐                  │
│  │   Part 1:    │         │   Part 2:    │                  │
│  │ Centralized  │────────▶│  Federated   │                  │
│  │   Training   │         │   Learning   │                  │
│  └──────────────┘         └──────────────┘                  │
│         │                         │                          │
│         ▼                         ▼                          │
│  ┌──────────────────────────────────────┐                   │
│  │      Deepfake Detection Model        │                   │
│  │  (EfficientNet / ResNet / Hybrid)    │                   │
│  └──────────────────────────────────────┘                   │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## 🏗️ Module Structure

### 1. Models Module (`models/`)

#### `deepfake_detector.py`
Contains three model architectures:

**DeepfakeDetector (EfficientNet-based)**
```
Input (3, 224, 224)
    ↓
EfficientNet-B0 Backbone (pretrained on ImageNet)
    ↓
Feature Vector (1280 dimensions)
    ↓
Custom Classification Head:
    - Linear(1280 → 512) + ReLU + Dropout(0.5)
    - Linear(512 → 256) + ReLU + Dropout(0.5)
    - Linear(256 → 2)
    ↓
Output Logits (2 classes: Real/Fake)
```

**HybridDeepfakeDetector**
- Adds attention mechanism
- Batch normalization layers
- Enhanced feature extraction

**ResNetDeepfakeDetector**
- ResNet-50 backbone
- Similar classification head
- Alternative architecture for comparison

#### `model_utils.py`
Utility functions:
- `save_checkpoint()`: Save model state
- `load_checkpoint()`: Load model state
- `calculate_metrics()`: Compute evaluation metrics
- `EarlyStopping`: Prevent overfitting
- `AverageMeter`: Track running statistics

### 2. Data Module (`data/`)

#### `preprocessing.py`

**VideoFrameExtractor**
```
Video File (.mp4, .avi, etc.)
    ↓
Extract N frames (uniform/random/first)
    ↓
Resize to (224, 224)
    ↓
RGB Frames (N, 224, 224, 3)
```

**Data Augmentation Pipeline**
```
Training:
- Resize(224, 224)
- HorizontalFlip(p=0.5)
- RandomRotate90(p=0.3)
- Brightness/Contrast adjustment
- Gaussian Noise/Blur
- ShiftScaleRotate
- CoarseDropout
- Normalize (ImageNet stats)
- ToTensor

Validation/Test:
- Resize(224, 224)
- Normalize
- ToTensor
```

**FaceDetector**
- Haar Cascade face detection
- Extract largest face
- Focus on facial regions

#### `data_loader.py`

**DeepfakeDataset**
```
Directory Structure:
data/
├── train/
│   ├── real/
│   └── fake/
├── val/
│   ├── real/
│   └── fake/
└── test/
    ├── real/
    └── fake/

Dataset loads images/videos and applies transforms
Returns: (image_tensor, label)
```

**Data Partitioning for FL**
```
IID Partitioning:
- Randomly shuffle all data
- Split evenly among N clients
- Each client has balanced distribution

Non-IID Partitioning:
- Use Dirichlet distribution with parameter α
- Lower α → more heterogeneous
- Each client has skewed class distribution
```

### 3. Federated Learning Module (`federated/`)

#### Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Federated Server                      │
│  - Aggregates model updates                             │
│  - Manages global model                                 │
│  - Coordinates training rounds                          │
└────────────┬────────────────────────────────────────────┘
             │
             │ Global Model
             ▼
    ┌────────┴────────┐
    │                 │
┌───▼───┐  ┌───▼───┐  ┌───▼───┐
│Client1│  │Client2│  │ClientN│
│       │  │       │  │       │
│ Data1 │  │ Data2 │  │ DataN │
└───┬───┘  └───┬───┘  └───┬───┘
    │          │          │
    │ Updates  │ Updates  │ Updates
    └──────────┴──────────┘
```

#### `client.py`

**DeepfakeClient**
```python
class DeepfakeClient:
    def fit(parameters, config):
        # 1. Receive global model parameters
        # 2. Train on local data for E epochs
        # 3. Return updated parameters
        
    def evaluate(parameters, config):
        # 1. Receive model parameters
        # 2. Evaluate on local validation data
        # 3. Return loss and metrics
```

#### `server.py`

**FedAvg Strategy**
```
For each round t = 1, 2, ..., T:
    1. Server sends global model w_t to clients
    
    2. Each client k:
       - Trains on local data D_k
       - Computes update Δw_k
       
    3. Server aggregates:
       w_{t+1} = Σ (n_k / n) * w_k
       where n_k = |D_k|, n = Σ n_k
       
    4. Update global model
```

#### `strategy.py`

**Available Strategies:**

1. **FedAvg**: Standard weighted averaging
2. **FedProx**: Adds proximal term for heterogeneous data
3. **FedAdagrad**: Adaptive learning rate
4. **FedYogi**: Adam-like optimization
5. **SecureAggregation**: Differential privacy

### 4. Training Module (`training/`)

#### `train_centralized.py`

**Training Loop**
```
For each epoch:
    For each batch in train_loader:
        1. Forward pass
        2. Compute loss
        3. Backward pass
        4. Update weights
        
    Validate on validation set
    Save checkpoint if best model
    Check early stopping
    
Final evaluation on test set
```

#### `evaluate.py`

**Evaluation Pipeline**
```
Load trained model
    ↓
For each batch in test_loader:
    - Forward pass
    - Collect predictions and probabilities
    ↓
Calculate metrics:
    - Accuracy, Precision, Recall, F1
    - AUC-ROC
    - Confusion Matrix
    ↓
Generate visualizations:
    - Confusion matrix heatmap
    - ROC curve
    - Prediction distribution
```

## 🔄 Data Flow

### Centralized Training

```
Raw Data
    ↓
[Data Loader]
    ↓
Preprocessing & Augmentation
    ↓
Batches (32 images)
    ↓
[Model]
    ↓
Predictions
    ↓
[Loss Function]
    ↓
Gradients
    ↓
[Optimizer]
    ↓
Updated Model
```

### Federated Training

```
Global Model (Server)
    ↓
Broadcast to Clients
    ↓
┌─────────┬─────────┬─────────┐
│Client 1 │Client 2 │Client N │
│         │         │         │
│Local    │Local    │Local    │
│Training │Training │Training │
│         │         │         │
│Updates  │Updates  │Updates  │
└────┬────┴────┬────┴────┬────┘
     │         │         │
     └─────────┴─────────┘
              ↓
    Aggregate Updates (FedAvg)
              ↓
    Updated Global Model
              ↓
         Next Round
```

## 🧠 Model Architecture Details

### EfficientNet-B0 Backbone

```
Parameters: ~5.3M
Input: (B, 3, 224, 224)

Stem: Conv3x3 + BN + Swish
    ↓
MBConv Blocks (7 stages):
    - Mobile Inverted Bottleneck
    - Squeeze-and-Excitation
    - Skip connections
    ↓
Head: Conv1x1 + BN + Swish + GlobalAvgPool
    ↓
Output: (B, 1280)
```

### Custom Classification Head

```
Input: (B, 1280)
    ↓
Linear(1280 → 512)
    ↓
ReLU
    ↓
Dropout(0.5)
    ↓
Linear(512 → 256)
    ↓
ReLU
    ↓
Dropout(0.5)
    ↓
Linear(256 → 2)
    ↓
Output: (B, 2) [logits for Real/Fake]
```

## 📊 Metrics and Evaluation

### Classification Metrics

**Confusion Matrix**
```
                Predicted
              Real    Fake
Actual Real    TN      FP
       Fake    FN      TP
```

**Computed Metrics**
- Accuracy = (TP + TN) / (TP + TN + FP + FN)
- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)
- F1-Score = 2 * (Precision * Recall) / (Precision + Recall)
- Specificity = TN / (TN + FP)

**ROC-AUC**
- Plot True Positive Rate vs False Positive Rate
- Area under curve (higher is better)
- Threshold-independent metric

## 🔐 Privacy in Federated Learning

### Data Privacy
```
✓ Raw data never leaves client devices
✓ Only model parameters are shared
✓ Server cannot reconstruct individual data
✗ Model updates may leak some information
```

### Differential Privacy (Optional)
```
For each parameter update:
    1. Clip gradient to bound sensitivity
    2. Add Gaussian noise: N(0, σ²)
    3. Noise scale proportional to privacy budget ε
    
Privacy-Accuracy Trade-off:
    More noise → Better privacy, Lower accuracy
    Less noise → Worse privacy, Higher accuracy
```

## 🎯 Training Strategies

### Learning Rate Scheduling

**ReduceLROnPlateau**
```
If validation loss doesn't improve for N epochs:
    lr = lr * factor (e.g., 0.5)
```

**CosineAnnealing**
```
lr(t) = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(πt/T))
```

### Early Stopping
```
Track best validation metric
If no improvement for patience epochs:
    Stop training
    Restore best model
```

## 🔧 Optimization

### Optimizers

**Adam**
- Adaptive learning rate
- Momentum + RMSprop
- Good default choice

**SGD with Momentum**
- Classic optimizer
- May need careful tuning
- Can achieve better generalization

**AdamW**
- Adam with decoupled weight decay
- Better regularization
- Recommended for transformers

## 📈 Performance Considerations

### Memory Optimization
- Gradient accumulation for large batches
- Mixed precision training (FP16)
- Gradient checkpointing

### Speed Optimization
- Data loading: Multiple workers
- Pin memory for GPU transfer
- Prefetching batches

### Distributed Training
- Data parallelism across GPUs
- Model parallelism for large models
- Federated learning for privacy

## 🔬 Research Extensions

### Possible Improvements

1. **Model Architecture**
   - Vision Transformers (ViT)
   - EfficientNet-B4/B7 (larger models)
   - Multi-modal fusion (audio + video)

2. **Federated Learning**
   - Personalized federated learning
   - Federated transfer learning
   - Byzantine-robust aggregation

3. **Privacy**
   - Secure multi-party computation
   - Homomorphic encryption
   - Trusted execution environments

4. **Data**
   - Temporal modeling for videos
   - Face alignment preprocessing
   - Synthetic data generation

## 📚 References

- **EfficientNet**: Tan & Le, 2019
- **Federated Learning**: McMahan et al., 2017
- **FedProx**: Li et al., 2020
- **Deepfake Detection**: Rossler et al., 2019 (FaceForensics++)

---

This architecture is designed to be:
- ✅ **Modular**: Easy to swap components
- ✅ **Scalable**: Works with any number of clients
- ✅ **Extensible**: Add new models and strategies
- ✅ **Privacy-preserving**: Federated learning by design
