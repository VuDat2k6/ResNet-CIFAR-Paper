# ResNet-20: Deep Residual Learning for Image Recognition

Implementation of **ResNet-20** (CIFAR-variant) from scratch using PyTorch, evaluated on **CIFAR-10** and **SVHN** with two training configurations: original (ReLU + SGD) and optimized (SiLU + AdamW + Label Smoothing).

## Key Results

| Dataset  | Config                    | Top-1 Error | Top-1 Accuracy |
|----------|---------------------------|-------------|----------------|
| CIFAR-10 | Original (ReLU + SGD)    | 14.24%      | 85.76%         |
| CIFAR-10 | Optimized (SiLU+AdamW+LS) | **11.74%**  | **88.26%**     |
| SVHN     | Original (ReLU + SGD)     | 5.49%       | 94.51%         |
| SVHN     | Optimized (SiLU+AdamW+LS) | **4.17%**  | **95.83%**     |

## Project Structure

```
Paper/
├── src/
│   ├── __init__.py
│   ├── resnet.py          # ResNet-20/32 implementation + LabelSmoothingCrossEntropy
│   ├── data.py            # CIFAR-10 & SVHN data loaders
│   └── utils.py           # Seed utilities
├── tests/
│   └── test_resnet.py     # Unit tests
├── outputs/               # Training results (JSON + plots)
├── plots/                 # Comparison plots
├── data/                  # Dataset storage (gitignored)
├── train_cifar10.py       # Train CIFAR-10 (original config)
├── train_optimized.py     # Train CIFAR-10 & SVHN (optimized config)
├── train_svhn.py          # Train SVHN (original config)
├── plot_comparison.py     # Generate comparison plots
├── REPORT.md             # Report in English
├── REPORT_VIETNAMESE.md  # Report in Vietnamese (detailed)
└── RESNET_TASK.md        # Assignment description
```

## How to Run

### 1. Install dependencies

```bash
pip install torch torchvision matplotlib tqdm
```

### 2. Train CIFAR-10 (Original Config)

```bash
python train_cifar10.py --epochs 200
```

### 3. Train CIFAR-10 (Optimized Config)

```bash
python train_optimized.py --dataset cifar10 --epochs 200
```

### 4. Train SVHN

**Original config:**
```bash
python train_svhn.py --epochs 200
```

**Optimized config:**
```bash
python train_optimized.py --dataset svhn --epochs 200
```

### 5. Generate Comparison Plots

```bash
python plot_comparison.py
```

## ResNet-20 Architecture (CIFAR-variant)

```
Input (3×32×32)
├── conv1 (3→16, 3×3)
├── layer1: 16→16, 3 residual blocks
├── layer2: 16→32, 3 residual blocks (stride=2)
├── layer3: 32→64, 3 residual blocks (stride=2)
├── AdaptiveAvgPool2d(1,1)
└── fc (64→10)

Total params: ~270K
```

## Optimization Techniques

| Technique              | Description                                                      |
|------------------------|------------------------------------------------------------------|
| **SiLU (Swish)**        | Smooth, self-gated activation function, avoids dying neurons     |
| **AdamW**               | Decoupled weight decay for better regularization                 |
| **Label Smoothing**     | ε=0.1, prevents overconfidence, improves generalization         |
| **CosineAnnealingLR**   | Smooth learning rate decay following cosine curve                |
| **Data Augmentation**   | RandomCrop + HorizontalFlip                                      |

## Dataset

| Dataset   | Train  | Test   | Size   | Classes |
|-----------|--------|--------|--------|---------|
| CIFAR-10  | 50,000 | 10,000 | 32×32  | 10      |
| SVHN      | 73,257 | 26,032 | 32×32  | 10      |

## Requirements

- Python 3.8+
- PyTorch 2.0+
- torchvision
- matplotlib
- tqdm

---

Detailed report: [REPORT_VIETNAMESE.md](REPORT_VIETNAMESE.md)
