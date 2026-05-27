# Deep Residual Learning for Image Recognition — ResNet-20 on CIFAR-10 & SVHN

This repository contains a professional PyTorch implementation of **ResNet-20** (CIFAR-variant) and a comprehensive suite of advanced optimization techniques, including **Squeeze-and-Excitation (SE) Attention**, **Stochastic Depth**, **CutMix/MixUp** regularization, and **Hinton's Knowledge Distillation (KD)**. 

Our optimized student model compressed via KD achieves **93.19%** test accuracy on CIFAR-10 while remaining extremely compact (272K parameters, 1.09 MB), surpassing even the heavy teacher model.

---

## 📈 Key Results

| Dataset | Model / Configuration | Top-1 Error | Top-1 Accuracy | Params | Size | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **CIFAR-10** | **ResNet-20 Baseline** (ReLU + SGD) | 8.07% | 91.93% | **272K** | **1.09 MB** | Completed |
| **CIFAR-10** | **ResNet-20 Optimized** (SiLU + AdamW + LS) | 9.72% | 90.28% | 272K | 1.09 MB | Completed |
| **CIFAR-10** | **SE-ResNet-20 Teacher** (V2 - SE + SD + CutMix) | 6.89% | 93.11% | 4.36M | 17.44 MB | Completed |
| **CIFAR-10** | **Student ResNet-20 + KD** (Proposed Champion) | **6.81%** | **93.19%** | **272K** | **1.09 MB** | **SOTA Nén** |
| **SVHN** | **ResNet-20 Baseline** (ReLU + SGD) | 3.76% | 96.24% | **272K** | **1.09 MB** | Completed |
| **SVHN** | **ResNet-20 Optimized** (SiLU + AdamW + LS) | 4.26% | 95.74% | 272K | 1.09 MB | Completed |
| **SVHN** | **SE-ResNet-20** (V2 - SE + SD + CutMix) | **3.56%** | **96.44%** | 4.36M | 17.44 MB | Completed |

---

## 📂 Project Structure

The repository has been restructured into a pristine, production-grade layout:

```
Paper/
├── documents/                  # Reference papers, guides, and academic reports
│   ├── 1512.03385v1.pdf        # Reference ResNet-20 paper
│   ├── REPORT.md               # Complete Research Report (English)
│   ├── SUMMARY.md              # Project summary & tasks map
│   ├── COLAB_GUIDE.md          # Guide for fast GPU execution on Google Colab/Kaggle
├── src/                        # Core PyTorch source code
│   ├── models/                 # Model architecture modules
│   │   ├── __init__.py         # Package entrypoint (exports architectures & criteria)
│   │   ├── resnet.py           # Standard ResNet-20/32 CIFAR-variant
│   │   └── se_resnet.py        # Advanced Squeeze-and-Excitation ResNet-20
│   ├── data.py                 # Data loaders + CutMix & MixUp augmentations
│   └── utils.py                # Random seed and helper utilities
├── scripts/                    # Clean executable scripts (training & analysis)
│   ├── train_baseline.py       # Trains standard ResNet-20 baseline
│   ├── train_optimized.py      # Trains optimized ResNet-20 (Task B config)
│   ├── train_se_resnet.py      # Trains heavy SE-ResNet-20 Teacher (V2 config)
│   ├── train_kd.py             # Distills Teacher into lightweight Student (V3 KD)
│   ├── train_svhn.py           # Trains baseline on SVHN dataset
│   ├── plot_comparison.py      # Generates comparison learning curves
│   ├── plot_distribution.py    # Plots class frequency distributions
│   ├── visualize_data.py       # Plots grid of dataset sample previews
│   ├── check_results.py        # Fast log metric parser
│   ├── download_datasets.py    # Standalone dataset downloader
│   └── generate_curves.py      # Reconstructs training curves from JSON logs
├── tests/                      # Validation suites
│   └── test_resnet.py          # Unit tests (gradient checks, architectures, dimensions)
├── plots/                      # Saved charts & visualization images
├── outputs/                    # Local training metrics (JSONs) and best weights (gitignored)
└── README.md                   # This file
```

---

## 🛠️ Installation & Setup

Ensure you have **Python 3.8+** and **PyTorch 2.0+** installed:

```bash
pip install torch torchvision matplotlib tqdm pytest
```

---

## 🚀 How to Run

All executable scripts are housed under `scripts/`. Always execute them from the project root.

### 1. Pre-download Datasets
```bash
python scripts/download_datasets.py
```

### 2. Train Standard ResNet-20 Baseline
```bash
python scripts/train_baseline.py --epochs 200 --batch_size 128
```

### 3. Train Optimized ResNet-20 (SiLU + AdamW + LS)
```bash
python scripts/train_optimized.py --dataset cifar10 --epochs 200 --batch_size 128
```

### 4. Train SOTA SE-ResNet-20 Teacher
```bash
python scripts/train_se_resnet.py --dataset cifar10 --epochs 200 --batch_size 128
```

### 5. Perform SOTA Knowledge Distillation (KD)
Once you have trained the SE-ResNet-20 teacher (weights saved at `outputs/cifar10_seresnet/best_model.pth`), distill its "dark knowledge" into the lightweight Student ResNet-20:
```bash
python scripts/train_kd.py --epochs 200 --batch_size 128 --alpha 0.6 --temp 4.0
```

### 6. Plot Comparisons & Visualizations
```bash
# Generate comparative training curves
python scripts/plot_comparison.py

# Visualize dataset sample grid
python scripts/visualize_data.py
```

### 7. Run Verification Unit Tests
```bash
python -m pytest
```

---

## 💡 Advanced Optimization & Compression Summary

### 1. Squeeze-and-Excitation (SE) Attention
*   Adds channel-wise attention to adaptively recalibrate feature responses.
*   Increases parameter size marginally ($<2\%$) but yields a significant boost ($+1.18\%$ on CIFAR-10).

### 2. Stochastic Depth & CutMix
*   **Stochastic Depth** randomly drops residual blocks during training, serving as a powerful structural regularizer.
*   **CutMix** blends random bounding-box sections of images and their labels, smoothing decision boundaries and fully eliminating overfitting.
*   Achieved a **Negative Generalization Gap** on SVHN (test accuracy **96.44%** > train accuracy **91.63%**).

### 3. Knowledge Distillation (KD)
*   Formulates a combined loss function mapping both hard targets (labels) and soft targets (teacher logits scaled by temperature $T=4.0$):
    $$L_{total} = (1 - \alpha) L_{CE}(S(x), y) + \alpha T^2 L_{KL}( \sigma(S(x)/T), \sigma(T(x)/T) )$$
*   **Student KD hits 93.19%** test accuracy — outperforming the original teacher model (**93.11%**) and baseline student (**91.93%**) while staying **16x smaller** during inference (272K params vs 4.36M params)!

---

## 📄 Academic Documentation

Detailed academic descriptions, slide scripts, and mathematical explanations can be found in the following documents:
*   [REPORT.md (English Research Report)](documents/REPORT.md)
*   [SUMMARY.md (Project Summary Map)](documents/SUMMARY.md)
*   [COLAB_GUIDE.md (Colab Execution Guide)](documents/COLAB_GUIDE.md)
