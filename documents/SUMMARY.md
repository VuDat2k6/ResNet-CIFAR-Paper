# Project Summary — ResNet CIFAR Paper

**Date**: Saturday, May 9, 2026 (updated Wednesday May 13, 2026)
**Project**: Deep Residual Learning for Image Recognition — ResNet on CIFAR-10 & SVHN

---

## Initial Prompt

> **Goal**: Reproduce results of the paper *"Deep Residual Learning for Image Recognition"* (He et al., arXiv:1512.03385), evaluate cross-dataset performance, and propose technical optimization methods with quantitative measurements.
>
> - Baseline Model: ResNet-20 / ResNet-32 (CIFAR variant)
> - Main Dataset: CIFAR-10
> - Extended Dataset: SVHN
> - Framework: PyTorch
> - Output: `REPORT.md` + plots (Matplotlib)
>
> **Task A**: Cross-Dataset Evaluation — train on CIFAR-10, evaluate on SVHN
> **Task B**: Model Optimization — modify activation (ReLU → SiLU), optimizer (SGD → AdamW), add Label Smoothing

---

## Summary of Completed Work

### 1. Original Version (V1)

**Code**: Implemented ResNet-20 (CIFAR-variant) from scratch using PyTorch
- `src/models/resnet.py` — Residual Block + LabelSmoothingCrossEntropy
- `src/data.py` — CIFAR-10 & SVHN data loaders
- `scripts/train_baseline.py` — Original CIFAR-10 training script (ReLU + SGD + MultiStepLR)
- `scripts/train_svhn.py` — Original SVHN training script
- `scripts/train_optimized.py` — Optimized training script (SiLU + AdamW + Label Smoothing + CosineAnnealingLR)
- `scripts/plot_comparison.py` — Comparison plotting script

**Training Results (200 epochs)**:

| Dataset | Config | Top-1 Error | Top-1 Accuracy |
|---------|--------|-------------|----------------|
| CIFAR-10 | Original (ReLU + SGD) | **8.07%** | 91.93% |
| CIFAR-10 | Optimized (SiLU+AdamW+LS) | 9.72% | 90.28% |
| SVHN | Original (ReLU + SGD) | **3.76%** | 96.24% |
| SVHN | Optimized (SiLU+AdamW+LS) | 4.26% | 95.74% |

### 2. Version V2: SE-ResNet + Stochastic Depth + CutMix + Cosine Warmup

**Derived from the following papers**:

| Technique | Paper | Improvements |
|-----------|-------|-----------|
| **SE Block** | SE-Net (Hu et al., CVPR 2018) | Channel attention, +1-2% accuracy |
| **Stochastic Depth** | Huang et al., ECCV 2016 | Reduces overfitting, faster training |
| **CutMix** | Yun et al., ICCV 2019 | Better regularization + localization than Mixup |
| **Cosine Warmup** | ResNet-RS (Bello et al., 2021) | Stabilizes training, +3.2% top-1 |
| **Pre-activation** | He et al., 2016 | Better than post-activation |
| **Label Smoothing** | Szegedy et al., 2016 | Prevents overconfidence |

**New V2 Files**:

| File | Description |
|------|--------|
| `src/models/se_resnet.py` | SE-ResNet + Stochastic Depth + LabelSmoothing + CutMixCriterion |
| `src/data.py` (updated) | Added CutMix and MixUp augmentations |
| `scripts/train_se_resnet.py` | Training script with cosine warmup |

**Model Variants**:

| Model | Channels | Params | Suitability |
|-------|----------|--------|-------------|
| `seresnet20` (1x) | 64-128-256 | ~4.4M | GPU 4GB |
| `seresnet20_wide` (2x) | 128-256-512 | ~17.4M | GPU 8GB+ |
| `seresnet20_wide` (3x) | 192-384-768 | ~39.2M | GPU 12GB+ |

**How to Run V2**:

```bash
# Lightweight SE-ResNet-20 (1x, ~4.4M params)
python scripts/train_se_resnet.py --dataset cifar10 --epochs 200

# Wide SE-ResNet-20 (2x, ~17M params)
python scripts/train_se_resnet.py --dataset cifar10 --wide --epochs 200

# Modify activation
python scripts/train_se_resnet.py --dataset cifar10 --activation silu --epochs 200
python scripts/train_se_resnet.py --dataset cifar10 --activation relu --epochs 200

# Modify stochastic depth
python scripts/train_se_resnet.py --dataset cifar10 --sd_prob 0.5 --epochs 200

# Modify CutMix probability
python scripts/train_se_resnet.py --dataset cifar10 --cutmix_prob 0.3 --epochs 200
```

**Differences between V1 and V2**:

| Feature | V1 (SiLU+AdamW) | V2 (SE+SD+CutMix) |
|-----------|------------------|--------------------|
| Architecture | Original ResNet-20 | SE-ResNet-20 + WRN-style width |
| Attention | SiLU (channel-wide) | SE Block (per-channel) |
| Regularization | Label Smoothing + Augmentation | + Stochastic Depth + CutMix |
| Scheduler | CosineAnnealingLR | CosineAnnealingLR + 5-epoch Warmup |
| Width | 64-128-256 (fixed) | Scalable: 1x, 2x, 3x |

### 3. Generated Reports

- `documents/REPORT.md` — Report in English
- `documents/REPORT_VIETNAMESE.md` — Detailed report in Vietnamese
- `documents/SUMMARY.md` — Project Summary (English)
- Folder `plots/` — Comparative plots

### 4. Project Files Map

| File | Description |
|------|--------|
| `src/models/resnet.py` | Original ResNet-20/32 + LabelSmoothingCrossEntropy |
| `src/models/se_resnet.py` | SE-ResNet + Stochastic Depth (V2) |
| `src/data.py` | Data loaders + CutMix/MixUp |
| `src/utils.py` | Seed utilities |
| `tests/test_resnet.py` | Unit tests |
| `scripts/train_baseline.py` | Training V1: Original CIFAR-10 baseline |
| `scripts/train_svhn.py` | Training V1: Original SVHN baseline |
| `scripts/train_optimized.py` | Training V1: Optimized configuration |
| `scripts/train_se_resnet.py` | Training V2: SE-ResNet + SD + CutMix |
| `scripts/train_kd.py` | Training V3: Knowledge Distillation training |
| `scripts/plot_comparison.py` | Comparison plotting script |

---

## Git Operations

```bash
# Add & commit all changes
git add .
git commit -m "Refactor repository structure and clean up files"
git push
```

---

*Generated on: Saturday, May 9, 2026 (updated Wednesday May 13, 2026)*
