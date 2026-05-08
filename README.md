# ResNet-20: Deep Residual Learning for Image Recognition

Triển khai mạng **ResNet-20** (phiên bản CIFAR) từ đầu bằng PyTorch, đánh giá trên **CIFAR-10** và **SVHN** với hai cấu hình huấn luyện: gốc (ReLU + SGD) và tối ưu (SiLU + AdamW + Label Smoothing).

## Kết quả nổi bật

| Dataset | Config | Top-1 Error | Top-1 Accuracy |
|---|---|---|---|
| CIFAR-10 | Original (ReLU+SGD) | 14.24% | 85.76% |
| CIFAR-10 | Optimized (SiLU+AdamW+LS) | **11.74%** | **88.26%** |
| SVHN | Original (ReLU+SGD) | 5.49% | 94.51% |
| SVHN | Optimized (SiLU+AdamW+LS) | **4.17%** | **95.83%** |

## Cấu trúc project

```
Paper/
├── src/
│   ├── __init__.py
│   ├── resnet.py        # ResNet-20/32 implementation + LabelSmoothingCrossEntropy
│   ├── data.py          # CIFAR-10 & SVHN data loaders
│   └── utils.py         # Seed utilities
├── tests/
│   └── test_resnet.py   # Unit tests
├── outputs/             # Training results (JSON + plots)
├── plots/               # Comparison plots
├── data/                # Dataset storage (gitignored)
├── train_cifar10.py     # Huấn luyện CIFAR-10 gốc
├── train_optimized.py   # Huấn luyện CIFAR-10 & SVHN tối ưu
├── train_svhn.py        # Huấn luyện SVHN gốc
├── plot_comparison.py   # Tạo biểu đồ so sánh
├── REPORT.md            # Báo cáo tiếng Anh
├── REPORT_VIETNAMESE.md # Báo cáo tiếng Việt (chi tiết)
└── RESNET_TASK.md       # Đề bài
```

## Cách chạy

### 1. Cài đặt dependencies

```bash
pip install torch torchvision matplotlib tqdm
```

### 2. Huấn luyện CIFAR-10 (cấu hình gốc)

```bash
python train_cifar10.py --epochs 40
```

### 3. Huấn luyện CIFAR-10 (cấu hình tối ưu)

```bash
python train_optimized.py --dataset cifar10 --epochs 40
```

### 4. Huấn luyện SVHN

```bash
python train_svhn.py --epochs 40
python train_optimized.py --dataset svhn --epochs 40
```

### 5. Tạo biểu đồ so sánh

```bash
python plot_comparison.py
```

## Kiến trúc ResNet-20 (CIFAR-variant)

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

## Các kỹ thuật tối ưu hóa

- **SiLU (Swish)**: Hàm kích hoạt mượt mà, self-gated, tránh dying neurons
- **AdamW**: Decoupled weight decay, regularization hiệu quả hơn
- **Label Smoothing (ε=0.1)**: Ngăn overconfidence, cải thiện generalization
- **CosineAnnealingLR**: Giảm LR mượt mà theo dạng cosine
- **Data Augmentation**: RandomCrop + HorizontalFlip

## Dataset

- **CIFAR-10**: 50K train / 10K test, 32×32 RGB, 10 lớp
- **SVHN**: 73K train / 26K test, 32×32 RGB, 10 lớp (chữ số 0-9)

## Yêu cầu

- Python 3.8+
- PyTorch 2.0+
- torchvision
- matplotlib
- tqdm

---

Báo cáo chi tiết: [REPORT_VIETNAMESE.md](REPORT_VIETNAMESE.md)
