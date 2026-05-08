# 🔬 Deep Residual Learning — Research & Optimization Report

> **Paper gốc**: *"Deep Residual Learning for Image Recognition"* — He et al., arXiv:1512.03385  
> **Mục tiêu**: Tái lập kết quả, đánh giá cross-dataset, và đề xuất cải tiến kỹ thuật có đo lường định lượng.

---

## 📋 Tổng quan nhiệm vụ

| Hạng mục | Chi tiết |
|---|---|
| **Model baseline** | ResNet-20 / ResNet-32 (CIFAR variant) |
| **Dataset chính** | CIFAR-10 |
| **Dataset mở rộng** | SVHN hoặc Fashion-MNIST |
| **Framework** | PyTorch |
| **Output cuối** | `REPORT.md` + plots (Matplotlib) |

---

## 🤖 Agent & Sub-agent Phân công

```
📦 Main Agent
├── 🗓️  planner.md            → Lập lịch training pipeline & data preprocessing
├── 🐍  python-reviewer.md    → Review code PyTorch, kiểm tra tính tối ưu
├── ⚙️  pytorch-build-resolver.md → Xử lý CUDA errors, OOM, dependency conflicts
└── 📝  doc-updater.md        → Tổng hợp kết quả vào báo cáo cuối
```

**Skills được dùng:**

- `python-testing` — Unit test cho Residual Block và SE Block
- `article-writing` — Soạn thảo báo cáo học thuật
- `verification-loop` — Xác thực độ chính xác của số liệu thực nghiệm

---

## 🗺️ Workflow chi tiết

### Bước 1 — Thiết lập môi trường & Baseline Reproduction

**Mục tiêu**: Cài đặt ResNet đúng thông số paper, train trên CIFAR-10.

```python
# Thông số chuẩn theo paper
model    = ResNet20()          # hoặc ResNet32
dataset  = CIFAR-10
optimizer = SGD(lr=0.1, momentum=0.9, weight_decay=1e-4)
scheduler = MultiStepLR(milestones=[100, 150], gamma=0.1)
epochs   = 200
batch_size = 128
```

**Checklist:**
- [ ] Implement đúng Residual Block (shortcut connection + identity mapping)
- [ ] Log `Top-1 Error` và `Training Loss` theo từng epoch
- [ ] Gọi `pytorch-build-resolver` nếu gặp lỗi CUDA / version conflict

---

### Bước 2 — Cross-Dataset Evaluation

**Dataset lựa chọn**: SVHN *(Street View House Numbers)* hoặc Fashion-MNIST

| Tiêu chí | SVHN | Fashion-MNIST |
|---|---|---|
| Kích thước ảnh | 32×32 RGB | 28×28 Grayscale |
| Số class | 10 | 10 |
| Đặc điểm | Nhiễu thực tế, phân phối lệch | Cấu trúc đơn giản hơn |
| Lý do chọn | Gần với real-world data | Baseline so sánh nhanh |

**Yêu cầu phân tích:**
- [ ] Giải thích lý do chọn dataset (dựa trên đặc tính phân phối)
- [ ] Đánh giá **Generalization Gap**: `Train Acc` vs `Test Acc`
- [ ] So sánh với kết quả CIFAR-10 baseline

---

### Bước 3 — Cải tiến mô hình (Chọn 1 trong 2 hướng)

#### 🅐 Hướng Kiến trúc
```
Thay đổi Activation: ReLU → SiLU / Mish
Hoặc: Thêm Squeeze-and-Excitation (SE) Block sau mỗi Residual Block
```

#### 🅑 Hướng Training Strategy
```
Optimizer:     SGD → AdamW (lr=1e-3, weight_decay=0.01)
Loss:          CrossEntropy → CrossEntropy + Label Smoothing (ε=0.1)
```

**Ràng buộc bắt buộc:**
- [ ] Không được phá vỡ cấu trúc Residual Learning (shortcut phải được giữ nguyên)
- [ ] Review qua `python-reviewer.md` trước khi chạy thực nghiệm

---

### Bước 4 — So sánh & Phân tích

Chạy **song song** 2 cấu hình trên cả 2 dataset:

```
┌─────────────┬───────────┬───────────┐
│             │  CIFAR-10 │  SVHN     │
├─────────────┼───────────┼───────────┤
│  Original   │  __.__%   │  __.__%   │
│  Optimized  │  __.__%   │  __.__%   │
└─────────────┴───────────┴───────────┘
         (điền sau khi có số liệu thực)
```

**Câu hỏi phân tích cần trả lời:**
1. Cải tiến có giảm Top-1 Error không? Giảm bao nhiêu % tuyệt đối?
2. Tốc độ hội tụ (epoch đạt 90% accuracy) thay đổi như thế nào?
3. Trong bối cảnh dataset mới, "Residual mapping" vs "Identity mapping" ảnh hưởng ra sao?

---

## 📄 Cấu trúc báo cáo cuối (`REPORT.md`)

```
REPORT.md
│
├── Abstract          — Tóm tắt mục tiêu và kết quả chính (≤ 200 từ)
├── 1. Introduction   — Vấn đề Vanishing Gradient & đóng góp của ResNet
├── 2. Methodology
│   ├── 2.1 Model Architecture   — Sơ đồ Residual Block
│   ├── 2.2 Hyperparameters      — Bảng đầy đủ lr, batch, decay...
│   └── 2.3 Optimization Variant — Mô tả cải tiến đã áp dụng
├── 3. Experimental Results
│   ├── 3.1 Bảng Accuracy/Error (4 cấu hình × 2 dataset)
│   └── 3.2 Đồ thị Loss/Accuracy curve (Matplotlib)
├── 4. Analysis & Discussion    ← Phần quan trọng nhất
│   ├── Lý do cải tiến hiệu quả / không hiệu quả
│   ├── Convergence speed analysis
│   └── Generalization analysis across datasets
└── 5. Conclusion     — Tính bền vững của ResNet + bài học rút ra
```

---

## ✅ Guardrails & Tiêu chuẩn

> ⚠️ **Bắt buộc tuân thủ — không được bỏ qua**

| Quy tắc | Chi tiết |
|---|---|
| **Số liệu cụ thể** | Không dùng "kết quả tốt". Phải viết: *"giảm 2.5% Top-1 error"* |
| **Code standard** | Tuân thủ `coding-standards/python` (type hints, docstrings, no magic numbers) |
| **Xác thực kết quả** | Mọi con số phải qua `verification-loop` trước khi đưa vào báo cáo |
| **Reproducibility** | Set `random_seed = 42` cho tất cả experiments |
| **Plots** | Lưu dưới dạng `.png` 300 DPI, có title, xlabel, ylabel, legend |

---

## 📌 Placeholders cần điền sau khi có số liệu

```markdown
<!-- TODO: Điền sau khi chạy xong -->
- [ ] CIFAR-10 Baseline Top-1 Error:        _____% (paper: 8.75% cho ResNet-32)
- [ ] CIFAR-10 Optimized Top-1 Error:       _____%
- [ ] SVHN Baseline Top-1 Error:            _____%
- [ ] SVHN Optimized Top-1 Error:           _____%
- [ ] Epoch hội tụ Baseline:                _____
- [ ] Epoch hội tụ Optimized:               _____
- [ ] Training time / epoch (GPU):          _____ sec
```

---

*Được thiết kế theo chuẩn `article-writing` skill — everything-claude-code framework*

Cách chạy
Bước 1: Cài đặt dependencies
pip install torch torchvision matplotlib tqdm pytest numpy
Bước 2: Chạy unit tests (xác minh model đúng)
cd c:\Users\ACER\Desktop\CS114\Paper
pytest tests/test_resnet.py -v
Bước 3: Chạy 4 experiments
# Task A: Baseline CIFAR-10 (Original)
python train_cifar10.py --epochs 200
# Task A: Cross-dataset SVHN (Original)
python train_svhn.py --epochs 200
# Task B: CIFAR-10 Optimized
python train_optimized.py --dataset cifar10 --epochs 200
# Task B: SVHN Optimized
python train_optimized.py --dataset svhn --epochs 200
Bước 4: Generate plots

python plot_comparison.py