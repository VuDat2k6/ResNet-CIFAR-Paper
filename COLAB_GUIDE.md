# Hướng dẫn chạy thực nghiệm Knowledge Distillation trên Google Colab

Để huấn luyện mô hình **Student ResNet-20** bằng kỹ thuật **Knowledge Distillation (KD)** với 200 epochs trong thời gian cực nhanh, bạn nên tận dụng GPU miễn phí (như T4 GPU) của Google Colab. Dưới đây là hướng dẫn từng bước:

---

## Bước 1: Chuẩn bị Repository trên GitHub
Do bạn đã cấu hình Git trong project (`VuDat2k6/ResNet-CIFAR-Paper`), hãy lưu lại các thay đổi mới nhất (bao gồm cả file `train_kd.py` vừa tạo) và push lên GitHub:

```bash
git add .
git commit -m "Add Knowledge Distillation (KD) experiment code and configs"
git push
```

---

## Bước 2: Khởi tạo Google Colab
1. Truy cập vào [Google Colab](https://colab.research.google.com/).
2. Chọn **New Notebook** (Sổ tay mới).
3. Đổi tên sổ tay thành `ResNet20_KD_CIFAR10.ipynb`.
4. Vào menu **Runtime** (Thời gian chạy) $\to$ **Change runtime type** (Thay đổi loại thời gian chạy) $\to$ Chọn **T4 GPU** (hoặc L4/A100 nếu có Colab Pro) $\to$ Nhấn **Save**.

---

## Bước 3: Thiết lập Môi trường và Tải Code trên Colab
Tạo một ô code mới (code cell) trong Colab và chạy các lệnh sau để clone repository và chuẩn bị dữ liệu:

```python
# 1. Clone code từ GitHub của bạn
!git clone https://github.com/VuDat2k6/ResNet-CIFAR-Paper.git
%cd ResNet-CIFAR-Paper

# 2. Xác minh CUDA hoạt động tốt trên GPU của Colab
import torch
print("CUDA Available:", torch.cuda.is_available())
print("GPU Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
```

---

## Bước 4: Chạy huấn luyện chưng cất tri thức (Knowledge Distillation)
Trong repo của bạn đã chứa sẵn checkpoint của mô hình **Teacher** (`outputs/cifar10_seresnet/best_model.pth`). Khi chạy file `train_kd.py`, chương trình sẽ tự động nạp mô hình Teacher này làm người hướng dẫn cho Student.

Tạo một ô code mới và chạy lệnh huấn luyện **200 epochs**:

```python
!python train_kd.py --epochs 200 --batch_size 128 --lr 0.1 --alpha 0.6 --temp 4.0
```

> [!NOTE]
> *   Trên Colab T4 GPU, mỗi epoch chỉ mất khoảng **5 đến 8 giây** (so với khoảng 108 giây của GTX 1650 cục bộ). 
> *   Tổng thời gian huấn luyện 200 epochs trên Colab sẽ chỉ mất khoảng **15 - 25 phút**!

---

## Bước 5: Xem kết quả và Vẽ biểu đồ so sánh
Sau khi huấn luyện xong, kết quả và biểu đồ đường cong huấn luyện sẽ được lưu tại thư mục `outputs/cifar10_kd/`. 
Bạn có thể xem trực tiếp biểu đồ so sánh bằng cách chạy lệnh sau trong Colab:

```python
from IPython.display import Image, display
display(Image("outputs/cifar10_kd/training_curves.png"))
```

Sau khi hoàn tất, bạn có thể tải toàn bộ thư mục kết quả `outputs/cifar10_kd` về máy tính hoặc commit ngược trở lại GitHub để cập nhật báo cáo LaTeX của bạn.
