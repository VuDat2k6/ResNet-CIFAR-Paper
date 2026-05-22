# Hướng Dẫn Thuyết Trình Chuyên Sâu Theo Báo Cáo REPORT.md
## (Tập Trung Giải Thích Các Cơ Chế Tối Ưu Hóa — Optimization)

Tài liệu này đóng vai trò là **cẩm nang đồng hành** khi bạn thuyết trình dựa trên file báo cáo tiếng Anh [REPORT.md](file:///c:/Users/ACER/Desktop/CS114/Paper/documents/REPORT.md). Dưới đây là hướng dẫn chi tiết cách thuyết minh và giải thích bằng tiếng Việt các thuật ngữ, công thức và biểu đồ tối ưu hóa (Optimize) có trong báo cáo, giúp bạn trả lời trôi chảy trước mọi câu hỏi chất vấn của Hội đồng.

---

## PHẦN 1: GIẢI THÍCH LIÊN HỆ GIỮA CÁC PHẦN TRONG REPORT.MD

### 1. Section 2.4: Modern Optimization Techniques (Trang tổng quan lý thuyết tối ưu)
*   **Điểm chính trong REPORT.md**: Đề cập đến lịch sử và lý thuyết đằng sau các thuật toán tối ưu hóa (AdamW, Cosine Warmup, Squeeze-and-Excitation, CutMix, Knowledge Distillation).
*   **Cách bạn thuyết minh bằng Tiếng Việt**:
    > *"Trong phần **Section 2.4** của báo cáo, tôi đã xây dựng cơ sở lý thuyết vững chắc bằng cách tổng hợp các công trình khoa học hiện đại nhất về tối ưu hóa mạng tích chập. Cụ thể, tôi nghiên cứu 4 nhóm tối ưu hóa: 
    > 1. Thuật toán tối ưu hóa thích ứng (**AdamW** đấu với **SGD**).
    > 2. Cơ chế chú ý kênh (**Squeeze-and-Excitation**).
    > 3. Các kỹ thuật điều hòa hóa chống quá khớp (**Stochastic Depth, CutMix**).
    > 4. Kỹ thuật nén và truyền thụ tri thức (**Knowledge Distillation**).
    > 
    > Đây chính là 'bản đồ công nghệ' làm tiền đề để tôi thiết lập các cấu hình thực nghiệm trong phần Methodology."*

---

### 2. Section 3.3 & 3.4: Hyperparameters – Task A, B, C & D (Cấu hình siêu tham số)
*   **Bảng thông số trong REPORT.md**:
    *   *Baseline/Optimized (Section 3.3)*: SGD vs AdamW, ReLU vs SiLU, Label Smoothing $\epsilon = 0.1$, Weight Decay $10^{-4}$ vs $10^{-2}$.
    *   *SE-ResNet-20 (Section 3.4)*: Cosine Warmup 5 epochs, $LR = 0.05$ (được hạ xuống để ổn định SE blocks), Stochastic Depth survival prob = 0.8, CutMix prob = 0.3.
    *   *Knowledge Distillation (Section 4.10)*: Student ResNet-20 ($272\text{K}$ params), Teacher seresnet20 ($4.36\text{M}$ params), $T = 4.0$, $\alpha = 0.6$.
*   **Cách bạn giải thích sâu về sự thay đổi siêu tham số (Optimize)**:
    > *"Thưa thầy cô, điểm mấu chốt trong **Section 3.3 & 3.4** là sự thay đổi tinh tế của các siêu tham số tối ưu hóa:
    > - Tại sao tốc độ học (LR) của mạng SE-ResNet-20 lại hạ từ `0.1` xuống `0.05`? Đó là vì các trọng số của khối Excitation (FC Layers) trong SE Block khi mới khởi tạo rất nhạy cảm. Nếu dùng LR quá lớn (`0.1`) ở những epochs đầu, dòng gradient cực mạnh sẽ làm lệch hướng tối ưu của Attention, gây mất ổn định huấn luyện.
    > - Tại sao chọn Stochastic Depth với xác suất sống sót $p_l = 0.8$? Con số này được tối ưu hóa thực nghiệm để cân bằng giữa việc bỏ qua các lớp dư thừa nhằm tăng tốc độ truyền gradient và việc giữ lại đủ chiều sâu của mạng để học đặc trưng phức tạp."*

---

## PHẦN 2: GIẢI THÍCH SỰ THẤT BẠI CỦA V1 & SỰ THÀNH CÔNG CỦA V2

### 3. Section 4.2 & 5.2: Why SiLU + AdamW + Label Smoothing Underperformed (Giải thích V1 thất bại)
*   **Dẫn chứng trong REPORT.md**:
    *   CIFAR-10: **90.28%** (Optimized V1) so với **91.93%** (Baseline) $\to$ Giảm $1.65\%$.
    *   SVHN: **95.74%** (Optimized V1) so với **96.24%** (Baseline) $\to$ Giảm $0.50\%$.
*   **Cách giải thích chi tiết cơ chế thất bại (Huấn luyện tối ưu)**:
    > *"Thầy cô có thể quan sát thấy trong **Section 4.2 & 5.2**, cấu hình tối ưu V1 (Task B) lại cho kết quả đi lùi. Tôi xin giải thích cặn kẽ 3 nguyên nhân cốt lõi dưới góc độ toán học tối ưu hóa:
    > 
    > 1. **Cơ chế cập nhật của AdamW**: AdamW tự động chia tốc độ học cho căn bậc hai của trung bình động bình phương gradient lịch sử ($\sqrt{v_t}$). Trên mạng cực hẹp như ResNet-20, khi đi về các epochs cuối, mẫu số này lớn lên làm cho LR hiệu dụng bị thu nhỏ quá mức (**Conservative LR decay**). Kết quả là mạng bị bóp chết dòng học tại các lớp đầu tiên, trong khi đó SGD với lịch MultiStep vẫn duy trì bước nhảy mạnh mẽ ở milestones 100 và 150 để 'thoát' khỏi các cực tiểu phẳng cục bộ.
    > 2. **Sự quá đà của Label Smoothing ($\epsilon=0.1$)**: Với bài toán chỉ có 10 lớp như CIFAR-10, việc phân bổ 10% xác suất mục tiêu cho các lớp sai là quá lớn. Nó làm mờ ranh giới quyết định (decision boundary) giữa các lớp vốn dĩ rất dễ phân biệt (như 'tàu thủy' vs 'xe tải'), vô tình làm giảm độ tự tin của các dự đoán chính xác.
    > 3. **Hạn chế của SiLU**: SiLU mượt hơn ReLU nhờ hàm tự cổng ($x \cdot \sigma(x)$), nhưng đối với một mạng nông chỉ có 20 tầng như ResNet-20, hiện tượng tiêu biến gradient không phải là vấn đề nghiêm trọng nhờ đã có các đường tắt (Shortcuts). Do đó, SiLU không mang lại bất kỳ ưu thế toán học nào mà chỉ làm tăng chi phí tính toán khi forward."*

---

### 4. Section 4.3 & 5.3: Why SE-ResNet-20 Achieved the Best Results (Giải thích V2 thành công)
*   **Dẫn chứng trong REPORT.md**:
    *   CIFAR-10: Đạt **93.11%** (Tốt nhất).
    *   SVHN: Đạt **96.44%** (Tốt nhất).
    *   Generalization Gap cực nhỏ: Chỉ **0.67%** trên CIFAR-10, và **âm (Test > Train)** trên SVHN.
*   **Cách giải thích chi tiết cơ chế thành công**:
    > *"Trong phần **Section 4.3 & 5.3**, tôi trình bày lý do tại sao cấu hình V2 (Task C) lại đạt đỉnh cao hiệu năng:
    > 
    > 1. **Cơ chế Attention linh hoạt của SE Block**: SE block đã bổ sung khả năng học trọng số kênh động. Mạng không còn nhìn nhận mọi kênh đặc trưng có vai trò như nhau mà tự học cách khuếch đại các kênh quan trọng và triệt tiêu các kênh nhiễu.
    > 2. **Sự kết hợp hoàn hảo giữa CutMix và Stochastic Depth**: 
    >    *   **Stochastic Depth** tạo ra một cơ chế điều hòa cấu trúc giống như Dropout, triệt tiêu sự đồng phụ thuộc của nơ-ron.
    >    *   **CutMix** ép mô hình không được tập trung vào một đặc điểm cục bộ (ví dụ: chỉ nhìn đầu con chim để đoán chim) mà phải nhận diện dựa trên các đặc trưng phân bố xung quanh. Điều này giải thích tại sao mạng có **Generalization Gap gần như bằng 0** (thậm chí âm trên SVHN), chứng tỏ mô hình học được bản chất ngữ nghĩa bất biến chứ hoàn toàn không học vẹt dữ liệu."*

---

## PHẦN 3: GIẢI THÍCH ĐỘT PHÁ CỦA KNOWLEDGE DISTILLATION (TASK D)

### 5. Section 4.10: SOTA Model Compression via Knowledge Distillation
*   **Dẫn chứng trong REPORT.md**:
    *   Student ResNet-20: 272,474 tham số (1.09 MB).
    *   Teacher seresnet20: 4,359,242 tham số (17.44 MB).
    *   **Kết quả**: Student KD đạt **93.19%**, vượt Teacher (**93.11%**) và tăng **+1.26%** so với baseline gốc.
*   **Cách giải thích cơ chế đỉnh cao này**:
    > *"Kính thưa thầy cô, **Section 4.10** mô tả phần cốt lõi và giá trị nhất của dự án: Nén tri thức thông qua Knowledge Distillation. 
    > 
    > Mục tiêu của tôi là nâng tầm mạng Student ResNet-20 siêu nhẹ (272K tham số) bằng cách chuyển giao tri thức từ Teacher SE-ResNet-20 khổng lồ (4.36M tham số, tỉ lệ nén **16 lần**).
    > 
    > Kết quả thực nghiệm cuối cùng vô cùng đột phá: **Student KD đạt 93.19%**, không chỉ vượt baseline ban đầu mà còn vượt qua chính Teacher của nó. 
    > 
    > Tại sao Student nhỏ hơn lại vượt qua được Teacher lớn? 
    > Về mặt toán học tối ưu hóa, hàm phân phối xác suất mềm (Soft Targets) của Teacher đóng vai trò là một **màng lọc nhiễu cực mạnh (super-regularizer)**. Nhãn mềm ngăn không cho Student hội tụ vào các vùng cực tiểu nhọn hẹp (sharp minima) và ép mạng Student tìm đến các thung lũng phẳng rộng lớn (**flat minima**) trong không gian trọng số. Nhờ vậy, Student nhỏ thừa hưởng toàn bộ khả năng tổng quát hóa tinh túy nhất từ Teacher mà hoàn toàn không chịu gánh nặng về số lượng tham số hay độ trễ tính toán khi triển khai thực tế."*

---

## PHẦN 4: BỘ CÂU HỎI PHẢN BIỆN GIẢ ĐỊNH & CÁCH TRẢ LỜI XUẤT SẮC

### Câu hỏi 1: Tại sao em lại chọn Nhiệt độ $T = 4.0$ và $\alpha = 0.6$ cho Knowledge Distillation? Có cơ sở khoa học nào không hay chỉ là thử ngẫu nhiên?
*   **Cách trả lời chuyên nghiệp**:
    > *"Thưa thầy cô, bộ siêu tham số $T = 4.0$ và $\alpha = 0.6$ được tôi lựa chọn dựa trên cả cơ sở lý thuyết toán học của Geoffrey Hinton và thực nghiệm kiểm chứng:
    > - **Nhiệt độ $T = 4.0$**: Trong phân phối softmax, nếu $T$ quá nhỏ (gần 1.0), phân phối xác suất mềm sẽ bị kéo sát về nhãn cứng (one-hot), làm mất đi thông tin 'Dark Knowledge' (độ tương đồng giữa các lớp sai). Ngược lại, nếu $T$ quá lớn (ví dụ $>10$), phân phối xác suất sẽ bị san phẳng hoàn toàn thành phân phối đều, khiến thông tin hữu ích bị biến thành nhiễu trắng. Khoảng giá trị $T \in [3, 5]$ là khoảng tối ưu để trích xuất cấu trúc lớp tốt nhất.
    > - **Trọng số $\alpha = 0.6$**: Con số này thể hiện sự ưu tiên nhẹ ($60\%$) cho dòng gradient từ tri thức mềm của Teacher so với nhãn cứng thực tế ($40\%$). Thực nghiệm cho thấy nếu $\alpha$ quá lớn (ví dụ $0.9$), mô hình Student dễ bị 'học vẹt' theo cả các lỗi sai ngẫu nhiên của Teacher. Nếu $\alpha$ quá nhỏ, Student lại không tận dụng được tri thức của Teacher và quay về giống baseline thông thường. Tỷ lệ $60/40$ mang lại sự cân bằng hoàn hảo giúp mô hình Student tự điều chỉnh linh hoạt."*

---

### Câu hỏi 2: Tại sao mạng SE-ResNet-20 trong Task C lại đạt hiện tượng "Negative Generalization Gap" (độ chính xác tập Test cao hơn tập Train) trên SVHN? Điều này có bình thường không?
*   **Cách trả lời chuyên nghiệp**:
    > *"Thưa thầy cô, hiện tượng 'Negative Generalization Gap' (Sai số tập Test nhỏ hơn tập Train) là hoàn toàn bình thường và là minh chứng cho thấy các kỹ thuật điều hòa hóa của tôi hoạt động cực kỳ hiệu quả:
    > 1. **Sự tác động của CutMix**: Trong quá trình huấn luyện, mô hình phải học trên các ảnh bị cắt dán ngẫu nhiên (CutMix) rất khó và có nhiều nhiễu biên. Điều này khiến độ chính xác trên tập Train bị kéo giảm xuống một cách nhân tạo (chỉ đạt 91.63%).
    > 2. **Đánh giá trên nhãn sạch**: Khi đánh giá trên tập kiểm thử (Test set), mô hình được test trên các ảnh gốc sạch sẽ, không bị cắt dán hay áp dụng Stochastic Depth. Vì mô hình đã được trui rèn cực tốt qua các thử thách khó của CutMix trong lúc train, nên khi gặp ảnh sạch dễ hơn, nó nhận diện cực kỳ chính xác và đạt tới **96.44%**.
    > 
    > Hiện tượng này chứng tỏ mô hình của tôi hoàn toàn không bị quá khớp (overfitting) mà trái lại, có khả năng tổng quát hóa cực kỳ mạnh mẽ."*

---

### Câu hỏi 3: Trong Section 5.2, em phân tích AdamW hoạt động kém hơn SGD trên ResNet-20. Vậy có phải AdamW là thuật toán lỗi thời và SGD luôn tốt hơn không?
*   **Cách trả lời chuyên nghiệp**:
    > *"Thưa thầy cô, hoàn toàn không phải AdamW lỗi thời. Mỗi optimizer sinh ra đều có một 'miền tối ưu' riêng:
    > - **AdamW** là tiêu chuẩn vàng cho các kiến trúc mạng rất lớn, nhiều tham số và phi tuyến tính cao như Transformers (ViT, BERT, LLMs). Trên các mạng lớn này, không gian tham số khổng lồ cần một bộ tối ưu hóa thích ứng nhanh để hội tụ trong thời gian thực thi cho phép.
    > - **SGD với Momentum** lại cực kỳ thích hợp cho các mạng tích chập (CNN) cổ điển loại vừa và nhỏ như gia đình ResNet trên các tập dữ liệu ảnh tiêu chuẩn. Việc cập nhật trọng số đồng đều, kết hợp với quán tính momentum giúp SGD vượt qua các nhiễu tần số cao của không gian tham số hẹp và tìm ra các điểm cực tiểu phẳng rộng lớn có tính tổng quát hóa cao.
    > 
    > Vì vậy, bài học rút ra từ Section 5.2 không phải là thuật toán nào tốt hơn, mà là **'sự tương thích giữa không gian tham số mô hình và thuật toán cập nhật'**. Đối với ResNet-20, SGD vẫn là sự lựa chọn tối ưu nhất."*
