# Kịch Bản Thuyết Trình Chuyên Sâu: Tối Ưu Hóa & Nén Mô Hình ResNet-20 Trên CIFAR-10 & SVHN

Tài liệu này tập trung làm rõ và mở rộng tối đa phần **TỐI ƯU HÓA (OPTIMIZATION)** trong bài báo cáo của bạn. Kịch bản được thiết kế với các thuật ngữ chuyên sâu, công thức toán học và lập luận logic chặt chẽ để thuyết phục hội đồng chuyên môn.

---

## TỔNG QUAN PHÂN BỔ TRỌNG TÂM THUYẾT TRÌNH (TẬP TRUNG VÀO OPTIMIZATION)
*   **Phần 1: Giới thiệu & Khảo sát Baseline** (15% thời lượng)
*   **Phần 2: Thất bại của Chiến lược Tối ưu V1 (Task B - Phân tích Bản chất tối ưu hóa)** (25% thời lượng) $\to$ *Tập trung giải thích sự không tương thích giữa cấu trúc hẹp và các thuật toán hội tụ nhanh.*
*   **Phần 3: Cải tiến Kiến trúc V2 (Task C - Sự kết hợp giữa Attention & Regularization)** (30% thời lượng) $\to$ *Mổ xẻ toán học của Squeeze-and-Excitation, Stochastic Depth và hiện tượng tổng quát hóa ngược (Negative Generalization Gap) của CutMix.*
*   **Phần 4: Đột phá Nén mô hình bằng Knowledge Distillation (Task D)** (30% thời lượng) $\to$ *Phân tích chi tiết phương trình loss của Hinton, giải mã "Dark Knowledge" và lý do Student vượt qua Teacher.*

---

## CHI TIẾT KỊCH BẢN THUYẾT TRÌNH TẬP TRUNG VÀO OPTIMIZE

### Slide 1: Giới Thiệu Đề Tài & Trọng Tâm Báo Cáo
*   **Nội dung Slide**:
    *   **Tiêu đề**: Nghiên cứu Tái thiết, Tối ưu hóa nâng cao và Nén Mô hình ResNet-20 trên bộ dữ liệu CIFAR-10 & SVHN.
    *   **Trọng tâm báo cáo**: Phân tích thực nghiệm các kỹ thuật tối ưu hóa từ mức độ Giải thuật (Optimizers), mức độ Kiến trúc (Attention Blocks), mức độ Điều hòa hóa (Regularization) cho đến mức độ Chuyển giao tri thức (Knowledge Distillation).
*   **Lời thoại thuyết trình**:
    > *"Kính chào thầy cô và các bạn. Tôi tên là [Tên của bạn]. Hôm nay tôi xin báo cáo đề tài nghiên cứu về ResNet-20. 
    >
    > Điểm khác biệt lớn nhất của nghiên cứu này là không dừng lại ở việc tái thiết mô hình nguyên bản, mà tập trung nghiên cứu sâu sắc vào **quá trình tối ưu hóa mạng nơ-ron (Optimization Pathways)**. Tôi sẽ làm rõ từ các thử nghiệm thất bại của hệ tối ưu hiện đại, đến việc thiết kế lại kiến trúc mạng chú ý, và cuối cùng là giải pháp nén mô hình thông qua Chuyển giao Tri thức (Knowledge Distillation). 
    >
    > Trọng tâm bài thuyết trình này sẽ mổ xẻ sâu về mặt toán học và trực giác kỹ thuật của các giải pháp tối ưu hóa mà tôi đã áp dụng."*

---

### Slide 2: Khảo Sát Baseline & Thách Thức của ResNet-20 gốc
*   **Nội dung Slide**:
    *   **Đặc trưng ResNet-20 (CIFAR variant)**: 272K tham số, 41M FLOPs.
    *   **Cấu trúc**: 3 giai đoạn tích chập với kích thước kênh cực hẹp ($16 \to 32 \to 64$).
    *   **Hạn chế cốt lõi của tối ưu hóa baseline**:
        *   *Nút thắt dung lượng biểu diễn (Representation Capacity)*: Số kênh hẹp làm giới hạn không gian trích xuất đặc trưng semantic.
        *   *Hiện tượng quá khớp (Overfitting)*: Mạng nhanh chóng đạt 99%+ độ chính xác trên tập train nhưng bị bão hòa sớm ở mức 91.93% trên tập test CIFAR-10 (Generalization gap ~7.95%).
*   **Lời thoại thuyết trình**:
    > *"Để bắt đầu phần tối ưu hóa, trước hết chúng ta cần hiểu rõ 'đối tượng nghiên cứu' - mô hình ResNet-20 gốc. Đây là một mạng tích chập rất gọn nhẹ với 272 nghìn tham số. Trong quá trình chạy thử nghiệm baseline bằng thuật toán SGD truyền thống, tôi phát hiện ra một thách thức lớn trong tối ưu hóa: 
    >
    > Mạng rất dễ đạt trạng thái bão hòa độ chính xác trên tập huấn luyện (gần 99%) nhưng trên tập kiểm thử chỉ đạt 91.93%. Khoảng cách tổng quát hóa (Generalization Gap) lên tới gần 8%. Nguyên nhân là do cấu trúc kênh cực kỳ hẹp (tối đa chỉ 64 kênh ở Stage cuối) làm hạn chế không gian biểu diễn đặc trưng của mạng, khiến mạng có xu hướng học vẹt các nhiễu tần số cao của tập train. Đây chính là động lực để tôi thực hiện các bước tối ưu hóa tiếp theo."*

---

### Slide 3: Chiến Lược Tối Ưu V1 (Task B) – Giải Mã Thất Bại Thực Nghiệm (Bài Học Học Thuật Rất Sâu)
*   **Nội dung Slide**:
    *   **Đề xuất tối ưu hóa ban đầu**:
        *   *ReLU $\to$ SiLU*: $f(x) = x \cdot \sigma(x)$. Hàm mượt hơn, đạo hàm khác 0 tại mọi điểm giúp ổn định dòng gradient.
        *   *SGD $\to$ AdamW*: Tối ưu hóa thích ứng tích hợp suy hao trọng số (decoupled weight decay).
        *   *Label Smoothing ($\epsilon=0.1$)*: Điều chỉnh nhãn mục tiêu từ nhãn cứng $[0, 1]$ sang nhãn mềm $[0.05, 0.95]$ để chống quá tự tin.
    *   **Kết quả nghịch lý**: Độ chính xác CIFAR-10 giảm từ **91.93% xuống còn 90.28%** (Lỗi tăng từ 8.07% lên 9.72%).
    *   **Phân tích sâu về không gian tối ưu (Loss Landscape)**:
        *   AdamW điều chỉnh tốc độ học thích ứng dựa trên bình phương gradient lịch sử. Tuy nhiên, trên mạng siêu nhỏ như ResNet-20, không gian tham số quá hẹp. AdamW dễ tạo ra các bước cập nhật trọng số quá lớn tại các vùng biên, làm nơ-ron rơi vào các vùng cực tiểu nhọn (sharp minima) thay vì các thung lũng phẳng (flat minima).
        *   Khi không đi kèm các kỹ thuật tăng cường dữ liệu mạnh, AdamW làm trầm trọng hơn hiện tượng overfitting trên tập dữ liệu nhỏ như CIFAR-10.
*   **Lời thoại thuyết trình**:
    > *"Ở giai đoạn tối ưu hóa đầu tiên (Nhiệm vụ B), tôi đã thử nghiệm một tổ hợp kỹ thuật rất phổ biến trong các mạng nơ-ron hiện đại: thay thế ReLU bằng SiLU để lấy hàm kích hoạt mượt hơn, chuyển SGD sang AdamW để tăng tốc độ hội tụ thích ứng, và thêm Label Smoothing để chống overfit. 
    >
    > Kết quả thực nghiệm trả về lại giảm sút: Độ chính xác giảm xuống còn 90.28%. 
    >
    > Phân tích sâu về lý thuyết tối ưu hóa, tôi rút ra kết luận quan trọng: Thuật toán AdamW tính toán tốc độ học riêng cho từng trọng số dựa trên các mô-men động lượng lịch sử. Đối với các kiến trúc mạng siêu nhỏ và hẹp như ResNet-20, không gian hàm mất mát (Loss Landscape) chứa đầy các cực tiểu cục bộ rất dốc và nhọn (sharp minima). AdamW với các bước học thích ứng nhanh dễ đẩy mạng rơi vào các hố cực tiểu nhọn này. Trên tập huấn luyện thì sai số rất nhỏ, nhưng khả năng tổng quát hóa trên tập kiểm thử lại cực kỳ tệ. Ngược lại, SGD với momentum tạo ra một động lượng quán tính lớn, giúp mô hình lướt qua các cực tiểu nhọn để tìm đến các vùng thung lũng phẳng rộng lớn (flat minima) có tính tổng quát hóa cao hơn. Đây là một phát hiện thực nghiệm vô cùng quý giá."*

---

### Slide 4: Kiến Trúc Tối Ưu V2 (Task C) – Tích Hợp Chú Ý Kênh Squeeze-and-Excitation (SE Block)
*   **Nội dung Slide**:
    *   **Mổ xẻ toán học của SE Block**:
        1.  *Squeeze (Nén không gian)*: Tính toán vector đặc trưng kênh bằng Global Average Pooling:
            $$z_c = F_{sq}(u_c) = \frac{1}{H \times W} \sum_{i=1}^{H} \sum_{j=1}^{W} u_c(i, j)$$
        2.  *Excitation (Kích hoạt phi tuyến)*: Học mối quan hệ phụ thuộc giữa các kênh thông qua 2 tầng Tuyến tính (Linear) với hệ số giảm chiều $r=16$:
            $$s = F_{ex}(z, W) = \sigma(W_2 \cdot \text{ReLU}(W_1 \cdot z))$$
        3.  *Scale (Tái cấu trúc)*: Nhân ánh xạ kênh với đặc trưng gốc để hiệu chỉnh độ lớn kích hoạt:
            $$\tilde{x}_c = F_{scale}(u_c, s_c) = s_c \cdot u_c$$
    *   **Ý nghĩa tối ưu**: SE Block chỉ tiêu tốn thêm **dưới 2% lượng tham số** nhưng giúp mạng tự động học cách 'lọc' thông tin nhiễu, tập trung vào các kênh mang tính semantic cao.
*   **Lời thoại thuyết trình**:
    > *"Từ bài học tối ưu hóa giải thuật, tôi chuyển sang tối ưu hóa cấu trúc mạng ở Nhiệm vụ C thông qua cơ chế chú ý kênh Squeeze-and-Excitation (SE Block) kết hợp việc mở rộng nhẹ số kênh để làm mạng Teacher. 
    >
    > Hãy nhìn vào toán học của khối SE Block. Đầu tiên là bước **Squeeze**, chúng ta sử dụng phép Global Average Pooling để nén toàn bộ thông tin không gian ảnh $32 \times 32$ của mỗi kênh thành một giá trị thực duy nhất. Tiếp theo là bước **Excitation**, vector này được đi qua hai lớp Tuyến tính với một nút thắt cổ chai giảm chiều (reduction ratio = 16) và kích hoạt bằng ReLU và Sigmoid để học mối quan hệ phi tuyến giữa các kênh. Cuối cùng, chúng ta **Scale** lại đặc trưng ban đầu bằng cách nhân trực tiếp với trọng số kênh vừa học được.
    >
    > Trực giác kỹ thuật ở đây là: mạng sẽ tự động biết được kênh nào đang chứa các đường nét quan trọng của con vật/vật thể để nhân hệ số lớn lên, và kênh nào chỉ chứa nhiễu hoặc nền để dập tắt đi. Khối SE Block này hoạt động cực kỳ hiệu quả, nó chỉ tốn thêm chưa đầy 2% lượng tham số nhưng lại tăng cường đáng kể năng lực biểu diễn của mô hình."*

---

### Slide 5: Tối Ưu Hóa Điều Hòa V2 (Task C) – Stochastic Depth & Tăng Cường Dữ Liệu CutMix
*   **Nội dung Slide**:
    *   **Stochastic Depth (Độ sâu ngẫu nhiên)**:
        *   *Nguyên lý*: Tắt ngẫu nhiên một số khối Residual Block trong quá trình train với xác suất sống sót $p_l$ giảm dần theo độ sâu mạng:
            $$x_{l+1} = \text{ReLU}(x_l + b_l \cdot F(x_l))$$
            Trong đó $b_l \in \{0, 1\}$ là biến ngẫu nhiên Bernoulli.
        *   *Ý nghĩa*: Tạo ra sự đa dạng cấu trúc (ensemble) của hàng ngàn mạng con nông hơn trong lúc huấn luyện, triệt tiêu hiện tượng đồng thích ứng (co-adaptation) của các bộ lọc.
    *   **Tăng cường CutMix**:
        *   Trộn 2 ảnh ngẫu nhiên theo một vùng chữ nhật $\mathbf{M}$:
            $$\tilde{x} = \mathbf{M} \odot x_i + (\mathbf{1} - \mathbf{M}) \odot x_j$$
        *   *Sự tổng quát hóa ngược (Negative Generalization Gap) trên SVHN*: Đạt **96.44%** test accuracy trong khi train accuracy chỉ đạt **91.63%**! Hiện tượng cực kỳ hiếm gặp này chứng minh CutMix đã ép mô hình học đặc trưng tổng quát xuất sắc, triệt tiêu hoàn toàn overfitting.
*   **Lời thoại thuyết trình**:
    > *"Bên cạnh Attention, tôi áp dụng hai kỹ thuật tối ưu hóa điều hòa (Regularization) vô cùng mạnh mẽ là Stochastic Depth và CutMix.
    >
    > Với **Stochastic Depth**, trong quá trình huấn luyện, tôi tắt ngẫu nhiên các khối Residual Block dựa trên phân phối Bernoulli. Việc này ép mạng phải tự tìm ra các con đường truyền gradient tối ưu khác nhau, hoạt động giống như một cơ chế học tập tập thể (ensemble learning) của hàng ngàn mạng nơ-ron nông hơn. Nhờ đó, mạng không bị phụ thuộc vào bất kỳ một liên kết cố định nào.
    >
    > Kết hợp với **CutMix**, chúng ta cắt một vùng chữ nhật của ảnh này đè lên ảnh kia và trộn nhãn theo tỷ lệ diện tích. Việc này giải quyết triệt để điểm yếu của mạng CNN thông thường là chỉ tập trung nhận diện một phần nhỏ trung tâm của đối tượng. 
    >
    > Kết quả thực nghiệm trên SVHN mang lại một hiện tượng vô cùng thú vị và hiếm gặp trong nghiên cứu AI: **Độ chính xác trên tập kiểm thử đạt 96.44% trong khi tập huấn luyện chỉ đạt 91.63%** - tức là có một khoảng cách tổng quát hóa âm (Negative Generalization Gap). Điều này khẳng định cơ chế điều hòa hóa của chúng ta hoạt động cực tốt, giúp mạng học được bản chất thực sự của ký tự số mà hoàn toàn không bị học vẹt nhãn huấn luyện."*

---

### Slide 6: Giải Pháp Đột Phá: Tối Ưu Hóa Chuyển Giao Tri Thức (Knowledge Distillation)
*   **Nội dung Slide**:
    *   **Mổ xẻ toán học của Hinton's KD Loss**:
        $$L_{\text{total}} = (1 - \alpha) L_{\text{CE}}(y, \sigma(z_s)) + \alpha T^2 D_{\text{KL}}\left(\sigma\left(\frac{z_t}{T}\right) \middle\| \sigma\left(\frac{z_s}{T}\right)\right)$$
    *   **Giải mã các biến số tối ưu**:
        *   $z_s, z_t$: Lớp logit chưa qua kích hoạt của Student và Teacher.
        *   $T = 4.0$ (Temperature): Biến số kiểm soát độ mịn của phân phối xác suất. Nếu $T$ lớn, phân phối xác suất sẽ mượt hơn, làm lộ ra các mối quan hệ tương đồng giữa các lớp (Dark Knowledge).
        *   $\alpha = 0.6$: Trọng số ưu tiên học từ Teacher hơn học từ nhãn cứng.
        *   Nhân với $T^2$: Bù đắp lại độ giảm của dòng gradient khi chia logits cho $T$.
*   **Lời thoại thuyết trình**:
    > *"Bây giờ, chúng ta đi đến phần đột phá nhất của đề tài - **Knowledge Distillation (Nén tri thức)** ở Nhiệm vụ D. Mục tiêu của tôi là truyền toàn bộ tri thức tối ưu của mô hình Teacher rộng lớn (SE-ResNet-20, 4.36 triệu tham số) vào mạng Student ResNet-20 gọn nhẹ ban đầu (chỉ 272 nghìn tham số). 
    >
    > Hãy phân tích toán học của hàm mất mát KD. Nó gồm hai thành phần được cân bằng bởi trọng số $\alpha = 0.6$:
    >
    > Thành phần thứ nhất là sai số Cross-Entropy truyền thống giữa dự đoán của Student và nhãn thực tế $y$. 
    >
    > Thành phần thứ hai - và cũng là cốt lõi của KD - là khoảng cách phân kỳ Kullback-Leibler ($D_{KL}$) giữa phân phối xác suất mềm của Teacher và Student. Ở đây, tôi sử dụng một tham số cực kỳ quan trọng là Nhiệt độ $T = 4.0$. 
    >
    > Tại sao phải cần Nhiệt độ $T$? Nếu dùng hàm softmax thông thường ($T=1$), xác suất dự đoán của lớp đúng sẽ rất gần 1, các lớp khác sẽ gần như bằng 0. Khi chúng ta tăng $T$ lên 4.0, phân phối xác suất sẽ được 'làm mềm' và trải đều ra. Nhờ vậy, Student sẽ học được những thông tin ẩn (Dark Knowledge) từ Teacher, ví dụ như: 'Ảnh con mèo này có 70% giống mèo, nhưng có 29% giống con chó và chỉ có 1% giống cái xe tải'. Thông tin 29% tương đồng giữa mèo và chó chính là tri thức cấu trúc vô giá giúp Student học nhanh hơn và thông minh hơn rất nhiều."*

---

### Slide 7: Kết Quả & Đối Chiếu Đường Cong Hội Tụ Tối Ưu
*   **Nội dung Slide**:
    *   **Biểu đồ đường cong học tập (Learning Curves)**:
        *   Student KD hội tụ nhanh hơn, đường loss mượt hơn hẳn baseline gốc.
        *   Độ chính xác vượt trội và ổn định qua các epochs cuối nhờ sự dẫn dắt của Teacher.
    *   **Bảng đối chiếu số liệu cuối cùng**:

| Cấu hình | Tham số | Kích thước File | Tỉ lệ nén | Độ chính xác CIFAR-10 |
| :--- | :---: | :---: | :---: | :---: |
| **ResNet-20 Baseline** | 272,474 | 1.09 MB | 1.0x (Gốc) | 91.93% |
| **SE-ResNet-20 Teacher (V2)** | 4,359,242 | 17.44 MB | 16.0x (Nặng) | 93.11% |
| **Student ResNet-20 + KD (Đề xuất)** | **272,474** | **1.09 MB** | **1.0x (Nhẹ)** | **93.19%** |

    *   **Lý do Student vượt qua cả Teacher**: Tri thức mềm từ Teacher đóng vai trò như một hàm phạt điều hòa (Regularizer) cực mạnh. Nó giới hạn không gian tìm kiếm trọng số của Student, giúp mạng Student nhỏ tránh xa hiện tượng tự tin thái quá (overconfidence) và tìm được cấu hình trọng số có khả năng tổng quát hóa xuất sắc hơn cả mạng lớn huấn luyện độc lập.
*   **Lời thoại thuyết trình**:
    > *"Hãy cùng tôi nhìn vào bảng kết quả thực nghiệm cuối cùng này. Đây là minh chứng rõ ràng nhất cho sức mạnh của tối ưu hóa chuyển giao tri thức.
    >
    > Mô hình Student ResNet-20 sau khi chắt lọc tri thức (KD) đã đạt **93.19%** độ chính xác trên CIFAR-10. Kết quả này không những tăng vượt bậc **+1.26%** so với baseline ban đầu, mà còn **vượt qua chính mô hình Teacher của nó (93.11%)**.
    >
    > Tại sao một mô hình Student nhỏ hơn 16 lần về mặt tham số lại có thể vượt qua mô hình Teacher khổng lồ của nó? 
    >
    > Câu trả lời nằm ở bản chất của quá trình tối ưu hóa. Khi Teacher huấn luyện độc lập, nó sử dụng nhãn cứng và CutMix nên vẫn có một không gian tìm kiếm trọng số rất rộng, dẫn tới một mức độ nhiễu nhất định. Khi Student học qua KD, phân phối xác suất mềm của Teacher đã hoạt động như một bộ lọc nhiễu và một hàm điều hòa hóa cực mạnh. Nó thu hẹp không gian tối ưu của Student, chỉ cho phép Student học các đặc trưng tinh túy nhất. Nhờ đó, Student tìm được một điểm cực tiểu phẳng vô cùng lý tưởng mà trước đây huấn luyện độc lập nó không bao giờ chạm tới được. 
    >
    > Về mặt ứng dụng thực tế, mô hình Student này hoàn toàn giữ nguyên kích thước 1.09 MB và độ trễ suy luận siêu thấp của baseline gốc. Chúng ta đã nén thành công mạng nơ-ron gấp 16 lần mà không hề mất đi một phần trăm hiệu năng nào!"*

---

### Slide 8: Bài Học Thực Tiễn Về Tối Ưu Hóa & Nén Mô Hình Dành Cho Triển Khai Edge
*   **Nội dung Slide**:
    *   **3 Bài học xương máu rút ra từ dự án**:
        1.  *Không áp dụng công nghệ mù quáng*: Các thuật toán tối ưu tiên tiến (như AdamW) không phải là vạn năng; chúng cần đi kèm cấu trúc mạng đủ rộng hoặc kỹ thuật điều hòa cực mạnh để phát huy hiệu quả.
        2.  *Attention & Regularization luôn là cặp bài trùng*: Cơ chế chú ý (SE block) nâng trần khả năng biểu diễn đặc trưng, trong khi CutMix và Stochastic Depth bảo vệ mạng khỏi việc quá khớp nhãn.
        3.  *Nén mô hình thông qua KD là tối ưu nhất cho Edge*: Đạt điểm cân bằng hoàn hảo (Pareto frontier) giữa Độ chính xác (Accuracy) và Tài nguyên tính toán (Computation Budget).
*   **Lời thoại thuyết trình**:
    > *"Kết thúc phần tối ưu hóa, tôi xin tổng kết 3 bài học thực tiễn vô cùng giá trị đóng góp vào kỹ nghệ triển khai AI nhúng:
    >
    > Thứ nhất, chúng ta không được phép áp dụng các công nghệ tối ưu hóa một cách mù quáng. Thực nghiệm V1 của tôi đã chứng minh rằng các thuật toán SOTA như AdamW hay SiLU hoàn toàn có thể phản tác dụng nếu không phù hợp với không gian tham số của mạng.
    >
    > Thứ hai, sự kết hợp giữa mạng chú ý (Attention) và các phương pháp điều hòa hóa (Regularization) như CutMix và Stochastic Depth là cặp bài trùng không thể tách rời để đẩy giới hạn trần độ chính xác của mạng tích chập.
    >
    > Thứ ba, phương pháp Nén tri thức (Knowledge Distillation) chính là 'chìa khóa vàng' cho các kỹ sư IoT. Nó giúp chúng ta đạt được điểm tối ưu tuyệt đối trên đường biên Pareto giữa năng lực phần cứng hạn chế và nhu cầu độ chính xác cao của doanh nghiệp."*

---

### Slide 9: Hướng Phát Triển Đề Tài - Lộ Trình Lượng Tử Hóa INT8
*   **Nội dung Slide**:
    *   **Lượng tử hóa nhận biết huấn luyện (Quantization-Aware Training - QAT)**:
        *   Mục tiêu: Chuyển đổi tham số từ FP32 (số thực 32-bit) sang INT8 (số nguyên 8-bit).
        *   Nén mô hình thêm **4 lần nữa** (Dung lượng giảm từ 1.09 MB xuống còn **0.27 MB**).
        *   Tối ưu hóa phần cứng nhúng: Các phép toán cộng nhân tích chập sẽ được thực hiện trực tiếp trên các thanh ghi số nguyên của vi điều khiển (STM32, ESP32), tăng tốc độ suy luận lên **2x - 3x**.
*   **Lời thoại thuyết trình**:
    > *"Cuối cùng, hướng phát triển tiếp theo của đề tài này là tối ưu hóa sâu hơn nữa ở mức độ biểu diễn số học phần cứng. 
    >
    > Tôi dự kiến sẽ áp dụng kỹ thuật **Lượng tử hóa nhận biết huấn luyện (QAT)** để đưa toàn bộ trọng số mạng từ số thực 32-bit về số nguyên 8-bit. Việc này sẽ tiếp tục nén mô hình của chúng ta thêm 4 lần nữa, đưa dung lượng file trọng số xuống mức không tưởng là dưới 0.3 MB. 
    >
    > Khi đó, các phép tính tích chập sẽ được thực thi cực kỳ nhanh bằng tập lệnh số nguyên trực tiếp trên các chip vi điều khiển giá rẻ như STM32 hay ESP32, biến mô hình ResNet-20 tối ưu này thành một giải pháp thị giác máy tính nhúng thương mại hoàn hảo."*
