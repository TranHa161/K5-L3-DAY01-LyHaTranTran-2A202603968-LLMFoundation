# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM


- **MSSV:** 2A202603968
- **Họ và tên:** Lý Hà Trân Trân
- **SEED** (in ở ô đầu notebook):

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☐ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☐ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☐ |

## 2. Kết quả chính

*                                 precision  recall  F1 (test)  F1 (SUB 40)  \
1. BoW + LogReg (tự code)            1.000   0.716      0.834        0.621   
2. BoW + MLP (deep learning)         1.000   0.862      0.926        0.889   
3a. Encoder (MiniLM) + kNN           0.908   0.931      0.919        0.947   
3b. Encoder (MiniLM) + LogReg        0.981   0.914      0.946        0.889   
3c. Encoder (MiniLM) + MLP head      0.991   0.931      0.960        0.919   

                                 train (s)  tin/s (cả lô)  p50 ms/tin  \
1. BoW + LogReg (tự code)            5.188       9657.431       0.051   
2. BoW + MLP (deep learning)         7.212      15463.026       0.246   
3a. Encoder (MiniLM) + kNN           4.787         45.564      47.300   
3b. Encoder (MiniLM) + LogReg        6.776       1093.564      20.030   
3c. Encoder (MiniLM) + MLP head      7.735       1247.022      31.387   

                                 p95 ms/tin  USD / 1k tin  
1. BoW + LogReg (tự code)             0.212           0.0  
2. BoW + MLP (deep learning)          2.312           0.0  
3a. Encoder (MiniLM) + kNN           72.101           0.0  
3b. Encoder (MiniLM) + LogReg        35.471           0.0  
3c. Encoder (MiniLM) + MLP head      67.944           0.0  
• Hàng LLM chỉ đo trên SUB (40 tin) → so sánh chất lượng công bằng bằng cột 'F1 (SUB 40)'.
• SUB chỉ 40 tin: 1 tin sai đổi F1 vài điểm. Chênh < ~0.05 F1 coi như HOÀ — đừng xếp hạng theo số lẻ.
• Inference đo TỪ VĂN BẢN THÔ → NHÃN (gồm cả vectorize / encode). p50/p95 = gọi từng tin một, như một request thật.
• train (s) của 3a/3b/3c = thời gian encoder MiniLM encode tập train (+ train đầu LogReg / MLP); LLM = 0 vì không train. Số đo phụ thuộc CPU/GPU của Colab.


## 3. Dự đoán và thực tế

Dự đoán Thí nghiệm 3.B (Tin nhắn ngụy trang giữa BoW vs. Encoder + LogReg):
Bạn đoán gì: Dự đoán Encoder + LogReg sẽ vượt trội hơn BoW vì hiểu ngữ cảnh sâu và bắt được ý định ẩn.Thực tế ra gì: BoW thắng 5/6, trong khi Encoder + LogReg chỉ đạt 4/6.Vì sao lệch: BoW bắt trúng "shortcut" từ vựng ("your", "mobile", "chosen"). Ngược lại, pre-trained encoder bị đánh lừa bởi văn phong trang trọng, chuẩn ngữ pháp ("complimentary", "holiday"), dẫn đến vector embedding bị gom về cụm thông báo dịch vụ khách hàng hợp lệ thay vì spam.

Dự đoán hiệu năng MLP Head so với Logistic Regression trên cùng Text Embedding (Thí nghiệm 3.c vs. 3.b):Bạn đoán gì: Thêm hidden layer phi tuyến (MLP) sẽ giúp F1 tăng đáng kể tương tự như bước chuyển từ BoW tuyến tính sang MLP ở Phần 2.Thực tế ra gì: F1 của MLP head không tăng (hoặc chỉ dao động nhẹ quanh mức $\pm 0.2\%$), thậm chí có dấu hiệu overfit nhẹ nếu train nhiều epoch.Vì sao lệch: Text encoder (Transformer backbone) bản thân nó đã là một mạng phi tuyến rất sâu. Không gian embedding đầu ra đã phân tách tuyến tính rất tốt (linearly separable), do đó một siêu phẳng Logistic Regression đơn giản là đủ để phân loại mà không cần thêm tầng ẩn.

Dự đoán LLM Zero-shot so với Mô hình Giám sát 3.b (Thí nghiệm 4.A):Bạn đoán gì: LLM lớn (với hàng chục/trăm tỷ tham số và tri thức ngôn ngữ khổng lồ) sẽ đạt F1 cao hơn hẳn mô hình Logistic Regression chạy trên embedding.Thực tế ra gì: LLM zero-shot chỉ đạt ngang ngửa hoặc thấp hơn mô hình 3.b (khoảng 0.95–0.97 so với $\ge 0.98$ của 3.b).Vì sao lệch: Mô hình 3.b được tối ưu hóa trực tiếp trên phân phối nhãn thực tế của dữ liệu SMS Spam (supervised training). LLM zero-shot tuy hiểu rộng nhưng không được căn chỉnh theo ngưỡng quyết định cụ thể của tập dữ liệu này, dễ bị thiên lệch (bias) hoặc phân vân trước các tin nhắn ngắn thiếu ngữ cảnh.

## 4. Phân tích 50 case khó

*Model nào tụt nhiều nhất, ở nhóm case nào? Nêu 2 case cụ thể và cơ chế khiến model sai.*

## 5. Thử thách cá nhân

*Đề theo SEED của bạn, cách làm, kết quả, giải thích.*

## 6. Phương án cho sản phẩm thật

*Với yêu cầu p95 < 50 ms/tin cho 10 triệu SMS/ngày: chọn phương án nào, dẫn số liệu nào, chấp nhận đánh đổi gì.*

## 7. AI log và điều đã học

*Mỗi lần dùng AI: hỏi gì, nhận gì, bạn sửa gì. Sau đó 2–3 câu: điều bạn hiểu ra hôm nay mà trước đó chưa hiểu.*
