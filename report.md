# Báo Cáo Kết Quả Phân Loại Dữ Liệu Nano

## 1. Mô tả bài toán
Bài toán đặt ra là phân loại 15 chất hóa học dựa trên dữ liệu quang phổ Raman. Để nâng cao hiệu quả và tính chuyên biệt, các chất này được chia thành 3 nhóm chính:
- **Nhóm 1 (PPs/OPs)**: Các thuốc bả vệ thực vật và hợp chất liên quan (Carbaryl, 4-nitrophenol, Chloramphenicol, Tricyclazole, Glyphosate).
- **Nhóm 2 (BTEX)**: Các hợp chất hữu cơ dễ bay hơi (Benzene, Toluen, E-Benzene, Xylene, Styrene).
- **Nhóm 3 (ODs/ICs)**: Các chất màu hữu cơ và vô cơ (Congo Red, Crystal Violet, Methylene Blue, Urea, Thiram).

Mục tiêu là xây dựng các mô hình học máy (Machine Learning) để phân loại chính xác các chất trong từng nhóm.

## 2. Mô tả dữ liệu (Data)
- **Nguồn dữ liệu**: Thư mục `Data for Nano-AI`.
- **Số lượng**: 15 nhãn (chất), mỗi nhãn có 5 mẫu dữ liệu (files `.txt`). Tổng cộng 75 mẫu.
- **Định dạng**: Mỗi file chứa dữ liệu quang phổ với hai cột:
    - Cột 0: Raman shift (số sóng).
    - Cột 1: Cường độ (Intensity).
- **Đặc điểm**: Mỗi mẫu có khoảng 2048 điểm dữ liệu quang phổ.

## 3. Xử lý dữ liệu (Data Processing)
- **Đọc dữ liệu**: Dữ liệu được đọc từ các file text, lấy cột cường độ làm đặc trưng đầu vào.
- **Lựa chọn đặc trưng (Feature Selection)**: Thay vì sử dụng toàn bộ ~2048 điểm, mô hình sử dụng **40 điểm đặc trưng** quan trọng:
    - **34 điểm cố định**: Đây là các đỉnh phổ đặc trưng (peaks) được xác định trước (ví dụ: tại các vị trí 1885, 1856, 1463, ...).
    - **6 điểm ngẫu nhiên**: Được chọn thêm từ các điểm còn lại để tăng tính đa dạng cho dữ liệu đầu vào.
- **Chuẩn hóa (Normalization)**: Dữ liệu sau khi chọn lọc được chuẩn hóa theo phương pháp **Min-Max Scaling** để đưa giá trị về khoảng [0, 1], giúp mô hình hội tụ tốt hơn.
- **Chia tập dữ liệu**: Dữ liệu của từng nhóm được chia thành tập huấn luyện (Train) và tập kiểm tra (Test) với tỷ lệ 80/20 (20 mẫu train, 5 mẫu test cho mỗi nhóm), có phân tầng (stratify) theo nhãn.

## 4. Kỹ thuật trích xuất đặc trưng (Feature Extraction)
Ngoài việc sử dụng 40 đặc trưng thô (Original), hai kỹ thuật trích xuất đặc trưng nâng cao đã được áp dụng để giảm chiều dữ liệu xuống còn **15 chiều**:

### a. PCA (Principal Component Analysis)
- Phân tích thành phần chính giúp giảm chiều dữ liệu trong khi vẫn giữ lại phần lớn thông tin (phương sai) của dữ liệu gốc.
- Số thành phần (components): 15.

### b. AutoEncoder (Mạng nơ-ron tự mã hóa)
- Sử dụng kiến trúc mạng nơ-ron sâu để học biểu diễn nén của dữ liệu.
- **Kiến trúc**:
    - **Encoder**: Input (40) $\to$ Dense(100) $\to$ ReLU $\to$ Dense(50) $\to$ ReLU $\to$ Dense(15).
    - **Decoder**: Input(15) $\to$ Dense(50) $\to$ ReLU $\to$ Dense(100) $\to$ ReLU $\to$ Output(40).
- **Huấn luyện**: 50 epochs, hàm mất mát MSE, tối ưu hóa bằng Adam.
- Đặc trưng được lấy từ lớp đầu ra của Encoder (15 chiều).

## 5. Mô hình thực hiện (Models)
Các mô hình học máy phổ biến được sử dụng để huấn luyện và đánh giá trên cả 3 bộ đặc trưng (Original, PCA, AutoEncoder):
1.  **Logistic Regression**
2.  **SVM (Support Vector Machine)**
3.  **KNN (K-Nearest Neighbors)**
4.  **Random Forest**
5.  **Decision Tree**
6.  **Naive Bayes (BernoulliNB)**

### Kỹ thuật Ensemble & Boosting
Để cải thiện hiệu suất, các kỹ thuật kết hợp mô hình (Ensemble) đã được áp dụng:
- **Voting Classifier (Soft Voting)**: Kết hợp dự đoán xác suất của Logistic Regression, SVM, KNN, và Random Forest. Kết quả cuối cùng là lớp có tổng xác suất cao nhất.
- **AdaBoost Classifier**: Sử dụng kỹ thuật Boosting để xây dựng một bộ phân loại mạnh từ các bộ phân loại yếu hơn. Trong dự án này, Voting Classifier (phiên bản không có KNN) được sử dụng làm bộ phân loại cơ sở (base estimator) cho AdaBoost.

## 6. Kết quả và Nhận xét

Kết quả đánh giá chi tiết được lưu trong file `evaluation_results_groups.csv`. Dưới đây là tóm tắt và nhận xét:

### Kết quả tóm tắt (Accuracy trên tập Test):

**Group 1: PPs/OPs**
- **Original & PCA**: Hầu hết các mô hình (Logistic Regression, Voting Classifier) đạt độ chính xác tuyệt đối **100%**. SVM và KNN cũng hoạt động rất tốt.
- **AutoEncoder**: Hiệu suất thấp hơn (khoảng 60-80%), có thể do chưa tối ưu hóa kiến trúc mạng cho tập dữ liệu nhỏ.

**Group 2: BTEX**
- **Original & PCA**: Logistic Regression, KNN, Random Forest và Voting Classifier đều đạt độ chính xác **100%**.
- **AutoEncoder**: Hiệu suất dao động (60-80%), chưa ổn định bằng các phương pháp truyền thống.

**Group 3: ODs/ICs**
- **Tất cả các phương pháp (Original, PCA, AutoEncoder)**: Đều cho kết quả xuất sắc, thường xuyên đạt **100%** độ chính xác với Logistic Regression, Voting Classifier và cả AutoEncoder (đạt 100% với Logistic Regression).

### Nhận xét chung:
1.  **Hiệu suất phân loại**: Các mô hình đạt độ chính xác rất cao, thường xuyên đạt tuyệt đối (100%) trên tập kiểm tra, chứng tỏ bộ đặc trưng 40 điểm được chọn lọc rất hiệu quả.
2.  **So sánh đặc trưng**:
    - **PCA**: Giữ nguyên được hiệu suất cao của dữ liệu gốc (Original) mặc dù đã giảm số chiều từ 40 xuống 15.
    - **AutoEncoder**: Tuy có tiềm năng nhưng trên tập dữ liệu nhỏ (20 mẫu train/nhóm), nó chưa vượt qua được sự ổn định và hiệu quả của PCA hay dữ liệu gốc. Tuy nhiên, ở nhóm ODs/ICs, nó vẫn hoạt động rất tốt.
3.  **Mô hình tốt nhất**:
    - **Voting Classifier** và **Logistic Regression** là những lựa chọn tốt nhất, luôn duy trì độ chính xác cao và ổn định trên cả 3 nhóm chất.

## 7. Các Mô Hình Đạt Độ Chính Xác Tuyệt Đối (100%)

Dưới đây là danh sách các mô hình đã phân loại chính xác hoàn toàn trên tập kiểm tra. Tất cả các chỉ số (Accuracy, Precision, Recall, F1-Score) đều đạt **1.0**.

| Nhóm | Feature Set | Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **PPs/OPs** | Original | Logistic Regression, SVM, KNN, VotingClassifier_Full | 1.0 | 1.0 | 1.0 | 1.0 |
| | PCA | Logistic Regression, SVM, Random Forest, Naive Bayes, VotingClassifier_Full | 1.0 | 1.0 | 1.0 | 1.0 |
| | AutoEncoder | VotingClassifier_Full | 1.0 | 1.0 | 1.0 | 1.0 |
| **BTEX** | Original | Logistic Regression, KNN, Random Forest, Decision Tree, VotingClassifier_Full | 1.0 | 1.0 | 1.0 | 1.0 |
| | PCA | Logistic Regression, KNN, Random Forest, VotingClassifier_Full | 1.0 | 1.0 | 1.0 | 1.0 |
| **ODs/ICs** | Original | Logistic Regression, KNN, Random Forest, VotingClassifier_Full, AdaBoostClassifier | 1.0 | 1.0 | 1.0 | 1.0 |
| | PCA | Logistic Regression, KNN, Random Forest, Decision Tree, Naive Bayes, VotingClassifier_Full, AdaBoostClassifier | 1.0 | 1.0 | 1.0 | 1.0 |
| | AutoEncoder | Logistic Regression, KNN, Decision Tree, VotingClassifier_Full | 1.0 | 1.0 | 1.0 | 1.0 |

## 8. Đường Dẫn Kết Quả Chi Tiết

Bạn có thể xem chi tiết kết quả và các biểu đồ confusion matrix tại các đường dẫn sau:

- **File kết quả (CSV)**: [evaluation_results_groups.csv](https://github.com/nguyencong1227/nanoclassification/blob/main/evaluation_results_groups.csv)
- **Thư mục biểu đồ (Confusion Matrix)**: [plots/](https://github.com/nguyencong1227/nanoclassification/tree/main/plots)
