# 🫀 Hệ thống AI Dự đoán Nguy cơ Suy tim dựa trên Chỉ số Lâm sàng
> **Bài tập lớn học phần: Trí tuệ nhân tạo (Artificial Intelligence)**  
> **Đề tài:** Nghiên cứu và Xây dựng Hệ thống AI Dự đoán Nguy cơ Suy tim dựa trên Chỉ số Lâm sàng

---

## 📌 1. Giới thiệu & Mục tiêu đề tài
Suy tim là một trong những nguyên nhân hàng đầu gây tử vong và tái nhập viện trên toàn cầu. Việc sàng lọc sớm bệnh nhân có nguy cơ cao giúp bác sĩ đưa ra phác đồ can thiệp kịp thời, giảm thiểu tỷ lệ tử vong.

Dự án này giải quyết hai thách thức cốt lõi của bài toán học máy trong y tế (**Medical AI**):
1. **Xử lý mất cân bằng dữ liệu (Class Imbalance):** Tỷ lệ bệnh nhân tử vong/nguy cơ cao chiếm thiểu số (~32%) so với nhóm an toàn (~68%), đòi hỏi quy trình huấn luyện chống rò rỉ dữ liệu (*Data Leakage*) và ưu tiên tối đa hóa độ nhạy (**Recall**) để hạn chế bỏ sót ca bệnh nặng (*False Negative*).
2. **Tính giải thích của mô hình (Explainable AI - XAI):** Ứng dụng **SHAP** (*SHapley Additive exPlanations*) để minh bạch hóa mức độ đóng góp của từng chỉ số sinh hóa đối với quyết định dự đoán của AI trên từng ca bệnh cụ thể.

---

## 📊 2. Tập dữ liệu (Dataset)
Dự án sử dụng bộ dữ liệu chuẩn mực **Heart Failure Clinical Records** từ kho dữ liệu UCI / Kaggle gồm **299 hồ sơ bệnh nhân** với **12 đặc trưng lâm sàng** và **1 biến mục tiêu** (`DEATH_EVENT`):

| Nhóm đặc trưng | Tên cột | Mô tả chỉ số y khoa |
| :--- | :--- | :--- |
| **Sinh hóa & Xét nghiệm** | `ejection_fraction` | Phân suất tống máu (% lượng máu bơm ra khỏi tim mỗi nhịp) |
| | `serum_creatinine` | Nồng độ Creatinine huyết thanh (mg/dL) |
| | `serum_sodium` | Nồng độ Natri huyết thanh (mEq/L) |
| | `creatinine_phosphokinase` | Mức độ enzyme CPK trong máu (mcg/L) |
| | `platelets` | Số lượng tiểu cầu trong máu (kiloplatelets/mL) |
| **Bệnh nền & Nhân trắc** | `age` | Tuổi bệnh nhân (năm) |
| | `anaemia` | Tình trạng thiếu máu (`0`: Không, `1`: Có) |
| | `diabetes` | Bệnh tiểu đường (`0`: Không, `1`: Có) |
| | `high_blood_pressure` | Cao huyết áp (`0`: Không, `1`: Có) |
| | `sex` | Giới tính (`0`: Nữ, `1`: Nam) |
| | `smoking` | Tiền sử hút thuốc (`0`: Không, `1`: Có) |
| **Theo dõi điều trị** | `time` | Thời gian theo dõi bệnh nhân (ngày) |
| **Biến mục tiêu** | `DEATH_EVENT` | `0`: Sống sót / Nguy cơ thấp — `1`: Tử vong / Nguy cơ cao |

---

## ⚙️ 3. Phương pháp & Quy trình Thực nghiệm

### 3.1. Kiến trúc chống rò rỉ dữ liệu (Zero Data Leakage Pipeline)
Toàn bộ quá trình tiền xử lý và cân bằng dữ liệu được đóng gói bên trong `imblearn.pipeline.Pipeline` kết hợp kiểm định chéo **Stratified 10-Fold Cross-Validation**:
* Chỉ tính toán tham số chuẩn hóa (`StandardScaler`) và sinh mẫu tổng hợp (`SMOTE`) trên **9 folds huấn luyện**.
* Giữ nguyên **1 fold kiểm thử** độc lập hoàn toàn để đánh giá khách quan.

### 3.2. Mô hình hóa & Tối ưu hóa
* **Mô hình cơ sở (Baseline):** Logistic Regression, Random Forest.
* **Mô hình nâng cao (Gradient Boosting):** XGBoost, LightGBM, CatBoost.
* **Tinh chỉnh siêu tham số (Hyperparameter Tuning):** Sử dụng **Optuna** tối ưu hóa hàm mục tiêu tùy chỉnh ưu tiên độ nhạy y khoa: $\text{Score} = 0.6 \times \text{Recall} + 0.4 \times \text{ROC-AUC}$.
* **Kết hợp mô hình (Ensemble Learning):** Soft Voting kết hợp xác suất dự đoán từ XGBoost, LightGBM và CatBoost.

---

## 📈 4. Kết quả Thực nghiệm (Stratified 10-Fold CV)

Hệ thống được thực nghiệm trên **2 kịch bản lâm sàng** (ghi nhận kết quả dạng `Mean ± Std` qua 10 folds):

### Kịch bản 1: Đầy đủ 12 đặc trưng (Bao gồm thời gian theo dõi `time`)
| Mô hình | Recall (Ưu tiên) | Precision | F1-Score | ROC-AUC | PR-AUC | Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression (Baseline) | 0.7700 ± 0.1204 | 0.6756 ± 0.0994 | 0.7121 ± 0.0782 | 0.8694 ± 0.0584 | 0.7878 ± 0.0857 | 0.7994 ± 0.0593 |
| Random Forest (Baseline) | 0.7389 ± 0.1728 | 0.7583 ± 0.0975 | 0.7429 ± 0.1285 | 0.9068 ± 0.0461 | 0.8158 ± 0.0876 | 0.8429 ± 0.0699 |
| XGBoost (Optuna Tuned + SMOTE) | 0.8211 ± 0.1312 | 0.7126 ± 0.0808 | 0.7570 ± 0.0800 | 0.8913 ± 0.0404 | 0.8023 ± 0.0658 | 0.8329 ± 0.0514 |
| **LightGBM (Optuna Tuned + SMOTE)** | **0.8422 ± 0.1250** | 0.7146 ± 0.0922 | 0.7667 ± 0.0821 | 0.8994 ± 0.0452 | **0.8288 ± 0.0821** | 0.8362 ± 0.0564 |
| **Soft Voting Ensemble** | 0.8011 ± 0.1169 | **0.7637 ± 0.0804** | **0.7791 ± 0.0877** | **0.9000 ± 0.0408** | 0.8241 ± 0.0772 | **0.8562 ± 0.0538** |

### Kịch bản 2: 11 đặc trưng lâm sàng (Loại bỏ biến `time` — Sàng lọc tức thời khi nhập viện)
| Mô hình | Recall (Ưu tiên) | Precision | F1-Score | ROC-AUC | PR-AUC | Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression (Baseline) | 0.7011 ± 0.1684 | 0.5372 ± 0.1048 | 0.6011 ± 0.1135 | 0.7660 ± 0.0689 | 0.6583 ± 0.1135 | 0.7061 ± 0.0777 |
| Random Forest (Baseline) | 0.5778 ± 0.1447 | 0.5826 ± 0.1039 | 0.5751 ± 0.1125 | 0.7753 ± 0.0834 | 0.6354 ± 0.1216 | 0.7293 ± 0.0731 |
| XGBoost (Optuna Tuned + SMOTE) | 0.7000 ± 0.1200 | **0.6037 ± 0.0744** | 0.6441 ± 0.0791 | **0.7981 ± 0.0883** | **0.6967 ± 0.1274** | **0.7528 ± 0.0573** |
| **LightGBM (Optuna Tuned + SMOTE)** | **0.7422 ± 0.0980** | 0.5877 ± 0.1065 | **0.6512 ± 0.0855** | 0.7893 ± 0.0787 | 0.6802 ± 0.1266 | 0.7395 ± 0.0793 |

---

## 📁 5. Cấu trúc Thư mục Dự án

```text
├── train_pipeline.py                     # Pipeline huấn luyện, tối ưu Optuna & đánh giá 10-Fold CV
├── heart_failure_model_full_features.joblib  # Mô hình đã huấn luyện (12 đặc trưng)
├── heart_failure_model_no_time.joblib        # Mô hình đã huấn luyện (11 đặc trưng lâm sàng)
├── requirements.txt                      # Danh sách thư viện môi trường Python
└── README.md                             # Tài liệu hướng dẫn dự án
