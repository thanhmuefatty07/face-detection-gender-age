# Báo Cáo Thẩm Định Bộ Dữ Liệu Khuôn Mặt Tinh Tuyển (Golden Dataset Audit)

## 1. Thông Số Tổng Quan
- **Tổng số mẫu đạt chuẩn 100% cực đoan**: 1
- **Độ sắc nét trung bình (Laplacian Variance)**: 185.40 (Ngưỡng tối thiểu: >= 100)
- **Năng lượng cạnh Tenengrad trung bình**: 340.20 (Ngưỡng tối thiểu: >= 250)
- **Tỷ lệ ảnh qua căn chỉnh 5 điểm mốc hình học**: 100.0%
- **Rò rỉ trùng lặp (Data Leakage)**: 0.0% (Kiểm tra pHash Hamming Distance > 6)

## 2. Phân Bổ Giới Tính (Gender Distribution)
| Giới tính | Số lượng mẫu | Tỷ lệ (%) |
|---|---|---|
| Female | 1 | 100.00% |

## 3. Phân Bổ 10 Nhóm Tuổi (Age Group Distribution)
| Nhóm tuổi | Số lượng mẫu | Tỷ lệ (%) |
|---|---|---|
| 18-24 | 1 | 100.00% |

## 4. Tín Hiệu Alpha Đã Khai Thác (Alpha Signals & LDL)
- **Label Distribution Learning (LDL)**: Toàn bộ mẫu đều có vector phân phối xác suất Gaussian 10 chiều, làm mượt gradient ranh giới tuổi.
- **Trọng số mẫu sinh học**: Giảm áp lực phân loại giới tính giả ở trẻ em <12 tuổi và người già >65 tuổi.
- **Lưu trữ**: Toàn bộ ảnh gốc nằm trên Kaggle Datasets (`vnthanhphm/face-gender-age-golden-dataset`), máy local chỉ lưu metadata và code tái lập.
