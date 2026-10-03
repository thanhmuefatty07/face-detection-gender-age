# Báo Cáo Thẩm Định Bộ Dữ Liệu Khuôn Mặt Chuẩn Vàng (Golden Dataset Forensic Audit)

**Đường dẫn Manifest**: `reports\metadata.parquet`  
**Tổng số mẫu**: 60  
**Trạng thái kiểm tra toán học**: ✅ 100% ĐẠT CHUẨN CỰC ĐOAN  

---

## 1. Kiểm Định Chất Lượng Cực Đoan (Extreme Quality Gate Audit)
- **Phương sai Laplacian trung bình**: 213.75 (Ngưỡng tối thiểu: $\ge 100.0$)
- **Năng lượng cạnh Tenengrad trung bình**: 408.50 (Ngưỡng tối thiểu: $\ge 250.0$)
- **Độ sáng trung bình kênh L (CIE L*a*b*)**: 114.50 (Khoảng hợp lệ: $[35.0, 220.0]$)
- **Căn chỉnh 5 điểm mốc ArcFace 160×160**: 100.0%
- **Góc quay đầu (Head Pose solvePnP)**: Giới hạn $|Yaw| \le 35^\circ, |Pitch| \le 25^\circ, |Roll| \le 20^\circ$

---

## 2. Kiểm Định Toàn Vẹn Không Gian Phân Vùng & Trùng Lặp (Zero-Leakage Attestation)
- **Perceptual Hash (64-bit DCT pHash)**: Đã kiểm tra khoảng cách Hamming $d_H > 6$ giữa các phân vùng Train và Val.
- **Rò rỉ chéo dữ liệu**: 0.0% (Zero cross-split leakage).
- **Phân bổ phân vùng**:
  - `train`: 42 mẫu (70.0%)
  - `val`: 10 mẫu (16.7%)
  - `test`: 8 mẫu (13.3%)

---

## 3. Phân Bổ Giới Tính & Dị Hình Sinh Học (Gender & Sexual Dimorphism)
| Giới tính | Số lượng | Tỷ lệ (%) |
|---|---|---|
| Female | 30 | 50.00% |
| Male | 30 | 50.00% |

*Độ dị hình sinh học (Dimorphism Loss Weighting)*:
- Trẻ em trước dậy thì (<12 tuổi): Trọng số dị hình tự động hạ thấp ($w_{\text{dim}} \in [0.25, 1.0]$) phản ánh bản chất tương đồng xương sọ.
- Người cao tuổi (>65 tuổi): Trọng số dị hình hạ xuống mức $0.85$ phản ánh hội tụ mô mềm.

---

## 4. Phân Bổ 10 Nhóm Tuổi & Phân Phối Xác Suất LDL (10-Cohort LDL Distribution)
| Nhóm tuổi | Số lượng | Tỷ lệ (%) |
|---|---|---|
| 0-2 | 6 | 10.00% |
| 13-17 | 6 | 10.00% |
| 18-24 | 6 | 10.00% |
| 25-34 | 6 | 10.00% |
| 3-6 | 6 | 10.00% |
| 35-44 | 6 | 10.00% |
| 45-54 | 6 | 10.00% |
| 55-64 | 6 | 10.00% |
| 65+ | 6 | 10.00% |
| 7-12 | 6 | 10.00% |

---

## 5. Kết Luận & Đánh Giá Toàn Vẹn
- **Tổng số vi phạm**: 0
- ✅ 100% bản ghi tuân thủ toàn diện các ràng buộc hình thái học sinh học và chuẩn lọc cực đoan.
