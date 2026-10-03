# Báo Cáo Phân Tích Thống Kê Nhân Khẩu Học (Demographic & Statistical Report)

**Tệp nguồn**: `reports\metadata.parquet`  
**Tổng số mẫu thẩm định**: 60  

---

## 1. Phân Tích Phân Bổ Độ Tuổi (Age Distribution Analysis)
- **Độ tuổi trung bình (Mean Age)**: 30.60 tuổi
- **Độ lệch chuẩn (Age Std)**: 23.27
- **Trung vị (Median Age)**: 25.75 tuổi
- **Kiểm định Kolmogorov-Smirnov (so với Uniform [0, 100])**:
  - $D$-statistic: 0.2950
  - $p$-value: 4.0154e-05

---

## 2. Phân Bổ 10 Nhóm Tuổi Rời Rạc (10-Cohort Group Breakdown)
| Nhóm tuổi | Tên nhóm | Số lượng mẫu | Tỷ lệ (%) |
|---|---|---|---|
| 0 | `0-2` | 6 | 10.00% |
| 1 | `3-6` | 6 | 10.00% |
| 2 | `7-12` | 6 | 10.00% |
| 3 | `13-17` | 6 | 10.00% |
| 4 | `18-24` | 6 | 10.00% |
| 5 | `25-34` | 6 | 10.00% |
| 6 | `35-44` | 6 | 10.00% |
| 7 | `45-54` | 6 | 10.00% |
| 8 | `55-64` | 6 | 10.00% |
| 9 | `65+` | 6 | 10.00% |

---

## 3. Cân Bằng Giới Tính (Gender Distribution)
| Giới tính | Số lượng | Tỷ lệ (%) |
|---|---|---|
| Female | 30 | 50.00% |
| Male | 30 | 50.00% |

---

## 4. Phân Bổ Chủng Tộc & Sắc Tộc (Ethnic Demographics)
| Nhóm nhân khẩu / Chủng tộc | Số lượng | Tỷ lệ (%) |
|---|---|---|
| East Asian | 10 | 16.67% |
| Caucasian | 10 | 16.67% |
| African | 10 | 16.67% |
| South Asian | 10 | 16.67% |
| Latino | 10 | 16.67% |
| Middle Eastern | 10 | 16.67% |

---

## 5. Phân Vùng Dữ Liệu Không Rò Rỉ (Split Isolation)
| Phân vùng (Split) | Số lượng | Tỷ lệ (%) |
|---|---|---|
| `train` | 42 | 70.00% |
| `val` | 10 | 16.67% |
| `test` | 8 | 13.33% |

---

## 6. Đánh Giá Trọng Số Dị Hình Sinh Học (Dimorphism Loss Reweighting)
- Trọng số dị hình trung bình: 0.8468
- Khoảng biến thiên: [0.2500, 1.0000]
- Các mẫu tuổi trước dậy thì (<12) và người già (>65) đã được áp dụng giảm trừ trọng số phạt nhị phân theo đúng sinh học hình thái.
