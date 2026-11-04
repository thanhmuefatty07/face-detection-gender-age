# Báo Cáo Phân Tích Thống Kê Nhân Khẩu Học (Demographic & Statistical Report)

**Tệp nguồn**: `reports\metadata.parquet`  
**Tổng số mẫu thẩm định**: 8,331  

---

## 1. Phân Tích Phân Bổ Độ Tuổi (Age Distribution Analysis)
- **Độ tuổi trung bình (Mean Age)**: 40.06 tuổi
- **Độ lệch chuẩn (Age Std)**: 23.32
- **Trung vị (Median Age)**: 38.00 tuổi
- **Kiểm định Kolmogorov-Smirnov (so với Uniform [0, 100])**:
  - $D$-statistic: 0.1894
  - $p$-value: 3.9802e-262

---

## 2. Phân Bổ 10 Nhóm Tuổi Rời Rạc (10-Cohort Group Breakdown)
| Nhóm tuổi | Tên nhóm | Số lượng mẫu | Tỷ lệ (%) |
|---|---|---|---|
| 0 | `0-2` | 272 | 3.26% |
| 1 | `3-6` | 294 | 3.53% |
| 2 | `7-12` | 364 | 4.37% |
| 3 | `13-17` | 676 | 8.11% |
| 4 | `18-24` | 881 | 10.57% |
| 5 | `25-34` | 1,275 | 15.30% |
| 6 | `35-44` | 1,135 | 13.62% |
| 7 | `45-54` | 993 | 11.92% |
| 8 | `55-64` | 956 | 11.48% |
| 9 | `65+` | 1,485 | 17.82% |

---

## 3. Cân Bằng Giới Tính (Gender Distribution)
| Giới tính | Số lượng | Tỷ lệ (%) |
|---|---|---|
| Male | 3,970 | 47.65% |
| Female | 2,512 | 30.15% |
| unknown | 1,849 | 22.19% |

---

## 4. Phân Bổ Chủng Tộc & Sắc Tộc (Ethnic Demographics)
| Nhóm nhân khẩu / Chủng tộc | Số lượng | Tỷ lệ (%) |
|---|---|---|
| Diverse | 7,092 | 85.13% |
| East Asian | 851 | 10.21% |
| African | 68 | 0.82% |
| South Asian | 68 | 0.82% |
| Latino | 66 | 0.79% |
| Southeast Asian | 63 | 0.76% |
| Caucasian | 62 | 0.74% |
| Middle Eastern | 61 | 0.73% |

---

## 5. Phân Vùng Dữ Liệu Không Rò Rỉ (Split Isolation)
| Phân vùng (Split) | Số lượng | Tỷ lệ (%) |
|---|---|---|
| `train` | 6,845 | 82.16% |
| `test` | 764 | 9.17% |
| `val` | 722 | 8.67% |

---

## 6. Đánh Giá Trọng Số Dị Hình Sinh Học (Dimorphism Loss Reweighting)
- Trọng số dị hình trung bình: 0.9207
- Khoảng biến thiên: [0.2500, 1.0000]
- Các mẫu tuổi trước dậy thì (<12) và người già (>65) đã được áp dụng giảm trừ trọng số phạt nhị phân theo đúng sinh học hình thái.
