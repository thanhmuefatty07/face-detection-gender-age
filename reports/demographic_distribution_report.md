# Báo Cáo Phân Tích Thống Kê Nhân Khẩu Học (Demographic & Statistical Report)

**Tệp nguồn**: `reports/metadata.parquet`  
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

## 2. Phân Bổ 23 Khoảng Tuổi Chi Tiết 3–5 Năm (23 Fine Age Intervals)
| Chỉ số | Khoảng tuổi | Số lượng mẫu | Tỷ lệ (%) |
|---|---|---|---|
| 0 | `0-2` | 272 | 3.26% |
| 1 | `3-5` | 202 | 2.42% |
| 2 | `6-8` | 188 | 2.26% |
| 3 | `9-11` | 209 | 2.51% |
| 4 | `12-14` | 295 | 3.54% |
| 5 | `15-17` | 464 | 5.57% |
| 6 | `18-21` | 512 | 6.15% |
| 7 | `22-25` | 513 | 6.16% |
| 8 | `26-30` | 729 | 8.75% |
| 9 | `31-35` | 536 | 6.43% |
| 10 | `36-40` | 596 | 7.15% |
| 11 | `41-45` | 522 | 6.27% |
| 12 | `46-50` | 493 | 5.92% |
| 13 | `51-55` | 497 | 5.97% |
| 14 | `56-60` | 481 | 5.77% |
| 15 | `61-65` | 462 | 5.55% |
| 16 | `66-70` | 317 | 3.81% |
| 17 | `71-75` | 240 | 2.88% |
| 18 | `76-80` | 279 | 3.35% |
| 19 | `81-85` | 278 | 3.34% |
| 20 | `86-90` | 142 | 1.70% |
| 21 | `91-95` | 85 | 1.02% |
| 22 | `96-100` | 19 | 0.23% |

---

## 3. Phân Bổ 10 Nhóm Tuổi Sinh Học (10-Cohort Group Breakdown)
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

## 4. Phân Bổ Theo 13 Bộ Dữ Liệu Thực Tế (Benchmark Dataset Sources)
| Bộ dữ liệu (`source_dataset`) | Số lượng | Tỷ lệ (%) |
|---|---|---|
| `appa_real` | 1,345 | 16.14% |
| `imdb_wiki` | 1,103 | 13.24% |
| `agedb` | 855 | 10.26% |
| `adience` | 773 | 9.28% |
| `morph` | 714 | 8.57% |
| `cacd` | 699 | 8.39% |
| `b3fd` | 618 | 7.42% |
| `afad` | 489 | 5.87% |
| `fgnet` | 456 | 5.47% |
| `fairface` | 445 | 5.34% |
| `lagenda` | 366 | 4.39% |
| `megaage_asian` | 305 | 3.66% |
| `axon_age` | 163 | 1.96% |

---

## 5. Cân Bằng Giới Tính (Gender Distribution)
| Giới tính | Số lượng | Tỷ lệ (%) |
|---|---|---|
| Male | 3,970 | 47.65% |
| Female | 2,512 | 30.15% |
| unknown | 1,849 | 22.19% |

---

## 6. Phân Bổ Chủng Tộc & Sắc Tộc (Ethnic Demographics)
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

## 7. Phân Vùng Dữ Liệu Không Rò Rỉ (Split Isolation)
| Phân vùng (Split) | Số lượng | Tỷ lệ (%) |
|---|---|---|
| `train` | 6,845 | 82.16% |
| `test` | 764 | 9.17% |
| `val` | 722 | 8.67% |

---

## 8. Đánh Giá Trọng Số Dị Hình Sinh Học (Dimorphism Loss Reweighting)
- Trọng số dị hình trung bình: 0.9207
- Khoảng biến thiên: [0.2500, 1.0000]
- Các mẫu tuổi trước dậy thì (<12) và người già (>65) đã được áp dụng giảm trừ trọng số phạt nhị phân theo đúng sinh học hình thái.
