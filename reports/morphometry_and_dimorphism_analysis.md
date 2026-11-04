# Báo Cáo Phân Tích Hình Thái Học Sinh Học & Đường Cong Học Phân Phối Nhãn (Biological Morphometry & LDL Analysis)

## 1. Dị Hình Giới Tính Theo Độ Tuổi (Craniofacial Sexual Dimorphism Across Lifespan)
- **Trẻ em trước tuổi dậy thì (<12 tuổi)**:
  - Cấu trúc xương mặt (hàm dưới, góc hàm, xương trán) ở nam và nữ có sự tương đồng phát triển đẳng cự (isometric growth).
  - Tín hiệu thị giác giới tính thực sự gần như bằng 0.
  - Áp dụng hàm trọng số $w_{\text{dimorphism}}(a) = \max(0.25, a / 12.0)$ và độ không chắc chắn ngẫu nhiên $\sigma_{\text{aleatoric}}(a) \in [0.05, 0.50]$ giúp mô hình không bị quá khớp vào kiểu tóc hay phông nền.
- **Người cao tuổi (>65 tuổi)**:
  - Sự sụt giảm hormone sinh dục và tiêu biến mô mỡ/collagen gây hội tụ hình thái.
  - Trọng số phạt $w_{\text{dimorphism}}$ giảm có giới hạn xuống mức $0.85$.

## 2. Động Học Lão Hóa Phi Tuyến Tính Theo Chủng Tộc (Ethnic Non-Linear Aging Kinetics)
- **Đông Á (East Asian)**:
  - Hiện tượng *Asian aging cliff*: duy trì độ trẻ trung thị giác ổn định trước 48 tuổi, sau đó diễn ra hiện tượng suy thoái dây chằng treo cơ mặt (SMAS ptosis).
- **Da trắng (Caucasian)**:
  - Lão hóa do quang hóa (photoaging) diễn ra sớm và tuyến tính do chỉ số melanin thấp.
- **Da đen (African)**:
  - Nồng độ eumelanin cao cung cấp độ bảo vệ tự nhiên SPF > 13, trì hoãn nếp nhăn bề mặt từ 10-15 năm.

## 3. Học Phân Phối Nhãn Phân Cấp (Hierarchical Label Distribution Learning - LDL)
- **101-Bin Continuous LDL ($0\text{--}100$ tuổi)**: Ánh xạ tuổi thực và khoảng tuổi sang phân phối xác suất liên tục 101 bins ($\Delta = 1$ năm) kết hợp tinh chỉnh từ Teacher (`convnextv2_base`).
- **23 Khoảng Tuổi Chi Tiết ($3\text{--}5$ năm)**: Tích hợp chính xác qua ma trận phân hoạch nhị phân $M \in \{0,1\}^{101 \times 23}$ (`0-2`, `3-5`, ..., `96-100`) và cửa sổ tập trung Gauss ($\sigma = 1.6$).
- **10 Nhóm Tuổi Sinh Học (10 Biological Cohorts)**: Tích phân hàm Gauss $\text{erf}$ và Laplace chuẩn hóa bằng $1.0$ trên không gian simplex.
- **Hàm Mất Mát Thứ Bậc**: Kết hợp KL Divergence, Jensen-Shannon Consistency, và khoảng cách Wasserstein $W_1$ (Earth Mover's Distance - EMD) bảo toàn trật tự khoảng cách giữa các độ tuổi.
