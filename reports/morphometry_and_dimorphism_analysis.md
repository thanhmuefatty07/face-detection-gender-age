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

## 3. Học Phân Phối Nhãn (Label Distribution Learning - LDL)
- Ánh xạ tuổi liên tục sang 10 bins xác suất qua tích phân hàm Gauss $\text{erf}$ và hàm Laplace.
- Tổng xác suất trên 10 nhóm luôn chuẩn hóa bằng $1.0$ trên không gian simplex.
- Sử dụng hàm mất mát khoảng cách Wasserstein (Earth Mover's Distance - EMD) kết hợp KL Divergence bảo toàn trật tự khoảng cách thứ bậc giữa các nhóm tuổi.
