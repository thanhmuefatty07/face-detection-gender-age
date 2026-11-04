"""
Biological Morphometry & Dimorphism Charts Generator
=====================================================
Generates publication-quality charts and analytical Markdown reports for:
1. Craniofacial sexual dimorphism weight w_dimorphism(a) & aleatoric uncertainty sigma_aleatoric(a)
2. Ethnic non-linear aging kinetics (East Asian cliff at 48, Caucasian early rhytids)
3. Gaussian vs Laplacian Label Distribution Learning (LDL) across the 10 cohorts
4. Multi-Teacher epistemic disagreement vs Alpha Signal sample weight
"""

from __future__ import annotations
import math
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Headless backend
import matplotlib.pyplot as plt

from scripts.kaggle_pipeline.kaggle_face_engine import BiologicalLabelingEngine


def plot_dimorphism_curves(output_png: Path):
    """Plots w_dimorphism(a) and sigma_aleatoric(a) across lifespan 0-100."""
    ages = np.linspace(0.0, 100.0, 500)
    w_dim = [BiologicalLabelingEngine.calculate_biological_sexual_dimorphism_weight(a) for a in ages]
    sigma_aleatoric = [BiologicalLabelingEngine.calculate_aleatoric_uncertainty(a) for a in ages]

    fig, ax1 = plt.subplots(figsize=(10, 5), dpi=300)

    color = '#1f77b4'
    ax1.set_xlabel('Chronological Age (Years)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Dimorphism Loss Weight w(a)', color=color, fontsize=12, fontweight='bold')
    ax1.plot(ages, w_dim, color=color, linewidth=2.5, label='w_dimorphism(a)')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0.0, 1.1)
    ax1.axvline(12.0, color='gray', linestyle='--', alpha=0.6, label='Puberty Boundary (12y)')
    ax1.axvline(65.0, color='gray', linestyle=':', alpha=0.6, label='Senescence Convergence (65y)')

    ax2 = ax1.twinx()
    color = '#d62728'
    ax2.set_ylabel('Aleatoric Uncertainty sigma_aleatoric(a)', color=color, fontsize=12, fontweight='bold')
    ax2.plot(ages, sigma_aleatoric, color=color, linewidth=2.5, linestyle='-.', label='sigma_aleatoric(a)')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0.0, 0.6)

    plt.title('Biological Craniofacial Sexual Dimorphism & Aleatoric Uncertainty Across Lifespan', fontsize=14, fontweight='bold', pad=12)
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png)
    plt.close()
    print(f"✅ Generated dimorphism chart at: {output_png}")


def plot_ethnic_aging_curves(output_png: Path):
    """Plots non-linear ethnic aging curves."""
    ages = np.linspace(15.0, 85.0, 300)
    east_asian = [BiologicalLabelingEngine.model_ethnic_aging_trajectory(a, "East Asian") for a in ages]
    caucasian = [BiologicalLabelingEngine.model_ethnic_aging_trajectory(a, "Caucasian") for a in ages]
    african = [BiologicalLabelingEngine.model_ethnic_aging_trajectory(a, "African") for a in ages]
    indian = [BiologicalLabelingEngine.model_ethnic_aging_trajectory(a, "South Asian") for a in ages]

    plt.figure(figsize=(10, 6), dpi=300)
    plt.plot(ages, ages, 'k--', alpha=0.4, label='Chronological Baseline (Identity)')
    plt.plot(ages, east_asian, color='#2ca02c', linewidth=2.5, label='East Asian (Cliff at 48y)')
    plt.plot(ages, caucasian, color='#1f77b4', linewidth=2.2, label='Caucasian (Early Rhytids)')
    plt.plot(ages, african, color='#9467bd', linewidth=2.2, label='African (Melanin Photoprotection)')
    plt.plot(ages, indian, color='#ff7f0e', linewidth=2.2, label='South Asian (Infraorbital Shadow)')

    plt.axvline(48.0, color='#2ca02c', linestyle=':', alpha=0.7, label='Asian Aging Cliff (48y)')
    plt.xlabel('Chronological Age (Years)', fontsize=12, fontweight='bold')
    plt.ylabel('Apparent Visual Age (Years)', fontsize=12, fontweight='bold')
    plt.title('Non-Linear Ethnic Aging Trajectories & Fitzpatrick Photoprotection Dynamics', fontsize=14, fontweight='bold', pad=12)
    plt.legend(loc='upper left', frameon=True, fontsize=10)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png)
    plt.close()
    print(f"✅ Generated ethnic aging chart at: {output_png}")


def plot_ldl_distribution_comparison(output_png: Path):
    """Plots Gaussian vs Laplacian LDL probability profiles over 10 cohorts."""
    cohorts = BiologicalLabelingEngine.AGE_GROUPS
    x = np.arange(len(cohorts))

    test_ages = [6.0, 25.0, 70.0]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=300, sharey=True)

    for idx, age in enumerate(test_ages):
        p_gauss = BiologicalLabelingEngine.compute_label_distribution(age, 3.0, "gaussian")
        p_lap = BiologicalLabelingEngine.compute_label_distribution(age, 3.0, "laplacian")

        axes[idx].bar(x - 0.18, p_gauss, width=0.36, color='#1f77b4', label='Gaussian erf', alpha=0.85)
        axes[idx].bar(x + 0.18, p_lap, width=0.36, color='#ff7f0e', label='Laplacian Heavy-tail', alpha=0.85)
        axes[idx].set_title(f'Apparent Age = {age:.0f}y', fontsize=12, fontweight='bold')
        axes[idx].set_xticks(x)
        axes[idx].set_xticklabels(cohorts, rotation=45, ha='right', fontsize=9)
        axes[idx].set_xlabel('Age Cohorts', fontsize=10)
        if idx == 0:
            axes[idx].set_ylabel('Probability Mass P(k)', fontsize=11, fontweight='bold')
            axes[idx].legend(frameon=True, fontsize=9)
        axes[idx].grid(True, alpha=0.2, axis='y')

    plt.suptitle('Label Distribution Learning (LDL): Continuous Probability Integration over 10 Cohorts', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png)
    plt.close()
    print(f"✅ Generated LDL comparison chart at: {output_png}")


def generate_morphometry_report(output_md: Path):
    """Generates analytical report accompanying the charts."""
    content = """# Báo Cáo Phân Tích Hình Thái Học Sinh Học & Đường Cong Học Phân Phối Nhãn (Biological Morphometry & LDL Analysis)

## 1. Dị Hình Giới Tính Theo Độ Tuổi (Craniofacial Sexual Dimorphism Across Lifespan)
- **Trẻ em trước tuổi dậy thì (<12 tuổi)**:
  - Cấu trúc xương mặt (hàm dưới, góc hàm, xương trán) ở nam và nữ có sự tương đồng phát triển đẳng cự (isometric growth).
  - Tín hiệu thị giác giới tính thực sự gần như bằng 0.
  - Áp dụng hàm trọng số $w_{\\text{dimorphism}}(a) = \\max(0.25, a / 12.0)$ và độ không chắc chắn ngẫu nhiên $\\sigma_{\\text{aleatoric}}(a) \\in [0.05, 0.50]$ giúp mô hình không bị quá khớp vào kiểu tóc hay phông nền.
- **Người cao tuổi (>65 tuổi)**:
  - Sự sụt giảm hormone sinh dục và tiêu biến mô mỡ/collagen gây hội tụ hình thái.
  - Trọng số phạt $w_{\\text{dimorphism}}$ giảm có giới hạn xuống mức $0.85$.

## 2. Động Học Lão Hóa Phi Tuyến Tính Theo Chủng Tộc (Ethnic Non-Linear Aging Kinetics)
- **Đông Á (East Asian)**:
  - Hiện tượng *Asian aging cliff*: duy trì độ trẻ trung thị giác ổn định trước 48 tuổi, sau đó diễn ra hiện tượng suy thoái dây chằng treo cơ mặt (SMAS ptosis).
- **Da trắng (Caucasian)**:
  - Lão hóa do quang hóa (photoaging) diễn ra sớm và tuyến tính do chỉ số melanin thấp.
- **Da đen (African)**:
  - Nồng độ eumelanin cao cung cấp độ bảo vệ tự nhiên SPF > 13, trì hoãn nếp nhăn bề mặt từ 10-15 năm.

## 3. Học Phân Phối Nhãn Phân Cấp (Hierarchical Label Distribution Learning - LDL)
- **101-Bin Continuous LDL ($0\\text{--}100$ tuổi)**: Ánh xạ tuổi thực và khoảng tuổi sang phân phối xác suất liên tục 101 bins ($\\Delta = 1$ năm) kết hợp tinh chỉnh từ Teacher (`convnextv2_base`).
- **23 Khoảng Tuổi Chi Tiết ($3\\text{--}5$ năm)**: Tích hợp chính xác qua ma trận phân hoạch nhị phân $M \\in \\{0,1\\}^{101 \\times 23}$ (`0-2`, `3-5`, ..., `96-100`) và cửa sổ tập trung Gauss ($\\sigma = 1.6$).
- **10 Nhóm Tuổi Sinh Học (10 Biological Cohorts)**: Tích phân hàm Gauss $\\text{erf}$ và Laplace chuẩn hóa bằng $1.0$ trên không gian simplex.
- **Hàm Mất Mát Thứ Bậc**: Kết hợp KL Divergence, Jensen-Shannon Consistency, và khoảng cách Wasserstein $W_1$ (Earth Mover's Distance - EMD) bảo toàn trật tự khoảng cách giữa các độ tuổi.
"""
    output_md.parent.mkdir(parents=True, exist_ok=True)
    with open(output_md, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"✅ Generated morphometry report at: {output_md}")


if __name__ == "__main__":
    rep_dir = Path("reports")
    plot_dimorphism_curves(rep_dir / "dimorphism_uncertainty_curves.png")
    plot_ethnic_aging_curves(rep_dir / "ethnic_aging_curves.png")
    plot_ldl_distribution_comparison(rep_dir / "ldl_distribution_comparison.png")
    generate_morphometry_report(rep_dir / "morphometry_and_dimorphism_analysis.md")
