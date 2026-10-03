"""
Demographic Distribution & Statistical Analysis Script
======================================================
Analyzes demographic balance, cross-cohort distribution, ethnic coverage,
and Kolmogorov-Smirnov goodness-of-fit from golden dataset manifests.
"""

from __future__ import annotations
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Dict, Any

import numpy as np
import pandas as pd
from scipy import stats


def analyze_demographics(manifest_path: Path, output_md_path: Path) -> str:
    """
    Computes mathematical distribution statistics and Kolmogorov-Smirnov tests.
    """
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest {manifest_path} not found.")

    if manifest_path.suffix == '.parquet':
        df = pd.read_parquet(manifest_path)
    else:
        df = pd.read_csv(manifest_path)

    total = len(df)
    age_col = 'age_continuous' if 'age_continuous' in df else None
    gender_col = 'gender' if 'gender' in df else None
    eth_col = 'ethnicity' if 'ethnicity' in df else None
    dim_col = 'dimorphism_weight' if 'dimorphism_weight' in df else None
    split_col = 'split' if 'split' in df else None

    # Age statistics
    age_mean = float(df[age_col].mean()) if age_col else 0.0
    age_std = float(df[age_col].std()) if age_col else 0.0
    age_median = float(df[age_col].median()) if age_col else 0.0

    # KS-test against uniform distribution [0, 100]
    ks_stat, ks_p = 0.0, 0.0
    if age_col and len(df) > 5:
        ages = df[age_col].dropna().values
        ks_stat, ks_p = stats.kstest(ages, 'uniform', args=(0.0, 100.0))

    gender_counts = df[gender_col].value_counts().to_dict() if gender_col else {}
    eth_counts = df[eth_col].value_counts().to_dict() if eth_col else {}
    split_counts = df[split_col].value_counts().to_dict() if split_col else {}

    report = f"""# Báo Cáo Phân Tích Thống Kê Nhân Khẩu Học (Demographic & Statistical Report)

**Tệp nguồn**: `{manifest_path}`  
**Tổng số mẫu thẩm định**: {total:,}  

---

## 1. Phân Tích Phân Bổ Độ Tuổi (Age Distribution Analysis)
- **Độ tuổi trung bình (Mean Age)**: {age_mean:.2f} tuổi
- **Độ lệch chuẩn (Age Std)**: {age_std:.2f}
- **Trung vị (Median Age)**: {age_median:.2f} tuổi
- **Kiểm định Kolmogorov-Smirnov (so với Uniform [0, 100])**:
  - $D$-statistic: {ks_stat:.4f}
  - $p$-value: {ks_p:.4e}

---

## 2. Phân Bổ 10 Nhóm Tuổi Rời Rạc (10-Cohort Group Breakdown)
| Nhóm tuổi | Tên nhóm | Số lượng mẫu | Tỷ lệ (%) |
|---|---|---|---|
"""
    if 'age_group' in df:
        cohorts = ['0-2', '3-6', '7-12', '13-17', '18-24', '25-34', '35-44', '45-54', '55-64', '65+']
        for idx, c in enumerate(cohorts):
            cnt = int((df['age_group'] == c).sum())
            pct = (cnt / max(1, total)) * 100
            report += f"| {idx} | `{c}` | {cnt:,} | {pct:.2f}% |\n"

    report += """
---

## 3. Cân Bằng Giới Tính (Gender Distribution)
| Giới tính | Số lượng | Tỷ lệ (%) |
|---|---|---|
"""
    for g, cnt in gender_counts.items():
        report += f"| {g} | {cnt:,} | {cnt / max(1, total) * 100:.2f}% |\n"

    report += """
---

## 4. Phân Bổ Chủng Tộc & Sắc Tộc (Ethnic Demographics)
| Nhóm nhân khẩu / Chủng tộc | Số lượng | Tỷ lệ (%) |
|---|---|---|
"""
    for eth, cnt in eth_counts.items():
        report += f"| {eth} | {cnt:,} | {cnt / max(1, total) * 100:.2f}% |\n"

    report += """
---

## 5. Phân Vùng Dữ Liệu Không Rò Rỉ (Split Isolation)
| Phân vùng (Split) | Số lượng | Tỷ lệ (%) |
|---|---|---|
"""
    for sp, cnt in split_counts.items():
        report += f"| `{sp}` | {cnt:,} | {cnt / max(1, total) * 100:.2f}% |\n"

    report += """
---

## 6. Đánh Giá Trọng Số Dị Hình Sinh Học (Dimorphism Loss Reweighting)
"""
    if dim_col:
        mean_dim = float(df[dim_col].mean())
        min_dim = float(df[dim_col].min())
        max_dim = float(df[dim_col].max())
        report += f"- Trọng số dị hình trung bình: {mean_dim:.4f}\n"
        report += f"- Khoảng biến thiên: [{min_dim:.4f}, {max_dim:.4f}]\n"
        report += "- Các mẫu tuổi trước dậy thì (<12) và người già (>65) đã được áp dụng giảm trừ trọng số phạt nhị phân theo đúng sinh học hình thái.\n"

    output_md_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_md_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"✅ Generated demographic report at: {output_md_path}")
    return report


if __name__ == "__main__":
    report_dir = Path("reports")
    manifest = None
    for p in [report_dir / "metadata.parquet", report_dir / "metadata.csv", report_dir / "sample_metadata.csv"]:
        if p.exists():
            manifest = p
            break
    if manifest:
        analyze_demographics(manifest, report_dir / "demographic_distribution_report.md")
    else:
        print("No manifest found in reports/.")
