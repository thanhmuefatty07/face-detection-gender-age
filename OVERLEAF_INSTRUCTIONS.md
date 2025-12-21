# 📄 LaTeX Report - Hướng Dẫn Sử Dụng Overleaf

## Cách upload lên Overleaf

### Bước 1: Truy cập Overleaf
1. Vào https://www.overleaf.com
2. Đăng ký/Đăng nhập (miễn phí)

### Bước 2: Tạo Project Mới
1. Click **"New Project"** → **"Upload Project"**
2. Hoặc: **"New Project"** → **"Blank Project"** → Đặt tên project

### Bước 3: Upload File
1. Nếu chọn "Blank Project":
   - Xóa nội dung file `main.tex` mặc định
   - Copy toàn bộ nội dung từ file `TECHNICAL_REPORT.tex` vào
2. Hoặc upload trực tiếp file `.tex`

### Bước 4: Compile
1. Click nút **"Recompile"** (màu xanh) góc phải
2. Đợi ~10-20 giây để render PDF
3. Nếu có lỗi:  Check **"Logs and output files"**

### Bước 5: Download PDF
1. Sau khi compile thành công, click **"Download PDF"**

---

## Cấu trúc Report

```
1. Title Page
   - Tên trường, khoa
   - Tên đề tài
   - Thông tin giảng viên và sinh viên

2. Abstract
   - Tóm tắt ngắn gọn về dự án

3. Introduction (Giới thiệu)
   - Motivation
   - Problem Statement
   - Proposed Solution
   - Contributions

4. Related Work (Công trình liên quan)
   - Face Detection Methods (3 thế hệ)
   - Gender/Age Estimation
   - Knowledge Distillation
   - Label Distribution Learning

5. Methodology (Phương pháp)
   - System Architecture
   - YOLOv11s Face Detection
   - MobileNetV3 Gender/Age Model
   - Multi-Face Tracking Algorithm

6. Implementation (Triển khai)
   - Training Configuration
   - Data Augmentation
   - Inference Optimizations

7. Experiments (Thí nghiệm)
   - Datasets
   - Evaluation Metrics
   - Comparative Analysis
   - Ablation Study

8. Industrial Applications (Ứng dụng thực tế)
   - Tesla, Amazon, Google, TikTok, etc.

9. Conclusion (Kết luận)
   - Kết quả đạt được
   - Hạn chế
   - Hướng phát triển

10. References (14 trích dẫn học thuật)
```

---

## Thông tin Report

| Mục | Chi tiết |
|-----|----------|
| **Số trang** | ~12-15 trang |
| **Số bảng** | 8 bảng |
| **Số hình** | 1 TikZ diagram |
| **Số công thức** | 10+ equations |
| **Số trích dẫn** | 14 papers |
| **Ngôn ngữ** | Tiếng Anh (chuẩn học thuật) |

---

## Files trong project

```
widerface_yolo_dataset/
├── TECHNICAL_REPORT.tex    ← File LaTeX chính (upload lên Overleaf)
├── TECHNICAL_REPORT.md     ← Phiên bản Markdown
├── TECHNICAL_REPORT.html   ← Phiên bản HTML
├── scripts/
│   ├── main.py             ← Script chính (PyTorch)
│   └── main_gpu.py         ← Script GPU (OpenVINO)
├── models/
│   ├── yolov11s-face.pt    ← Face detection model
│   └── best_model.pth      ← Gender/Age model
└── notebooks/
    └── train_optimized.ipynb ← Training notebook
```

---

## Lưu ý quan trọng

1. **Overleaf hỗ trợ TikZ** - Diagram sẽ tự động render
2. **Packages đã được include** - Không cần cài thêm
3. **Vietnamese support** - Đã có `\usepackage[vietnamese]{babel}`
4. **Hyperlinks** - Tất cả links trong PDF có thể click được

---

## Troubleshooting

**Lỗi "Missing $ inserted":**
- Check các ký tự đặc biệt như `_` hoặc `%` - cần escape với `\_` hoặc `\%`

**Lỗi "Undefined control sequence":**
- Đảm bảo compile với pdfLaTeX (mặc định của Overleaf)

**Lỗi font tiếng Việt:**
- Thử đổi compiler sang XeLaTeX trong Settings → Compiler

---

**Happy LaTeXing! 📝**
