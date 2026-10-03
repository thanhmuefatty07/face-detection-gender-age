# SOTA Biometric Rebuild Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Xây dựng lại toàn diện hệ thống nhận diện khuôn mặt và ước lượng tuổi/giới tính chuẩn SOTA tại `C:\Users\ADMIN\face-detection-gender-age` với cấu trúc mô-đun hóa `src/biometrics/` và `src/mining/`, loại bỏ phụ thuộc cứng vào YOLO, áp dụng Dual-Engine Detector (SCRFD + YuNet), Umeyama similarity alignment 160×160, và Continuous Label Distribution Learning (LDL) kết hợp trọng số dị hình sinh học.

**Architecture:** Kiến trúc 2 chặng tách biệt (Decoupled Two-Stage Biometric Architecture). Chặng 1 sử dụng SCRFD/YuNet phát hiện khuôn mặt và 5 điểm mốc để căn chỉnh hình học Umeyama sang chuẩn ArcFace 160×160 kèm bộ lọc chất lượng cực đoan. Chặng 2 sử dụng backbone hiện đại (ConvNeXt-V2 / MobileNetV4) với các đầu ra đa nhiệm: Continuous 101-bin LDL Age (hàm mất mát Wasserstein/EMD), Giới tính có trọng số dị hình sinh học $w_{\text{dimorphism}}$, và định lượng độ bất định Aleatoric.

**Tech Stack:** Python 3.10+, PyTorch, OpenCV (YuNet C++ ONNX native), NumPy, SciPy, Timm, Pandas, PyArrow/FastParquet, Pytest.

**Spec:** [`docs/superpowers/specs/2026-10-04-biometric-rebuild-design.md`](file:///C:/Users/ADMIN/face-detection-gender-age/docs/superpowers/specs/2026-10-04-biometric-rebuild-design.md)

---

## Global Constraints
- Tất cả các tác vụ tính toán nặng (cào, trích xuất dữ liệu, huấn luyện mô hình) chạy 100% trên Kaggle Cloud GPU.
- Không tải gigabytes ảnh thô về máy local; repo local chỉ chứa mã nguồn, tests, manifests (Parquet/CSV) và báo cáo (Zero-Junk guarantee).
- Định dạng căn chỉnh khuôn mặt: ArcFace chuẩn hóa $160 \times 160$, 5 điểm mốc.
- Phân phối nhãn tuổi: Simplex xác suất 101 bins liên tục $[0, 100]$, tổng xác suất bằng $1.0 \pm 10^{-4}$.
- Bộ lọc chất lượng cực đoan: $\text{Var}(\text{Laplacian}) \ge 100$, Tenengrad $\ge 250$, Pose: $|Yaw| \le 35^\circ, |Pitch| \le 25^\circ, |Roll| \le 20^\circ$.
- Khử trùng lặp đa tầng: 64-bit DCT pHash (khoảng cách Hamming $d_H \le 6$ bị loại; $d_H > 6$ đảm bảo cách ly Train-Val).

---

## Review Focus
1. **Lỗi góc quay Pitch xấp xỉ $\pm 180^\circ$ khi camera chiếu thẳng**: Cần hàm `normalize_angle_to_frontal` để đưa góc về $[-90^\circ, 90^\circ]$.
2. **Sai số dấu chấm động IEEE 754 tại biên cực đoan**: Áp dụng $\epsilon = 10^{-4}$ tại các ngưỡng so sánh biên góc đầu.
3. **Hiện tượng clipping cảm biến tại tóc/đồng tử**: Ngưỡng cắt bóng tối $L < 8$ (thay vì $L < 15$) để không bắt nhầm mắt/tóc đen châu Á.
4. **Trẻ em $<12$ tuổi không phân hóa giới tính rõ nét**: Trọng số mất mát $w_{\text{dimorphism}} \in [0.25, 1.0]$ với độ bất định cao.
5. **Độc lập bộ phát hiện khi không có GPU rời**: Đảm bảo YuNet chạy mượt mà ngay trên CPU với zero-dependency.

---

## Task Decomposition

### Task 1: Dọn Dẹp Mã Nguồn Di Sản & Khởi Tạo Khung Gói Chuẩn (`src/`)
**Mục tiêu**: Xóa bỏ toàn bộ boilerplate cũ (`scripts/main.py`, `scripts/main_gpu.py`, các file `.bat` di sản), khởi tạo cấu trúc thư mục sạch sẽ `src/biometrics/` và `src/mining/`.
- [ ] Xóa `scripts/main.py`, `scripts/main_gpu.py`, `run_advanced.bat`, `run_gpu.bat`.
- [ ] Tạo cấu trúc thư mục `src/biometrics/core/`, `src/biometrics/detection/`, `src/biometrics/estimation/`, `src/mining/filtering/`, `src/mining/connectors/`.
- [ ] Tạo các file `__init__.py` tương ứng.
- [ ] Chạy `git status` và commit.

### Task 2: Core Biometrics & Căn Chỉnh Hình Học Umeyama (`src/biometrics/core/`)
**Mục tiêu**: Hiện thực hóa các cấu trúc dữ liệu nền tảng và bộ căn chỉnh hình học Umeyama 5 điểm mốc đưa về chuẩn ArcFace 160×160.
- [ ] Tạo `src/biometrics/core/types.py`: Dataclasses `BoundingBox`, `FaceLandmarks`, `BiometricPrediction`, `FaceAuditResult`.
- [ ] Tạo `src/biometrics/core/geometry.py`: Phép biến đổi đồng dạng Umeyama SVD giải nghiệm đóng, ước lượng 3D Euler angles solvePnP với hàm `normalize_angle_to_frontal`.
- [ ] Viết test `tests/unit/test_geometry.py` kiểm tra độ bất biến quay, co giãn, tịnh tiến và nghiệm đóng Umeyama.
- [ ] Chạy pytest và commit.

### Task 3: Động Cơ Phát Hiện Đa Tầng Dual-Engine (`src/biometrics/detection/`)
**Mục tiêu**: Hiện thực hóa bộ phát hiện khuôn mặt trừu tượng cùng 2 động cơ: SCRFD (SOTA chính xác cao) và YuNet (OpenCV C++ Native siêu nhẹ 85 KB).
- [ ] Tạo `src/biometrics/detection/base.py`: Interface `FaceDetectorBase`.
- [ ] Tạo `src/biometrics/detection/yunet_detector.py`: Tích hợp `cv2.FaceDetectorYN` siêu tốc, trích xuất 5 điểm mốc.
- [ ] Tạo `src/biometrics/detection/scrfd_detector.py`: Tích hợp bộ phát hiện InsightFace SCRFD ONNX Runtime.
- [ ] Viết test `tests/unit/test_detection.py` kiểm thử cả hai detector trên ảnh tổng hợp và kiểm tra định dạng 5 điểm mốc.
- [ ] Chạy pytest và commit.

### Task 4: Continuous LDL & Mô Hình Sinh Trắc Học Đa Nhiệm (`src/biometrics/estimation/`)
**Mục tiêu**: Hiện thực hóa động cơ học phân phối nhãn tuổi (Continuous LDL 101 bins), hàm mất mát Wasserstein/EMD, trọng số dị hình sinh học, và kiến trúc mạng ConvNeXt-V2 / MobileNetV4.
- [ ] Tạo `src/biometrics/estimation/dimorphism.py`: Hàm $w_{\text{dimorphism}}(age)$, mô hình hóa động học lão hóa chủng tộc (East Asian cliff, Caucasian, African).
- [ ] Tạo `src/biometrics/estimation/ldl_engine.py`: Sinh phân phối Gauss/Laplace liên tục trên simplex, hàm mất mát Earth Mover's Distance $\mathcal{L}_{\text{EMD}}$, kỳ vọng tuổi $\mu = \sum i \cdot p_i$.
- [ ] Tạo `src/biometrics/estimation/backbone.py`: Mô hình PyTorch đa nhiệm hỗ trợ ConvNeXt-V2 và MobileNetV4 với các head chuyên biệt (Age LDL, Gender BCE, Uncertainty).
- [ ] Viết test `tests/unit/test_ldl_dimorphism.py` kiểm thử bảo toàn xác suất $\sum p_i = 1.0 \pm 10^{-4}$ và hành vi biên tuổi.
- [ ] Chạy pytest và commit.

### Task 5: Bộ Lọc Chất Lượng Cực Đoan & Khử Trùng Lặp (`src/mining/filtering/`)
**Mục tiêu**: Tích hợp module kiểm định chất lượng quang học và khử trùng lặp 2 tầng (pHash + Cosine ArcFace) vào cấu trúc gói mới.
- [ ] Tạo `src/mining/filtering/quality_auditor.py`: Laplacian $\ge 100$, Tenengrad $\ge 250$, Pose Euler, trắc quang CIE L*a*b*, che khuất gò má $\Delta E$.
- [ ] Tạo `src/mining/filtering/deduplicator.py`: 64-bit DCT pHash ($d_H > 6$) và Cosine distance ($d_{\cos} \ge 0.35$).
- [ ] Viết test `tests/unit/test_auditor_dedup.py`.
- [ ] Chạy pytest và commit.

### Task 6: Unified Biometric Pipeline & CLI Runner (`src/biometrics/pipeline.py`)
**Mục tiêu**: Kết nối tất cả các khâu thành pipeline thống nhất, hỗ trợ xử lý ảnh đơn, thư mục, video và webcam trực tiếp.
- [ ] Tạo `src/biometrics/pipeline.py`: Lớp `BiometricPipeline` tự động chuyển đổi giữa YuNet và SCRFD, tự động căn chỉnh và dự đoán tuổi/giới tính.
- [ ] Tạo `src/biometrics/cli.py`: Giao diện dòng lệnh trực quan với argparse.
- [ ] Viết test tích hợp `tests/integration/test_pipeline_e2e.py`.
- [ ] Chạy pytest và commit.

### Task 7: Đồng Bộ Hóa Kaggle & Hoàn Tất Kiểm Thử Toàn Diện
**Mục tiêu**: Bảo đảm notebook trên Kaggle đồng bộ với kiến trúc mới, chạy toàn bộ test suite và xuất báo cáo hoàn thiện.
- [ ] Cập nhật notebook `notebooks/kaggle_face_mining_pipeline.ipynb` theo kiến trúc mới.
- [ ] Chạy toàn bộ test suite `pytest tests -v` (đạt 100% pass rate).
- [ ] Kiểm tra working tree git sạch sẽ, không có file rác untracked.
