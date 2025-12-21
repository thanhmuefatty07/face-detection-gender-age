# Face Detection & Gender/Age Estimation

Real-time face detection with gender and age estimation using deep learning.

## 🚀 Features
- **Face Detection**: YOLOv11s-face (optimized for speed)
- **Gender/Age Estimation**: MobileNetV3 with Knowledge Distillation
- **Hardware Acceleration**: OpenVINO (Intel CPU/GPU), DirectML (AMD GPU)
- **Tracking**: Multi-face IOU tracking with temporal smoothing
- **Preprocessing**: CLAHE histogram equalization for better lighting

## 📂 Project Structure
```
├── models/
│   ├── yolov11s-face.pt          # Face detector
│   ├── best_model.pth            # Gender/Age model (PyTorch)
│   └── gender_age.onnx           # Gender/Age model (ONNX)
├── scripts/
│   ├── main.py                   # PyTorch version
│   └── main_gpu.py               # OpenVINO accelerated version
├── run_gpu.bat                   # Auto-detect GPU launcher
├── run_advanced.bat              # PyTorch launcher
└── requirements.txt
```

## 🛠️ Installation

```bash
# Create virtual environment
python -m venv venv

# Activate it
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

## 🎮 Usage

### Option 1: Auto-detect GPU (Recommended)
Double-click `run_gpu.bat` or run:
```bash
python scripts/main_gpu.py --source 0 --track --device HYBRID
```

### Option 2: PyTorch only
```bash
python scripts/main.py --source 0 --track
```

### Command Line Options
| Argument | Default | Description |
|----------|---------|-------------|
| `--source` | 0 | Camera ID or video path |
| `--conf` | 0.6 | Detection confidence threshold (0.6 optimal) |
| `--track` | False | Enable face tracking |
| `--device` | HYBRID | AMD, HYBRID, AUTO, CPU, GPU |
| `--max-faces` | 3 | Maximum faces to track |

## ⌨️ Controls
- `Q`: Quit application
- `L`: Lock focused face
- `U`: Unlock face
- `R`: Reset tracker

## 👥 Authors
- Phạm Văn Thanh - 104240479
- Huỳnh Minh Đức - 104240434
- Hồ Tâm Nguyên - 104240573
- Lê Sỹ Hiếu - 104240169
- Hoàng Đào Ngọc Minh - 1042315

**Course**: Introduction to Computer Science  
**Instructor**: Dr. Phạm Trung Kiên  
**University**: Vietnamese-German University (VGU)
