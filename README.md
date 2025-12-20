# 🎯 Face Detection + Gender + Age

Real-time face detection with gender classification and age estimation using YOLO + EfficientNetV2.

## 🚀 Quick Start

### Option 1: Double-Click (Easiest!)
- **`run_advanced.bat`** - Run with tracking + face locking

### Option 2: Terminal
```bash
cd c:\Users\ADMIN\Downloads\widerface_yolo_dataset
python scripts/inference.py --source 0 --track
```

## 🎮 Controls

| Key | Function |
|-----|----------|
| `Q` | Quit |
| `T` | Toggle tracking (webcam.py) |
| `L` | Lock nearest face (inference.py) |
| `U` | Unlock face (inference.py) |

## 📊 Features

- ✅ Real-time face detection (YOLOv11n-Face trained on WiderFace)
- ✅ Gender classification (Male/Female)
- ✅ Age estimation (10 groups: 0-2, 3-6, 7-12, 13-17, 18-24, 25-34, 35-44, 45-54, 55-64, 65+)
- ✅ Multi-face tracking with ID persistence
- ✅ Face locking (follow specific person)
- ✅ Smoothed predictions (temporal averaging)

## 📁 Project Structure

```
├── run_advanced.bat      # Launcher (tracking + face locking)
├── gender_age_model.pth  # Gender/Age model (84MB)
├── yolov11n-face.pt      # YOLO Face Detector (5.5MB)
├── scripts/
│   ├── webcam.py         # Basic webcam detection
│   ├── inference.py      # Advanced features
│   └── train.py          # Training script
├── configs/
│   └── widerface.yaml    # Dataset config
├── images/               # Dataset images
└── labels/               # Dataset labels
```

## 🔧 Requirements

- Python 3.11+
- PyTorch
- OpenCV
- timm
- ultralytics
