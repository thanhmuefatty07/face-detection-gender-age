# Face Detection & Age/Gender Estimation (Optimized)

## 🚀 Features
- **Face Detection**: YOLOv11s-face (Downscaled for speed)
- **Age/Gender**: MobileNetV3 + Dynamic Quantization + JIT
- **Performance**: Ultra low latency (Threaded I/O)
- **Tracking**: Real-time multi-face tracking
- **Optimization**: Zero DLL dependency issues

## 📂 Project Structure
```
├── models/
│   ├── yolov11s-face.pt          # Face detector
│   └── best_model.pth            # Age/Gender model (Main)
├── notebooks/
│   └── train_optimized.ipynb     # Training notebook (Kaggle)
├── scripts/
│   └── main.py                   # Main application script
├── run_advanced.bat              # One-click launcher
└── requirements.txt              # Dependencies
```

## 🛠️ Usage

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Application:**
   Double click `run_advanced.bat` or run:
   ```bash
   python scripts/main.py --source 0 --track
   ```

## ⌨️ Controls
- `Q`: Quit
- `L`: Lock focused face
- `U`: Unlock face
- `R`: Reset tracker
