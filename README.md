# SOTA Biometric Face Perception: Age, Gender & Re-ID

Production-grade real-time facial biometric perception and target tracking engine powered by **ConvNeXt-V2-Tiny** with Global Response Normalization (GRN), Multi-Scale Mixed Dual-Pooling, Continuous 101-Bin Gaussian Label Distribution Learning (LDL), and Epistemic Target Locking.

---

## 🚀 Key Architectural Highlights

- **SOTA Vision Backbone**: `ConvNeXt-V2-Tiny` (27.87M params, depths `[3, 3, 9, 3]`, channels `[96, 192, 384, 768]`) capturing fine epidermal micro-wrinkles and cranial morphology.
- **Multi-Scale Feature Fusion**: Mixed Dual-Pooling (`GAP || GMP`) across Stage 2 ($20 \times 20$), Stage 3 ($10 \times 10$), and Stage 4 ($5 \times 5$), preserving peak wrinkle depth without spatial dilution.
- **Strict Head Routing Isolation**:
  - *Age & Uncertainty*: Fused multi-scale biometric embedding (512-D).
  - *Gender & Re-ID (64-D)*: Strictly bound to Stage 4 macro-structure for facial expression and motion blur invariance.
- **Unified Mathematical Loss**: Cramér Distance (Squared EMD) + Soft-KL Divergence (LogSumExp FP16-safe) + Soft-L1 MAE + Masked Dimorphic BCE + L1-Detached Laplace Uncertainty Loss.
- **Multi-Target Tracking & Anti-Jitter**: 2-tier ByteTrack with 64-D Re-ID cosine recovery, Epistemic-Kalman continuous age filter, and Bounded Leaky Bayesian gender accumulator.
- **Dual-Mode Deployment**:
  - **Cloud Kaggle GPU Inference Tunnel**: iVCam DirectShow capture locally, YuNet C++ face detection (11ms), WebSocket streaming to Kaggle Dual Tesla T4 (<3.5ms inference), rendering a 60 FPS real-time HUD.
  - **Local Edge Standalone**: OpenVINO 2025 AVX2 acceleration running offline.

---

## 📂 Project Structure

```
├── configs/
│   └── pipeline_config.yaml         # Dataset mining and pipeline configuration
├── models/
│   └── .gitkeep                     # Local weights directory (downloaded from Kaggle)
├── notebooks/
│   ├── dataset-metadata.json        # Kaggle Dataset metadata
│   ├── kaggle_face_mining_pipeline.ipynb # Mining kernel for 54.9k ArcFace faces
│   ├── kernel-metadata.json         # Mining kernel metadata
│   └── training/
│       ├── kaggle_training_convnextv2.ipynb # Distributed DDP training notebook
│       └── kernel-metadata.json     # Training kernel metadata
├── reports/                         # Dataset audit & demographic reports
├── src/
│   ├── biometrics/
│   │   ├── core/                    # Umeyama SVD 1991 & 3D pose geometry
│   │   ├── detection/               # YuNet C++ & SCRFD face detectors
│   │   ├── estimation/              # ConvNeXt-V2, LDL engine, dimorphism, losses
│   │   ├── inference/               # Cloud client & OpenVINO async pipeline
│   │   └── tracking/                # ByteTrack & Epistemic TargetLocker
│   └── mining/
│       ├── augmentation/            # Photometric physical environmental transforms
│       └── filtering/               # Sharpness auditor & pHash deduplicator
├── tests/                           # Unit, adversarial & integration test suite (432 tests)
├── run_client.py                    # One-click desktop launcher for iVCam HUD
└── requirements.txt
```

---

## 🛠️ Installation & Setup

```bash
# Clone the repository and enter directory
cd face-detection-gender-age

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate      # Windows
source venv/bin/activate    # Linux / macOS

# Install dependencies
pip install -r requirements.txt
```

---

## 🎮 Usage

### Mode 1: Cloud Kaggle GPU Tunnel (Recommended for iVCam)
Connect your mobile camera via iVCam and launch the real-time HUD connected to the Kaggle Dual T4 WebSocket:
```bash
python run_client.py --server wss://<your-kaggle-tunnel-url>/ws/infer --camera 0
```
*Press `[ESC]` on the video window to quit cleanly.*

### Mode 2: Automated Testing
Execute the complete regression and verification test suite:
```bash
python -m pytest tests scripts/tests -q
```

---

## 👥 Authors
- Phạm Văn Thanh - 104240479
- Huỳnh Minh Đức - 104240434
- Hồ Tâm Nguyên - 104240573
- Lê Sỹ Hiếu - 104240169
- Hoàng Đào Ngọc Minh - 1042315

**Course**: Introduction to Computer Science  
**Instructor**: Dr. Phạm Trung Kiên  
**University**: Vietnamese-German University (VGU)
