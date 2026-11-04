# Face Detection, Age & Gender Biometric Estimation (`ConvNeXt-V2` @ 224×224)

End-to-end facial detection, $224 \times 224$ 5-point Umeyama similarity alignment, hierarchical age estimation (continuous $0\text{--}100$ years via 101-bin Label Distribution Learning + 23 fine-grained $3\text{--}5$ year intervals + 10 biological cohorts), sexual-dimorphism-weighted gender classification, uncertainty quantification, and multi-target tracking.

---

## 🚀 Architecture & Core Design

### 1. Multi-Task Vision Architecture (`BiometricMultiTaskModel`)
- **Photometric Retinex Stem (`PhotometricRetinexStem`)**: Models Lambertian + Specular surface optics via Multi-Scale Retinex log-reflectance ($\sigma \in \{1.2, 2.5\}$), a differentiable luminance exposure gate $w(Y) = 4Y(1-Y)$ suppressing highlight bloom and shadow read-noise, and ConvNeXt-V2 Global Response Normalization (GRN) residual gating.
- **Multi-Scale Backbone & Bilateral Symmetry (`ConvNeXt-V2-Tiny` + `MultiScaleBiometricFusion`)**:
  - Operates at canonical **$224 \times 224$** resolution (`ARCFACE_CANONICAL_224`).
  - Fuses **Stage 2** ($28 \times 28$, micro-epidermal rhytids & pores), **Stage 3** ($14 \times 14$, mesoscopic contours & landmarks), and **Stage 4** ($7 \times 7$, macro craniofacial bone structure) using **Coordinate Attention**, **Cauchy-Lorentzian Bilateral Craniofacial Symmetry** gating, Mixed Dual-Pooling ($\text{GAP} \parallel \text{GMP}$), and 1D GRN residual projection ($512\text{-D}$).
- **Decoupled Head Routing & Strict Gradient Isolation**:
  - **Head 1A — 23 Fine Age Intervals ($3\text{--}5$ Year Brackets)**: Predicts 23 developmental and aging intervals (`0-2`, `3-5`, ..., `96-100`) from the $512\text{-D}$ multi-scale embedding.
  - **Head 1B — 101-Bin Continuous Age LDL ($0\text{--}100$ Years)**: Coupled with Head 1A through an exact binary partition matrix $M \in \{0,1\}^{101 \times 23}$ and Gaussian interval focus ($\sigma = 1.6$).
  - **Head 2 — Craniofacial Dimorphic Gender**: Strictly routed from **Stage 4 ($256\text{-D}$) only**, isolating gender classification from high-frequency skin texture or motion blur.
  - **Head 3 — Aleatoric & Epistemic Uncertainty**: Predicts heteroscedastic log-variance $\log \sigma^2$, combined with bilateral Flip-TTA epistemic variance (`predict_smooth`).
  - **Head 4 — 64-D Unit-Hypersphere Re-ID**: Strictly routed from **Stage 4 ($256\text{-D}$) only** onto $\mathcal{S}^{63}$ for occlusion-resilient identity recovery.

### 2. Two-Stage Teacher–Student Knowledge Distillation & Model Soup
- **Stage 1 (`convnextv2_base` Teacher, 89.25M params)**: Trained on the curated $224 \times 224$ dataset to refine coarse interval annotations into smooth unimodal 101-bin age distributions (`refine_interval_ldl_with_teacher`) and impute high-confidence missing gender labels.
- **Stage 2 (`convnextv2_tiny` Student, 29.34M params, $3.04\times$ compression)**: Distilled via `BiometricDistillationLoss` (temperature-scaled 101-bin & 23-interval KL, confidence-gated gender distillation, and pairwise Re-ID relational similarity) combined with **Top-3 SWA Model Soup** weight averaging (`models/convnextv2_tiny_biometrics.pt`).

### 3. Temporal Anti-Jitter & Multi-Target Tracking
- **2-Tier ByteTrack + 64-D Re-ID**: Combines Kalman spatial IoU association with 64-D cosine distance recovery across temporary occlusions.
- **Epistemic Target Locker (`TargetLocker`)**: Uses an Epistemic-Kalman continuous age filter, 0.75-year hysteresis deadband for 23-bracket interval locking, and a Bounded Leaky Bayesian Log-Odds accumulator for zero-flip gender locking.

---

## 📊 Dataset Curation & Evaluation Benchmarks

### Curated Real-Human Dataset (`8,331` Faces @ $224 \times 224$)
Audited in [`reports/kaggle_dataset_audit_20261103.json`](reports/kaggle_dataset_audit_20261103.json) and [`reports/metadata.parquet`](reports/metadata.parquet):
- **13 Real-Human Benchmark Sources**: `appa_real` (1,345), `imdb_wiki` (1,103), `agedb` (855), `adience` (773), `morph` (714), `cacd` (699), `b3fd` (618), `afad` (489), `fgnet` (456), `fairface` (445), `lagenda` (366), `megaage_asian` (305), `axon_age` (163).
- **Zero Synthetic / Zero UTKFace Policy**: Strictly `0` UTKFace or synthetic records.
- **Zero Cross-Split Leakage**: `6,845` train / `722` val / `764` test (`0` cross-split subjects across 4,828 tracked identities; `0` cross-split pHash pairs across 389,661 comparisons at Hamming threshold $\le 6$).
- **100% Physical Quality Gate**: Verified $224 \times 224$ resolution, Laplacian variance $\ge 100$ (median $1,760.8$), Tenengrad energy $\ge 250$ (median $5,353.4$), CIE $L^* \in [35, 220]$, clipping $< 8\%$, and 3D `solvePnP` Euler pose bounds.

### Holdout Test Set Performance (`N = 764`, $224 \times 224$)
Evaluated in [`reports/training_evaluation_report.json`](reports/training_evaluation_report.json):

| Metric | Teacher (`convnextv2_base`) | Student (`convnextv2_tiny` Soup) |
| :--- | :---: | :---: |
| **Continuous Age MAE** ($0\text{--}100\text{y}$) | **5.86 years** | **5.96 years** |
| **Cumulative Score CS@5 / CS@8** | 59.55% / 75.52% | **59.69% / 76.31%** |
| **23-Interval Exact / $\pm 1$ / $\pm 2$ Acc** | 29.97% / 68.19% / 87.57% | **32.33% / 69.50% / 87.17%** |
| **23-Interval Mean Bracket Distance** | 1.29 brackets | **1.27 brackets** |
| **10-Cohort Exact / $\pm 1$ Cohort Acc** | 53.14% / 92.02% | **54.84% / 90.97%** |
| **Gender Accuracy (Overall / Adult Dimorphic)** | 93.20% / 93.37% | **91.87% / 93.19%** |
| **Gender ROC-AUC / F1 Score** | 0.9724 / 0.9158 | **0.9710 / 0.9018** |
| **Gender ECE / Brier Score** | — | **0.0528 / 0.0684** |

### Extreme Environmental Stress & Temporal Smoothness Benchmarks
- **6 Physical Corruption Regimes** (*raking directional light, extreme overexposure glare, extreme low-light sensor noise, motion blur PSF, partial occlusion, compound storm*):
  - Mean Corrupted Age MAE: **6.80 years** ($\Delta = 2.84\text{y}$ drift vs. clean).
  - Mean Corrupted Gender Accuracy: **91.21%** (**0.9656 ROC-AUC**, **95.99%** agreement vs. clean).
  - Monotonic uncertainty expansion under noise: **Verified (`true`)**.
- **Demographic Fairness (Fitzpatrick / ITA Skin-Tone Types I–VI)**:
  - Cross-group Gender Accuracy Std: **1.67%** | Cross-group Age MAE Std: **1.17 years**.
- **Inference Latency & TargetLocker Anti-Jitter ($224 \times 224$)**:
  - Batched GPU Latency ($B=32$): **6.34 ms / face** (**157.9 faces/sec**); Single-face latency: **16.75 ms** (**19.20 ms** with Flip-TTA).
  - Temporal Age Jitter: reduced by **89.65%** ($3.86\text{y} \rightarrow \mathbf{0.40\text{y}}$).
  - 23-Interval Flicker Rate: reduced by **96.33%** ($47.81\% \rightarrow \mathbf{1.75\%}$).
  - Temporal Gender Flip Rate: reduced from $2.63\%$ to **0.00%**.

---

## 📂 Project Structure

```text
├── configs/
│   └── pipeline_config.yaml                  # Dataset curation, 23-interval LDL & quality thresholds
├── models/
│   ├── convnextv2_tiny_biometrics.pt         # Trained ConvNeXt-V2-Tiny Student Soup checkpoint (224x224, 56.27 MiB)
│   └── face_detection_yunet_2023mar.onnx     # YuNet 5-point landmark face detector (227 KB)
├── notebooks/
│   ├── dataset-metadata.json                 # Kaggle Dataset metadata
│   ├── kaggle_face_mining_pipeline.ipynb     # 224x224 real-human face mining & filtering kernel
│   ├── kernel-metadata.json                  # Mining kernel metadata
│   └── training/
│       ├── kaggle_training_convnextv2.ipynb  # Two-Stage Teacher-Student KD + SWA training kernel
│       └── kernel-metadata.json              # Training kernel metadata
├── reports/
│   ├── dataset_audit_report.md               # Manifest & cross-split integrity summary
│   ├── demographic_distribution_report.md    # 8,331-sample 23-interval, 10-cohort & 13-source breakdown
│   ├── kaggle_dataset_audit_20261103.json    # Full forensic audit of 8,331 decoded 224x224 images
│   ├── metadata.csv                          # Master 8,331-record dataset manifest (CSV)
│   ├── metadata.parquet                      # Master 8,331-record dataset manifest (Parquet)
│   ├── morphometry_and_dimorphism_analysis.md
│   ├── training_evaluation_report.json       # Full clean, stress, fairness & latency evaluation report
│   └── *.png                                 # LDL, dimorphism & ethnic aging curve visualizations
├── scripts/
│   ├── kaggle_pipeline/                      # Dataset connectors, face engine, identity split, server & training kernel builders
│   ├── reports/                              # Demographic & morphometry report generators
│   └── tests/                                # Tier 1–5 requirement & adversarial test suites
├── src/
│   ├── biometrics/
│   │   ├── core/                             # 5-point Umeyama SVD alignment (112/160/224) & 3D pose geometry
│   │   ├── detection/                        # YuNet & SCRFD face detectors
│   │   ├── estimation/                       # ConvNeXt-V2 backbone, 23-interval & 101-bin LDL engine, dimorphism, KD & losses
│   │   ├── inference/                        # Cloud WebSocket client & local Edge/OpenVINO async pipeline
│   │   ├── tracking/                         # 2-tier ByteTrack & Epistemic TargetLocker
│   │   ├── cli.py                            # Single-image & webcam CLI inference entrypoint
│   │   └── pipeline.py                       # Unified BiometricPipeline
│   └── mining/
│       ├── augmentation/                     # Physical optics, illumination & sensor noise transforms
│       ├── filtering/                        # Extreme quality auditor & 64-bit DCT pHash deduplicator
│       ├── connectors.py                     # Multi-source dataset ingestion connectors
│       ├── curation.py                       # ExtremeDatasetCurator (224x224)
│       └── intervals.py                      # Canonical 23 fine age interval definitions
├── tests/                                    # Unit, integration & adversarial boundary test suites
├── run_client.py                             # Real-time video / iVCam HUD launcher (224x224)
├── pytest.ini
└── requirements.txt
```

---

## 🛠️ Installation

```bash
# Clone the repository (includes trained ConvNeXt-V2-Tiny and YuNet weights in models/)
git clone https://github.com/thanhmuefatty07/face-detection-gender-age.git
cd face-detection-gender-age

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate      # Windows
source venv/bin/activate     # Linux / macOS

# Install dependencies
pip install -r requirements.txt
```

---

## 🎮 Usage

### 1. Local Image or Webcam Inference (`src.biometrics.cli`)
Automatically loads `models/convnextv2_tiny_biometrics.pt` and `models/face_detection_yunet_2023mar.onnx` at $224 \times 224$ canonical resolution:
```bash
# Run on a single image
python -m src.biometrics.cli --image path/to/photo.jpg

# Run live on local webcam / iVCam
python -m src.biometrics.cli --webcam 0
```

### 2. Real-Time Multi-Target Locked HUD (`run_client.py`)
Run the FastAPI inference server locally (or on a Kaggle GPU tunnel) and launch the 60 FPS target-locked HUD at $224 \times 224$:
```bash
# Terminal 1: Start local or cloud FastAPI + WebSocket inference server (auto-loads models/convnextv2_tiny_biometrics.pt)
python scripts/kaggle_pipeline/kaggle_inference_server.py

# Terminal 2: Launch real-time HUD client (aligns faces at 224x224)
python run_client.py --server ws://localhost:8000/ws/infer --camera 0 --target-size 224
```
*Press `[ESC]` on the HUD window to exit.*

### 3. Python API (`BiometricPipeline` & `EdgeAsyncPipeline`)
```python
import cv2
from src.biometrics.pipeline import BiometricPipeline
from src.biometrics.inference.async_pipeline import EdgeAsyncPipeline

# Single-frame / image biometric audit & estimation (auto-sets target_size=224 with checkpoint)
pipeline = BiometricPipeline(checkpoint_path="models/convnextv2_tiny_biometrics.pt")
results = pipeline.process_frame(cv2.imread("path/to/photo.jpg"))

# Multi-target video tracking & TargetLocker anti-jitter (auto-loads models/convnextv2_tiny_biometrics.pt @ 224x224)
edge = EdgeAsyncPipeline()
locked_targets = edge.process_frame(cv2.imread("path/to/frame.jpg"))
```

### 4. Automated Test Suite
Run all unit, integration, pipeline, and adversarial tests across `tests/` and `scripts/tests/`:
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
