#!/usr/bin/env python3
"""
Universal Accelerated Inference with OpenVINO
Supports: Intel CPU (AVX optimized) | Intel GPU | AMD GPU* | NVIDIA GPU*
* GPU support depends on driver and OpenVINO plugins

Usage:
    python scripts/main_gpu.py --source 0 --track
"""

import cv2
import numpy as np
import argparse
import gc
import time
import platform
import torch
import torch.nn as nn
import timm
from pathlib import Path
from collections import deque, defaultdict
from ultralytics import YOLO

# Try importing OpenVINO
try:
    from openvino import Core
    HAS_OPENVINO = True
except ImportError:
    HAS_OPENVINO = False
    print("⚠️ OpenVINO not installed, using PyTorch fallback")

# Config
FACE_MODEL = "models/yolov11s-face.pt"
PYTORCH_MODEL = Path(__file__).parent.parent / "models/best_model.pth"
ONNX_MODEL = Path(__file__).parent.parent / "models/gender_age.onnx"
OV_MODEL = Path(__file__).parent.parent / "models/gender_age_ov"
AGE_GROUPS = ['0-2', '3-6', '7-12', '13-17', '18-24', '25-34', '35-44', '45-54', '55-64', '65+']
GENDERS = ['Male', 'Female']
COLORS = {'Male': (255, 150, 50), 'Female': (180, 105, 255), 'Locked': (0, 255, 0)}

# Validation - Tighter thresholds for higher accuracy
MIN_FACE_SIZE = 60
MIN_CLASSIFY_SIZE = 80  # Larger face = better accuracy
MAX_ASPECT_RATIO = 1.6
MIN_ASPECT_RATIO = 0.6


class ChannelAttention(nn.Module):
    """Channel attention module from training notebook"""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class GenderAgeStudent(nn.Module):
    """Student model definition matching the trained model"""
    def __init__(self, backbone: str = 'mobilenetv3_large_100', num_ages: int = 10):
        super().__init__()
        
        self.backbone = timm.create_model(backbone, pretrained=False, features_only=True)
        self.num_features = self.backbone.feature_info[-1]['num_chs']
        
        self.attention = ChannelAttention(self.num_features)
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        self.shared_fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.num_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )
        
        self.gender_head = nn.Linear(256, 2)
        self.age_head = nn.Linear(256, num_ages)
    
    def forward(self, x):
        features = self.backbone(x)[-1]
        features = self.attention(features)
        features = self.pool(features).flatten(1)
        shared = self.shared_fc(features)
        return self.gender_head(shared), self.age_head(shared)


def export_to_onnx(pytorch_model, onnx_path, img_size=160):
    """Export PyTorch model to ONNX"""
    print(f"🔄 Exporting model to ONNX: {onnx_path}")
    
    pytorch_model.eval()
    dummy_input = torch.randn(1, 3, img_size, img_size)
    
    torch.onnx.export(
        pytorch_model,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['gender', 'age'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'gender': {0: 'batch_size'},
            'age': {0: 'batch_size'}
        }
    )
    print(f"✅ ONNX model exported: {onnx_path}")


class FaceTracker:
    def __init__(self, max_disappeared=15, iou_threshold=0.4, max_faces=3):
        self.next_id = 0
        self.faces = {}
        self.disappeared = defaultdict(int)
        self.max_disappeared = max_disappeared
        self.iou_threshold = iou_threshold
        self.locked_id = None
        self.max_faces = max_faces
        self.history = defaultdict(lambda: {'gender': deque(maxlen=8), 'age': deque(maxlen=8)})
        self.cache = {}
    
    def iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2-x1) * max(0, y2-y1)
        area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
        area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
        return inter / (area1 + area2 - inter + 1e-6)
    
    def cleanup(self):
        active = set(self.faces.keys())
        for k in list(self.history.keys()):
            if k not in active:
                del self.history[k]
        for k in list(self.cache.keys()):
            if k not in active:
                del self.cache[k]
    
    def update(self, boxes):
        if len(boxes) == 0:
            for fid in list(self.faces.keys()):
                self.disappeared[fid] += 1
                if self.disappeared[fid] > self.max_disappeared:
                    del self.faces[fid]
                    del self.disappeared[fid]
            return self.faces
        
        boxes = boxes[:self.max_faces]
        if len(self.faces) == 0:
            for box in boxes:
                self.faces[self.next_id] = box
                self.next_id += 1
            return self.faces
        
        matched = set()
        for fid, old_box in list(self.faces.items()):
            best_iou, best_idx = 0, -1
            for idx, new_box in enumerate(boxes):
                if idx in matched:
                    continue
                score = self.iou(old_box, new_box)
                if score > best_iou:
                    best_iou, best_idx = score, idx
            
            if best_iou > self.iou_threshold:
                self.faces[fid] = boxes[best_idx]
                self.disappeared[fid] = 0
                matched.add(best_idx)
            else:
                self.disappeared[fid] += 1
                if self.disappeared[fid] > self.max_disappeared:
                    del self.faces[fid]
                    del self.disappeared[fid]
        
        for idx, box in enumerate(boxes):
            if idx not in matched and len(self.faces) < self.max_faces:
                self.faces[self.next_id] = box
                self.next_id += 1
        
        if self.next_id % 30 == 0:
            self.cleanup()
        return self.faces
    
    def needs_classification(self, face_id, current_frame, cache_frames=20):
        if face_id not in self.cache:
            return True
        _, _, cached_frame = self.cache[face_id]
        return (current_frame - cached_frame) >= cache_frames
    
    def add_prediction(self, face_id, gender_idx, age_idx, current_frame):
        self.history[face_id]['gender'].append(gender_idx)
        self.history[face_id]['age'].append(age_idx)
        g, a = self.get_smoothed(face_id)
        self.cache[face_id] = (g, a, current_frame)
    
    def get_smoothed(self, face_id):
        h = self.history[face_id]
        if len(h['gender']) == 0:
            return None, None
        gender = max(set(h['gender']), key=list(h['gender']).count)
        age = max(set(h['age']), key=list(h['age']).count)
        return gender, age
    
    def get_cached(self, face_id):
        if face_id in self.cache:
            g, a, _ = self.cache[face_id]
            return g, a
        return self.get_smoothed(face_id)
    
    def lock_face(self, face_id):
        if face_id in self.faces:
            self.locked_id = face_id
            return True
        return False
    
    def unlock(self):
        self.locked_id = None
    
    def reset(self):
        self.faces.clear()
        self.disappeared.clear()
        self.history.clear()
        self.cache.clear()
        self.locked_id = None
        self.next_id = 0


def should_classify(x1, y1, x2, y2):
    w, h = x2 - x1, y2 - y1
    return w >= MIN_CLASSIFY_SIZE and h >= MIN_CLASSIFY_SIZE


def preprocess_face(face_bgr, img_size=160):
    """Preprocess face for inference with enhanced quality"""
    # Resize first
    face = cv2.resize(face_bgr, (img_size, img_size), interpolation=cv2.INTER_LANCZOS4)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for better lighting
    lab = cv2.cvtColor(face, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    face = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # Convert to RGB
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    
    # Normalize with ImageNet stats
    face = face.astype(np.float32) / 255.0
    face = (face - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    face = face.transpose(2, 0, 1)  # HWC to CHW
    face = np.expand_dims(face, axis=0)  # Add batch dimension
    return face.astype(np.float32)


class OpenVINOModel:
    """OpenVINO model wrapper for GPU-accelerated inference
    
    Supports:
    - AUTO: Automatic device selection (best available)
    - CPU: Intel CPU with AVX/AVX2 optimization
    - GPU: Intel iGPU/dGPU
    - HYBRID: Combine CPU + GPU for maximum performance
    """
    
    def __init__(self, model_path, device='AUTO'):
        self.core = Core()
        self.available = self.core.available_devices
        self.device, self.device_display = self._select_best_device(device)
        
        print(f"📋 Available devices: {self.available}")
        print(f"✅ Selected device: {self.device_display}")
        
        # Load model
        if Path(model_path).is_dir():
            model = self.core.read_model(str(Path(model_path) / "model.xml"))
        else:
            model = self.core.read_model(str(model_path))
        
        # Compile for target device (no custom config for compatibility)
        try:
            self.compiled_model = self.core.compile_model(model, self.device)
        except Exception as e:
            print(f"⚠️ Failed to compile for {self.device}: {e}")
            print("⚠️ Falling back to CPU...")
            self.device = 'CPU'
            self.device_display = 'CPU (Fallback)'
            try:
                self.compiled_model = self.core.compile_model(model, 'CPU')
            except Exception as e2:
                raise RuntimeError(f"Failed to compile model: {e2}")
        
        self.infer_request = self.compiled_model.create_infer_request()
        
        # Get output nodes
        self.output_gender = self.compiled_model.output(0)
        self.output_age = self.compiled_model.output(1)
    
    def _select_best_device(self, requested):
        """Select best available device with HYBRID support"""
        available = self.available
        has_gpu = 'GPU' in available
        has_cpu = 'CPU' in available
        
        # HYBRID mode: Combine CPU and GPU
        if requested == 'HYBRID':
            if has_gpu and has_cpu:
                # MULTI plugin distributes inference across devices
                return 'MULTI:GPU,CPU', 'HYBRID (GPU+CPU)'
            elif has_gpu:
                return 'GPU', 'GPU (no CPU fallback)'
            else:
                return 'CPU', 'CPU (no GPU available)'
        
        # AUTO mode: Best single device
        if requested == 'AUTO':
            if has_gpu:
                return 'GPU', 'GPU (Auto)'
            return 'CPU', 'CPU (Auto)'
        
        # Specific device requested
        if requested == 'GPU':
            if has_gpu:
                return 'GPU', 'GPU'
            print("⚠️ GPU not available, using CPU")
            return 'CPU', 'CPU (GPU unavailable)'
        
        if requested == 'CPU':
            return 'CPU', 'CPU'
        
        # Unknown device
        print(f"⚠️ Unknown device {requested}, using AUTO")
        return self._select_best_device('AUTO')
    
    def __call__(self, input_tensor):
        """Run inference"""
        self.infer_request.infer({0: input_tensor})
        gender = self.infer_request.get_output_tensor(0).data
        age = self.infer_request.get_output_tensor(1).data
        return gender.copy(), age.copy()


def load_models(img_size=160, device='AUTO'):
    """Load face detection and gender/age models"""
    
    # Face model (YOLO)
    model_path = Path(__file__).parent.parent / FACE_MODEL
    try:
        if model_path.exists():
            face_model = YOLO(str(model_path))
        else:
            face_model = YOLO("yolov11n.pt")
            print("⚠️ Using fallback YOLO model")
    except Exception as e:
        print(f"⚠️ Error loading face model: {e}")
        face_model = None

    # Gender/Age model
    ga_model = None
    model_device = "CPU"  # Default
    
    if HAS_OPENVINO:
        # Try OpenVINO first
        if not ONNX_MODEL.exists():
            # Export from PyTorch to ONNX first
            if PYTORCH_MODEL.exists():
                print(f"🔄 Exporting PyTorch model to ONNX...")
                try:
                    checkpoint = torch.load(PYTORCH_MODEL, map_location='cpu')
                    pytorch_model = GenderAgeStudent(backbone='mobilenetv3_large_100', num_ages=10)
                    pytorch_model.load_state_dict(checkpoint['model_state_dict'])
                    pytorch_model.eval()
                    export_to_onnx(pytorch_model, ONNX_MODEL, img_size)
                except Exception as e:
                    print(f"❌ Error exporting to ONNX: {e}")
        
        if ONNX_MODEL.exists():
            try:
                print(f"🚀 Loading model with OpenVINO...")
                ga_model = OpenVINOModel(str(ONNX_MODEL), device)
                model_device = ga_model.device_display
            except Exception as e:
                print(f"❌ Error loading OpenVINO model: {e}")
    
    # Fallback to PyTorch
    if ga_model is None and PYTORCH_MODEL.exists():
        print("📦 Falling back to PyTorch...")
        try:
            checkpoint = torch.load(PYTORCH_MODEL, map_location='cpu')
            pytorch_model = GenderAgeStudent(backbone='mobilenetv3_large_100', num_ages=10)
            pytorch_model.load_state_dict(checkpoint['model_state_dict'])
            pytorch_model.eval()
            
            # Quantize + JIT
            pytorch_model = torch.quantization.quantize_dynamic(
                pytorch_model, {nn.Linear}, dtype=torch.qint8
            )
            dummy = torch.randn(1, 3, img_size, img_size)
            pytorch_model = torch.jit.trace(pytorch_model, dummy)
            pytorch_model = torch.jit.freeze(pytorch_model)
            
            ga_model = ('pytorch', pytorch_model)
            model_device = "CPU (PyTorch)"
        except Exception as e:
            print(f"❌ Error loading PyTorch model: {e}")
    
    return face_model, ga_model, model_device


class ThreadedCamera:
    """Threaded camera for non-blocking I/O"""
    def __init__(self, src=0, width=640, height=480):
        from threading import Thread
        
        if platform.system() == 'Windows':
            self.stream = cv2.VideoCapture(src, cv2.CAP_DSHOW)
            self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        else:
            self.stream = cv2.VideoCapture(src)
        
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.stream.set(cv2.CAP_PROP_FPS, 30)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False
        self._thread = Thread(target=self.update, daemon=True)
        
    def start(self):
        self._thread.start()
        return self
        
    def update(self):
        while not self.stopped:
            if not self.stream.isOpened():
                continue
            (self.grabbed, self.frame) = self.stream.read()
            
    def read(self):
        return self.frame
        
    def stop(self):
        self.stopped = True
        self.stream.release()


def main():
    parser = argparse.ArgumentParser(description="GPU-Accelerated Face Detection & Gender/Age")
    parser.add_argument('--source', default='0', help='Camera ID')
    parser.add_argument('--conf', type=float, default=0.6, help='Confidence threshold (0.6 optimal)')
    parser.add_argument('--track', action='store_true', help='Enable tracking')
    parser.add_argument('--skip', type=int, default=3, help='Process every N frames')
    parser.add_argument('--max-faces', type=int, default=3, help='Max faces')
    parser.add_argument('--width', type=int, default=640, help='Camera width')
    parser.add_argument('--height', type=int, default=480, help='Camera height')
    parser.add_argument('--det-width', type=int, default=320, help='Detection width')
    parser.add_argument('--img-size', type=int, default=160, help='Model input size')
    parser.add_argument('--device', default='HYBRID', help='Device: HYBRID (CPU+GPU), AUTO, CPU, GPU')
    args = parser.parse_args()
    
    print("=" * 60)
    print("🚀 UNIVERSAL ACCELERATED MODE (OpenVINO)")
    print("   Modes: HYBRID (CPU+GPU) | AUTO | CPU | GPU")
    print("   HYBRID = Maximum performance (combines all hardware)")
    print("=" * 60)
    
    face_model, ga_model, model_device = load_models(args.img_size, args.device)
    
    if face_model is None:
        print("❌ Face model not loaded!")
        return
    
    # Check model type
    is_openvino = not isinstance(ga_model, tuple)
    
    tracker = FaceTracker(max_faces=args.max_faces) if args.track else None
    
    source = int(args.source) if args.source.isdigit() else args.source
    
    print("📷 Starting Threaded Camera...")
    cap = ThreadedCamera(source, args.width, args.height).start()
    time.sleep(1.0)  # Warmup
    
    frame_count = 0
    last_tracked = {}
    fps = 0
    fps_counter = 0
    fps_time = time.time()
    
    try:
        while True:
            frame = cap.read()
            if frame is None:
                continue
            
            frame_count += 1
            fps_counter += 1
            
            now = time.time()
            if now - fps_time >= 1.0:
                fps = fps_counter
                fps_counter = 0
                fps_time = now
            
            if frame_count % args.skip == 0:
                # Downscale for detection
                h, w = frame.shape[:2]
                scale = args.det_width / w
                det_h = int(h * scale)
                frame_small = cv2.resize(frame, (args.det_width, det_h))
                
                results = face_model(frame_small, conf=args.conf, verbose=False, max_det=args.max_faces * 2)
                boxes = []
                
                for r in results:
                    for box in r.boxes:
                        if float(box.conf[0]) >= args.conf:
                            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                            boxes.append([
                                int(x1 / scale), int(y1 / scale),
                                int(x2 / scale), int(y2 / scale)
                            ])
                
                if tracker:
                    last_tracked = tracker.update(boxes)
                else:
                    last_tracked = {i: b for i, b in enumerate(boxes)}
            
            # Rendering Loop
            for face_id, (x1, y1, x2, y2) in last_tracked.items():
                gender_idx, age_idx = None, None
                
                if ga_model and tracker and should_classify(x1, y1, x2, y2):
                    if tracker.needs_classification(face_id, frame_count):
                        h, w = frame.shape[:2]
                        fx1, fy1, fx2, fy2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
                        face = frame[fy1:fy2, fx1:fx2]
                        if face.size > 0:
                            try:
                                inp = preprocess_face(face, args.img_size)
                                
                                if is_openvino:
                                    g_out, a_out = ga_model(inp)
                                else:
                                    # PyTorch fallback
                                    _, pytorch_model = ga_model
                                    inp_torch = torch.from_numpy(inp)
                                    with torch.no_grad():
                                        g_out, a_out = pytorch_model(inp_torch)
                                        g_out = g_out.numpy()
                                        a_out = a_out.numpy()
                                
                                gi = int(np.argmax(g_out, axis=1)[0])
                                ai = int(np.argmax(a_out, axis=1)[0])
                                tracker.add_prediction(face_id, gi, ai, frame_count)
                            except Exception as e:
                                pass
                    
                    gender_idx, age_idx = tracker.get_cached(face_id)
                
                # Draw
                color = (0, 255, 0)
                label = f"ID:{face_id}"
                if gender_idx is not None:
                    label = f"{GENDERS[gender_idx]} {AGE_GROUPS[age_idx]}"
                    color = COLORS.get(GENDERS[gender_idx], color)
                
                if tracker and tracker.locked_id == face_id:
                    label = "[LOCK] " + label
                    color = COLORS['Locked']
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Info overlay
            info = f"FPS:{fps} | {model_device} | Skip:{args.skip}"
            cv2.putText(frame, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.imshow('Face Detection (OpenVINO Accelerated)', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('l') and tracker and len(last_tracked) > 0:
                h, w = frame.shape[:2]
                cx, cy = w // 2, h // 2
                nearest_id = min(last_tracked.items(), 
                              key=lambda x: abs((x[1][0]+x[1][2])/2 - cx) + abs((x[1][1]+x[1][3])/2 - cy))[0]
                tracker.lock_face(nearest_id)
                print(f"🔒 Locked: ID{nearest_id}")
            elif key == ord('u') and tracker:
                tracker.unlock()
            elif key == ord('r') and tracker:
                tracker.reset()
                
            if frame_count % 300 == 0:
                gc.collect()

    except KeyboardInterrupt:
        pass
    finally:
        cap.stop()
        cv2.destroyAllWindows()
        print("👋 Goodbye!")


if __name__ == "__main__":
    main()
