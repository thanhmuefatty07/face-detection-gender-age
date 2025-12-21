#!/usr/bin/env python3
"""
Optimized Gender/Age Inference using PyTorch
This script uses the optimized MobileNetV3 model trained on Kaggle.

Usage:
    python scripts/main.py --source 0 --track
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

# Config
FACE_MODEL = "models/yolov11s-face.pt"
PYTORCH_MODEL = Path(__file__).parent.parent / "models/best_model.pth"
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
        
        # Backbone
        self.backbone = timm.create_model(backbone, pretrained=False, features_only=True)
        self.num_features = self.backbone.feature_info[-1]['num_chs']
        
        # Attention
        self.attention = ChannelAttention(self.num_features)
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # Shared
        self.shared_fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.num_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )
        
        # Heads
        self.gender_head = nn.Linear(256, 2)
        self.age_head = nn.Linear(256, num_ages)
    
    def forward(self, x):
        features = self.backbone(x)[-1]
        features = self.attention(features)
        features = self.pool(features).flatten(1)
        shared = self.shared_fc(features)
        return self.gender_head(shared), self.age_head(shared)


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
    
    def needs_classification(self, face_id, current_frame, cache_frames=25):
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


def is_valid_face(x1, y1, x2, y2):
    w, h = x2 - x1, y2 - y1
    if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
        return False
    aspect = w / (h + 1e-6)
    return MIN_ASPECT_RATIO <= aspect <= MAX_ASPECT_RATIO


def should_classify(x1, y1, x2, y2):
    w, h = x2 - x1, y2 - y1
    return w >= MIN_CLASSIFY_SIZE and h >= MIN_CLASSIFY_SIZE


def preprocess_face_torch(face_bgr, img_size=160, device='cpu'):
    """Preprocess face for inference with enhanced quality"""
    # Resize with high quality
    face = cv2.resize(face_bgr, (img_size, img_size), interpolation=cv2.INTER_LANCZOS4)
    
    # Apply CLAHE for better lighting normalization
    lab = cv2.cvtColor(face, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    face = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # BGR to RGB
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    # Normalize (0-1) and standardized with ImageNet stats
    face = face.astype(np.float32) / 255.0
    face = (face - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    # HWC to CHW
    face = face.transpose(2, 0, 1)
    # To Tensor
    face_tensor = torch.from_numpy(face).unsqueeze(0).float().to(device)
    return face_tensor


def load_models(img_size=160, device='cpu'):
    # Face model
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

    # Gender/Age PyTorch model
    ga_model = None
    if PYTORCH_MODEL.exists():
        try:
            print(f"Loading PyTorch model from {PYTORCH_MODEL}...")
            checkpoint = torch.load(PYTORCH_MODEL, map_location=device)
            
            # Init model structure
            ga_model = GenderAgeStudent(backbone='mobilenetv3_large_100', num_ages=10)
            ga_model.load_state_dict(checkpoint['model_state_dict'])
            ga_model.to(device)
            ga_model.eval()
            print("✅ PyTorch model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading PyTorch model: {e}")
    else:
        print(f"⚠️ PyTorch model not found: {PYTORCH_MODEL}")
    
    return face_model, ga_model


def open_camera(source, width, height):
    if platform.system() == 'Windows':
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(source)
    
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    if platform.system() == 'Windows':
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    
    return cap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='0', help='Camera ID')
    parser.add_argument('--conf', type=float, default=0.6, help='Confidence threshold (0.6 optimal)')
    parser.add_argument('--track', action='store_true', help='Enable tracking')
    parser.add_argument('--skip', type=int, default=5, help='Process every N frames (Higher=Faster)')
    parser.add_argument('--max-faces', type=int, default=3, help='Max faces')
    parser.add_argument('--width', type=int, default=640, help='Camera width')
    parser.add_argument('--height', type=int, default=480, help='Camera height')
    parser.add_argument('--det-width', type=int, default=320, help='Detection width (Smaller=Faster)')
    parser.add_argument('--img-size', type=int, default=160, help='Model input size')
    args = parser.parse_args()
    
    # 🚀 Performance Optimizations
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 50)
    print(f"🚀 ULTRA High Perf Mode (PyTorch {device})")
    print(f"   Skip Frames: {args.skip}")
    print(f"   Detect Res: {args.det_width}px (Native: {args.width}px)")
    print("=" * 50)
    
    face_model, ga_model = load_models(args.img_size, device)
    
    # 🔥 Dynamic Quantization + JIT (The Ultimate CPU Combo)
    if ga_model is not None and device.type == 'cpu':
        try:
            print("🚀 Optimizing: Dynamic Quantization + JIT...")
            # 1. Dynamic Quantization (Linear layers -> INT8)
            ga_model_quant = torch.quantization.quantize_dynamic(
                ga_model, {nn.Linear}, dtype=torch.qint8
            )
            # 2. JIT Trace
            dummy_input = torch.randn(1, 3, args.img_size, args.img_size).to(device)
            ga_model = torch.jit.trace(ga_model_quant, dummy_input)
            ga_model = torch.jit.freeze(ga_model)
            print("✅ Model FULLY optimized (Quantized + JIT)!")
        except Exception as e:
            print(f"⚠️ Optimization warning: {e}")

    tracker = FaceTracker(max_faces=args.max_faces) if args.track else None
    
    if face_model is None:
        return

    source = int(args.source) if args.source.isdigit() else args.source
    
    # 🚀 Threaded Camera (Decouples I/O from Inference)
    from threading import Thread
    class ThreadedCamera:
        def __init__(self, src=0, width=640, height=480):
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
            
        def start(self):
            Thread(target=self.update, args=(), daemon=True).start()
            return self
            
        def update(self):
            while not self.stopped:
                if not self.stream.isOpened(): continue
                (self.grabbed, self.frame) = self.stream.read()
                
        def read(self):
            return self.frame
            
        def stop(self):
            self.stopped = True
            self.stream.release()

    print("📷 Starting Threaded Camera...")
    cap = ThreadedCamera(source, args.width, args.height).start()
    time.sleep(1.0) # Warmup
    
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
                # 🚀 Downscale for detection only (Huge speedup)
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
                            # Scale back to original
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
                        # Clamp coords
                        h, w = frame.shape[:2]
                        fx1, fy1, fx2, fy2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
                        face = frame[fy1:fy2, fx1:fx2]
                        if face.size > 0:
                            try:
                                inp = preprocess_face_torch(face, args.img_size, device)
                                with torch.no_grad():
                                    g_out, a_out = ga_model(inp)
                                    gi = int(g_out.argmax(1).item())
                                    ai = int(a_out.argmax(1).item())
                                tracker.add_prediction(face_id, gi, ai, frame_count)
                            except: pass
                    
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
            
            info = f"FPS:{fps} | Skip:{args.skip} | Det:{args.det_width}px"
            cv2.putText(frame, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Face Detection (Ultra Fast)', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('l') and tracker and len(last_tracked) > 0:
                h, w = frame.shape[:2]
                cx, cy = w // 2, h // 2
                # Lock face closest to center
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


if __name__ == "__main__":
    main()
