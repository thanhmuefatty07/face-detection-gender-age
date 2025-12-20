#!/usr/bin/env python3
"""
Advanced Face Detection with Tracking + Gender + Age
Optimized for CPU (no GPU required)
Features:
- Multi-target face tracking with ID assignment
- Face locking (follow specific face)
- Smoothed predictions (temporal averaging)
- Memory optimized for long-running sessions
"""

import cv2
import torch
import torch.nn as nn
import argparse
import gc
import time
from pathlib import Path
from collections import deque, defaultdict
from torchvision import transforms
import timm
from ultralytics import YOLO

# Config - Use larger model for better accuracy
FACE_MODEL = "yolov11s-face.pt"  # Upgraded from yolov11n for better detection
GA_MODEL = Path(__file__).parent.parent / "gender_age_model.pth"
AGE_GROUPS = ['0-2', '3-6', '7-12', '13-17', '18-24', '25-34', '35-44', '45-54', '55-64', '65+']
GENDERS = ['Male', 'Female']
COLORS = {'Male': (255, 150, 50), 'Female': (180, 105, 255), 'Locked': (0, 255, 0)}

# Face validation constants
MIN_FACE_SIZE = 40  # Minimum face width/height in pixels
MAX_ASPECT_RATIO = 2.0  # Maximum width/height ratio (filter non-face boxes)
MIN_ASPECT_RATIO = 0.5  # Minimum width/height ratio


class FaceTracker:
    """Optimized face tracker with memory management"""
    
    def __init__(self, max_disappeared=20, iou_threshold=0.35, max_faces=10):
        self.next_id = 0
        self.faces = {}
        self.disappeared = defaultdict(int)
        self.max_disappeared = max_disappeared
        self.iou_threshold = iou_threshold
        self.locked_id = None
        self.max_faces = max_faces
        
        # Smaller history buffer for faster processing
        self.history = defaultdict(lambda: {'gender': deque(maxlen=5), 'age': deque(maxlen=5)})
    
    def iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter = max(0, x2-x1) * max(0, y2-y1)
        area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
        area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
        
        return inter / (area1 + area2 - inter + 1e-6)
    
    def cleanup_old_history(self):
        """Remove history for faces no longer tracked"""
        active_ids = set(self.faces.keys())
        old_ids = [k for k in self.history.keys() if k not in active_ids]
        for old_id in old_ids:
            del self.history[old_id]
    
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
            best_iou = 0
            best_idx = -1
            for idx, new_box in enumerate(boxes):
                if idx in matched:
                    continue
                iou_score = self.iou(old_box, new_box)
                if iou_score > best_iou:
                    best_iou = iou_score
                    best_idx = idx
            
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
        
        # Cleanup every 50 new IDs
        if self.next_id % 50 == 0:
            self.cleanup_old_history()
        
        return self.faces
    
    def add_prediction(self, face_id, gender_idx, age_idx):
        self.history[face_id]['gender'].append(gender_idx)
        self.history[face_id]['age'].append(age_idx)
    
    def get_smoothed(self, face_id):
        h = self.history[face_id]
        if len(h['gender']) == 0:
            return None, None
        gender = max(set(h['gender']), key=list(h['gender']).count)
        age = max(set(h['age']), key=list(h['age']).count)
        return gender, age
    
    def lock_face(self, face_id):
        if face_id in self.faces:
            self.locked_id = face_id
            return True
        return False
    
    def unlock(self):
        self.locked_id = None
    
    def reset(self):
        """Full reset of tracker state"""
        self.faces.clear()
        self.disappeared.clear()
        self.history.clear()
        self.locked_id = None
        self.next_id = 0


class GenderAgeModel(nn.Module):
    def __init__(self, backbone='tf_efficientnetv2_s', num_ages=10):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=False, num_classes=0)
        n = self.backbone.num_features
        self.fc = nn.Sequential(
            nn.BatchNorm1d(n),
            nn.Dropout(0.3),
            nn.Linear(n, 512),
            nn.ReLU()
        )
        self.gender = nn.Linear(512, 2)
        self.age = nn.Linear(512, num_ages)
    
    def forward(self, x):
        f = self.fc(self.backbone(x))
        return self.gender(f), self.age(f)


def is_valid_face(x1, y1, x2, y2):
    """Check if bounding box is likely a face based on size and aspect ratio"""
    w = x2 - x1
    h = y2 - y1
    
    # Check minimum size
    if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
        return False
    
    # Check aspect ratio (faces are roughly square)
    aspect = w / (h + 1e-6)
    if aspect < MIN_ASPECT_RATIO or aspect > MAX_ASPECT_RATIO:
        return False
    
    return True


def load_models():
    device = torch.device('cpu')  # Force CPU
    print(f"🖥️ Running on CPU")
    
    # Check if upgraded model exists, fallback to nano
    model_path = Path(__file__).parent.parent / FACE_MODEL
    if not model_path.exists():
        fallback = "yolov11n-face.pt"
        print(f"⚠️ {FACE_MODEL} not found, using {fallback}")
        face_model = YOLO(fallback)
    else:
        face_model = YOLO(str(model_path))
        print(f"✅ Face model: {FACE_MODEL}")
    
    ga_model = None
    img_size = 224
    
    if GA_MODEL.exists():
        cp = torch.load(GA_MODEL, map_location=device, weights_only=False)
        img_size = cp.get('img_size', 224)
        backbone = cp.get('backbone', 'tf_efficientnetv2_s')
        
        ga_model = GenderAgeModel(backbone=backbone, num_ages=10)
        ga_model.load_state_dict(cp['model_state_dict'])
        ga_model.eval()
        
        # Set to inference mode for better CPU performance
        for param in ga_model.parameters():
            param.requires_grad = False
        
        print(f"✅ Gender/Age model loaded")
    else:
        print("⚠️ Gender/Age model not found - face detection only")
    
    return face_model, ga_model, device, img_size


def get_transform(img_size):
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='0', help='Camera ID or video path')
    parser.add_argument('--conf', type=float, default=0.5, help='Detection confidence (0.3-0.8)')
    parser.add_argument('--track', action='store_true', help='Enable tracking')
    parser.add_argument('--skip', type=int, default=3, help='Process every N frames (CPU: 3-5)')
    parser.add_argument('--max-faces', type=int, default=5, help='Max faces to track')
    parser.add_argument('--width', type=int, default=640, help='Camera width')
    parser.add_argument('--height', type=int, default=480, help='Camera height')
    args = parser.parse_args()
    
    print("=" * 50)
    print("🚀 Face Detection + Gender + Age (CPU Optimized)")
    print("=" * 50)
    print("Controls: Q=Quit, L=Lock, U=Unlock, R=Reset")
    print(f"Settings: conf={args.conf}, skip={args.skip}, max={args.max_faces}")
    print("=" * 50)
    
    face_model, ga_model, device, img_size = load_models()
    transform = get_transform(img_size)
    tracker = FaceTracker(max_faces=args.max_faces) if args.track else None
    
    # Video source
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    
    # Optimize camera settings for CPU
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    if not cap.isOpened():
        print(f"❌ Cannot open: {args.source}")
        return
    
    frame_count = 0
    last_tracked = {}
    fps = 0
    fps_counter = 0
    fps_time = time.time()
    
    # Auto-reset counter
    frames_since_reset = 0
    AUTO_RESET_FRAMES = 3000  # Reset every ~100 seconds at 30fps
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        frames_since_reset += 1
        fps_counter += 1
        
        # FPS calculation
        now = time.time()
        if now - fps_time >= 1.0:
            fps = fps_counter
            fps_counter = 0
            fps_time = now
        
        # Process frames at interval
        if frame_count % args.skip == 0:
            # Detect faces
            results = face_model(frame, conf=args.conf, verbose=False, max_det=args.max_faces * 2)
            boxes = []
            
            for r in results:
                for box in r.boxes:
                    conf = float(box.conf[0])
                    if conf >= args.conf:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                        
                        # Validate face box
                        if is_valid_face(x1, y1, x2, y2):
                            boxes.append([x1, y1, x2, y2])
            
            # Update tracker
            if tracker:
                last_tracked = tracker.update(boxes)
            else:
                last_tracked = {i: box for i, box in enumerate(boxes)}
        
        # Draw cached results
        for face_id, (x1, y1, x2, y2) in last_tracked.items():
            # Bounds check
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue
            
            gender_idx, age_idx = None, None
            
            # Classify only on detection frames
            if ga_model and frame_count % args.skip == 0:
                try:
                    inp = transform(cv2.cvtColor(face, cv2.COLOR_BGR2RGB)).unsqueeze(0)
                    with torch.no_grad():
                        g, a = ga_model(inp)
                        gender_idx = g.argmax(1).item()
                        age_idx = a.argmax(1).item()
                    
                    if tracker:
                        tracker.add_prediction(face_id, gender_idx, age_idx)
                except Exception:
                    pass
            
            # Get smoothed prediction
            if tracker:
                gender_idx, age_idx = tracker.get_smoothed(face_id)
            
            # Draw
            gender = GENDERS[gender_idx] if gender_idx is not None else None
            age = AGE_GROUPS[age_idx] if age_idx is not None else None
            
            is_locked = tracker and tracker.locked_id == face_id
            color = COLORS['Locked'] if is_locked else COLORS.get(gender, (0, 255, 0))
            
            label = f"ID:{face_id}"
            if gender:
                label += f" {gender}"
            if age:
                label += f" {age}"
            if is_locked:
                label = "[LOCK] " + label
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2 if not is_locked else 3)
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Info overlay
        info = f"FPS:{fps} | Faces:{len(last_tracked)}"
        if tracker:
            info += f" | Track:ON"
            if tracker.locked_id is not None:
                info += f" | Lock:ID{tracker.locked_id}"
        cv2.putText(frame, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow('Face Detection (CPU)', frame)
        
        # Key handling
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('l') and tracker and len(last_tracked) > 0:
            h, w = frame.shape[:2]
            cx, cy = w // 2, h // 2
            nearest = min(last_tracked.items(), 
                         key=lambda x: abs((x[1][0]+x[1][2])/2 - cx) + abs((x[1][1]+x[1][3])/2 - cy))
            tracker.lock_face(nearest[0])
            print(f"🔒 Locked: ID{nearest[0]}")
        elif key == ord('u') and tracker:
            tracker.unlock()
            print("🔓 Unlocked")
        elif key == ord('r') and tracker:
            tracker.reset()
            frames_since_reset = 0
            gc.collect()
            print("🔄 Tracker reset")
        
        # Auto-reset to prevent drift
        if tracker and frames_since_reset >= AUTO_RESET_FRAMES:
            tracker.cleanup_old_history()
            frames_since_reset = 0
            gc.collect()
        
        # Periodic memory cleanup
        if frame_count % 300 == 0:
            gc.collect()
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
