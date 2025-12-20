#!/usr/bin/env python3
"""
Simple Webcam Face Detection (CPU Optimized)
Lightweight version with tracking support
"""

import cv2
import torch
import torch.nn as nn
import gc
import time
from pathlib import Path
from collections import deque, defaultdict
from torchvision import transforms
import timm
from ultralytics import YOLO

# Config
FACE_MODEL = "yolov11s-face.pt"
GA_MODEL = Path(__file__).parent.parent / "gender_age_model.pth"
AGE_GROUPS = ['0-2', '3-6', '7-12', '13-17', '18-24', '25-34', '35-44', '45-54', '55-64', '65+']
GENDERS = ['Male', 'Female']

# Validation
MIN_FACE_SIZE = 40
MAX_ASPECT_RATIO = 2.0
MIN_ASPECT_RATIO = 0.5


class FaceTracker:
    def __init__(self, max_lost=15, max_faces=5):
        self.next_id = 0
        self.faces = {}
        self.lost = defaultdict(int)
        self.max_lost = max_lost
        self.max_faces = max_faces
        self.history = defaultdict(lambda: {'g': deque(maxlen=5), 'a': deque(maxlen=5)})
    
    def iou(self, b1, b2):
        x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
        x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
        inter = max(0, x2-x1) * max(0, y2-y1)
        union = (b1[2]-b1[0])*(b1[3]-b1[1]) + (b2[2]-b2[0])*(b2[3]-b2[1]) - inter
        return inter / (union + 1e-6)
    
    def update(self, boxes):
        boxes = boxes[:self.max_faces]
        
        if not boxes:
            for k in list(self.faces):
                self.lost[k] += 1
                if self.lost[k] > self.max_lost:
                    del self.faces[k]
            return self.faces
        
        if not self.faces:
            for b in boxes:
                self.faces[self.next_id] = b
                self.next_id += 1
            return self.faces
        
        matched = set()
        for fid, old in list(self.faces.items()):
            best = max(enumerate(boxes), key=lambda x: self.iou(old, x[1]) if x[0] not in matched else 0, default=(-1, None))
            if best[0] >= 0 and self.iou(old, best[1]) > 0.35:
                self.faces[fid] = best[1]
                self.lost[fid] = 0
                matched.add(best[0])
            else:
                self.lost[fid] += 1
                if self.lost[fid] > self.max_lost:
                    del self.faces[fid]
        
        for i, b in enumerate(boxes):
            if i not in matched and len(self.faces) < self.max_faces:
                self.faces[self.next_id] = b
                self.next_id += 1
        
        # Cleanup old history
        if self.next_id % 50 == 0:
            active = set(self.faces.keys())
            for k in list(self.history.keys()):
                if k not in active:
                    del self.history[k]
        
        return self.faces
    
    def smooth(self, fid, g, a):
        self.history[fid]['g'].append(g)
        self.history[fid]['a'].append(a)
        return max(set(self.history[fid]['g']), key=list(self.history[fid]['g']).count), \
               max(set(self.history[fid]['a']), key=list(self.history[fid]['a']).count)
    
    def reset(self):
        self.faces.clear()
        self.lost.clear()
        self.history.clear()
        self.next_id = 0


class GenderAgeModel(nn.Module):
    def __init__(self, backbone='tf_efficientnetv2_s'):
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
        self.age = nn.Linear(512, 10)
    
    def forward(self, x):
        f = self.fc(self.backbone(x))
        return self.gender(f), self.age(f)


def is_valid_face(x1, y1, x2, y2):
    w, h = x2 - x1, y2 - y1
    if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
        return False
    aspect = w / (h + 1e-6)
    return MIN_ASPECT_RATIO <= aspect <= MAX_ASPECT_RATIO


def main():
    print("=" * 50)
    print("🚀 Webcam Face Detection (CPU Optimized)")
    print("=" * 50)
    print("Controls: Q=Quit, T=Toggle Tracking, R=Reset")
    print("=" * 50)
    
    device = torch.device('cpu')
    
    # Load face model
    model_path = Path(__file__).parent.parent / FACE_MODEL
    if model_path.exists():
        face_model = YOLO(str(model_path))
    else:
        face_model = YOLO("yolov11n-face.pt")
        print("⚠️ Using fallback model")
    
    # Load gender/age model
    ga_model, img_size = None, 224
    if GA_MODEL.exists():
        cp = torch.load(GA_MODEL, map_location=device, weights_only=False)
        img_size = cp.get('img_size', 224)
        backbone = cp.get('backbone', 'tf_efficientnetv2_s')
        ga_model = GenderAgeModel(backbone)
        ga_model.load_state_dict(cp['model_state_dict'])
        ga_model.eval()
        for p in ga_model.parameters():
            p.requires_grad = False
        print("✅ Models loaded")
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    tracker = FaceTracker()
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    tracking = True
    frame_count = 0
    skip = 3  # Process every 3rd frame
    last_tracked = {}
    fps = 0
    fps_counter = 0
    fps_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        fps_counter += 1
        
        # FPS
        if time.time() - fps_time >= 1.0:
            fps = fps_counter
            fps_counter = 0
            fps_time = time.time()
        
        # Detect on interval
        if frame_count % skip == 0:
            results = face_model(frame, conf=0.5, verbose=False, max_det=10)
            boxes = []
            for r in results:
                for b in r.boxes:
                    x1, y1, x2, y2 = map(int, b.xyxy[0].cpu().numpy())
                    if is_valid_face(x1, y1, x2, y2):
                        boxes.append([x1, y1, x2, y2])
            
            last_tracked = tracker.update(boxes) if tracking else {i: b for i, b in enumerate(boxes)}
        
        # Draw
        for fid, (x1, y1, x2, y2) in last_tracked.items():
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue
            
            label, color = f"ID:{fid}", (0, 255, 0)
            
            if ga_model and frame_count % skip == 0:
                try:
                    inp = transform(cv2.cvtColor(face, cv2.COLOR_BGR2RGB)).unsqueeze(0)
                    with torch.no_grad():
                        g, a = ga_model(inp)
                        gi, ai = g.argmax(1).item(), a.argmax(1).item()
                        if tracking:
                            gi, ai = tracker.smooth(fid, gi, ai)
                        label = f"{fid}:{GENDERS[gi]} {AGE_GROUPS[ai]}"
                        color = (255, 150, 50) if gi == 0 else (180, 105, 255)
                except Exception:
                    pass
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        info = f"FPS:{fps} Faces:{len(last_tracked)} Track:{'ON' if tracking else 'OFF'}"
        cv2.putText(frame, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow('Webcam', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('t'):
            tracking = not tracking
            print(f"Tracking: {'ON' if tracking else 'OFF'}")
        elif key == ord('r'):
            tracker.reset()
            gc.collect()
            print("🔄 Reset")
        
        # Cleanup
        if frame_count % 300 == 0:
            gc.collect()
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
