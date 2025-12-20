#!/usr/bin/env python3
"""
Advanced Face Detection with Tracking + Gender + Age
Features:
- Multi-target face tracking with ID assignment
- Face locking (follow specific face)
- Smoothed predictions (temporal averaging)
"""

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import numpy as np
from pathlib import Path
from collections import deque, defaultdict
from torchvision import transforms
import timm
from ultralytics import YOLO

# Config
FACE_MODEL = "yolov11n-face.pt"
GA_MODEL = Path(__file__).parent.parent / "gender_age_model.pth"
AGE_GROUPS = ['0-2', '3-6', '7-12', '13-17', '18-24', '25-34', '35-44', '45-54', '55-64', '65+']
GENDERS = ['Male', 'Female']
COLORS = {'Male': (255, 150, 50), 'Female': (180, 105, 255), 'Locked': (0, 255, 0)}


class FaceTracker:
    """Simple face tracker using IoU matching"""
    
    def __init__(self, max_disappeared=30, iou_threshold=0.3):
        self.next_id = 0
        self.faces = {}  # id -> box
        self.disappeared = defaultdict(int)
        self.max_disappeared = max_disappeared
        self.iou_threshold = iou_threshold
        self.locked_id = None
        
        # Prediction history for smoothing
        self.history = defaultdict(lambda: {'gender': deque(maxlen=10), 'age': deque(maxlen=10)})
    
    def iou(self, box1, box2):
        """Calculate IoU between two boxes [x1,y1,x2,y2]"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter = max(0, x2-x1) * max(0, y2-y1)
        area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
        area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
        
        return inter / (area1 + area2 - inter + 1e-6)
    
    def update(self, boxes):
        """Update tracker with new detections"""
        if len(boxes) == 0:
            for fid in list(self.faces.keys()):
                self.disappeared[fid] += 1
                if self.disappeared[fid] > self.max_disappeared:
                    del self.faces[fid]
                    del self.disappeared[fid]
            return self.faces
        
        if len(self.faces) == 0:
            for box in boxes:
                self.faces[self.next_id] = box
                self.next_id += 1
            return self.faces
        
        # Match existing faces with new detections
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
        
        # Add new faces
        for idx, box in enumerate(boxes):
            if idx not in matched:
                self.faces[self.next_id] = box
                self.next_id += 1
        
        return self.faces
    
    def add_prediction(self, face_id, gender_idx, age_idx):
        """Add prediction to history for smoothing"""
        self.history[face_id]['gender'].append(gender_idx)
        self.history[face_id]['age'].append(age_idx)
    
    def get_smoothed(self, face_id):
        """Get smoothed prediction from history"""
        h = self.history[face_id]
        if len(h['gender']) == 0:
            return None, None
        
        # Majority vote
        gender = max(set(h['gender']), key=list(h['gender']).count)
        age = max(set(h['age']), key=list(h['age']).count)
        return gender, age
    
    def lock_face(self, face_id):
        """Lock onto a specific face"""
        if face_id in self.faces:
            self.locked_id = face_id
            return True
        return False
    
    def unlock(self):
        """Unlock face tracking"""
        self.locked_id = None


class GenderAgeModel(nn.Module):
    """Model compatible with trained checkpoint"""
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


def load_models():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    face_model = YOLO(FACE_MODEL)
    
    ga_model = None
    img_size = 224
    backbone = 'tf_efficientnetv2_s'
    
    if GA_MODEL.exists():
        cp = torch.load(GA_MODEL, map_location=device)
        img_size = cp.get('img_size', 224)
        backbone = cp.get('backbone', 'tf_efficientnetv2_s')
        
        ga_model = GenderAgeModel(backbone=backbone, num_ages=10)
        ga_model.load_state_dict(cp['model_state_dict'])
        ga_model.eval().to(device)
        print(f"✅ Gender/Age model loaded (backbone: {backbone})")
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
    parser.add_argument('--source', default='0', help='Image/video path or camera ID')
    parser.add_argument('--conf', type=float, default=0.3, help='Face detection confidence')
    parser.add_argument('--track', action='store_true', help='Enable face tracking')
    args = parser.parse_args()
    
    print("🚀 Face + Gender + Age Detection (Advanced)")
    print("Controls: Q=Quit, L=Lock face, U=Unlock")
    
    face_model, ga_model, device, img_size = load_models()
    transform = get_transform(img_size)
    tracker = FaceTracker() if args.track else None
    
    # Video source
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print(f"❌ Cannot open source: {args.source}")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect faces
        results = face_model(frame, conf=args.conf, verbose=False)
        boxes = []
        
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                boxes.append([x1, y1, x2, y2])
        
        # Update tracker
        if tracker:
            tracked = tracker.update(boxes)
        else:
            tracked = {i: box for i, box in enumerate(boxes)}
        
        # Process each face
        for face_id, (x1, y1, x2, y2) in tracked.items():
            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue
            
            # Classify
            gender_idx, age_idx = None, None
            if ga_model:
                inp = transform(cv2.cvtColor(face, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)
                with torch.no_grad():
                    g, a = ga_model(inp)
                    gender_idx = g.argmax(1).item()
                    age_idx = a.argmax(1).item()
                
                # Smoothing
                if tracker:
                    tracker.add_prediction(face_id, gender_idx, age_idx)
                    gender_idx, age_idx = tracker.get_smoothed(face_id)
            
            # Draw
            gender = GENDERS[gender_idx] if gender_idx is not None else None
            age = AGE_GROUPS[age_idx] if age_idx is not None else None
            
            is_locked = tracker and tracker.locked_id == face_id
            color = COLORS['Locked'] if is_locked else COLORS.get(gender, (0, 255, 0))
            
            label = f"ID:{face_id} {gender or 'Face'} {age or ''}"
            if is_locked:
                label = f"🔒 {label}"
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2 if not is_locked else 3)
            cv2.putText(frame, label.strip(), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Info overlay
        info = f"Faces: {len(tracked)} | Track: {'ON' if tracker else 'OFF'}"
        if tracker and tracker.locked_id is not None:
            info += f" | Locked: ID{tracker.locked_id}"
        cv2.putText(frame, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow('Face Detection', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('l') and tracker and len(tracked) > 0:
            # Lock nearest face to center
            h, w = frame.shape[:2]
            cx, cy = w // 2, h // 2
            nearest = min(tracked.items(), key=lambda x: abs((x[1][0]+x[1][2])/2 - cx) + abs((x[1][1]+x[1][3])/2 - cy))
            tracker.lock_face(nearest[0])
            print(f"🔒 Locked face ID: {nearest[0]}")
        elif key == ord('u') and tracker:
            tracker.unlock()
            print("🔓 Unlocked")
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
