#!/usr/bin/env python3
"""Real-time Webcam Face Detection with Tracking"""

import cv2
import torch
import torch.nn as nn
from pathlib import Path
from collections import deque, defaultdict
from torchvision import transforms
import timm
from ultralytics import YOLO

FACE_MODEL = "yolov11n-face.pt"
GA_MODEL = Path(__file__).parent.parent / "gender_age_model.pth"
AGE_GROUPS = ['0-2', '3-6', '7-12', '13-17', '18-24', '25-34', '35-44', '45-54', '55-64', '65+']
GENDERS = ['Male', 'Female']


class FaceTracker:
    def __init__(self, max_lost=15):
        self.next_id = 0
        self.faces = {}
        self.lost = defaultdict(int)
        self.max_lost = max_lost
        self.history = defaultdict(lambda: {'g': deque(maxlen=8), 'a': deque(maxlen=8)})
    
    def iou(self, b1, b2):
        x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
        x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
        inter = max(0, x2-x1) * max(0, y2-y1)
        union = (b1[2]-b1[0])*(b1[3]-b1[1]) + (b2[2]-b2[0])*(b2[3]-b2[1]) - inter
        return inter / (union + 1e-6)
    
    def update(self, boxes):
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
            if best[0] >= 0 and self.iou(old, best[1]) > 0.3:
                self.faces[fid] = best[1]
                self.lost[fid] = 0
                matched.add(best[0])
            else:
                self.lost[fid] += 1
                if self.lost[fid] > self.max_lost:
                    del self.faces[fid]
        
        for i, b in enumerate(boxes):
            if i not in matched:
                self.faces[self.next_id] = b
                self.next_id += 1
        return self.faces
    
    def smooth(self, fid, g, a):
        self.history[fid]['g'].append(g)
        self.history[fid]['a'].append(a)
        return max(set(self.history[fid]['g']), key=list(self.history[fid]['g']).count), \
               max(set(self.history[fid]['a']), key=list(self.history[fid]['a']).count)


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


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    face_model = YOLO(FACE_MODEL)
    
    ga_model, img_size = None, 224
    if GA_MODEL.exists():
        cp = torch.load(GA_MODEL, map_location=device)
        img_size = cp.get('img_size', 224)
        backbone = cp.get('backbone', 'tf_efficientnetv2_s')
        ga_model = GenderAgeModel(backbone)
        ga_model.load_state_dict(cp['model_state_dict'])
        ga_model.eval().to(device)
        print("✅ Gender/Age model loaded")
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    tracker = FaceTracker()
    cap = cv2.VideoCapture(0)
    print("Press Q to quit, T to toggle tracking")
    
    tracking = True
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        results = face_model(frame, conf=0.3, verbose=False)
        boxes = [[int(x) for x in b.xyxy[0].cpu().numpy()] for r in results for b in r.boxes]
        
        tracked = tracker.update(boxes) if tracking else {i: b for i, b in enumerate(boxes)}
        
        for fid, (x1, y1, x2, y2) in tracked.items():
            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue
            
            label, color = f"ID:{fid}", (0, 255, 0)
            
            if ga_model:
                inp = transform(cv2.cvtColor(face, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)
                with torch.no_grad():
                    g, a = ga_model(inp)
                    gi, ai = g.argmax(1).item(), a.argmax(1).item()
                    if tracking:
                        gi, ai = tracker.smooth(fid, gi, ai)
                    label = f"{fid}:{GENDERS[gi]} {AGE_GROUPS[ai]}"
                    color = (255, 150, 50) if gi == 0 else (180, 105, 255)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        cv2.putText(frame, f"Faces:{len(tracked)} Track:{'ON' if tracking else 'OFF'}", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow('Webcam', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('t'):
            tracking = not tracking
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
