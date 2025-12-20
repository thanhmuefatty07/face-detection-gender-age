#!/usr/bin/env python3
"""Train YOLO Face Detection on WiderFace"""

from pathlib import Path
from ultralytics import YOLO

def main():
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    config = project_dir / "configs" / "widerface.yaml"
    
    model = YOLO("yolov11n-face.pt")
    
    model.train(
        data=str(config),
        epochs=50,
        imgsz=640,
        batch=8,
        device="0" if __import__('torch').cuda.is_available() else "cpu",
        workers=0,
        patience=10,
        project=str(project_dir / "runs"),
        name="face_detect",
        exist_ok=True
    )
    
    print("Done! Model saved to runs/face_detect/weights/best.pt")

if __name__ == "__main__":
    main()
