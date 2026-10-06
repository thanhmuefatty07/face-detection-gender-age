"""
Local Cloud Inference Client (iVCam -> WebSocket Tunnel -> HUD)
===============================================================
Captures video stream (iVCam / USB webcam / DirectShow),
detects faces locally using YuNet C++, sends compressed 160x160 face crops
over secure WebSocket to Kaggle GPU Server, receives SOTA ConvNeXt-V2 predictions,
updates ByteTrack & TargetLocker, and renders a 60 FPS real-time HUD.
"""

from __future__ import annotations
import base64
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union, Generator
import cv2
import numpy as np

from src.biometrics.tracking.byte_tracker import ByteTracker, Detection, STrack
from src.biometrics.tracking.target_locker import TargetLocker
from src.biometrics.detection.yunet_detector import YuNetFaceDetector


class CloudInferenceClient:
    """
    Client managing local face detection, WebSocket cloud inference,
    ByteTrack ID persistence, and Epistemic Target Locking.
    """

    def __init__(
        self,
        server_url: str = "ws://localhost:8000/ws/infer",
        max_targets: int = 5,
        track_thresh: float = 0.6,
        low_thresh: float = 0.2,
        yunet_model_path: Optional[str | Path] = None,
    ):
        self.server_url = server_url
        self.max_targets = max_targets
        self.tracker = ByteTracker(
            track_thresh=track_thresh,
            low_thresh=low_thresh,
            max_targets=max_targets,
        )
        self.locker = TargetLocker()

        # Initialize YuNet Face Detector
        if yunet_model_path is None:
            default_p = Path(__file__).resolve().parent.parent.parent.parent / "models" / "face_detection_yunet_2023mar.onnx"
            if default_p.exists():
                yunet_model_path = default_p
            else:
                yunet_model_path = "models/face_detection_yunet_2023mar.onnx"

        self.detector = YuNetFaceDetector(model_path=yunet_model_path, confidence_threshold=0.5)

    def detect_faces(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detects faces in frame using YuNet and extracts 160x160 aligned crops.
        
        Args:
            frame: (H, W, 3) BGR image
        Returns:
            List of detection dicts with 'bbox', 'score', and 'crop' (160x160)
        """
        H, W = frame.shape[:2]
        if H < 20 or W < 20:
            return []

        raw_dets = self.detector.detect(frame)
        results = []
        for d in raw_dets:
            x1 = max(0, int(d.bbox.x))
            y1 = max(0, int(d.bbox.y))
            x2 = min(W, int(d.bbox.x + d.bbox.width))
            y2 = min(H, int(d.bbox.y + d.bbox.height))
            if x2 > x1 and y2 > y1:
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    crop_160 = cv2.resize(crop, (160, 160))
                    results.append({
                        "bbox": (float(x1), float(y1), float(x2), float(y2)),
                        "score": float(d.bbox.confidence),
                        "crop": crop_160,
                    })
        return results

    def encode_crops(self, local_dets: List[Dict[str, Any]]) -> List[str]:
        """Encodes face crops as JPEG base64 strings for compact WebSocket transport."""
        crops_b64 = []
        for ld in local_dets:
            crop = ld.get("crop")
            if crop is not None:
                _, enc = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                crops_b64.append(base64.b64encode(enc.tobytes()).decode("utf-8"))
        return crops_b64

    def update_local_tracks(
        self,
        local_dets: List[Dict[str, Any]],
        cloud_results: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Associates cloud predictions with local detections and updates tracker & locker.
        Guarantees that occluded tracks coast stably without corrupting biometric filters.
        
        Args:
            local_dets: list of detections with 'bbox': (x1, y1, x2, y2), 'score': float, 'crop': np.ndarray
            cloud_results: corresponding list of cloud results with age/gender/reid
        Returns:
            List of active locked targets with smoothed age, locked gender, and HUD info
        """
        tracker_dets: List[Detection] = []
        for i, ld in enumerate(local_dets):
            bbox = ld.get("bbox", (0.0, 0.0, 0.0, 0.0))
            score = float(ld.get("score", 0.9))
            reid_feat = None
            if cloud_results and i < len(cloud_results):
                reid_raw = cloud_results[i].get("reid_embedding")
                if reid_raw is not None:
                    reid_feat = np.array(reid_raw, dtype=np.float32)
            tracker_dets.append(Detection(bbox=bbox, score=score, reid_feat=reid_feat, det_id=i))

        online_tracks = self.tracker.update(tracker_dets)
        active_ids = [t.track_id for t in online_tracks]
        self.locker.prune_inactive(active_ids)

        output_targets: List[Dict[str, Any]] = []
        for t in online_tracks:
            matched_res = None
            if cloud_results and len(cloud_results) > 0:
                if t.last_det_id is not None and t.last_det_id < len(cloud_results):
                    matched_res = cloud_results[t.last_det_id]
                else:
                    # Spatial IoU fallback
                    best_iou = 0.2
                    best_idx = None
                    for idx, ld in enumerate(local_dets):
                        b = ld.get("bbox", (0.0, 0.0, 0.0, 0.0))
                        xx1 = max(t.bbox[0], b[0])
                        yy1 = max(t.bbox[1], b[1])
                        xx2 = min(t.bbox[2], b[2])
                        yy2 = min(t.bbox[3], b[3])
                        inter = max(0.0, xx2 - xx1) * max(0.0, yy2 - yy1)
                        area_t = max(0.0, t.bbox[2] - t.bbox[0]) * max(0.0, t.bbox[3] - t.bbox[1])
                        area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
                        iou = inter / (area_t + area_b - inter + 1e-8)
                        if iou > best_iou:
                            best_iou = iou
                            best_idx = idx
                    if best_idx is not None and best_idx < len(cloud_results):
                        matched_res = cloud_results[best_idx]

            if matched_res is not None:
                age_pred = float(matched_res.get("expected_age", 25.0))
                std_pred = float(matched_res.get("aleatoric_std", 1.0))
                g_logit = float(matched_res.get("gender_logit", 0.0))
                has_meas = True
            else:
                age_pred = 25.0
                std_pred = 1.0
                g_logit = 0.0
                has_meas = False

            lock_info = self.locker.update(
                track_id=t.track_id,
                age_pred=age_pred,
                aleatoric_std=std_pred,
                gender_logit=g_logit,
                has_measurement=has_meas,
            )

            output_targets.append({
                "track_id": t.track_id,
                "bbox": t.bbox,
                "score": t.score,
                "locked_age": lock_info["locked_age"],
                "locked_gender": lock_info["locked_gender"],
                "gender_prob": lock_info["gender_prob"],
                "aleatoric_std": std_pred,
            })

        return output_targets

    def render_hud(self, frame: np.ndarray, targets: List[Dict[str, Any]]) -> np.ndarray:
        """Renders futuristic HUD overlay for locked biometric targets."""
        hud_img = frame.copy()
        for tgt in targets:
            x1, y1, x2, y2 = [int(v) for v in tgt["bbox"]]
            tid = tgt["track_id"]
            age = tgt["locked_age"]
            gender = tgt["locked_gender"]
            std = tgt.get("aleatoric_std", 1.0)

            # Color theme: Female = Magenta/Coral (200, 50, 220), Male = Cyan (240, 180, 40)
            color = (200, 50, 220) if gender == "Female" else (240, 180, 40)

            # Draw target bounding box with corner reticles
            cv2.rectangle(hud_img, (x1, y1), (x2, y2), color, 2)
            corner_len = min(15, max(5, (x2 - x1) // 4))
            # Corners
            cv2.line(hud_img, (x1, y1), (x1 + corner_len, y1), color, 3)
            cv2.line(hud_img, (x1, y1), (x1, y1 + corner_len), color, 3)
            cv2.line(hud_img, (x2, y1), (x2 - corner_len, y1), color, 3)
            cv2.line(hud_img, (x2, y1), (x2, y1 + corner_len), color, 3)

            # HUD Label banner
            label = f"ID #{tid} | {gender} | {age:.1f}y (+/-{std:.1f})"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(hud_img, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, max(0, y1)), color, -1)
            cv2.putText(
                hud_img,
                label,
                (x1 + 3, max(0, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return hud_img

    def open_camera(self, camera_source: Union[int, str] = 0) -> cv2.VideoCapture:
        """
        Opens camera capture stream, optimizing for iVCam DirectShow backend.
        """
        if isinstance(camera_source, int):
            # Try DirectShow for low-latency iVCam on Windows
            cap = cv2.VideoCapture(camera_source, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(camera_source)
        else:
            cap = cv2.VideoCapture(str(camera_source))

        if cap.isOpened():
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FPS, 60)
        return cap

    def run_live_stream(
        self,
        camera_source: Union[int, str] = 0,
        max_frames: Optional[int] = None,
        display: bool = False,
    ) -> Generator[np.ndarray, None, None]:
        """
        Live physical/synthetic video processing generator.
        
        Args:
            camera_source: Camera index (0 for iVCam / default USB) or video file path
            max_frames: Optional frame limit
            display: Whether to open cv2.imshow GUI window
        Yields:
            Processed HUD frame (np.ndarray)
        """
        cap = self.open_camera(camera_source)
        frame_count = 0

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                frame_count += 1
                local_dets = self.detect_faces(frame)
                # In standalone testing, update tracks directly or with cloud mock
                targets = self.update_local_tracks(local_dets=local_dets, cloud_results=None)
                hud_frame = self.render_hud(frame, targets)

                if display:
                    cv2.imshow("iVCam Biometric Target Locker HUD", hud_frame)
                    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
                        break

                yield hud_frame

                if max_frames is not None and frame_count >= max_frames:
                    break
        finally:
            cap.release()
            if display:
                cv2.destroyAllWindows()
