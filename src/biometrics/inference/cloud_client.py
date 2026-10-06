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
from typing import List, Dict, Any, Optional, Tuple, Union
import cv2
import numpy as np

from src.biometrics.tracking.byte_tracker import ByteTracker, Detection, STrack
from src.biometrics.tracking.target_locker import TargetLocker


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
    ):
        self.server_url = server_url
        self.max_targets = max_targets
        self.tracker = ByteTracker(
            track_thresh=track_thresh,
            low_thresh=low_thresh,
            max_targets=max_targets,
        )
        self.locker = TargetLocker()

    def update_local_tracks(
        self,
        local_dets: List[Dict[str, Any]],
        cloud_results: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Associates cloud predictions with local detections and updates tracker & locker.
        
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
            tracker_dets.append(Detection(bbox=bbox, score=score, reid_feat=reid_feat))

        online_tracks = self.tracker.update(tracker_dets)
        active_ids = [t.track_id for t in online_tracks]
        self.locker.prune_inactive(active_ids)

        output_targets: List[Dict[str, Any]] = []
        for t in online_tracks:
            # Match with closest detection or cloud result
            matched_res = None
            if cloud_results and len(cloud_results) > 0:
                # Find matching detection by IoU
                best_iou = -1.0
                best_idx = -1
                for idx, ld in enumerate(local_dets):
                    b = ld.get("bbox", (0.0, 0.0, 0.0, 0.0))
                    # simple IoU
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

                if best_idx >= 0 and best_idx < len(cloud_results):
                    matched_res = cloud_results[best_idx]

            if matched_res is not None:
                age_pred = float(matched_res.get("expected_age", 25.0))
                std_pred = float(matched_res.get("aleatoric_std", 1.0))
                g_logit = float(matched_res.get("gender_logit", 0.0))
            else:
                age_pred = 25.0
                std_pred = 2.0
                g_logit = 0.0

            lock_info = self.locker.update(
                track_id=t.track_id,
                age_pred=age_pred,
                aleatoric_std=std_pred,
                gender_logit=g_logit,
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
            corner_len = min(15, (x2 - x1) // 4)
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
