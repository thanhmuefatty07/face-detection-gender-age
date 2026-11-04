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

from src.biometrics.core.geometry import UmeyamaAligner
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
        target_size: int = 160,
    ):
        self.server_url = server_url
        self.max_targets = max_targets
        self.target_size = int(target_size)
        self.tracker = ByteTracker(
            track_thresh=track_thresh,
            low_thresh=low_thresh,
            max_targets=max_targets,
        )
        self.locker = TargetLocker()
        self.aligner = UmeyamaAligner(target_size=self.target_size)

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
        Detects faces in frame using YuNet and extracts aligned crops (224x224 or 160x160).
        
        Args:
            frame: (H, W, 3) BGR image
        Returns:
            List of detection dicts with 'bbox', 'score', and 'crop'
        """
        if frame is None or frame.size == 0 or frame.ndim != 3:
            return []
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
                crop_aligned = None
                lmks_obj = getattr(d, "landmarks", None)
                if lmks_obj is not None:
                    try:
                        lmks_pts = lmks_obj.points if hasattr(lmks_obj, "points") else np.asarray(lmks_obj, dtype=np.float32)
                        if lmks_pts.shape == (5, 2) and np.all(np.isfinite(lmks_pts)):
                            crop_aligned = self.aligner.align(frame, lmks_pts)
                    except Exception:
                        crop_aligned = None
                if crop_aligned is None:
                    crop = frame[y1:y2, x1:x2]
                    if crop.size > 0:
                        crop_aligned = cv2.resize(crop, (self.target_size, self.target_size))
                if crop_aligned is not None:
                    results.append({
                        "bbox": (float(x1), float(y1), float(x2), float(y2)),
                        "score": float(d.bbox.confidence),
                        "crop": crop_aligned,
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
        retain_ids = [t.track_id for t in online_tracks] + [t.track_id for t in self.tracker.lost_tracks]
        self.locker.prune_inactive(retain_ids)
        for lost_t in self.tracker.lost_tracks:
            self.locker.coast(lost_t.track_id)

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
                "locked_age_interval": lock_info["locked_age_interval"],
                "locked_age_interval_idx": lock_info["locked_age_interval_idx"],
                "locked_gender": lock_info["locked_gender"],
                "gender_prob": lock_info["gender_prob"],
                "aleatoric_std": std_pred,
            })

        return output_targets

    def query_cloud(
        self,
        local_dets: List[Dict[str, Any]],
        ws_conn: Any = None,
    ) -> Tuple[Optional[List[Dict[str, Any]]], float]:
        """
        Transmits encoded face crops to Kaggle cloud server and receives biometric predictions.
        
        Args:
            local_dets: list of face detections containing 160x160 crops
            ws_conn: active persistent WebSocket connection if available
        Returns:
            Tuple of (cloud_results, rtt_latency_ms)
        """
        if not local_dets:
            return [], 0.0

        crops_b64 = self.encode_crops(local_dets)
        if not crops_b64:
            return [], 0.0

        t0 = time.perf_counter()

        # 1. Active WebSocket persistent connection
        if ws_conn is not None:
            try:
                payload = json.dumps({
                    "request_id": f"req_{int(time.time() * 1000)}",
                    "crops": crops_b64,
                })
                ws_conn.send(payload)
                resp_raw = ws_conn.recv(timeout=1.0)
                resp = json.loads(resp_raw)
                lat = (time.perf_counter() - t0) * 1000.0
                return resp.get("results", []), lat
            except Exception:
                pass

        # 2. REST HTTP fallback if server_url is http/https
        if self.server_url.startswith("http://") or self.server_url.startswith("https://"):
            try:
                import requests
                resp = requests.post(
                    f"{self.server_url.rstrip('/')}/infer",
                    json={
                        "request_id": f"req_{int(time.time() * 1000)}",
                        "crops_base64": crops_b64,
                    },
                    timeout=1.0,
                )
                lat = (time.perf_counter() - t0) * 1000.0
                if resp.status_code == 200:
                    return resp.json().get("results", []), lat
            except Exception:
                pass

        return None, 0.0

    def render_hud(
        self,
        frame: np.ndarray,
        targets: List[Dict[str, Any]],
        fps: float = 0.0,
        latency_ms: float = 0.0,
        cloud_connected: bool = True,
    ) -> np.ndarray:
        """Renders futuristic HUD overlay for locked biometric targets with telemetry banner."""
        hud_img = frame.copy()

        # Top Telemetry Header
        header_text = f"FPS: {fps:4.1f} | RTT: {latency_ms:4.1f}ms | Cloud: {'ONLINE' if cloud_connected else 'COASTING'}"
        cv2.rectangle(hud_img, (10, 10), (320, 34), (20, 20, 20), -1)
        stat_color = (0, 255, 120) if cloud_connected else (0, 180, 255)
        cv2.putText(hud_img, header_text, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, stat_color, 1, cv2.LINE_AA)

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
            from ..estimation.intervals import get_fine_interval_for_age
            age_iv = tgt.get("locked_age_interval") or get_fine_interval_for_age(age)[0]
            label = f"ID #{tid} | {gender} | {age_iv}y ({age:.1f}y +/-{std:.1f})"
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
        use_cloud: bool = True,
    ) -> Generator[np.ndarray, None, None]:
        """
        Live physical/synthetic video processing generator with cloud WebSocket tunnel.
        
        Args:
            camera_source: Camera index (0 for iVCam / default USB) or video file path
            max_frames: Optional frame limit
            display: Whether to open cv2.imshow GUI window
            use_cloud: Whether to transmit crops to Kaggle cloud server
        Yields:
            Processed HUD frame (np.ndarray)
        """
        cap = self.open_camera(camera_source)
        frame_count = 0

        # Attempt to establish persistent WebSocket connection
        ws_conn = None
        if use_cloud and (self.server_url.startswith("ws://") or self.server_url.startswith("wss://")):
            try:
                from websockets.sync.client import connect
                ws_conn = connect(self.server_url, open_timeout=2.0)
            except Exception:
                ws_conn = None

        t_prev = time.perf_counter()
        fps = 0.0
        rtt_lat = 0.0

        try:
            while cap.isOpened():
                t_start = time.perf_counter()
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                frame_count += 1
                local_dets = self.detect_faces(frame)

                cloud_res = None
                if use_cloud and local_dets:
                    cloud_res, rtt_lat = self.query_cloud(local_dets, ws_conn=ws_conn)

                targets = self.update_local_tracks(local_dets=local_dets, cloud_results=cloud_res)

                # Compute smoothed FPS
                dt = time.perf_counter() - t_start
                if dt > 0:
                    inst_fps = 1.0 / dt
                    fps = 0.9 * fps + 0.1 * inst_fps if fps > 0 else inst_fps

                hud_frame = self.render_hud(
                    frame,
                    targets,
                    fps=fps,
                    latency_ms=rtt_lat,
                    cloud_connected=(cloud_res is not None or not use_cloud),
                )

                if display:
                    cv2.imshow("iVCam Biometric Target Locker HUD [Kaggle Cloud GPU]", hud_frame)
                    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
                        break

                yield hud_frame

                if max_frames is not None and frame_count >= max_frames:
                    break
        finally:
            cap.release()
            if ws_conn is not None:
                try:
                    ws_conn.close()
                except Exception:
                    pass
            if display:
                cv2.destroyAllWindows()
