"""
Edge Asynchronous Inference Pipeline (OpenVINO / Local Standalone)
==================================================================
Runs completely offline on local CPU / iGPU without cloud network dependency.
Supports OpenVINO runtime acceleration (GPU/CPU) and PyTorch AVX2 fallback.
Pipelined multi-stage execution with accurate multi-target biometric tracking.
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import cv2
import numpy as np
import torch

from src.biometrics.estimation.backbone import BiometricMultiTaskModel
from src.biometrics.tracking.byte_tracker import ByteTracker, Detection
from src.biometrics.tracking.target_locker import TargetLocker
from src.biometrics.detection.yunet_detector import YuNetFaceDetector

try:
    import openvino as ov
except ImportError:
    ov = None


class EdgeAsyncPipeline:
    """
    Local edge standalone biometric estimation and multi-target locking pipeline.
    Seamlessly integrates OpenVINO iGPU/CPU acceleration, YuNet detection,
    and Epistemic Target Locking.
    """

    def __init__(
        self,
        backbone_name: str = "convnextv2_nano",
        max_targets: int = 5,
        use_openvino: bool = False,
        device: str = "cpu",
        yunet_model_path: Optional[str | Path] = None,
    ):
        self.backbone_name = backbone_name
        self.max_targets = max_targets
        self.device = device
        self.use_openvino = use_openvino

        # 1. Initialize local PyTorch model
        self.model = BiometricMultiTaskModel(backbone_name=backbone_name, pretrained=False)
        self.model.to(self.device)
        self.model.eval()

        # 2. OpenVINO runtime acceleration
        self.ov_compiled_model = None
        self.ov_infer_request = None
        if self.use_openvino and ov is not None:
            self._init_openvino(target_device=device)

        # 3. YuNet Face Detector
        if yunet_model_path is None:
            default_p = Path(__file__).resolve().parent.parent.parent.parent / "models" / "face_detection_yunet_2023mar.onnx"
            if default_p.exists():
                yunet_model_path = default_p
            else:
                yunet_model_path = "models/face_detection_yunet_2023mar.onnx"

        self.detector = YuNetFaceDetector(model_path=yunet_model_path, confidence_threshold=0.5)

        # 4. Multi-target tracker and Epistemic Locker
        self.tracker = ByteTracker(max_targets=max_targets)
        self.locker = TargetLocker()

    def _init_openvino(self, target_device: str = "cpu"):
        """Initializes and compiles OpenVINO execution engine on CPU or iGPU."""
        try:
            core = ov.Core()
            available = core.available_devices
            ov_dev = target_device.upper()
            if ov_dev not in available:
                ov_dev = "GPU" if "GPU" in available else "CPU"

            dummy = torch.zeros(1, 3, 160, 160)
            ov_model = ov.convert_model(self.model, example_input=dummy)
            # Make batch dimension dynamic
            ov_model.inputs[0].get_node().set_partial_shape(ov.PartialShape([-1, 3, 160, 160]))
            ov_model.validate_nodes_and_infer_types()

            compiled = core.compile_model(ov_model, ov_dev)
            self.ov_compiled_model = compiled
            self.ov_infer_request = compiled.create_infer_request()
        except Exception:
            self.ov_compiled_model = None
            self.ov_infer_request = None

    def detect_faces(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detects faces in frame using YuNet with fallback.
        Returns list of {'bbox': (x1, y1, x2, y2), 'score': float}.
        """
        H, W = frame.shape[:2]
        if H < 20 or W < 20:
            return []

        dets = self.detector.detect(frame)
        results = []
        for d in dets:
            x1 = float(d.bbox.x)
            y1 = float(d.bbox.y)
            x2 = float(d.bbox.x + d.bbox.width)
            y2 = float(d.bbox.y + d.bbox.height)
            results.append({
                "bbox": (x1, y1, x2, y2),
                "score": float(d.bbox.confidence),
            })
        return results

    def _infer_crops(self, crops: List[np.ndarray]) -> Tuple[List[float], List[float], List[float], List[Optional[np.ndarray]]]:
        """Runs forward inference on 160x160 face crops via OpenVINO or PyTorch."""
        N = len(crops)
        if N == 0:
            return [], [], [], []

        batch_arr = np.stack(crops, axis=0).transpose(0, 3, 1, 2)
        batch_float = batch_arr.astype(np.float32) / 127.5 - 1.0

        if self.ov_compiled_model is not None:
            try:
                ov_out = self.ov_compiled_model(batch_float)
                out_by_name = {port.any_name: arr for port, arr in ov_out.items()}
                age_preds = out_by_name.get("expected_age", out_by_name.get("age_expected", np.zeros(N))).flatten().tolist()
                std_preds = out_by_name.get("aleatoric_std", np.ones(N)).flatten().tolist()
                g_logits = out_by_name.get("gender_logit", np.zeros(N)).flatten().tolist()
                reid_embs = [out_by_name["reid_embedding"][i] for i in range(N)] if "reid_embedding" in out_by_name else [None] * N
                return age_preds, std_preds, g_logits, reid_embs
            except Exception:
                pass

        # PyTorch fallback
        tensor = torch.from_numpy(batch_float).to(self.device)
        with torch.no_grad():
            out = self.model(tensor)

        age_preds = out["expected_age"].cpu().numpy().flatten().tolist()
        std_preds = out["aleatoric_std"].cpu().numpy().flatten().tolist()
        g_logits = out["gender_logit"].cpu().numpy().flatten().tolist()
        reid_arr = out["reid_embedding"].cpu().numpy()
        reid_embs = [reid_arr[i] for i in range(N)]

        return age_preds, std_preds, g_logits, reid_embs

    def process_frame(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Processes a single video frame locally.
        Guarantees correct biometric alignment across multiple targets.
        
        Args:
            frame: (H, W, 3) BGR image
        Returns:
            List of locked biometric target dictionaries
        """
        H, W = frame.shape[:2]
        raw_dets = self.detect_faces(frame)

        crops: List[np.ndarray] = []
        valid_det_indices: List[int] = []

        for i, d in enumerate(raw_dets):
            x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(W, x2)
            y2 = min(H, y2)
            if x2 > x1 and y2 > y1:
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    crop_160 = cv2.resize(crop, (160, 160))
                    crop_rgb = cv2.cvtColor(crop_160, cv2.COLOR_BGR2RGB)
                    crops.append(crop_rgb)
                    valid_det_indices.append(i)

        age_preds, std_preds, gender_logits, reid_feats = self._infer_crops(crops)

        # Build tracker detections, linking det_id to crop index
        crop_index_by_det: Dict[int, int] = {}
        for crop_idx, det_i in enumerate(valid_det_indices):
            crop_index_by_det[det_i] = crop_idx

        tracker_dets: List[Detection] = []
        for det_i, d in enumerate(raw_dets):
            rf = None
            if det_i in crop_index_by_det:
                c_idx = crop_index_by_det[det_i]
                rf = reid_feats[c_idx] if c_idx < len(reid_feats) else None
            tracker_dets.append(Detection(
                bbox=d["bbox"],
                score=d["score"],
                reid_feat=rf,
                det_id=det_i,
            ))

        online_tracks = self.tracker.update(tracker_dets)
        active_ids = [t.track_id for t in online_tracks]
        self.locker.prune_inactive(active_ids)

        output_targets: List[Dict[str, Any]] = []
        for t in online_tracks:
            matched_crop_idx = None
            if t.last_det_id is not None and t.last_det_id in crop_index_by_det:
                matched_crop_idx = crop_index_by_det[t.last_det_id]
            else:
                # Spatial IoU fallback for matched detection
                best_iou = 0.2
                best_idx = None
                for det_i in valid_det_indices:
                    d_box = raw_dets[det_i]["bbox"]
                    xx1 = max(t.bbox[0], d_box[0])
                    yy1 = max(t.bbox[1], d_box[1])
                    xx2 = min(t.bbox[2], d_box[2])
                    yy2 = min(t.bbox[3], d_box[3])
                    inter = max(0.0, xx2 - xx1) * max(0.0, yy2 - yy1)
                    area_t = max(0.0, t.bbox[2] - t.bbox[0]) * max(0.0, t.bbox[3] - t.bbox[1])
                    area_d = max(0.0, d_box[2] - d_box[0]) * max(0.0, d_box[3] - d_box[1])
                    iou = inter / (area_t + area_d - inter + 1e-8)
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = det_i
                if best_idx is not None and best_idx in crop_index_by_det:
                    matched_crop_idx = crop_index_by_det[best_idx]

            if matched_crop_idx is not None and matched_crop_idx < len(age_preds):
                age_val = float(age_preds[matched_crop_idx])
                std_val = float(std_preds[matched_crop_idx])
                g_logit = float(gender_logits[matched_crop_idx])
                has_meas = True
            else:
                age_val = 25.0
                std_val = 1.0
                g_logit = 0.0
                has_meas = False

            lock_info = self.locker.update(
                track_id=t.track_id,
                age_pred=age_val,
                aleatoric_std=std_val,
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
                "aleatoric_std": std_val,
            })

        return output_targets
