"""
Edge Asynchronous Inference Pipeline (OpenVINO / CUDA / Local Standalone)
=========================================================================
Runs completely offline on local CPU / iGPU / dGPU without cloud network dependency.
Integrates:
1. HardwareResourceManager (CPU thread/affinity/priority tuning, OpenVINO disk caching, CUDA AMP)
2. EnvironmentNormalizer (multi-environment low-light, backlight, glare, color-cast & blur compensation)
3. Multi-Scale YuNet Face Detection + 5-Point Umeyama Similarity Alignment
4. Multi-Target ByteTrack (with Tier-1 Re-ID anti-swap gating) + Epistemic TargetLocker
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import cv2
import numpy as np
import torch

from src.biometrics.core.geometry import UmeyamaAligner
from src.biometrics.estimation.backbone import BiometricMultiTaskModel
from src.biometrics.tracking.byte_tracker import ByteTracker, Detection
from src.biometrics.tracking.target_locker import TargetLocker
from src.biometrics.detection.yunet_detector import YuNetFaceDetector
from src.biometrics.runtime import HardwareResourceManager, EnvironmentNormalizer

try:
    import openvino as ov
except ImportError:
    ov = None


class EdgeAsyncPipeline:
    """
    Local edge standalone biometric estimation and multi-target locking pipeline.
    Seamlessly integrates OpenVINO/CUDA/CPU acceleration, multi-environment normalization,
    YuNet multi-scale detection, 5-point Umeyama affine alignment, and Epistemic Target Locking.
    """

    def __init__(
        self,
        backbone_name: str = "convnextv2_tiny",
        max_targets: int = 16,
        use_openvino: bool = False,
        device: str = "cpu",
        yunet_model_path: Optional[str | Path] = None,
        enable_flip_tta: bool = True,
        checkpoint_path: Optional[str | Path] = None,
        target_size: Optional[int] = None,
        enable_env_normalization: bool = True,
        max_batch_size: int = 16,
    ):
        self.hw_profile = HardwareResourceManager.optimize_process_resources()
        self.hw_config = HardwareResourceManager.resolve_device_and_backend(
            requested_device=device,
            use_openvino=use_openvino,
        )

        self.backbone_name = backbone_name
        self.max_targets = int(max_targets)
        self.torch_device = self.hw_config["torch_device"]
        self.device = self.torch_device
        self.ov_target_device = self.hw_config["ov_device"]
        self.use_openvino = bool(use_openvino)
        self.enable_flip_tta = bool(enable_flip_tta)
        self.enable_env_normalization = bool(enable_env_normalization)
        self.max_batch_size = max(1, int(max_batch_size))

        if target_size is None:
            self.target_size = 224 if backbone_name == "convnextv2_tiny" else 160
        else:
            self.target_size = int(target_size)

        # 1. Initialize local PyTorch model
        self.model = BiometricMultiTaskModel(backbone_name=backbone_name, pretrained=False)
        self._loaded_checkpoint_path: Optional[Path] = None
        if checkpoint_path is None and backbone_name == "convnextv2_tiny":
            default_ckpt = Path(__file__).resolve().parent.parent.parent.parent / "models" / "convnextv2_tiny_biometrics.pt"
            if default_ckpt.exists():
                checkpoint_path = default_ckpt
        if checkpoint_path is not None and Path(checkpoint_path).exists():
            ckpt_p = Path(checkpoint_path)
            ckpt = torch.load(ckpt_p, map_location=self.torch_device, weights_only=False)
            state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            try:
                self.model.load_state_dict(state_dict, strict=True)
            except RuntimeError:
                self.model.load_state_dict(state_dict, strict=False)
            self._loaded_checkpoint_path = ckpt_p

        self.model.to(self.torch_device)
        self.model.eval()

        # 2. OpenVINO runtime acceleration
        self.ov_compiled_model = None
        self.ov_infer_request = None
        if self.use_openvino and ov is not None:
            self._init_openvino(target_device=self.ov_target_device)

        # 3. YuNet Face Detector, 5-Point Umeyama Aligner & Multi-Environment Normalizer
        if yunet_model_path is None:
            default_p = Path(__file__).resolve().parent.parent.parent.parent / "models" / "face_detection_yunet_2023mar.onnx"
            if default_p.exists():
                yunet_model_path = default_p
            else:
                yunet_model_path = "models/face_detection_yunet_2023mar.onnx"

        self.detector = YuNetFaceDetector(model_path=yunet_model_path, confidence_threshold=0.5)
        self.aligner = UmeyamaAligner(target_size=self.target_size)
        self.env_normalizer = EnvironmentNormalizer()

        # 4. Multi-target tracker and Epistemic Locker
        self.tracker = ByteTracker(max_targets=self.max_targets)
        self.locker = TargetLocker()

    def _init_openvino(self, target_device: str = "CPU"):
        """Initializes and compiles OpenVINO execution engine with optimal CPU/GPU properties."""
        try:
            core = ov.Core()
            available = list(core.available_devices)
            ov_dev = str(target_device).upper()
            if ov_dev not in available:
                ov_dev = "CPU" if "CPU" in available else (available[0] if available else "CPU")

            # Enable persistent disk cache only when loading a deterministic checkpoint file
            if self._loaded_checkpoint_path is not None:
                cache_dir = self._loaded_checkpoint_path.parent / ".ov_cache"
                try:
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    core.set_property({"CACHE_DIR": str(cache_dir)})
                except Exception:
                    pass

            dummy = torch.zeros(1, 3, self.target_size, self.target_size, device=self.torch_device)
            ov_model = ov.convert_model(self.model, example_input=dummy)
            if ov_dev == "CPU":
                ov_model.inputs[0].get_node().set_partial_shape(
                    ov.PartialShape([-1, 3, self.target_size, self.target_size])
                )
            else:
                ov_model.inputs[0].get_node().set_partial_shape(
                    ov.PartialShape([ov.Dimension(1, 8), 3, self.target_size, self.target_size])
                )
            ov_model.validate_nodes_and_infer_types()

            compile_props: Dict[str, Any] = {"PERFORMANCE_HINT": "LATENCY"}
            if ov_dev == "CPU":
                compile_props["INFERENCE_NUM_THREADS"] = self.hw_profile.get("logical_cores", 4)

            try:
                compiled = core.compile_model(ov_model, ov_dev, compile_props)
            except Exception:
                compiled = core.compile_model(ov_model, ov_dev)

            self.ov_compiled_model = compiled
            self.ov_infer_request = compiled.create_infer_request()
        except Exception:
            self.ov_compiled_model = None
            self.ov_infer_request = None

    def detect_faces(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detects faces in frame using multi-scale, multi-environment YuNet.
        Returns list of {'bbox': (x1, y1, x2, y2), 'score': float, 'landmarks': Optional[np.ndarray]}.
        """
        if frame is None or frame.size == 0 or frame.ndim != 3:
            return []
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
            lmks = d.landmarks.points if getattr(d, "landmarks", None) is not None else None
            results.append({
                "bbox": (x1, y1, x2, y2),
                "score": float(d.bbox.confidence),
                "landmarks": lmks,
            })
        return results

    def _infer_single_batch(
        self,
        crops: List[np.ndarray],
        enable_flip_tta: Optional[bool] = None,
    ) -> Tuple[List[float], List[float], List[float], List[Optional[np.ndarray]]]:
        N = len(crops)
        if N == 0:
            return [], [], [], []

        use_tta = self.enable_flip_tta if enable_flip_tta is None else bool(enable_flip_tta)
        batch_arr = np.stack(crops, axis=0).transpose(0, 3, 1, 2)
        batch_float = np.ascontiguousarray(batch_arr.astype(np.float32) / 127.5 - 1.0)

        if self.ov_compiled_model is not None:
            try:
                if use_tta:
                    batch_flip = np.ascontiguousarray(batch_float[:, :, :, ::-1])
                    inp_ov = np.concatenate([batch_float, batch_flip], axis=0)
                    ov_out = self.ov_compiled_model(inp_ov)
                    vals = [np.asarray(v, dtype=np.float32) for v in ov_out.values()]
                    logits_cat = vals[0]
                    g_cat = vals[1].reshape(2 * N)
                    lv_cat = np.clip(vals[2].reshape(2 * N), -6.0, 4.0)
                    reid_cat = vals[3].reshape(2 * N, -1)

                    logits_shift = logits_cat - np.max(logits_cat, axis=-1, keepdims=True)
                    exp_l = np.exp(logits_shift)
                    probs_cat = exp_l / np.clip(np.sum(exp_l, axis=-1, keepdims=True), 1e-12, None)
                    bins_vec = np.arange(probs_cat.shape[-1], dtype=np.float32)
                    expected_cat = np.sum(probs_cat * bins_vec[None, :], axis=-1)
                    std_cat = np.exp(0.5 * lv_cat)

                    probs_avg = 0.5 * (probs_cat[:N] + probs_cat[N:])
                    probs_avg = probs_avg / np.clip(np.sum(probs_avg, axis=-1, keepdims=True), 1e-12, None)
                    age_preds = np.sum(probs_avg * bins_vec[None, :], axis=-1).flatten().tolist()

                    var_orig = std_cat[:N] ** 2
                    var_flip = std_cat[N:] ** 2
                    epistemic_age_diff = 0.5 * np.abs(expected_cat[:N] - expected_cat[N:])
                    w_half = batch_float.shape[3] // 2
                    if w_half > 0:
                        left_h = batch_float[:, :, :, :w_half]
                        right_h = batch_float[:, :, :, -w_half:][:, :, :, ::-1]
                        hemiface_var_gap = np.abs(np.var(left_h, axis=(1, 2, 3)) - np.var(right_h, axis=(1, 2, 3)))
                        bilateral_penalty = 0.50 * hemiface_var_gap
                    else:
                        bilateral_penalty = np.zeros_like(epistemic_age_diff)
                    epistemic_std = np.sqrt(epistemic_age_diff ** 2 + bilateral_penalty ** 2)
                    gender_dis = 0.05 * ((g_cat[:N] - g_cat[N:]) ** 2)
                    total_var = 0.5 * (var_orig + var_flip) + (epistemic_std ** 2) + gender_dis
                    std_preds = np.sqrt(np.clip(total_var, 1e-6, None)).flatten().tolist()

                    g_logits = (0.5 * (g_cat[:N] + g_cat[N:])).flatten().tolist()
                    reid_avg = 0.5 * (reid_cat[:N] + reid_cat[N:])
                    reid_norm = reid_avg / np.clip(np.linalg.norm(reid_avg, axis=-1, keepdims=True), 1e-12, None)
                    reid_embs = [reid_norm[i] for i in range(N)]
                    return age_preds, std_preds, g_logits, reid_embs
                else:
                    ov_out = self.ov_compiled_model(batch_float)
                    vals = [np.asarray(v, dtype=np.float32) for v in ov_out.values()]
                    logits = vals[0]
                    g_arr = vals[1].reshape(N)
                    lv_arr = np.clip(vals[2].reshape(N), -6.0, 4.0)
                    reid_arr = vals[3].reshape(N, -1)

                    logits_shift = logits - np.max(logits, axis=-1, keepdims=True)
                    exp_l = np.exp(logits_shift)
                    probs = exp_l / np.clip(np.sum(exp_l, axis=-1, keepdims=True), 1e-12, None)
                    bins_vec = np.arange(probs.shape[-1], dtype=np.float32)
                    age_preds = np.sum(probs * bins_vec[None, :], axis=-1).flatten().tolist()
                    std_preds = np.exp(0.5 * lv_arr).flatten().tolist()
                    g_logits = g_arr.flatten().tolist()
                    reid_norm = reid_arr / np.clip(np.linalg.norm(reid_arr, axis=-1, keepdims=True), 1e-12, None)
                    reid_embs = [reid_norm[i] for i in range(N)]
                    return age_preds, std_preds, g_logits, reid_embs
            except Exception:
                pass

        # PyTorch execution with torch.inference_mode()
        tensor = torch.from_numpy(batch_float).to(self.torch_device)
        with torch.inference_mode():
            if hasattr(self.model, "predict_smooth"):
                out = self.model.predict_smooth(tensor, enable_flip_tta=use_tta)
            else:
                out = self.model(tensor)

        if isinstance(out, (tuple, list)):
            age_logits, gender_logit, log_var, reid_t = out[0], out[1], out[2], out[3]
            age_ldl_t = torch.softmax(age_logits, dim=-1)
            bins_t = torch.arange(age_ldl_t.shape[-1], device=age_ldl_t.device, dtype=age_ldl_t.dtype)
            exp_age_t = torch.sum(age_ldl_t * bins_t.unsqueeze(0), dim=-1).reshape(-1)
            std_t = torch.exp(0.5 * torch.clamp(log_var.reshape(-1), -6.0, 4.0))
            age_preds = exp_age_t.cpu().numpy().flatten().tolist()
            std_preds = std_t.cpu().numpy().flatten().tolist()
            g_logits = gender_logit.reshape(-1).cpu().numpy().flatten().tolist()
            reid_arr = reid_t.cpu().numpy()
            reid_embs = [reid_arr[i] for i in range(N)]
            return age_preds, std_preds, g_logits, reid_embs

        age_preds = out["expected_age"].cpu().numpy().flatten().tolist()
        std_tensor = out.get("total_std", out["aleatoric_std"])
        std_preds = std_tensor.cpu().numpy().flatten().tolist()
        g_logits = out["gender_logit"].cpu().numpy().flatten().tolist()
        reid_arr = out["reid_embedding"].cpu().numpy()
        reid_embs = [reid_arr[i] for i in range(N)]

        return age_preds, std_preds, g_logits, reid_embs

    def _infer_crops(
        self,
        crops: List[np.ndarray],
    ) -> Tuple[List[float], List[float], List[float], List[Optional[np.ndarray]]]:
        """Runs forward inference on face crops via OpenVINO or PyTorch with bounded sub-batching."""
        N = len(crops)
        if N == 0:
            return [], [], [], []

        if N <= self.max_batch_size:
            return self._infer_single_batch(crops, enable_flip_tta=self.enable_flip_tta)

        all_ages: List[float] = []
        all_stds: List[float] = []
        all_g_logits: List[float] = []
        all_reids: List[Optional[np.ndarray]] = []
        for start in range(0, N, self.max_batch_size):
            chunk = crops[start : start + self.max_batch_size]
            ages, stds, g_logits, reids = self._infer_single_batch(chunk, enable_flip_tta=self.enable_flip_tta)
            all_ages.extend(ages)
            all_stds.extend(stds)
            all_g_logits.extend(g_logits)
            all_reids.extend(reids)
        return all_ages, all_stds, all_g_logits, all_reids

    def process_frame(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Processes a single video frame locally across multiple targets and lighting environments.
        Guarantees strict 1-to-1 biometric alignment without cross-track contamination.
        
        Args:
            frame: (H, W, 3) BGR image
        Returns:
            List of locked biometric target dictionaries
        """
        if frame is None or frame.size == 0 or frame.ndim != 3:
            return []
        H, W = frame.shape[:2]
        raw_dets = self.detect_faces(frame)

        crops: List[np.ndarray] = []
        valid_det_indices: List[int] = []
        env_unc_multipliers: List[float] = []

        for i, d in enumerate(raw_dets):
            crop_rgb = None
            lmks = d.get("landmarks")
            if lmks is not None:
                try:
                    lmks_arr = np.asarray(lmks, dtype=np.float32)
                    if lmks_arr.shape == (5, 2) and np.all(np.isfinite(lmks_arr)):
                        aligned_bgr = self.aligner.align(frame, lmks_arr)
                        crop_rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)
                except Exception:
                    crop_rgb = None

            if crop_rgb is None:
                x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(W, x2)
                y2 = min(H, y2)
                if x2 > x1 and y2 > y1:
                    crop = frame[y1:y2, x1:x2]
                    if crop.size > 0:
                        crop_resized = cv2.resize(crop, (self.target_size, self.target_size))
                        crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)

            if crop_rgb is not None:
                if self.enable_env_normalization:
                    crop_rgb, env_info = self.env_normalizer.normalize_crop_for_inference(crop_rgb, is_bgr=False)
                    env_unc_multipliers.append(float(env_info.get("uncertainty_multiplier", 1.0)))
                else:
                    env_unc_multipliers.append(1.0)
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
        retain_ids = [t.track_id for t in online_tracks] + [t.track_id for t in self.tracker.lost_tracks]
        self.locker.prune_inactive(retain_ids)
        for lost_t in self.tracker.lost_tracks:
            self.locker.coast(lost_t.track_id)

        used_crop_indices = set()
        output_targets: List[Dict[str, Any]] = []
        for t in online_tracks:
            matched_crop_idx = None
            if t.last_det_id is not None and t.last_det_id in crop_index_by_det:
                matched_crop_idx = crop_index_by_det[t.last_det_id]
                used_crop_indices.add(matched_crop_idx)
            else:
                # Strict 1-to-1 Spatial IoU fallback for matched detection
                best_iou = 0.2
                best_idx = None
                for det_i in valid_det_indices:
                    cand_c_idx = crop_index_by_det.get(det_i)
                    if cand_c_idx is None or cand_c_idx in used_crop_indices:
                        continue
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
                    used_crop_indices.add(matched_crop_idx)

            if matched_crop_idx is not None and matched_crop_idx < len(age_preds):
                age_val = float(age_preds[matched_crop_idx])
                env_mult = env_unc_multipliers[matched_crop_idx] if matched_crop_idx < len(env_unc_multipliers) else 1.0
                std_val = float(std_preds[matched_crop_idx]) * env_mult
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
                "locked_age_interval": lock_info["locked_age_interval"],
                "locked_age_interval_idx": lock_info["locked_age_interval_idx"],
                "locked_gender": lock_info["locked_gender"],
                "gender_prob": lock_info["gender_prob"],
                "aleatoric_std": std_val,
            })

        return output_targets
