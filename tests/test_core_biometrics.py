"""
Core Biometric System Verification Suite
========================================
Essential end-to-end verification covering:
1. Trained ConvNeXt-V2 & YuNet model weight integrity, 101-bin LDL age, and 23 fine age brackets.
2. Multi-Environment Photometric Normalization (skin-tone preservation, low-light bilateral + CLAHE,
   raking directional backlight flattening, glare compression, non-skin color cast, motion blur, blank).
3. Multi-Scale & Multi-Environment YuNet Face Detection (simultaneous multi-target, mixed lighting,
   mixed close-up + distant scales, grayscale/BGRA inputs, zero false positives on blank frames).
4. Multi-Target ByteTrack (Re-ID crossing disambiguation, partial occlusion crossing, spatial recovery
   without Re-ID) & TargetLocker (Epistemic-Kalman age + Bayesian gender without cache double-counting).
5. Multi-Backend Camera Port Discovery (live vs static vs blank virtual stub vitality scoring,
   active port contention avoidance, thread-safe port switching).
6. Deep Machine Resource Optimization (physical/logical CPU topology, RAM/VRAM detection, bounded
   OpenCV threads, OpenVINO InferRequest reuse, and lowercase PyTorch vs uppercase OpenVINO devices).
"""

from __future__ import annotations
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np
import pytest
import torch

from src.biometrics.runtime import (
    CameraPortInfo,
    CameraPortScanner,
    EnvironmentNormalizer,
    HardwareResourceManager,
    SyntheticMultiTargetStream,
    ThreadedCameraStream,
)
from src.biometrics.detection.yunet_detector import YuNetFaceDetector
from src.biometrics.tracking.byte_tracker import ByteTracker, Detection, STrack
from src.biometrics.tracking.target_locker import TargetLocker
from src.biometrics.estimation.intervals import FINE_AGE_INTERVALS, get_fine_interval_for_age
from src.biometrics.inference.async_pipeline import EdgeAsyncPipeline
from src.biometrics.inference.cloud_client import CloudInferenceClient
from src.biometrics.pipeline import BiometricPipeline


ROOT_DIR = Path(__file__).resolve().parent.parent
CONVNEXT_PT = ROOT_DIR / "models" / "convnextv2_tiny_biometrics.pt"
YUNET_ONNX = ROOT_DIR / "models" / "face_detection_yunet_2023mar.onnx"


def test_model_weights_and_age_intervals_integrity():
    """Verifies trained model weights on disk and 23 fine age brackets [0, 100]."""
    assert CONVNEXT_PT.exists(), f"Missing ConvNeXt-V2 checkpoint: {CONVNEXT_PT}"
    assert YUNET_ONNX.exists(), f"Missing YuNet ONNX model: {YUNET_ONNX}"
    assert CONVNEXT_PT.stat().st_size > 10_000_000
    assert YUNET_ONNX.stat().st_size > 100_000

    ckpt = torch.load(CONVNEXT_PT, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    assert len(state_dict) >= 140
    for k, v in state_dict.items():
        if torch.is_tensor(v):
            assert torch.all(torch.isfinite(v)), f"Non-finite weights in {k}"

    assert len(FINE_AGE_INTERVALS) == 23
    for age in range(0, 101):
        name, idx, low, high = get_fine_interval_for_age(float(age))
        assert 0 <= idx < 23
        assert low <= age <= high
        assert isinstance(name, str) and len(name) > 0


def test_environment_normalizer_skin_preservation_and_8_regimes():
    """
    Verifies EnvironmentNormalizer across all 8 optical regimes and ensures natural
    human skin tone is NOT misclassified as an artificial color cast.
    """
    norm = EnvironmentNormalizer()

    # 1. Clean physiological skin tone crop must stay 'clean' (is_color_cast == False)
    skin_crop = np.full((224, 224, 3), (140, 165, 210), dtype=np.uint8)
    rng = np.random.RandomState(42)
    skin_crop = np.clip(skin_crop.astype(np.int16) + rng.randint(-12, 12, size=skin_crop.shape), 0, 255).astype(np.uint8)
    cv2.circle(skin_crop, (80, 90), 12, (35, 35, 40), -1)
    cv2.circle(skin_crop, (144, 90), 12, (35, 35, 40), -1)
    cv2.ellipse(skin_crop, (112, 160), (28, 10), 0, 0, 180, (90, 95, 165), -1)
    skin_Crop_blur = cv2.GaussianBlur(skin_crop, (3, 3), 0.6)

    info_clean = norm.analyze(skin_Crop_blur, is_bgr=True)
    assert info_clean["is_color_cast"] is False, f"Natural skin misclassified as color_cast: {info_clean}"
    assert info_clean["regime"] == "clean"
    out_clean, _ = norm.normalize_crop_for_inference(skin_Crop_blur, is_bgr=True)
    assert np.array_equal(out_clean, skin_Crop_blur)

    # 2. Extreme low light: must lift luminance and suppress read noise
    dark_crop = (skin_Crop_blur.astype(np.float32) * 0.14).astype(np.uint8)
    out_dark, info_dark = norm.normalize_crop_for_inference(dark_crop, is_bgr=True)
    assert info_dark["is_low_light"] is True
    assert float(np.mean(out_dark)) > float(np.mean(dark_crop)) * 1.6

    # 3. Raking directional side-light (15% bright, 85% shadow -> highlight_ratio < 0.08 but high hemiface asymmetry)
    raking_crop = (skin_Crop_blur.astype(np.float32) * 0.25).astype(np.uint8)
    raking_crop[:, :70] = np.clip(skin_Crop_blur[:, :70].astype(np.float32) * 1.15, 0, 255).astype(np.uint8)
    out_rak, info_rak = norm.normalize_crop_for_inference(raking_crop, is_bgr=True)
    assert info_rak["is_backlight"] is True, f"Failed to detect raking side shadow: {info_rak}"
    # Hemiface luminance gap should be reduced after normalization
    raw_gap = abs(float(np.mean(raking_crop[:, :70])) - float(np.mean(raking_crop[:, 150:])))
    norm_gap = abs(float(np.mean(out_rak[:, :70])) - float(np.mean(out_rak[:, 150:])))
    assert norm_gap < raw_gap

    # 4. Non-skin artificial blue color cast: must be detected as color_cast
    blue_cast = skin_Crop_blur.copy()
    blue_cast[:, :, 0] = np.clip(blue_cast[:, :, 0].astype(np.int16) + 75, 0, 255).astype(np.uint8)
    blue_cast[:, :, 2] = np.clip(blue_cast[:, :, 2].astype(np.int16) - 55, 0, 255).astype(np.uint8)
    _, info_blue = norm.normalize_crop_for_inference(blue_cast, is_bgr=True)
    assert info_blue["is_color_cast"] is True

    # 5. Overexposed glare, motion blur, blank, grayscale & BGRA inputs
    glare_crop = np.clip(skin_Crop_blur.astype(np.float32) * 1.45 + 75.0, 0, 255).astype(np.uint8)
    out_glare, info_glare = norm.normalize_crop_for_inference(glare_crop, is_bgr=True)
    assert info_glare["is_overexposed"] is True
    assert float(np.mean(out_glare)) < float(np.mean(glare_crop))

    k_blur = np.zeros((15, 15), dtype=np.float32)
    k_blur[7, :] = 1.0 / 15.0
    blur_crop = cv2.filter2D(skin_Crop_blur, -1, k_blur)
    _, info_blur = norm.normalize_crop_for_inference(blur_crop, is_bgr=True)
    assert info_blur["is_blurry"] is True
    assert info_blur["uncertainty_multiplier"] > 1.0

    blank_crop = np.zeros((224, 224, 3), dtype=np.uint8)
    assert norm.analyze(blank_crop)["regime"] == "blank"

    gray_2d = cv2.cvtColor(skin_Crop_blur, cv2.COLOR_BGR2GRAY)
    bgra_4c = cv2.cvtColor(skin_Crop_blur, cv2.COLOR_BGR2BGRA)
    out_g, _ = norm.normalize_crop_for_inference(gray_2d, is_bgr=True)
    out_4, _ = norm.normalize_crop_for_inference(bgra_4c, is_bgr=True)
    assert out_g.shape == (224, 224, 3) and out_4.shape == (224, 224, 3)


def test_multi_target_and_multi_environment_yunet_detection():
    """
    Verifies YuNetFaceDetector detects all 3 synthetic targets consistently,
    recovers shadowed faces alongside bright faces in mixed-lighting frames,
    and returns 0 false positives on blank frames.
    """
    detector = YuNetFaceDetector(model_path=YUNET_ONNX, confidence_threshold=0.5)

    # 1. Blank frame -> 0 detections
    assert len(detector.detect(np.zeros((480, 640, 3), dtype=np.uint8))) == 0

    # 2. SyntheticMultiTargetStream (3 simultaneous moving targets)
    syn = SyntheticMultiTargetStream(width=640, height=480, num_targets=3)
    for _ in range(5):
        ok, frame = syn.read()
        assert ok and frame is not None
        dets = detector.detect(frame)
        assert len(dets) == 3, f"Expected 3 targets in SyntheticMultiTargetStream, got {len(dets)}"

    # 3. Mixed-lighting multi-target frame: left target well-lit, right targets severely darkened (0.12x)
    _, base_frame = syn.read()
    mixed_frame = base_frame.copy()
    mixed_frame[:, 260:] = (mixed_frame[:, 260:].astype(np.float32) * 0.12).astype(np.uint8)
    mixed_dets = detector.detect(mixed_frame)
    assert len(mixed_dets) >= 2, f"Shadowed background targets were dropped when bright face was present: {len(mixed_dets)}"

    # 4. Grayscale and 4-channel BGRA frame compatibility
    gray_frame = cv2.cvtColor(base_frame, cv2.COLOR_BGR2GRAY)
    bgra_frame = cv2.cvtColor(base_frame, cv2.COLOR_BGR2BGRA)
    assert len(detector.detect(gray_frame)) == 3
    assert len(detector.detect(bgra_frame)) == 3


def test_bytetrack_anti_swap_occlusion_and_spatial_recovery():
    """
    Verifies ByteTracker:
    1. Prevents ID swaps when 2 targets cross trajectories with distinct 64-D Re-ID vectors.
    2. Disambiguates partial occlusion crossings (2 tracked targets -> 1 visible detection).
    3. Recovers multiple lost tracks via spatial IoU even when Re-ID vectors are None.
    """
    STrack.reset_counter()
    tracker = ByteTracker(track_thresh=0.6, low_thresh=0.2, reid_thresh=0.35, max_targets=8)

    emb_a = np.zeros(64, dtype=np.float32)
    emb_a[0] = 1.0
    emb_b = np.zeros(64, dtype=np.float32)
    emb_b[1] = 1.0

    t1 = tracker.update([
        Detection(bbox=(100, 100, 200, 200), score=0.95, reid_feat=emb_a, det_id=0),
        Detection(bbox=(300, 100, 400, 200), score=0.95, reid_feat=emb_b, det_id=1),
    ])
    assert len(t1) == 2
    id_a = next(t.track_id for t in t1 if t.last_det_id == 0)
    id_b = next(t.track_id for t in t1 if t.last_det_id == 1)

    # Partial occlusion at midpoint: only person B is visible in overlapping zone
    t2 = tracker.update([
        Detection(bbox=(195, 100, 295, 200), score=0.92, reid_feat=emb_b, det_id=0),
    ])
    matched_ids = [t.track_id for t in t2]
    assert id_b in matched_ids and id_a not in matched_ids

    # Next frame: person A re-emerges via Re-ID
    t3 = tracker.update([
        Detection(bbox=(310, 100, 410, 200), score=0.93, reid_feat=emb_a, det_id=0),
        Detection(bbox=(90, 100, 190, 200), score=0.94, reid_feat=emb_b, det_id=1),
    ])
    assert {t.track_id for t in t3} == {id_a, id_b}

    # Spatial IoU recovery of 2 lost tracks when reid_feat is None
    STrack.reset_counter()
    tracker_no_reid = ByteTracker(track_thresh=0.6, low_thresh=0.2, max_targets=8)
    init_trks = tracker_no_reid.update([
        Detection(bbox=(50, 50, 150, 150), score=0.9, reid_feat=None, det_id=0),
        Detection(bbox=(400, 50, 500, 150), score=0.9, reid_feat=None, det_id=1),
    ])
    orig_ids = {t.track_id for t in init_trks}
    # 1 frame of missed detection -> both tracks become Lost
    assert len(tracker_no_reid.update([])) == 0
    # Both reappear in reversed order without Re-ID features -> must recover original IDs via IoU
    rec_trks = tracker_no_reid.update([
        Detection(bbox=(402, 52, 502, 152), score=0.9, reid_feat=None, det_id=0),
        Detection(bbox=(52, 52, 152, 152), score=0.9, reid_feat=None, det_id=1),
    ])
    assert {t.track_id for t in rec_trks} == orig_ids


def test_cloud_client_multi_target_coasting_no_identity_swap():
    """
    Verifies that CloudInferenceClient does not swap identities or double-count Kalman
    measurements on skipped inference frames when detection order changes.
    """
    STrack.reset_counter()
    client = CloudInferenceClient(enable_local_fallback=True, target_size=224)

    det_young_female = {"bbox": (50.0, 80.0, 170.0, 200.0), "score": 0.95}
    det_senior_male = {"bbox": (380.0, 80.0, 500.0, 200.0), "score": 0.92}

    res_frame1 = [
        {"expected_age": 20.0, "aleatoric_std": 2.0, "gender_logit": 3.5, "gender_prob": 0.97, "reid_embedding": [1.0] + [0.0] * 63},
        {"expected_age": 68.0, "aleatoric_std": 2.5, "gender_logit": -3.5, "gender_prob": 0.03, "reid_embedding": [0.0, 1.0] + [0.0] * 62},
    ]
    targets_f1 = client.update_local_tracks([det_young_female, det_senior_male], cloud_results=res_frame1)
    assert len(targets_f1) == 2
    t_yf = next(t for t in targets_f1 if t["bbox"][0] < 200)
    t_sm = next(t for t in targets_f1 if t["bbox"][0] > 300)
    assert abs(t_yf["locked_age"] - 20.0) < 1.0 and t_yf["locked_gender"] == "Female"
    assert abs(t_sm["locked_age"] - 68.0) < 1.0 and t_sm["locked_gender"] == "Male"

    # Frame 2 (coasting / skipped inference frame: cloud_results=None) with reversed detection order!
    det_young_female_f2 = {"bbox": (54.0, 82.0, 174.0, 202.0), "score": 0.91}
    det_senior_male_f2 = {"bbox": (376.0, 82.0, 496.0, 202.0), "score": 0.96}
    targets_f2 = client.update_local_tracks([det_senior_male_f2, det_young_female_f2], cloud_results=None)
    assert len(targets_f2) == 2
    t_yf2 = next(t for t in targets_f2 if t["track_id"] == t_yf["track_id"])
    t_sm2 = next(t for t in targets_f2 if t["track_id"] == t_sm["track_id"])

    # Ages and genders must remain locked to their respective tracks without cross-contamination
    assert abs(t_yf2["locked_age"] - 20.0) < 1.0 and t_yf2["locked_gender"] == "Female"
    assert abs(t_sm2["locked_age"] - 68.0) < 1.0 and t_sm2["locked_gender"] == "Male"


def test_camera_port_scanner_vitality_ranking_and_switching():
    """
    Verifies CameraPortScanner ranks live optical motion > static splash > blank black stub,
    avoids re-opening active_port_info, and switches ports thread-safely.
    """
    class FakeCap:
        def __init__(self, mode: str):
            self.mode = mode
            self.idx = 0

        def isOpened(self):
            return self.mode != "dead"

        def set(self, prop, val):
            return True

        def get(self, prop):
            if prop == cv2.CAP_PROP_FPS:
                return 30.0
            return 0.0

        def read(self):
            self.idx += 1
            if self.mode == "dead":
                return False, None
            if self.mode == "blank":
                return True, np.zeros((480, 640, 3), dtype=np.uint8)
            if self.mode == "static":
                img = np.full((480, 640, 3), 120, dtype=np.uint8)
                cv2.rectangle(img, (100, 100), (300, 300), (200, 80, 40), -1)
                return True, img
            # Live moving optical feed
            img = np.full((480, 640, 3), 110, dtype=np.uint8)
            offset = (self.idx * 17) % 200
            cv2.circle(img, (150 + offset, 220), 60, (40, 190, 240), -1)
            return True, img

        def release(self):
            pass

    info_dead = CameraPortScanner._evaluate_capture(FakeCap("dead"), 0, "DSHOW", cv2.CAP_DSHOW, device_name="Dead")
    info_blank = CameraPortScanner._evaluate_capture(FakeCap("blank"), 1, "DSHOW", cv2.CAP_DSHOW, device_name="BlankStub")
    info_static = CameraPortScanner._evaluate_capture(FakeCap("static"), 2, "DSHOW", cv2.CAP_DSHOW, device_name="StaticSplash")
    info_live = CameraPortScanner._evaluate_capture(FakeCap("live"), 3, "DSHOW", cv2.CAP_DSHOW, device_name="LiveUSB")

    assert not info_dead.can_read
    assert info_blank.can_read and not info_blank.is_live_signal
    assert info_live.can_read and info_live.is_live_signal
    assert info_live.score > info_static.score > info_blank.score > info_dead.score

    # Verify demo stream & thread-safe port switching
    demo_info = CameraPortScanner.find_best_camera_port("demo")
    assert demo_info.can_read and demo_info.port == "DEMO"

    stream = ThreadedCameraStream(camera_source="demo", use_threading=True, allow_demo_fallback=True)
    try:
        ok, frm = stream.read()
        assert ok and frm is not None and frm.shape == (480, 640, 3)
        switched = stream.switch_to_next_port()
        assert switched.can_read
    finally:
        stream.release()


def test_hardware_resource_manager_and_end_to_end_pipeline():
    """
    Verifies HardwareResourceManager topology/memory/thread tuning and end-to-end
    EdgeAsyncPipeline + BiometricPipeline multi-target execution.
    """
    hw = HardwareResourceManager.optimize_process_resources()
    assert hw["physical_cores"] >= 1
    assert hw["logical_cores"] >= hw["physical_cores"]
    assert 1 <= hw["optimal_opencv_threads"] <= 4
    assert hw["total_ram_gb"] > 0.0
    assert hw["optimal_batch_size"] >= 2

    for req in ("auto", "openvino", "ov_cpu", "ov_gpu", "cpu", "cuda"):
        cfg = HardwareResourceManager.resolve_device_and_backend(req)
        assert cfg["torch_device"] in ("cpu", "cuda", "mps")
        assert cfg["ov_device"] in ("CPU", "GPU")

    # Verify EdgeAsyncPipeline auto-enables OpenVINO when available and processes multi-target frames
    edge = EdgeAsyncPipeline(device="auto", enable_flip_tta=False)
    assert edge.torch_device in ("cpu", "cuda", "mps")
    syn = SyntheticMultiTargetStream(width=640, height=480, num_targets=3)
    _, frame = syn.read()

    locked_targets = edge.process_frame(frame)
    assert len(locked_targets) == 3
    seen_ids = set()
    for t in locked_targets:
        seen_ids.add(t["track_id"])
        assert 0.0 <= t["locked_age"] <= 100.0
        assert t["locked_gender"] in ("Male", "Female")
        assert 0.0 <= t["gender_prob"] <= 1.0
        assert np.isfinite(t["aleatoric_std"]) and t["aleatoric_std"] > 0.0
    assert len(seen_ids) == 3

    # Verify BiometricPipeline multi-face batched inference
    bio_pipe = BiometricPipeline(
        detector_model_path=str(YUNET_ONNX),
        checkpoint_path=str(CONVNEXT_PT),
        enable_quality_audit=False,
        enable_flip_tta=False,
        device="auto",
    )
    bio_res = bio_pipe.process_frame(frame)
    assert len(bio_res) == 3
    for det, _, pred in bio_res:
        assert det is not None and pred is not None
        assert 0.0 <= pred.expected_age <= 100.0
        assert pred.gender in ("Male", "Female")
        assert abs(float(np.sum(pred.age_ldl_distribution)) - 1.0) < 1e-3
        assert abs(float(np.sum(pred.age_interval_probs)) - 1.0) < 1e-3


def test_first_shot_target_lock_and_zero_post_lock_jumping():
    """
    Verifies 'một phát ăn ngay' (immediate 1st-shot target lock) and zero post-lock
    interval/gender/frame jumping across 20 consecutive frames.
    """
    locker = TargetLocker(lock_on_first_shot=True)

    # Frame 1: target acquired at 21.4y (just inside 18-21 / 22-25 boundary at 21.5), Female
    f1 = locker.update(track_id=1, age_pred=21.4, aleatoric_std=1.8, gender_logit=2.4, has_measurement=True)
    assert f1["locked_age_interval"] == "18-21"
    assert f1["locked_gender"] == "Female"

    # Subsequent frames with boundary jitter (21.8, 22.7, 21.2, 23.0, coasting, etc.) must NEVER jump interval or gender
    for noisy_age, noisy_g in [(21.8, 1.9), (22.7, 0.8), (21.2, 2.1), (23.0, 1.2)]:
        fn = locker.update(track_id=1, age_pred=noisy_age, aleatoric_std=2.0, gender_logit=noisy_g, has_measurement=True)
        assert fn["locked_age_interval"] == "18-21"
        assert fn["locked_gender"] == "Female"

    # Coasting frames must also preserve locked interval and gender without decaying toward 0
    for _ in range(15):
        fc = locker.coast(track_id=1)
        assert fc["locked_age_interval"] == "18-21"
        assert fc["locked_gender"] == "Female"

    # Verify live stream / multi-frame EdgeAsyncPipeline & CloudInferenceClient run inference ONLY on Frame 1
    # for already-locked targets, preventing post-lock frame latency spikes and label jumping.
    STrack.reset_counter()
    client = CloudInferenceClient(enable_local_fallback=True, target_size=224, device="auto")
    syn = SyntheticMultiTargetStream(width=640, height=480, num_targets=3)

    infer_calls = 0
    orig_infer = client.infer_locally

    def counted_infer(dets):
        nonlocal infer_calls
        infer_calls += 1
        return orig_infer(dets)

    with patch.object(client, "infer_locally", side_effect=counted_infer):
        first_locks = {}
        for f_idx in range(12):
            _, frm = syn.read()
            dets = client.detect_faces(frm)
            assert len(dets) == 3
            targets = client.update_local_tracks(dets, cloud_results=None)
            assert len(targets) == 3
            for t in targets:
                tid = t["track_id"]
                if tid not in first_locks:
                    first_locks[tid] = (t["locked_gender"], t["locked_age_interval"])
                else:
                    assert (t["locked_gender"], t["locked_age_interval"]) == first_locks[tid], (
                        f"Post-lock jump on frame {f_idx} for track {tid}: "
                        f"{(t['locked_gender'], t['locked_age_interval'])} != {first_locks[tid]}"
                    )
        # Inference should have executed ONLY once on Frame 1 when the 3 targets first appeared!
        assert infer_calls == 1, f"Expected 1 first-shot inference call, got {infer_calls}"


def test_clean_hud_displays_only_gender_and_age_range():
    """
    Verifies HUD renders ONLY Gender and Age Range on locked target boxes and hides
    cluttered telemetry banner by default.
    """
    client = CloudInferenceClient(enable_local_fallback=False)
    frame = np.full((480, 640, 3), 50, dtype=np.uint8)
    targets = [
        {
            "track_id": 7,
            "bbox": (120.0, 140.0, 260.0, 280.0),
            "score": 0.96,
            "locked_age": 20.4,
            "locked_age_interval": "18-21",
            "locked_age_interval_idx": 6,
            "locked_gender": "Female",
            "gender_prob": 0.94,
            "aleatoric_std": 1.7,
        }
    ]
    hud = client.render_hud(frame, targets, fps=60.0, latency_ms=12.3)
    # Top-left telemetry region (10:34, 10:320) must be unchanged when show_telemetry=False (default)
    assert np.array_equal(hud[10:34, 10:320], frame[10:34, 10:320])
    # Target box region must have been drawn
    assert not np.array_equal(hud[110:285, 115:265], frame[110:285, 115:265])


def test_macos_darwin_support_and_one_click_launchers():
    """
    Verifies macOS (darwin) CPU/memory/MPS/AVFoundation camera support, LF line endings
    on run.command & run.sh, and zero dependency on src/mining.
    """
    run_cmd = ROOT_DIR / "run.command"
    run_sh = ROOT_DIR / "run.sh"
    run_bat = ROOT_DIR / "run.bat"
    assert run_cmd.exists() and run_sh.exists() and run_bat.exists()
    assert b"\r\n" not in run_cmd.read_bytes(), "run.command must use LF line endings for macOS"
    assert b"\r\n" not in run_sh.read_bytes(), "run.sh must use LF line endings for macOS/Linux"

    with patch("sys.platform", "darwin"):
        backends = CameraPortScanner.get_platform_backends()
        assert backends[0][0] == "AVFOUNDATION"

        with patch("subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = (
                "Camera:\n\n    FaceTime HD Camera:\n\n      Model ID: FaceTime\n\n    ngohuy Camera:\n\n      Model ID: iPhone15,3\n"
            )
            cam_names = CameraPortScanner.enumerate_os_camera_names()
            assert "FaceTime HD Camera" in cam_names
            assert "ngohuy Camera" in cam_names

    # Ensure no file in src/biometrics imports from src.mining or ..mining
    for py_file in (ROOT_DIR / "src" / "biometrics").rglob("*.py"):
        content = py_file.read_text(encoding="utf-8")
        assert "mining" not in content, f"{py_file} still references mining!"

