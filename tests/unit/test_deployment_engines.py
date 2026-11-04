"""
Unit Tests for Dual-Mode Deployment Engines
===========================================
Tests:
1. Kaggle Training Kernel Generator (scripts/kaggle_pipeline/kaggle_training_kernel.py)
2. Kaggle Cloud Inference Server (scripts/kaggle_pipeline/kaggle_inference_server.py)
3. Local iVCam Cloud WebSocket Client (src/biometrics/inference/cloud_client.py)
4. Local Edge OpenVINO Async Pipeline (src/biometrics/inference/async_pipeline.py)
"""

from __future__ import annotations
import json
from pathlib import Path
import cv2
import numpy as np
import pytest
import torch

from scripts.kaggle_pipeline.kaggle_training_kernel import generate_training_notebook
from scripts.kaggle_pipeline.kaggle_inference_server import create_inference_app, process_face_crops
from src.biometrics.inference.cloud_client import CloudInferenceClient
from src.biometrics.inference.async_pipeline import EdgeAsyncPipeline


class TestKaggleTrainingKernel:
    def test_notebook_generation(self, tmp_path):
        nb_path = tmp_path / "train_convnextv2.ipynb"
        generate_training_notebook(output_path=nb_path)

        assert nb_path.exists()
        content = json.loads(nb_path.read_text(encoding="utf-8"))
        assert "cells" in content
        assert len(content["cells"]) >= 4

        # Verify key keywords are embedded in notebook cells
        all_text = " ".join("".join(c.get("source", [])) for c in content["cells"])
        assert "convnextv2" in all_text.lower()
        assert "FaceGoldenDataset" in all_text
        assert "UnifiedBiometricLoss" in all_text
        assert "gender_mask" in all_text


class TestKaggleInferenceServer:
    def test_crop_processing(self):
        # Create 2 dummy 160x160 JPEG face crops
        dummy_img = np.full((160, 160, 3), 120, dtype=np.uint8)
        _, enc = cv2.imencode(".jpg", dummy_img)
        jpeg_bytes = enc.tobytes()

        # Mock model or CPU model
        from src.biometrics.estimation.backbone import BiometricMultiTaskModel
        model = BiometricMultiTaskModel(backbone_name="convnextv2_nano", pretrained=False)
        model.eval()

        results = process_face_crops(model=model, crop_bytes_list=[jpeg_bytes, jpeg_bytes], device="cpu")
        assert len(results) == 2
        for r in results:
            assert "expected_age" in r
            assert "gender_prob" in r
            assert "gender_logit" in r
            assert "aleatoric_std" in r
            assert "reid_embedding" in r
            assert len(r["reid_embedding"]) == 64

    def test_fastapi_app_health(self):
        from starlette.testclient import TestClient
        app = create_inference_app()
        client = TestClient(app)
        res = client.get("/health")
        assert res.status_code == 200
        data = res.json()
        assert data.get("status") == "ok"


class TestCloudInferenceClient:
    def test_client_payload_and_local_tracking(self):
        client = CloudInferenceClient(server_url="ws://localhost:8000/ws/infer", max_targets=5)
        # Simulate local detections and mock cloud results
        dummy_crop = np.full((160, 160, 3), 120, dtype=np.uint8)
        dets = [
            {"bbox": (50.0, 50.0, 110.0, 110.0), "score": 0.9, "crop": dummy_crop}
        ]
        mock_cloud_results = [
            {
                "expected_age": 28.5,
                "gender_prob": 0.92,
                "gender_logit": 2.4,
                "aleatoric_std": 1.2,
                "reid_embedding": np.ones(64, dtype=np.float32).tolist()
            }
        ]

        active_targets = client.update_local_tracks(dets, mock_cloud_results)
        assert len(active_targets) == 1
        tgt = active_targets[0]
        assert "track_id" in tgt
        assert "locked_age" in tgt
        assert "locked_gender" in tgt
        assert tgt["locked_gender"] == "Female"

    def test_client_crop_encoding_and_hud(self):
        client = CloudInferenceClient()
        dummy_crop = np.full((160, 160, 3), 150, dtype=np.uint8)
        dets = [{"bbox": (20.0, 20.0, 80.0, 80.0), "score": 0.95, "crop": dummy_crop}]
        encoded = client.encode_crops(dets)
        assert len(encoded) == 1
        assert isinstance(encoded[0], str)
        assert len(encoded[0]) > 0

        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        targets = [{
            "track_id": 1,
            "bbox": (20.0, 20.0, 80.0, 80.0),
            "locked_age": 30.0,
            "locked_gender": "Male",
            "aleatoric_std": 1.0,
        }]
        hud = client.render_hud(frame, targets)
        assert hud.shape == (200, 200, 3)
        assert hud.sum() > 0

    def test_cloud_client_preserves_and_coasts_biometric_lock_across_temporary_occlusion(self):
        """Verifies temporarily occluded tracks in lost_tracks retain and coast their TargetLocker state upon Re-ID recovery."""
        client = CloudInferenceClient(server_url="ws://localhost:8000/ws/infer", max_targets=5)
        dummy_crop = np.full((160, 160, 3), 120, dtype=np.uint8)
        reid_vec = np.zeros(64, dtype=np.float32)
        reid_vec[0] = 1.0

        dets = [{"bbox": (50.0, 50.0, 110.0, 110.0), "score": 0.92, "crop": dummy_crop}]
        cloud_res = [{
            "expected_age": 64.0,
            "gender_prob": 0.95,
            "gender_logit": 3.0,
            "aleatoric_std": 0.8,
            "reid_embedding": reid_vec.tolist(),
        }]

        for _ in range(3):
            active = client.update_local_tracks(dets, cloud_res)
        assert len(active) == 1
        tid = active[0]["track_id"]
        assert active[0]["locked_age_interval"] == "61-65"
        log_odds_before_occ = client.locker.gender_accumulators[tid].log_odds

        # Occlude for 3 frames (no detections -> track moves to client.tracker.lost_tracks)
        for _ in range(3):
            occ_active = client.update_local_tracks([], [])
            assert occ_active == []
            # Track MUST still be preserved in TargetLocker while in lost_tracks!
            assert tid in client.locker.active_track_ids

        # Subject reappears at a shifted position with matching 64-D Re-ID embedding and slight boundary age 65.7
        dets_reappear = [{"bbox": (180.0, 50.0, 240.0, 110.0), "score": 0.92, "crop": dummy_crop}]
        cloud_reappear = [{
            "expected_age": 65.7,
            "gender_prob": 0.90,
            "gender_logit": 1.0,
            "aleatoric_std": 0.8,
            "reid_embedding": reid_vec.tolist(),
        }]
        recovered = client.update_local_tracks(dets_reappear, cloud_reappear)
        assert len(recovered) == 1
        assert recovered[0]["track_id"] == tid
        # Because TargetLocker preserved the 64.0 Kalman state and '61-65' hysteresis lock,
        # smoothed age stays inside the '61-65' hysteresis retention band instead of resetting to '66-70'
        assert recovered[0]["locked_age_interval"] == "61-65"
        assert client.locker.gender_accumulators[tid].log_odds > log_odds_before_occ * (0.95 ** 4)

    def test_cloud_client_detect_faces_uses_umeyama_alignment_with_landmarks(self):
        """Verifies CloudInferenceClient.detect_faces aligns crops via 5-point Umeyama when landmarks are present."""
        from unittest.mock import MagicMock
        from src.biometrics.core.types import BoundingBox, FaceLandmarks, FaceDetection

        client = CloudInferenceClient()
        assert client.detect_faces(None) == []
        assert client.detect_faces(np.array([])) == []

        lmks = FaceLandmarks(np.array(
            [[55.0, 65.0], [95.0, 65.0], [75.0, 85.0], [60.0, 105.0], [90.0, 105.0]],
            dtype=np.float32,
        ))
        det_obj = FaceDetection(
            bbox=BoundingBox(x=40, y=40, width=70, height=85, confidence=0.95),
            landmarks=lmks,
        )
        client.detector.detect = MagicMock(return_value=[det_obj])
        client.aligner.align = MagicMock(wraps=client.aligner.align)

        frame = np.full((200, 200, 3), 128, dtype=np.uint8)
        dets = client.detect_faces(frame)
        assert len(dets) == 1
        assert dets[0]["crop"].shape == (160, 160, 3)
        assert client.aligner.align.call_count == 1


class TestEdgeAsyncPipeline:
    def test_pipeline_execution(self):
        pipeline = EdgeAsyncPipeline(backbone_name="convnextv2_nano")
        dummy_frame = np.full((320, 240, 3), 100, dtype=np.uint8)
        
        # Test frame submission and processing
        output_targets = pipeline.process_frame(dummy_frame)
        assert isinstance(output_targets, list)
        assert pipeline.process_frame(None) == []
        assert pipeline.process_frame(np.array([])) == []

    def test_multi_target_biometric_alignment(self):
        """Verifies that multiple targets in a frame receive their respective biometrics without swap."""
        from unittest.mock import MagicMock
        pipeline = EdgeAsyncPipeline(backbone_name="convnextv2_nano", use_openvino=False)

        # Mock detections: Target A at x=200, Target B at x=50
        pipeline.detect_faces = MagicMock(return_value=[
            {"bbox": (200.0, 50.0, 260.0, 110.0), "score": 0.9},
            {"bbox": (50.0, 50.0, 110.0, 110.0), "score": 0.9},
        ])

        # Mock inference: Crop 0 is Target A (Age 60, Male), Crop 1 is Target B (Age 20, Female)
        pipeline._infer_crops = MagicMock(return_value=(
            [60.0, 20.0],
            [1.0, 1.0],
            [-2.0, 2.0],
            [None, None]
        ))

        dummy_frame = np.full((320, 320, 3), 120, dtype=np.uint8)
        targets = pipeline.process_frame(dummy_frame)

        assert len(targets) == 2
        tgt_b = [t for t in targets if abs(t["bbox"][0] - 50.0) < 5.0][0]
        tgt_a = [t for t in targets if abs(t["bbox"][0] - 200.0) < 5.0][0]

        assert pytest.approx(tgt_b["locked_age"], abs=1.0) == 20.0
        assert tgt_b["locked_gender"] == "Female"
        assert pytest.approx(tgt_a["locked_age"], abs=1.0) == 60.0
        assert tgt_a["locked_gender"] == "Male"

    def test_openvino_runtime_compilation_and_parity(self):
        """Verifies OpenVINO compilation and numerical parity with PyTorch predict_smooth on real crops."""
        pipeline_ov = EdgeAsyncPipeline(backbone_name="convnextv2_nano", use_openvino=True, device="cpu")
        import sys
        if "openvino" in sys.modules and pipeline_ov.ov_compiled_model is not None:
            rng = np.random.RandomState(42)
            crop = rng.randint(0, 256, size=(160, 160, 3), dtype=np.uint8)

            # Compute PyTorch reference using the exact same model weights
            pipeline_pt = EdgeAsyncPipeline(backbone_name="convnextv2_nano", use_openvino=False, device="cpu")
            pipeline_pt.model.load_state_dict(pipeline_ov.model.state_dict())
            pt_age, pt_std, pt_g, pt_reid = pipeline_pt._infer_crops([crop])

            # Disable PyTorch model on pipeline_ov so fallback cannot mask an OpenVINO failure
            pipeline_ov.model = None
            ov_age, ov_std, ov_g, ov_reid = pipeline_ov._infer_crops([crop])

            assert pytest.approx(ov_age[0], abs=1e-3) == pt_age[0]
            assert pytest.approx(ov_std[0], abs=1e-3) == pt_std[0]
            assert pytest.approx(ov_g[0], abs=1e-3) == pt_g[0]
            assert ov_reid[0] is not None
            assert np.allclose(ov_reid[0], pt_reid[0], atol=1e-3)

    def test_process_frame_uses_umeyama_alignment_when_landmarks_present(self):
        """Verifies process_frame aligns face crops via 5-point Umeyama similarity transform when landmarks exist."""
        from unittest.mock import MagicMock

        pipeline = EdgeAsyncPipeline(backbone_name="convnextv2_nano", use_openvino=False)
        lmks = np.array(
            [[55.0, 65.0], [95.0, 65.0], [75.0, 85.0], [60.0, 105.0], [90.0, 105.0]],
            dtype=np.float32,
        )
        pipeline.detect_faces = MagicMock(return_value=[
            {"bbox": (40.0, 40.0, 110.0, 125.0), "score": 0.95, "landmarks": lmks},
        ])
        dummy_frame = np.full((200, 200, 3), 128, dtype=np.uint8)
        targets = pipeline.process_frame(dummy_frame)
        assert len(targets) == 1
        assert 0.0 <= targets[0]["locked_age"] <= 100.0
        assert "locked_age_interval" in targets[0]

    def test_edge_pipeline_preserves_and_coasts_biometric_lock_across_occlusion_and_reid_recovery(self):
        """Verifies EdgeAsyncPipeline retains and coasts TargetLocker state during temporary occlusion and Re-ID recovery."""
        from unittest.mock import MagicMock

        pipeline = EdgeAsyncPipeline(backbone_name="convnextv2_nano", use_openvino=False)
        reid_vec = np.zeros(64, dtype=np.float32)
        reid_vec[0] = 1.0

        pipeline.detect_faces = MagicMock(return_value=[
            {"bbox": (40.0, 40.0, 100.0, 100.0), "score": 0.92},
        ])
        pipeline._infer_crops = MagicMock(return_value=([64.0], [0.8], [3.0], [reid_vec]))

        frame = np.full((240, 240, 3), 120, dtype=np.uint8)
        for _ in range(3):
            targets = pipeline.process_frame(frame)
        assert len(targets) == 1
        tid = targets[0]["track_id"]
        assert targets[0]["locked_age_interval"] == "61-65"

        # Temporary occlusion for 3 frames
        pipeline.detect_faces = MagicMock(return_value=[])
        pipeline._infer_crops = MagicMock(return_value=([], [], [], []))
        for _ in range(3):
            occ_targets = pipeline.process_frame(frame)
            assert occ_targets == []
            assert tid in pipeline.locker.active_track_ids

        # Reappearance at shifted location recovered via 64-D Re-ID matching with boundary measurement 65.7
        pipeline.detect_faces = MagicMock(return_value=[
            {"bbox": (150.0, 40.0, 210.0, 100.0), "score": 0.92},
        ])
        pipeline._infer_crops = MagicMock(return_value=([65.7], [0.8], [1.0], [reid_vec]))
        recovered = pipeline.process_frame(frame)
        assert len(recovered) == 1
        assert recovered[0]["track_id"] == tid
        assert recovered[0]["locked_age_interval"] == "61-65"

    def test_224x224_canonical_resolution_across_server_client_edge_and_curator(self, tmp_path):
        """Verifies 224x224 crops are preserved natively across server, client, edge pipeline, and curator."""
        from unittest.mock import MagicMock
        from src.biometrics.estimation.backbone import BiometricMultiTaskModel
        from src.biometrics.core.types import BoundingBox, FaceLandmarks, FaceDetection
        from src.biometrics.pipeline import BiometricPipeline
        from src.mining.curation import ExtremeDatasetCurator

        # 1. kaggle_inference_server.process_face_crops preserves 224x224 without downsampling to 160x160
        dummy_224 = np.full((224, 224, 3), 135, dtype=np.uint8)
        _, enc_224 = cv2.imencode(".jpg", dummy_224)
        bytes_224 = enc_224.tobytes()

        model = BiometricMultiTaskModel(backbone_name="convnextv2_nano", pretrained=False).eval()
        seen_shapes = []
        orig_predict_smooth = model.predict_smooth

        def _spy_predict_smooth(tensor, enable_flip_tta=True):
            seen_shapes.append(tuple(tensor.shape))
            return orig_predict_smooth(tensor, enable_flip_tta=enable_flip_tta)

        model.predict_smooth = _spy_predict_smooth
        res_224 = process_face_crops(model=model, crop_bytes_list=[bytes_224, bytes_224], device="cpu")
        assert len(res_224) == 2
        assert seen_shapes == [(2, 3, 224, 224)]

        # 2. CloudInferenceClient(target_size=224) produces (224, 224, 3) aligned crops
        client_224 = CloudInferenceClient(target_size=224)
        lmks = FaceLandmarks(np.array(
            [[75.0, 100.0], [145.0, 100.0], [110.0, 140.0], [85.0, 180.0], [135.0, 180.0]],
            dtype=np.float32,
        ))
        det_obj = FaceDetection(
            bbox=BoundingBox(x=40, y=40, width=140, height=160, confidence=0.95),
            landmarks=lmks,
        )
        client_224.detector.detect = MagicMock(return_value=[det_obj])
        frame = np.full((300, 300, 3), 128, dtype=np.uint8)
        dets_224 = client_224.detect_faces(frame)
        assert len(dets_224) == 1
        assert dets_224[0]["crop"].shape == (224, 224, 3)

        # 3. EdgeAsyncPipeline(target_size=224) and BiometricPipeline(target_size=224)
        edge_224 = EdgeAsyncPipeline(backbone_name="convnextv2_nano", use_openvino=False, target_size=224)
        assert edge_224.aligner.target_size == 224
        bio_224 = BiometricPipeline(estimator_model=model, enable_quality_audit=False, target_size=224)
        assert bio_224.aligner.target_size == 224
        curator_224 = ExtremeDatasetCurator(output_dir=tmp_path / "curated_224", target_size=224)
        assert curator_224.aligner.target_size == 224

    def test_training_kernel_source_dataset_resolution_and_evaluation_report_audit(self, tmp_path):
        """Verifies kaggle_training_kernel resolves source_dataset and training_evaluation_report.json has 11 test sources."""
        nb_path = tmp_path / "train_convnextv2.ipynb"
        generate_training_notebook(output_path=nb_path)
        content = json.loads(nb_path.read_text(encoding="utf-8"))
        cell5_text = "".join(content["cells"][5]["source"])
        assert 'row.get("source_dataset"' in cell5_text
        assert '["source_dataset", "dataset_source", "dataset_name", "source", "dataset"]' in cell5_text

        report_path = Path(__file__).resolve().parent.parent.parent / "reports" / "training_evaluation_report.json"
        if report_path.exists():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            per_source = report["calibration_and_per_source_audit"]["per_dataset_source_breakdown"]
            assert "unknown" not in per_source
            assert len(per_source) == 11
            assert sum(v["count"] for v in per_source.values()) == 764

    def test_default_student_checkpoint_auto_resolution_and_github_size_limit(self):
        """Verifies default 224x224 Student checkpoint loading across EdgeAsyncPipeline, BiometricPipeline, run_client.py, and <100 MiB size."""
        from src.biometrics.pipeline import BiometricPipeline

        root = Path(__file__).resolve().parent.parent.parent
        ckpt_path = root / "models" / "convnextv2_tiny_biometrics.pt"
        yunet_path = root / "models" / "face_detection_yunet_2023mar.onnx"
        assert ckpt_path.exists()
        assert yunet_path.exists()
        assert ckpt_path.stat().st_size < 100 * 1024 * 1024  # Strictly under GitHub 100 MiB limit

        edge = EdgeAsyncPipeline(use_openvino=False)
        assert edge.backbone_name == "convnextv2_tiny"
        assert edge.target_size == 224
        assert edge.aligner.target_size == 224

        bio = BiometricPipeline(checkpoint_path=ckpt_path, enable_quality_audit=False)
        assert bio.target_size == 224
        assert bio.aligner.target_size == 224
        assert bio.estimator_model is not None

        run_client_src = (root / "run_client.py").read_text(encoding="utf-8")
        assert '"--target-size"' in run_client_src
        assert "default=224" in run_client_src
        assert "target_size=args.target_size" in run_client_src


