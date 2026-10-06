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


class TestEdgeAsyncPipeline:
    def test_pipeline_execution(self):
        pipeline = EdgeAsyncPipeline(backbone_name="convnextv2_nano")
        dummy_frame = np.full((320, 240, 3), 100, dtype=np.uint8)
        
        # Test frame submission and processing
        output_targets = pipeline.process_frame(dummy_frame)
        assert isinstance(output_targets, list)
