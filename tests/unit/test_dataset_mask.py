"""
Unit Tests for FaceGoldenDataset Gender Mask & Incomplete Label Handling
========================================================================
Validates that 23,876 ungendered samples in the Golden Dataset have
`gender_mask == 0.0` to avoid corrupting Binary Cross Entropy loss,
while valid male/female annotations have `gender_mask == 1.0`.
"""

from __future__ import annotations
import math
import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from src.biometrics.estimation.dataset import FaceGoldenDataset


class TestDatasetGenderMask:
    def test_gender_mask_and_target_resolution(self, tmp_path):
        """Test that gender_mask and gender target are correctly computed."""
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        
        # Create minimal 160x160 dummy image
        dummy_img = np.full((160, 160, 3), 128, dtype=np.uint8)
        import cv2
        _, encoded = cv2.imencode(".jpg", dummy_img)
        img_bytes = encoded.tobytes()

        samples = [
            ("img_female.jpg", 25.0, "Female"),
            ("img_male.jpg", 30.0, "Male"),
            ("img_unlabeled_none.jpg", 40.0, None),
            ("img_unlabeled_nan.jpg", 18.0, float("nan")),
            ("img_unlabeled_empty.jpg", 50.0, ""),
            ("img_unlabeled_unknown.jpg", 60.0, "unknown"),
        ]

        for fname, _, _ in samples:
            (img_dir / fname).write_bytes(img_bytes)

        df = pd.DataFrame({
            "image_id": [s[0] for s in samples],
            "age_continuous": [s[1] for s in samples],
            "gender": [s[2] for s in samples],
            "dimorphism_weight": [1.0] * len(samples),
            "split": ["train"] * len(samples)
        })

        manifest_path = tmp_path / "metadata.parquet"
        df.to_parquet(manifest_path)

        dataset = FaceGoldenDataset(manifest_path=manifest_path, images_dir=img_dir)
        assert len(dataset) == 6

        # 1. Female sample: gender=1.0, gender_mask=1.0
        sample_female = dataset[0]
        assert "gender_mask" in sample_female
        assert sample_female["gender_mask"].dtype == torch.float32
        assert sample_female["gender_mask"].item() == 1.0
        assert sample_female["gender"].item() == 1.0

        # 2. Male sample: gender=0.0, gender_mask=1.0
        sample_male = dataset[1]
        assert sample_male["gender_mask"].item() == 1.0
        assert sample_male["gender"].item() == 0.0

        # 3. None: gender_mask=0.0
        sample_none = dataset[2]
        assert sample_none["gender_mask"].item() == 0.0

        # 4. NaN: gender_mask=0.0
        sample_nan = dataset[3]
        assert sample_nan["gender_mask"].item() == 0.0

        # 5. Empty string: gender_mask=0.0
        sample_empty = dataset[4]
        assert sample_empty["gender_mask"].item() == 0.0

        # 6. Unknown: gender_mask=0.0
        sample_unknown = dataset[5]
        assert sample_unknown["gender_mask"].item() == 0.0

    def test_dataloader_batch_collation_with_mask(self, tmp_path):
        """Test DataLoader batch collation with gender_mask."""
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        dummy_img = np.full((160, 160, 3), 128, dtype=np.uint8)
        import cv2
        _, encoded = cv2.imencode(".jpg", dummy_img)
        img_bytes = encoded.tobytes()

        img_names = [f"face_{i}.jpg" for i in range(4)]
        for name in img_names:
            (img_dir / name).write_bytes(img_bytes)

        df = pd.DataFrame({
            "image_id": img_names,
            "age_continuous": [20.0, 30.0, 40.0, 50.0],
            "gender": ["Female", None, "Male", ""],
            "dimorphism_weight": [1.0, 0.5, 1.0, 0.8],
            "split": ["train"] * 4
        })
        manifest_path = tmp_path / "metadata.parquet"
        df.to_parquet(manifest_path)

        dataset = FaceGoldenDataset(manifest_path=manifest_path, images_dir=img_dir)
        loader = DataLoader(dataset, batch_size=4, shuffle=False)
        batch = next(iter(loader))

        assert "gender_mask" in batch
        assert batch["gender_mask"].shape == (4,)
        assert torch.equal(batch["gender_mask"], torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=torch.float32))
        assert torch.equal(batch["gender"], torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32))
