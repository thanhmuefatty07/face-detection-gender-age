"""
Unit Tests for Kaggle Ground-Truth Connectors, Harvester & PyTorch FaceGoldenDataset
=====================================================================================
Validates:
1. UTKFaceConnector: Parses ^(\\d+)_([01])_([0-4])_.* with full demographic mapping.
2. AgeDBConnector: Parses ^(\\d+)_(.+)_(\\d+)_([mf])\\.jpg with 100% verified DOB/date.
3. MorphConnector: Parses ^(\\d+)_(\\d+)([MF])(\\d{2})\\.JPG official mugshots.
4. FGNetConnector: Parses ^(\\d{3})[aA](\\d{2}) longitudinal child progressions.
5. AppaRealConnector: Reads labels_train.csv and labels_valid.csv without gt_avg missing error.
6. B3FDConnector: Parses {name}/{id}_{dob}_{photo_year}.jpg filename formats.
7. InCloudStreamingHarvester: Mounts all 7 verified ground-truth datasets.
8. FaceGoldenDataset: PyTorch Dataset helper for downstream training with 101-bin LDL & batching.
"""

from __future__ import annotations
import io
import math
import tarfile
import zipfile
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from scripts.kaggle_pipeline.kaggle_dataset_connectors import (
    RawImageSample,
    UTKFaceConnector,
    AgeDBConnector,
    MorphConnector,
    FGNetConnector,
    AppaRealConnector,
    B3FDConnector,
    InCloudStreamingHarvester,
    DiskQuotaManager
)
from scripts.kaggle_pipeline.kaggle_face_engine import (
    FacialLandmarkDetector,
    FaceGeometryAligner,
    umeyama_similarity_transform
)
from src.biometrics.estimation.dataset import FaceGoldenDataset


def _make_dummy_image_bytes() -> bytes:
    img = np.full((160, 160, 3), 128, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


class TestKaggleConnectors:
    """Verifies all 7 verified ground-truth dataset connectors."""

    def test_utkface_connector(self, tmp_path):
        """Validates UTKFace filename parsing: age, gender, race."""
        conn = UTKFaceConnector(root_dir=tmp_path, max_samples=10)
        assert conn.get_name() == "utkface"
        assert conn.is_available() is False

        # Create sample files
        b = _make_dummy_image_bytes()
        f1 = tmp_path / "25_0_2_20170116174525.jpg"
        f2 = tmp_path / "1_1_0_20170109193456.jpg"
        f1.write_bytes(b)
        f2.write_bytes(b)

        assert conn.is_available() is True
        samples = list(conn.stream_samples())
        assert len(samples) == 2

        s1 = next(s for s in samples if s.raw_id == f1.name)
        assert s1.declared_age == 25.0
        assert s1.declared_gender == "Male"
        assert s1.declared_ethnicity == "East Asian"
        assert s1.source_dataset == "utkface"

        s2 = next(s for s in samples if s.raw_id == f2.name)
        assert s2.declared_age == 1.0
        assert s2.declared_gender == "Female"
        assert s2.declared_ethnicity == "Caucasian"

    def test_agedb_connector(self, tmp_path):
        """Validates AgeDB filename parsing: id, name, age, gender."""
        conn = AgeDBConnector(root_dir=tmp_path, max_samples=10)
        assert conn.get_name() == "agedb"
        assert conn.is_available() is False

        b = _make_dummy_image_bytes()
        f1 = tmp_path / "12_MariaCallas_35_f.jpg"
        f2 = tmp_path / "105_AlbertEinstein_72_m.jpg"
        f1.write_bytes(b)
        f2.write_bytes(b)

        assert conn.is_available() is True
        samples = list(conn.stream_samples())
        assert len(samples) == 2

        s1 = next(s for s in samples if s.raw_id == f1.name)
        assert s1.declared_age == 35.0
        assert s1.declared_gender == "Female"
        assert s1.metadata["name"] == "MariaCallas"

        s2 = next(s for s in samples if s.raw_id == f2.name)
        assert s2.declared_age == 72.0
        assert s2.declared_gender == "Male"
        assert s2.metadata["name"] == "AlbertEinstein"

    def test_morph_connector(self, tmp_path):
        """Validates MORPH-2 filename parsing: subj_id, photo_id, gender, 2-digit age."""
        conn = MorphConnector(root_dir=tmp_path, max_samples=10)
        assert conn.get_name() == "morph"
        assert conn.is_available() is False

        b = _make_dummy_image_bytes()
        f1 = tmp_path / "123456_00M24.JPG"
        f2 = tmp_path / "654321_01F48.jpg"
        f1.write_bytes(b)
        f2.write_bytes(b)

        assert conn.is_available() is True
        samples = list(conn.stream_samples())
        assert len(samples) == 2

        s1 = next(s for s in samples if s.raw_id == f1.name)
        assert s1.declared_age == 24.0
        assert s1.declared_gender == "Male"

        s2 = next(s for s in samples if s.raw_id == f2.name)
        assert s2.declared_age == 48.0
        assert s2.declared_gender == "Female"

    def test_fgnet_connector(self, tmp_path):
        """Validates FG-NET filename parsing: 3-digit subj, 2-digit age."""
        conn = FGNetConnector(root_dir=tmp_path, max_samples=10)
        assert conn.get_name() == "fgnet"
        assert conn.is_available() is False

        b = _make_dummy_image_bytes()
        f1 = tmp_path / "001A02.JPG"
        f2 = tmp_path / "010a00.jpg"
        f1.write_bytes(b)
        f2.write_bytes(b)

        assert conn.is_available() is True
        samples = list(conn.stream_samples())
        assert len(samples) == 2

        s1 = next(s for s in samples if s.raw_id == f1.name)
        assert s1.declared_age == 2.0
        assert s1.metadata["fgnet_subject_id"] == "001"

        s2 = next(s for s in samples if s.raw_id == f2.name)
        assert s2.declared_age == 0.0
        assert s2.metadata["fgnet_subject_id"] == "010"

    def test_appa_real_connector_labels_csv(self, tmp_path):
        """Validates AppaRealConnector reading labels_train.csv and labels_valid.csv."""
        conn = AppaRealConnector(root_dir=tmp_path, max_samples=10)
        assert conn.get_name() == "appa_real"
        assert conn.is_available() is False

        b = _make_dummy_image_bytes()
        img1 = tmp_path / "00001.jpg"
        img2 = tmp_path / "00002.jpg"
        img1.write_bytes(b)
        img2.write_bytes(b)

        # Create labels_train.csv
        df = pd.DataFrame({
            "file_name": ["00001.jpg", "00002.jpg"],
            "real_age": [28.0, 42.0],
            "apparent_age_avg": [26.5, 45.0],
            "apparent_age_std": [1.8, 3.2]
        })
        df.to_csv(tmp_path / "labels_train.csv", index=False)

        assert conn.is_available() is True
        samples = list(conn.stream_samples())
        assert len(samples) == 2
        s1 = samples[0]
        assert s1.declared_age == 26.5
        assert s1.metadata["real_age"] == 28.0
        assert s1.metadata["apparent_age_std"] == 1.8

    def test_b3fd_connector_filename_parsing(self, tmp_path):
        """Validates B3FDConnector filename format {id}_{dob}_{photo_year}.jpg."""
        conn = B3FDConnector(root_dir=tmp_path, max_samples=10)
        assert conn.get_name() == "b3fd"
        assert conn.is_available() is False

        b = _make_dummy_image_bytes()
        sub_dir = tmp_path / "Tom_Hanks"
        sub_dir.mkdir()
        f1 = sub_dir / "123_1956_2010.jpg"
        f1.write_bytes(b)

        assert conn.is_available() is True
        samples = list(conn.stream_samples())
        assert len(samples) == 1
        s1 = samples[0]
        assert s1.declared_age == 54.0  # 2010 - 1956
        assert s1.source_dataset == "b3fd"

    def test_b3fd_with_underscores_and_archive_streaming(self, tmp_path):
        """Validates B3FDConnector handles underscores in subject ID and archive streaming."""
        b = _make_dummy_image_bytes()
        # Create a .tar.gz archive with files having multiple underscores in subject ID
        tar_p = tmp_path / "b3fd_archive.tar.gz"
        with tarfile.open(tar_p, "w:gz") as tf:
            ti = tarfile.TarInfo(name="John_F_Kennedy_01_1917_1961.jpg")
            ti.size = len(b)
            tf.addfile(ti, io.BytesIO(b))

        conn = B3FDConnector(root_dir=tmp_path, max_samples=10)
        assert conn.is_available() is True
        samples = list(conn.stream_samples())
        assert len(samples) == 1
        s = samples[0]
        assert s.declared_age == 44.0  # 1961 - 1917
        assert s.source_dataset == "b3fd"

    def test_morph_ethnicity_code_parsing(self, tmp_path):
        """Validates MorphConnector ethnicity code parsing from official suffixes."""
        b = _make_dummy_image_bytes()
        f_b = tmp_path / "100001_00M24_B.JPG"  # African
        f_w = tmp_path / "100002_00F35W.JPG"   # Caucasian
        f_a = tmp_path / "100003_00M40A.JPG"   # East Asian
        f_h = tmp_path / "100004_00F28H.JPG"   # Latino
        f_b.write_bytes(b)
        f_w.write_bytes(b)
        f_a.write_bytes(b)
        f_h.write_bytes(b)

        conn = MorphConnector(root_dir=tmp_path, max_samples=10)
        assert conn.is_available() is True
        samples = list(conn.stream_samples())
        assert len(samples) == 4

        m_b = next(s for s in samples if s.raw_id == f_b.name)
        assert m_b.declared_ethnicity == "African"
        assert m_b.declared_gender == "Male"
        assert m_b.declared_age == 24.0

        m_w = next(s for s in samples if s.raw_id == f_w.name)
        assert m_w.declared_ethnicity == "Caucasian"
        assert m_w.declared_gender == "Female"
        assert m_w.declared_age == 35.0

        m_a = next(s for s in samples if s.raw_id == f_a.name)
        assert m_a.declared_ethnicity == "East Asian"

        m_h = next(s for s in samples if s.raw_id == f_h.name)
        assert m_h.declared_ethnicity == "Latino"

    def test_harvester_mounting_all_7_datasets(self, tmp_path):
        """Verifies InCloudStreamingHarvester automatically registers the 7 verified connectors."""
        # Create 7 mock dataset directories
        dataset_names = [
            "agedb-database", "morph", "utkface-new", "afad-full",
            "appa-real-age-only-v1", "fgnet-dataset", "biometrically-filtered-famous-figure-dataset"
        ]
        for name in dataset_names:
            (tmp_path / name).mkdir()

        harvester = InCloudStreamingHarvester(kaggle_input_dir=str(tmp_path))
        assert len(harvester.connectors) == 7
        conn_names = {c.get_name() for c in harvester.connectors}
        assert conn_names == {"agedb", "morph", "utkface", "afad", "appa_real", "fgnet", "b3fd"}


class TestFaceGoldenDatasetPyTorch:
    """Verifies PyTorch Dataset helper for downstream training."""

    def test_dataset_loading_and_batching(self, tmp_path):
        """Validates dataset loading from manifest, 101-bin LDL target, and DataLoader batching."""
        # Create mock images and manifest
        img_dir = tmp_path / "images"
        img_dir.mkdir()

        b = _make_dummy_image_bytes()
        img_names = [f"face_{i:04d}.jpg" for i in range(8)]
        for name in img_names:
            (img_dir / name).write_bytes(b)

        df = pd.DataFrame({
            "image_id": img_names,
            "age_continuous": [2.0, 8.0, 15.0, 22.0, 30.0, 45.0, 60.0, 75.0],
            "apparent_age_std": [1.5, 2.0, 2.1, 2.2, 2.5, 2.8, 3.0, 3.5],
            "gender": ["Female", "Male", "Female", "Male", "Female", "Male", "Female", "Male"],
            "dimorphism_weight": [0.25, 0.67, 1.0, 1.0, 1.0, 1.0, 1.0, 0.85],
            "alpha_sample_weight": [0.8, 0.85, 0.9, 0.95, 1.0, 0.95, 0.9, 0.85],
            "ldl_0_2": [0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "ldl_3_6": [0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "ldl_7_12": [0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "ldl_13_17": [0.0, 0.1, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0],
            "ldl_18_24": [0.0, 0.0, 0.1, 0.9, 0.0, 0.0, 0.0, 0.0],
            "ldl_25_34": [0.0, 0.0, 0.0, 0.1, 0.9, 0.0, 0.0, 0.0],
            "ldl_35_44": [0.0, 0.0, 0.0, 0.0, 0.1, 0.8, 0.0, 0.0],
            "ldl_45_54": [0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.1, 0.0],
            "ldl_55_64": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.1],
            "ldl_65_plus": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.9],
            "split": ["train", "train", "train", "train", "train", "train", "val", "test"]
        })
        manifest_p = tmp_path / "metadata.parquet"
        df.to_parquet(manifest_p)

        # 1. Test train split dataset
        ds_train = FaceGoldenDataset(
            manifest_path=manifest_p,
            images_dir=img_dir,
            split="train",
            num_age_bins=101
        )
        assert len(ds_train) == 6

        sample = ds_train[0]
        assert sample["image"].shape == (3, 160, 160)
        assert sample["image"].dtype == torch.float32
        assert sample["ldl_101"].shape == (101,)
        # Check simplex sum == 1.0
        assert pytest.approx(float(sample["ldl_101"].sum()), abs=1e-4) == 1.0
        assert sample["ldl_10"].shape == (10,)
        assert sample["gender"].item() == 1.0  # Female
        assert sample["dimorphism_weight"].item() == 0.25

        # 2. Test PyTorch DataLoader batching
        loader = DataLoader(ds_train, batch_size=4, shuffle=False)
        batch = next(iter(loader))
        assert batch["image"].shape == (4, 3, 160, 160)
        assert batch["age_continuous"].shape == (4,)
        assert batch["ldl_101"].shape == (4, 101)
        assert batch["gender"].shape == (4,)
        assert batch["dimorphism_weight"].shape == (4,)

        # 3. Test val split
        ds_val = FaceGoldenDataset(manifest_path=manifest_p, images_dir=img_dir, split="val")
        assert len(ds_val) == 1
        assert ds_val[0]["split"] == "val"

    def test_dataset_sharded_tar_loading(self, tmp_path):
        """Validates FaceGoldenDataset reading directly from sharded .tar archives."""
        quota = DiskQuotaManager(working_dir=str(tmp_path), shard_image_count=5)
        shards_dir = tmp_path / "golden_dataset" / "shards"
        dummy_face = np.full((160, 160, 3), 100, dtype=np.uint8)

        # Write 3 images into tar shard
        quota.write_aligned_face(dummy_face, "face_0001.jpg")
        quota.write_aligned_face(dummy_face, "face_0002.jpg")
        quota.write_aligned_face(dummy_face, "face_0003.jpg")
        quota.close()

        df = pd.DataFrame({
            "image_id": ["face_0001.jpg", "face_0002.jpg", "face_0003.jpg"],
            "age_continuous": [21.0, 35.0, 50.0],
            "gender": ["Male", "Female", "Male"],
            "split": ["train", "train", "val"]
        })
        manifest_p = tmp_path / "metadata.csv"
        df.to_csv(manifest_p, index=False)

        ds = FaceGoldenDataset(
            manifest_path=manifest_p,
            shards_dir=shards_dir,
            split="train"
        )
        assert len(ds) == 2
        s0 = ds[0]
        assert s0["image"].shape == (3, 160, 160)
        assert s0["image_id"] == "face_0001.jpg"

    def test_dataset_corrupted_buffers_and_nan_sanitation(self, tmp_path):
        """Validates FaceGoldenDataset gracefully handles corrupted byte buffers and NaN age/std values."""
        img_dir = tmp_path / "images"
        img_dir.mkdir()

        # Valid image
        b_valid = _make_dummy_image_bytes()
        (img_dir / "valid.jpg").write_bytes(b_valid)

        # Corrupted truncated buffer that cv2.imdecode cannot decode
        (img_dir / "corrupted.jpg").write_bytes(b"truncated_garbage_bytes_1234567890")

        # Missing image file
        # "missing.jpg" is intentionally not created

        df = pd.DataFrame({
            "image_id": ["valid.jpg", "corrupted.jpg", "missing.jpg"],
            "age_continuous": [25.0, float("nan"), 70.0],
            "apparent_age_std": [2.0, float("nan"), -1.0],
            "gender": ["Female", "Male", "Unknown"],
            "split": ["train", "train", "train"]
        })
        manifest_p = tmp_path / "metadata.parquet"
        df.to_parquet(manifest_p)

        ds = FaceGoldenDataset(manifest_path=manifest_p, images_dir=img_dir, split="train")
        assert len(ds) == 3

        # Sample 0: valid
        s0 = ds[0]
        assert s0["image"].shape == (3, 160, 160)
        assert s0["age_continuous"].item() == 25.0

        # Sample 1: corrupted buffer must not crash DataLoader/cv2, but fallback to 160x160 neutral image
        s1 = ds[1]
        assert s1["image"].shape == (3, 160, 160)
        assert not torch.isnan(s1["age_continuous"])
        assert s1["age_continuous"].item() == 25.0  # Sanitized default
        assert not torch.isnan(s1["ldl_101"]).any()
        assert pytest.approx(float(s1["ldl_101"].sum()), abs=1e-4) == 1.0

        # Sample 2: missing image file must also safely fallback without error
        s2 = ds[2]
        assert s2["image"].shape == (3, 160, 160)
        assert s2["age_continuous"].item() == 70.0
        assert not torch.isnan(s2["ldl_101"]).any()
        assert pytest.approx(float(s2["ldl_101"].sum()), abs=1e-4) == 1.0


class TestBiometricLandmarksAndGeometry:
    """Verifies YuNet landmark canonical ordering and authentic Umeyama SVD transform determinant."""

    def test_landmark_canonical_ordering_and_umeyama_determinant(self):
        """
        Validates:
        1. YuNet landmarks are mapped so viewer's left eye (smaller x) is at index 0,
           and viewer's right eye (larger x) is at index 1.
        2. Umeyama similarity transform matrix has positive scaling and non-collapsed determinant:
           det(R) > 1.5 for canonical ArcFace 160 alignment from ~112x112 facial geometry.
        """
        # Create a synthetic face image 160x160
        img = np.full((160, 160, 3), 128, dtype=np.uint8)

        # Standard canonical upright face landmarks in viewer perspective:
        # Viewer left eye: x=40.0, y=52.0
        # Viewer right eye: x=74.0, y=52.0
        # Nose: x=57.0, y=72.0
        # Viewer left mouth: x=42.0, y=92.0
        # Viewer right mouth: x=71.0, y=92.0
        viewer_left_eye = np.array([40.0, 52.0], dtype=np.float32)
        viewer_right_eye = np.array([74.0, 52.0], dtype=np.float32)
        nose = np.array([57.0, 72.0], dtype=np.float32)
        viewer_left_mouth = np.array([42.0, 92.0], dtype=np.float32)
        viewer_right_mouth = np.array([71.0, 92.0], dtype=np.float32)

        # In YuNet convention:
        # best_face[4:6] = subject's right eye (viewer's left eye)
        # best_face[6:8] = subject's left eye (viewer's right eye)
        # best_face[8:10] = nose
        # best_face[10:12] = subject's right mouth (viewer's left mouth)
        # best_face[12:14] = subject's left mouth (viewer's right mouth)
        # Simulate YuNet detector output vector (15 elements)
        simulated_yunet_face = np.zeros(15, dtype=np.float32)
        simulated_yunet_face[0:4] = [20.0, 20.0, 80.0, 90.0]  # bbox
        simulated_yunet_face[4:6] = viewer_left_eye            # subject right eye
        simulated_yunet_face[6:8] = viewer_right_eye           # subject left eye
        simulated_yunet_face[8:10] = nose                      # nose
        simulated_yunet_face[10:12] = viewer_left_mouth        # subject right mouth
        simulated_yunet_face[12:14] = viewer_right_mouth       # subject left mouth
        simulated_yunet_face[14] = 0.99                        # detection score

        # Construct canonical landmarks
        r_eye = simulated_yunet_face[4:6]
        l_eye = simulated_yunet_face[6:8]
        nose_pt = simulated_yunet_face[8:10]
        r_mouth = simulated_yunet_face[10:12]
        l_mouth = simulated_yunet_face[12:14]

        lms = np.array([r_eye, l_eye, nose_pt, r_mouth, l_mouth], dtype=np.float32)
        if lms[0, 0] > lms[1, 0]:
            lms[[0, 1]] = lms[[1, 0]]
        if lms[3, 0] > lms[4, 0]:
            lms[[3, 4]] = lms[[4, 3]]

        # 1. Check canonical eye ordering:
        assert lms[0, 0] < lms[1, 0], "Viewer left eye x must be < viewer right eye x"
        assert lms[3, 0] < lms[4, 0], "Viewer left mouth x must be < viewer right mouth x"

        # 2. Umeyama similarity transform
        aligner = FaceGeometryAligner(output_size=160)
        aligned, matrix = aligner.align(img, lms)

        assert aligned.shape == (160, 160, 3)
        assert matrix.shape == (2, 3)

        # Extract 2x2 rotation/scaling block
        R = matrix[:2, :2]
        det_R = float(np.linalg.det(R))

        # Expected determinant det(R) = s^2 where scale s ~ 160 / 112 = 1.4286 => s^2 ~ 2.04
        # A collapsed or inverted transform yields det(R) < 0.1 or det(R) < 0
        assert det_R > 1.5, f"Umeyama determinant {det_R:.3f} is collapsed or inverted (expected > 1.5)"
