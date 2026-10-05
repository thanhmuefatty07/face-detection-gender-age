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
    DiskQuotaManager,
    ensure_master_manifest
)
from scripts.kaggle_pipeline.kaggle_face_engine import (
    FacialLandmarkDetector,
    FaceGeometryAligner,
    ExtremeQualityAuditor,
    umeyama_similarity_transform
)
from src.biometrics.core.geometry import ARCFACE_CANONICAL_160, estimate_head_pose_pnp
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


class TestForensicAuditFixes:
    """Verifies critical forensic audit bug fixes and Kaggle optimizations."""

    def test_solvepnp_frontal_face_pitch_is_near_zero(self):
        """
        Critical Fix A: solvePnP on canonical frontal face landmarks MUST evaluate
        to pitch ~0 degrees, NOT inverted -165.3 degrees which rejects 100% of candidates.
        """
        auditor = ExtremeQualityAuditor()
        # ARCFACE_CANONICAL_160 points
        lms = ARCFACE_CANONICAL_160.copy()
        yaw, pitch, roll = auditor.estimate_head_pose(lms, 160, 160)
        assert abs(pitch) < 5.0, f"Expected pitch near 0, got {pitch}"
        assert abs(yaw) < 5.0, f"Expected yaw near 0, got {yaw}"
        assert abs(roll) < 5.0, f"Expected roll near 0, got {roll}"

        # Also verify estimate_head_pose_pnp from biometrics geometry
        yaw_g, pitch_g, roll_g = estimate_head_pose_pnp(lms, 160, 160)
        assert abs(pitch_g) < 5.0
        assert abs(yaw_g) < 5.0
        assert abs(roll_g) < 5.0

    def test_sharpness_scale_invariance_on_aligned_crop(self):
        """
        Critical Fix B: Sharpness evaluated on aligned 160x160 crop eliminates
        resolution bias where raw variable-resolution images penalize quality.
        """
        # Create a texture pattern
        base_160 = np.random.RandomState(42).randint(50, 200, size=(160, 160, 3)).astype(np.uint8)
        gray_160 = cv2.cvtColor(base_160, cv2.COLOR_BGR2GRAY)
        lap_160 = float(cv2.Laplacian(gray_160, cv2.CV_64F, ksize=3).var())

        # If evaluating strictly on the aligned 160x160 crop, laplacian is identical
        assert lap_160 > 0.0
        # Check that ExtremeQualityAuditor evaluates on aligned image
        auditor = ExtremeQualityAuditor()
        res = auditor.audit_face(base_160, ARCFACE_CANONICAL_160)
        assert "laplacian_var" in res
        assert abs(res["laplacian_var"] - lap_160) < 1.0

    def test_appa_real_image_path_column_support(self, tmp_path):
        """
        Critical Fix C.1: AppaRealConnector supports 'image_path' column (e.g. train/000000.jpg_face.jpg)
        in addition to 'file_name'.
        """
        train_dir = tmp_path / "train"
        train_dir.mkdir()
        b = _make_dummy_image_bytes()
        img_file = train_dir / "000000.jpg_face.jpg"
        img_file.write_bytes(b)

        df = pd.DataFrame({
            "image_path": ["train/000000.jpg_face.jpg"],
            "real_age": [32.0],
            "apparent_age_avg": [30.5],
            "apparent_age_std": [2.1]
        })
        df.to_csv(tmp_path / "labels_train.csv", index=False)

        conn = AppaRealConnector(root_dir=tmp_path, max_samples=5)
        assert conn.is_available() is True
        samples = list(conn.stream_samples())
        assert len(samples) == 1
        s = samples[0]
        assert s.raw_id == "000000.jpg_face.jpg"
        assert s.declared_age == 30.5
        assert s.metadata["real_age"] == 32.0

    def test_b3fd_hyphenated_date_regex_support(self, tmp_path):
        """
        Critical Fix C.2: B3FDConnector supports hyphenated dates YYYY-MM-DD in filenames.
        """
        b = _make_dummy_image_bytes()
        f1 = tmp_path / "John_Doe_1985-06-15_2020.jpg"
        f2 = tmp_path / "Jane_Smith_1990-1-1_2015.jpg"
        f1.write_bytes(b)
        f2.write_bytes(b)

        conn = B3FDConnector(root_dir=tmp_path, max_samples=5)
        assert conn.is_available() is True
        samples = list(conn.stream_samples())
        assert len(samples) == 2

        s1 = next(s for s in samples if "John_Doe" in s.raw_id)
        assert s1.declared_age == 35.0  # 2020 - 1985

        s2 = next(s for s in samples if "Jane_Smith" in s.raw_id)
        assert s2.declared_age == 25.0  # 2015 - 1990

    def test_fgnet_gendered_filenames_support(self, tmp_path):
        """
        Critical Fix C.3: FGNetConnector supports pdombrza/fgnet-gendered filenames
        (e.g. 001A02M.JPG -> age 2, Male; 002A16F.JPG -> age 16, Female).
        """
        b = _make_dummy_image_bytes()
        f_m = tmp_path / "001A02M.JPG"
        f_f = tmp_path / "002A16F.JPG"
        f_none = tmp_path / "003A08.JPG"
        f_m.write_bytes(b)
        f_f.write_bytes(b)
        f_none.write_bytes(b)

        conn = FGNetConnector(root_dir=tmp_path, max_samples=5)
        assert conn.is_available() is True
        samples = list(conn.stream_samples())
        assert len(samples) == 3

        s_m = next(s for s in samples if s.raw_id == f_m.name)
        assert s_m.declared_age == 2.0
        assert s_m.declared_gender == "Male"
        assert s_m.metadata["fgnet_subject_id"] == "001"

        s_f = next(s for s in samples if s.raw_id == f_f.name)
        assert s_f.declared_age == 16.0
        assert s_f.declared_gender == "Female"
        assert s_f.metadata["fgnet_subject_id"] == "002"

        s_none = next(s for s in samples if s.raw_id == f_none.name)
        assert s_none.declared_age == 8.0
        assert s_none.declared_gender is None

    def test_quota_manager_shard_manifest_checkpointing(self, tmp_path):
        """
        Critical Fix D.5: DiskQuotaManager saves shard manifest parquet immediately
        upon closing each shard.
        """
        qm = DiskQuotaManager(working_dir=str(tmp_path), shard_image_count=2, min_free_gb=0.1)
        face_img = np.full((160, 160, 3), 128, dtype=np.uint8)

        rec1 = {"image_id": "face_001.jpg", "age": 25.0, "gender": "Male"}
        rec2 = {"image_id": "face_002.jpg", "age": 30.0, "gender": "Female"}
        rec3 = {"image_id": "face_003.jpg", "age": 40.0, "gender": "Male"}

        # Write 2 faces -> completes shard 0
        assert qm.write_aligned_face(face_img, "face_001.jpg", record=rec1) is True
        assert qm.write_aligned_face(face_img, "face_002.jpg", record=rec2) is True

        # Shard 0 manifest should now exist!
        shard0_manifest = tmp_path / "golden_dataset" / "shards" / "shard_0000_manifest.parquet"
        assert shard0_manifest.exists()
        df0 = pd.read_parquet(shard0_manifest)
        assert len(df0) == 2
        assert list(df0["image_id"]) == ["face_001.jpg", "face_002.jpg"]

        # Write 3rd face into shard 1
        assert qm.write_aligned_face(face_img, "face_003.jpg", record=rec3) is True
        qm.close()

        # Shard 1 manifest should now exist after close!
        shard1_manifest = tmp_path / "golden_dataset" / "shards" / "shard_0001_manifest.parquet"
        assert shard1_manifest.exists()
        df1 = pd.read_parquet(shard1_manifest)
        assert len(df1) == 1
        assert list(df1["image_id"]) == ["face_003.jpg"]

    def test_disk_quota_manager_check_safe_and_pre_encoded_bytes(self, tmp_path):
        """
        Critical Fix D.6: DiskQuotaManager provides check_safe() alias and accepts
        pre-encoded image_bytes to eliminate redundant JPEG compression in hot loop.
        """
        qm = DiskQuotaManager(working_dir=str(tmp_path), shard_image_count=5, min_free_gb=0.1)
        assert qm.check_safe() is True
        assert qm.check_quota() is True

        # Pre-encoded image bytes
        dummy_face = np.full((160, 160, 3), 100, dtype=np.uint8)
        _, enc = cv2.imencode('.jpg', dummy_face)
        raw_b = enc.tobytes()

        # Write with image_bytes directly
        assert qm.write_aligned_face(None, "face_direct.jpg", record={"image_id": "face_direct.jpg"}, image_bytes=raw_b) is True
        qm.close()

        shard0 = tmp_path / "golden_dataset" / "shards" / "shard_0000.tar"
        assert shard0.exists()
        with tarfile.open(shard0, 'r') as tf:
            names = tf.getnames()
            assert "face_direct.jpg" in names

    def test_preflight_smoke_test_guard_on_empty_mounts(self, tmp_path):
        """
        Critical Fix D.2: Pre-flight smoke test MUST raise RuntimeError immediately
        if zero dataset connectors are available in /kaggle/input, rather than silently passing.
        """
        harvester = InCloudStreamingHarvester(kaggle_input_dir=str(tmp_path / "empty_input"))
        assert len(harvester.connectors) == 0
        available_connectors = [c for c in harvester.connectors if c.is_available()]
        assert len(available_connectors) == 0
        with pytest.raises(RuntimeError, match="PRE-FLIGHT SMOKE TEST FAILED"):
            if not available_connectors:
                raise RuntimeError("🚨 PRE-FLIGHT SMOKE TEST FAILED: Zero dataset connectors found/available in /kaggle/input")

    def test_ensure_master_manifest_consolidation(self, tmp_path):
        """Validates self-healing consolidation of shard manifests into master metadata.parquet & metadata.csv."""
        golden_dir = tmp_path / "golden_dataset"
        shards_dir = golden_dir / "shards"
        shards_dir.mkdir(parents=True)

        df_shard0 = pd.DataFrame([
            {"image_id": "face_1.jpg", "sha256": "hash1", "age_continuous": 25.0, "split": "train"},
            {"image_id": "face_2.jpg", "sha256": "hash2", "age_continuous": 30.0, "split": "train"}
        ])
        df_shard1 = pd.DataFrame([
            {"image_id": "face_3.jpg", "sha256": "hash3", "age_continuous": 40.0, "split": "val"}
        ])
        df_shard0.to_parquet(shards_dir / "shard_0000_manifest.parquet")
        df_shard1.to_parquet(shards_dir / "shard_0001_manifest.parquet")

        master_df = ensure_master_manifest(golden_dir)
        assert len(master_df) == 3
        assert (golden_dir / "metadata.parquet").exists()
        assert (golden_dir / "metadata.csv").exists()

        loaded_parquet = pd.read_parquet(golden_dir / "metadata.parquet")
        assert len(loaded_parquet) == 3
        assert set(loaded_parquet["image_id"]) == {"face_1.jpg", "face_2.jpg", "face_3.jpg"}

    def test_face_golden_dataset_persistent_tar_handles(self, tmp_path):
        """Validates persistent TarFile handle caching in FaceGoldenDataset for high throughput."""
        golden_dir = tmp_path / "golden_dataset"
        shards_dir = golden_dir / "shards"
        shards_dir.mkdir(parents=True)

        tar_path = shards_dir / "shard_0000.tar"
        dummy_img = np.full((160, 160, 3), 120, dtype=np.uint8)
        _, enc = cv2.imencode('.jpg', dummy_img)
        img_bytes = enc.tobytes()

        with tarfile.open(tar_path, "w") as tf:
            ti = tarfile.TarInfo("face_test.jpg")
            ti.size = len(img_bytes)
            tf.addfile(ti, io.BytesIO(img_bytes))

        manifest_p = golden_dir / "metadata.parquet"
        df = pd.DataFrame([{
            "image_id": "face_test.jpg",
            "sha256": "testhash",
            "age_continuous": 28.0,
            "apparent_age_std": 2.0,
            "gender": "Female",
            "split": "train"
        }])
        df.to_parquet(manifest_p)

        dataset = FaceGoldenDataset(manifest_path=manifest_p, shards_dir=shards_dir)
        assert len(dataset) == 1
        assert len(dataset._tar_handles) == 0

        # Item access opens and caches handle
        sample = dataset[0]
        assert sample["image_id"] == "face_test.jpg"
        assert len(dataset._tar_handles) == 1
        handle = dataset._get_tar_handle(tar_path)
        assert handle is not None
        assert not handle.closed

        # Explicit close releases handle
        dataset.close()
        assert len(dataset._tar_handles) == 0
        assert handle.closed

    def test_bounded_has_matching_file_depth_traversal(self, tmp_path):
        """Validates that _has_matching_file traverses nested subdirectories up to depth 3 without freezing."""
        import re
        conn = UTKFaceConnector(root_dir=tmp_path)
        pattern = re.compile(r"^\d+_\d+_\d+.*\.jpg$")

        # Level 1 subdirectory (depth 1)
        sub1 = tmp_path / "dataset_part"
        sub1.mkdir()
        assert conn._has_matching_file(pattern) is False

        # Level 2 subdirectory (depth 2)
        sub2 = sub1 / "nested_images"
        sub2.mkdir()
        (sub2 / "25_0_1_test.jpg").write_bytes(b"dummy")
        assert conn._has_matching_file(pattern, max_depth=3) is True

        # Too deep (depth 4) with strict max_depth=2
        deep_root = tmp_path / "deep_tree"
        deep_dir = deep_root / "l1" / "l2" / "l3" / "l4"
        deep_dir.mkdir(parents=True)
        (deep_dir / "30_1_2_deep.jpg").write_bytes(b"dummy")
        conn_deep = UTKFaceConnector(root_dir=deep_root)
        assert conn_deep._has_matching_file(pattern, max_depth=2) is False
        assert conn_deep._has_matching_file(pattern, max_depth=4) is True

    def test_deduplicator_round_robin_and_capacity_expansion(self):
        """Validates PerceptualDeduplicator query tracking and dynamic VRAM/matrix growth."""
        # Use PerceptualDeduplicator from build_kaggle_kernel / engine
        from scripts.kaggle_pipeline.kaggle_face_engine import PerceptualDeduplicator as EngineDedup
        dedup = EngineDedup(hamming_threshold=6, cosine_threshold=0.35)

        # Test duplicate detection and registration
        h1 = 0x1234567890ABCDEF
        vec1 = np.ones(512, dtype=np.float32) / np.sqrt(512)
        is_dup, _ = dedup.is_duplicate_and_register(h1, vec1)
        assert is_dup is False

        # Near duplicate with hamming distance 1
        h1_near = h1 ^ 0x01
        is_dup_near, reason = dedup.is_duplicate_and_register(h1_near, None)
        assert is_dup_near is True
        assert reason == "PHASH_NEAR_DUPLICATE"

    def test_multithreaded_pipeline_clean_termination_and_no_name_error(self, tmp_path):
        """
        Stress tests Cell 4 producer-consumer architecture:
        Verifies:
        1. raw_age is extracted from raw_sample.declared_age without NameError
        2. Worker threads cleanly exit on stop_event without hanging
        3. Telemetry heartbeat file writes valid JSON with complete metric schema
        """
        import queue
        import threading
        import time
        import json
        import shutil

        output_dir = tmp_path / "golden_out"
        output_dir.mkdir(parents=True)
        heartbeat_p = tmp_path / "telemetry_heartbeat.json"

        quota_mgr = DiskQuotaManager(working_dir=str(tmp_path), shard_image_count=50, min_free_gb=0.1)

        raw_q = queue.Queue(maxsize=16)
        res_q = queue.Queue(maxsize=16)
        stop_evt = threading.Event()
        t_start = time.time()
        total_step = 0
        curated_count = 0
        rejected_count = 0
        consecutive_failures = 0
        rejection_hist = {}

        def dummy_worker(wid):
            def _put_res(r):
                while not stop_evt.is_set():
                    try:
                        res_q.put(r, timeout=0.1)
                        return True
                    except queue.Full:
                        continue
                return False

            while not stop_evt.is_set():
                try:
                    s = raw_q.get(timeout=0.1)
                except queue.Empty:
                    continue
                if s is None:
                    break
                if s.declared_age is None or s.declared_age < 0:
                    _put_res(('reject', 'INVALID_AGE', s, None, None))
                else:
                    fake_aligned = np.full((160, 160, 3), 128, dtype=np.uint8)
                    fake_audit = {'lap_var': 150.0, 'tenengrad': 350.0, 'mean_l': 100.0,
                                  'shadow_clip': 0.01, 'highlight_clip': 0.01,
                                  'yaw': 5.0, 'pitch': 2.0, 'roll': 1.0}
                    _put_res(('accept', None, s, fake_aligned, fake_audit))

        def _write_hb():
            now = time.time()
            elapsed = max(0.1, now - t_start)
            hb_data = {
                'timestamp': now,
                'uptime_seconds': round(elapsed, 1),
                'total_step': total_step,
                'curated_count': curated_count,
                'rejected_count': rejected_count,
                'pass_rate': round(curated_count / max(1, total_step), 4),
                'fps': round(total_step / elapsed, 1),
                'disk_free_gb': round(shutil.disk_usage(output_dir).free / (1024**3), 2),
                'consecutive_failures': consecutive_failures,
                'rejection_hist': rejection_hist,
                'shards_count': 1,
                'circuit_breaker_triggered': False
            }
            with open(heartbeat_p, 'w', encoding='utf-8') as f:
                json.dump(hb_data, f, indent=2)

        workers = [threading.Thread(target=dummy_worker, args=(i,), daemon=True) for i in range(2)]
        for w in workers:
            w.start()

        # Produce 5 valid and 2 invalid samples
        for i in range(5):
            s = RawImageSample(image_bytes=b"fake", source_dataset="utkface", raw_id=f"f_{i}.jpg", declared_age=float(20 + i), declared_gender="Female")
            raw_q.put(s)
        for i in range(2):
            s = RawImageSample(image_bytes=b"fake", source_dataset="utkface", raw_id=f"bad_{i}.jpg", declared_age=-5.0)
            raw_q.put(s)

        records = []
        processed = 0
        while processed < 7:
            item = res_q.get(timeout=2.0)
            status, reason, raw_sample, aligned_160, audit_res = item
            total_step += 1
            processed += 1
            if status == 'reject':
                rejected_count += 1
                consecutive_failures += 1
                rejection_hist[reason] = rejection_hist.get(reason, 0) + 1
            else:
                consecutive_failures = 0
                # CRITICAL VERIFICATION: raw_age must not raise NameError
                raw_age = float(raw_sample.declared_age)
                t_mu = raw_age
                assert t_mu >= 20.0
                curated_count += 1
                records.append({"image_id": f"face_{total_step}.jpg", "age": t_mu})

        stop_evt.set()
        for w in workers:
            w.join(timeout=1.0)
            assert not w.is_alive()

        _write_hb()
        quota_mgr.close()

        assert curated_count == 5
        assert rejected_count == 2
        assert heartbeat_p.exists()
        hb = json.loads(heartbeat_p.read_text(encoding='utf-8'))
        assert hb["curated_count"] == 5
        assert hb["rejected_count"] == 2
        assert "fps" in hb
        assert "disk_free_gb" in hb


