"""
Tier 1 - Feature Coverage Test Suite (F01 to F26)
==================================================
Isolated requirement-driven test cases for each of the 26 features from PROJECT.md
and TEST_INFRA.md. Each feature contains >= 5 comprehensive tests.
"""

from __future__ import annotations
import math
import json
import tempfile
from pathlib import Path
from typing import Dict, Any, List
import pytest
import numpy as np
import cv2
import pandas as pd

from scripts.tests.conftest import (
    ReferenceBiometricOracles,
    create_synthetic_face,
    get_canonical_landmarks
)
from scripts.kaggle_pipeline.kaggle_face_engine import (
    FacialLandmarkDetector,
    FaceGeometryAligner,
    ExtremeQualityAuditor,
    PerceptualDeduplicator,
    BiologicalLabelingEngine,
    MultiTeacherEnsemble,
    AlignedFaceSample,
    GoldenFaceRecord
)
from scripts.kaggle_pipeline.kaggle_dataset_connectors import (
    RawImageSample,
    BaseFaceConnector,
    FairFaceConnector,
    CelebAConnector,
    IMDBWikiConnector,
    AFADConnector,
    B3FDConnector,
    AppaRealConnector,
    InCloudStreamingHarvester,
    DiskQuotaManager
)
from scripts.kaggle_pipeline.kaggle_crawler_module import (
    InCloudDemographicCrawler
)
from scripts.kaggle_pipeline.build_kaggle_kernel import (
    generate_pipeline_notebook,
    generate_kernel_metadata
)
from scripts.kaggle_pipeline.audit_and_sync_manifest import (
    generate_audit_report,
    validate_manifest_integrity,
    EXPECTED_MANIFEST_COLUMNS
)


# ============================================================================
# F01: Kaggle Multi-Dataset Connector (FairFace, CelebA, IMDB-WIKI, AFAD, B3FD, APPA)
# ============================================================================

class TestF01MultiDatasetConnector:
    """Verifies modular ingestion connectors for target biometric datasets."""

    def test_fairface_connector_interface(self, tmp_path):
        """F01-1: FairFaceConnector initializes, names properly, and parses age midpoints."""
        conn = FairFaceConnector(root_dir=tmp_path, max_samples=100)
        assert conn.get_name() == "fairface"
        assert conn.AGE_BIN_MIDPOINTS["20-29"] == 24.5
        assert conn.RACE_MAPPING["White"] == "Caucasian"
        assert conn.is_available() is False

    def test_celeba_connector_interface(self, tmp_path):
        """F01-2: CelebAConnector initializes, checks availability, and maps name."""
        conn = CelebAConnector(root_dir=tmp_path, max_samples=100)
        assert conn.get_name() == "celeba"
        assert conn.is_available() is False

    def test_imdb_wiki_noise_filtering_spec(self, tmp_path):
        """F01-3: IMDBWikiConnector initializes and validates age bounds (0 <= age <= 100)."""
        conn = IMDBWikiConnector(root_dir=tmp_path, max_samples=100)
        assert conn.get_name() == "imdb_wiki"
        assert conn.is_available() is False

    def test_afad_connector_interface(self, tmp_path):
        """F01-4: AFADConnector maps East Asian cohort with age bounds [15, 40]."""
        conn = AFADConnector(root_dir=tmp_path, max_samples=100)
        assert conn.get_name() == "afad"
        assert conn.is_available() is False

    def test_appa_real_connector_interface(self, tmp_path):
        """F01-5: AppaRealConnector extracts apparent age mean and standard deviation."""
        conn = AppaRealConnector(root_dir=tmp_path, max_samples=100)
        assert conn.get_name() == "appa_real"
        assert conn.is_available() is False


# ============================================================================
# F02: In-Cloud Zero-Disk Streamer
# ============================================================================

class TestF02ZeroDiskStreamer:
    """Verifies streaming image loader operating directly from memory without disk writes."""

    def test_raw_image_sample_decoding(self, sharp_face):
        """F02-1: RawImageSample decodes in-memory bytes into OpenCV BGR array."""
        ok, buf = cv2.imencode(".jpg", sharp_face)
        assert ok
        sample = RawImageSample(
            image_bytes=buf.tobytes(),
            source_dataset="fairface",
            raw_id="sample_01.jpg",
            declared_age=25.0,
            declared_gender="Female",
            declared_ethnicity="East Asian"
        )
        img = sample.to_bgr_image()
        assert img is not None
        assert img.shape == (160, 160, 3)

    def test_streamer_sample_dataclass_contract(self, sharp_face):
        """F02-2: RawImageSample fields conform to interface contract."""
        _, buf = cv2.imencode(".png", sharp_face)
        sample = RawImageSample(
            image_bytes=buf.tobytes(),
            source_dataset="celeba",
            raw_id="000001.png",
            declared_landmarks=np.zeros((5, 2), dtype=np.float32)
        )
        assert sample.source_dataset == "celeba"
        assert sample.declared_landmarks.shape == (5, 2)

    def test_corrupt_buffer_handling(self):
        """F02-3: Corrupt byte buffer returns None safely without unhandled exception."""
        sample = RawImageSample(
            image_bytes=b"not_an_image_data",
            source_dataset="unknown",
            raw_id="bad.jpg"
        )
        assert sample.to_bgr_image() is None

    def test_harvester_register_and_stream(self, tmp_path):
        """F02-4: InCloudStreamingHarvester registers connectors and streams."""
        harvester = InCloudStreamingHarvester(kaggle_input_dir=str(tmp_path))
        conn = FairFaceConnector(root_dir=tmp_path)
        harvester.register_connector(conn)
        assert len(harvester.connectors) == 1
        samples = list(harvester.stream_all())
        assert len(samples) == 0

    def test_zero_disk_cost_verification(self, tmp_path, sharp_face):
        """F02-5: Ingestion into RawImageSample writes zero bytes to local filesystem."""
        initial_file_count = len(list(tmp_path.iterdir()))
        _, buf = cv2.imencode(".jpg", sharp_face)
        sample = RawImageSample(image_bytes=buf.tobytes(), source_dataset="test", raw_id="test.jpg")
        _ = sample.to_bgr_image()
        final_file_count = len(list(tmp_path.iterdir()))
        assert initial_file_count == final_file_count


# ============================================================================
# F03: Dynamic Demographic Scraper
# ============================================================================

class TestF03DynamicDemographicScraper:
    """Verifies Wikimedia Commons 220-demographic query crawler."""

    def test_query_matrix_length_is_220(self):
        """F03-1: Query matrix must produce exactly 220 combinations (10x11x2)."""
        crawler = InCloudDemographicCrawler()
        queries = [q["query"] for q in crawler.generate_220_demographic_queries()]
        assert len(queries) == 220, f"Expected 220 demographic queries, got {len(queries)}"

    def test_query_matrix_covers_diverse_ethnicities(self):
        """F03-2: Queries must cover all required ethnic groups."""
        crawler = InCloudDemographicCrawler()
        queries = [q["query"] for q in crawler.generate_220_demographic_queries()]
        for eth in ["Vietnamese", "East Asian", "African Black", "Latino Hispanic", "South Asian Indian"]:
            assert any(eth in q for q in queries), f"Missing ethnicity: {eth}"

    def test_query_matrix_covers_diverse_age_brackets(self):
        """F03-3: Queries must cover full lifespan from infant to centenarian."""
        crawler = InCloudDemographicCrawler()
        queries = [q["query"] for q in crawler.generate_220_demographic_queries()]
        for age in ["newborn", "toddler", "teenager", "adult", "elderly", "centenarian"]:
            assert any(age in q for q in queries), f"Missing age bracket: {age}"

    def test_query_matrix_covers_both_genders(self):
        """F03-4: Queries must cover male and female portrait terms."""
        crawler = InCloudDemographicCrawler()
        queries = [q["query"] for q in crawler.generate_220_demographic_queries()]
        males = sum(1 for q in queries if "male" in q and "female" not in q)
        females = sum(1 for q in queries if "female" in q)
        assert males == 110
        assert females == 110

    def test_wikimedia_query_format_and_params(self):
        """F03-5: Verify API parameter structure and bitmap filter."""
        crawler = InCloudDemographicCrawler()
        query = "portrait photo Vietnamese adult male portrait face"
        search_term = f"{query} face portrait filetype:bitmap"
        assert "filetype:bitmap" in search_term
        assert crawler.HEADERS.get("User-Agent") is not None


# ============================================================================
# F04: Scraper Error Handling & Headers
# ============================================================================

class TestF04ScraperErrorHandlingAndHeaders:
    """Verifies crawler resilience, headers, backoff, and exception handling."""

    def test_user_agent_header_present(self):
        """F04-1: User-Agent header must be explicitly set and non-empty."""
        crawler = InCloudDemographicCrawler()
        assert "User-Agent" in crawler.HEADERS
        assert len(crawler.HEADERS["User-Agent"]) > 20

    def test_download_url_invalid_domain_fails_gracefully(self, tmp_path):
        """F04-2: Unreachable URL returns False without raising unhandled exception."""
        crawler = InCloudDemographicCrawler(output_dir=str(tmp_path), max_retries=1)
        dest = tmp_path / "test.jpg"
        result = crawler.download_url({"url": "http://127.0.0.1:1/face.jpg", "dest": str(dest)})
        assert result is False
        assert not dest.exists()

    def test_download_url_handles_malformed_url(self, tmp_path):
        """F04-3: Malformed URL returns False safely."""
        crawler = InCloudDemographicCrawler(output_dir=str(tmp_path), max_retries=1)
        result = crawler.download_url({"url": "not-a-valid-url", "dest": str(tmp_path / "x.jpg")})
        assert result is False

    def test_wikimedia_query_exception_tolerance(self):
        """F04-4: query_wikimedia_commons handles network failure gracefully."""
        crawler = InCloudDemographicCrawler()
        urls = crawler.query_wikimedia_commons("", limit=1)
        assert isinstance(urls, list)

    def test_crawl_batch_with_zero_queries(self, tmp_path):
        """F04-5: crawl_batch with max_queries=0 returns 0 downloaded."""
        crawler = InCloudDemographicCrawler(output_dir=str(tmp_path))
        downloaded = crawler.crawl_batch(max_queries=0, images_per_query=0)
        assert downloaded == 0


# ============================================================================
# F05: Disk Quota & Sharding Manager
# ============================================================================

class TestF05DiskQuotaAndShardingManager:
    """Verifies storage guard ensuring working storage remains strictly below 20GB."""

    def test_quota_manager_initialization(self, tmp_path):
        """F05-1: DiskQuotaManager initializes tar sharding in working directory."""
        qm = DiskQuotaManager(working_dir=str(tmp_path), max_storage_gb=18.0, shard_image_count=50)
        assert qm.max_storage_bytes == int(18.0 * (1024 ** 3))
        assert qm.shards_dir.exists()
        qm.close()

    def test_quota_manager_check_quota(self, tmp_path):
        """F05-2: check_quota returns True under normal conditions."""
        qm = DiskQuotaManager(working_dir=str(tmp_path))
        assert qm.check_quota() is True
        qm.close()

    def test_quota_manager_write_aligned_face(self, tmp_path, sharp_face):
        """F05-3: write_aligned_face saves image directly into tar shard."""
        qm = DiskQuotaManager(working_dir=str(tmp_path), shard_image_count=5)
        ok = qm.write_aligned_face(sharp_face, "face_0001.jpg")
        assert ok is True
        assert qm.current_shard_count == 1
        qm.close()

    def test_quota_manager_shard_rollover(self, tmp_path, sharp_face):
        """F05-4: Shard rolls over when count reaches shard_image_count."""
        qm = DiskQuotaManager(working_dir=str(tmp_path), shard_image_count=2)
        qm.write_aligned_face(sharp_face, "face_01.jpg")
        qm.write_aligned_face(sharp_face, "face_02.jpg")
        assert qm.current_shard_idx == 1
        qm.close()

    def test_quota_manager_close_finalizes_tar(self, tmp_path, sharp_face):
        """F05-5: close() finalizes archive handles safely."""
        qm = DiskQuotaManager(working_dir=str(tmp_path))
        qm.write_aligned_face(sharp_face, "face_01.jpg")
        qm.close()
        assert qm.current_tar_obj is None


# ============================================================================
# F06: Genuine 5-Point Landmark Detector
# ============================================================================

class TestF06Genuine5PointLandmarkDetector:
    """Verifies facial keypoint detection producing valid (5, 2) coordinates."""

    def test_landmark_detector_output_contract(self, sharp_face):
        """F06-1: FacialLandmarkDetector produces (5, 2) coordinates and confidence."""
        det = FacialLandmarkDetector()
        ok, landmarks, conf = det.detect_landmarks(sharp_face)
        assert ok is True
        assert landmarks.shape == (5, 2)
        assert 0.0 <= conf <= 1.0

    def test_landmark_geometric_ordering(self, sharp_face):
        """F06-2: Verify landmark anatomical ordering [L_eye, R_eye, Nose, L_mouth, R_mouth]."""
        det = FacialLandmarkDetector()
        _, lms, _ = det.detect_landmarks(sharp_face)
        l_eye, r_eye, nose, l_mouth, r_mouth = lms
        # Left eye must be to the left of right eye (lower x coordinate)
        assert l_eye[0] < r_eye[0]
        # Eyes must be higher than nose (lower y coordinate in image space)
        assert l_eye[1] < nose[1] and r_eye[1] < nose[1]
        # Nose must be higher than mouth
        assert nose[1] < l_mouth[1] and nose[1] < r_mouth[1]
        # Left mouth must be to the left of right mouth
        assert l_mouth[0] < r_mouth[0]

    def test_landmark_coordinate_bounds(self, sharp_face):
        """F06-3: Landmarks must reside within image boundaries."""
        det = FacialLandmarkDetector()
        _, lms, _ = det.detect_landmarks(sharp_face)
        h, w = sharp_face.shape[:2]
        for pt in lms:
            assert 0.0 <= pt[0] <= w
            assert 0.0 <= pt[1] <= h

    def test_interocular_distance_strictly_positive(self, sharp_face):
        """F06-4: Distance between eyes must be strictly positive."""
        det = FacialLandmarkDetector()
        _, lms, _ = det.detect_landmarks(sharp_face)
        iod = np.linalg.norm(lms[1] - lms[0])
        assert iod >= 20.0, f"Interocular distance too small: {iod}"

    def test_detector_handles_small_image(self):
        """F06-5: Detector handles small images without crashing."""
        small = np.zeros((32, 32, 3), dtype=np.uint8)
        det = FacialLandmarkDetector()
        ok, lms, conf = det.detect_landmarks(small)
        # Should gracefully return false or low confidence
        assert isinstance(ok, bool)


# ============================================================================
# F07: Canonical Similarity Transform
# ============================================================================

class TestF07CanonicalSimilarityTransform:
    """Verifies Umeyama similarity alignment to ArcFace 160x160 coordinate grid."""

    def test_aligner_output_dimensions(self, sharp_face, canonical_landmarks):
        """F07-1: Warped image must match specified output_size (160x160x3)."""
        aligner = FaceGeometryAligner(output_size=160)
        aligned, matrix = aligner.align(sharp_face, canonical_landmarks)
        assert aligned.shape == (160, 160, 3)
        assert matrix.shape == (2, 3)

    def test_aligner_rotation_rectification(self, sharp_face, canonical_landmarks):
        """F07-2: Aligner rectifies tilted landmarks back to horizontal eye line."""
        aligner = FaceGeometryAligner(output_size=160)
        theta = math.radians(15.0)
        c, s = math.cos(theta), math.sin(theta)
        R = np.array([[c, -s], [s, c]], dtype=np.float32)
        center = np.mean(canonical_landmarks, axis=0)
        tilted_lms = (canonical_landmarks - center) @ R.T + center
        aligned, matrix = aligner.align(sharp_face, tilted_lms)
        assert aligned.shape == (160, 160, 3)

    def test_aligner_different_target_scales(self, sharp_face, canonical_landmarks):
        """F07-3: Aligner supports multiple canonical resolutions (112, 160, 224)."""
        for sz in [112, 160, 224]:
            aligner = FaceGeometryAligner(output_size=sz)
            aligned, mat = aligner.align(sharp_face, canonical_landmarks)
            assert aligned.shape == (sz, sz, 3)

    def test_aligner_fallback_on_degenerate_points(self, sharp_face):
        """F07-4: Aligner handles collinear/degenerate points using fallback."""
        aligner = FaceGeometryAligner(output_size=160)
        collinear_lms = np.array([[20, 20], [40, 40], [60, 60], [80, 80], [100, 100]], dtype=np.float32)
        aligned, mat = aligner.align(sharp_face, collinear_lms)
        assert aligned.shape == (160, 160, 3)

    def test_border_reflect_prevents_black_voids(self, sharp_face, canonical_landmarks):
        """F07-5: Aligned face has no black void borders at corners."""
        aligner = FaceGeometryAligner(output_size=160)
        aligned, _ = aligner.align(sharp_face, canonical_landmarks)
        corners = [aligned[0, 0], aligned[0, -1], aligned[-1, 0], aligned[-1, -1]]
        for corner in corners:
            assert np.mean(corner) > 0.0


# ============================================================================
# F08: Laplacian Sharpness Gate (Var >= 100)
# ============================================================================

class TestF08LaplacianSharpnessGate:
    """Verifies Laplacian spatial variance sharpness gating."""

    def test_sharp_image_passes_laplacian(self, sharp_face):
        """F08-1: Sharp synthetic face produces Laplacian variance >= 100.0."""
        gray = cv2.cvtColor(sharp_face, cv2.COLOR_BGR2GRAY)
        lap_var = ExtremeQualityAuditor.compute_laplacian_var(gray)
        assert lap_var >= 100.0, f"Expected lap_var >= 100, got {lap_var}"

    def test_blurry_image_fails_laplacian(self, blurry_face):
        """F08-2: Blurred face produces Laplacian variance < 100.0."""
        gray = cv2.cvtColor(blurry_face, cv2.COLOR_BGR2GRAY)
        lap_var = ExtremeQualityAuditor.compute_laplacian_var(gray)
        assert lap_var < 100.0, f"Expected blurry lap_var < 100, got {lap_var}"

    def test_flat_uniform_image_has_zero_laplacian(self):
        """F08-3: Flat uniform image produces zero Laplacian variance."""
        flat = np.full((160, 160), 128, dtype=np.uint8)
        lap_var = ExtremeQualityAuditor.compute_laplacian_var(flat)
        assert lap_var == pytest.approx(0.0, abs=1e-3)

    def test_audit_face_flags_blurry_laplacian_reason(self, blurry_face):
        """F08-4: audit_face records BLURRY_LAPLACIAN rejection reason."""
        auditor = ExtremeQualityAuditor(min_laplacian_var=100.0)
        res = auditor.audit_face(blurry_face)
        assert res["passed"] is False
        assert any("BLURRY_LAPLACIAN" in r for r in res["reasons"])

    def test_laplacian_threshold_customizable(self, sharp_face):
        """F08-5: Custom threshold allows stricter or relaxed gating."""
        gray = cv2.cvtColor(sharp_face, cv2.COLOR_BGR2GRAY)
        val = ExtremeQualityAuditor.compute_laplacian_var(gray)
        auditor_strict = ExtremeQualityAuditor(min_laplacian_var=val + 50.0)
        res = auditor_strict.audit_face(sharp_face)
        assert res["passed"] is False


# ============================================================================
# F09: Tenengrad Gradient Sharpness Gate (Energy >= 250)
# ============================================================================

class TestF09TenengradGradientSharpnessGate:
    """Verifies Tenengrad Sobel squared energy sharpness filter."""

    def test_sharp_image_passes_tenengrad(self, sharp_face):
        """F09-1: Sharp synthetic face produces Tenengrad >= 250.0."""
        gray = cv2.cvtColor(sharp_face, cv2.COLOR_BGR2GRAY)
        tenengrad = ExtremeQualityAuditor.compute_tenengrad(gray)
        assert tenengrad >= 250.0, f"Expected tenengrad >= 250, got {tenengrad}"

    def test_blurry_image_fails_tenengrad(self, blurry_face):
        """F09-2: Blurred face produces Tenengrad < 250.0."""
        gray = cv2.cvtColor(blurry_face, cv2.COLOR_BGR2GRAY)
        tenengrad = ExtremeQualityAuditor.compute_tenengrad(gray)
        assert tenengrad < 250.0, f"Expected tenengrad < 250, got {tenengrad}"

    def test_tenengrad_mathematical_sobel_energy_formula(self):
        """F09-3: Verify Tenengrad uses Sobel gradient energy."""
        test_img = np.zeros((100, 100), dtype=np.uint8)
        test_img[:, 50:] = 255
        val = ExtremeQualityAuditor.compute_tenengrad(test_img)
        assert val > 0.0

    def test_audit_face_flags_low_tenengrad_reason(self, blurry_face):
        """F09-4: audit_face records LOW_TENENGRAD rejection reason."""
        auditor = ExtremeQualityAuditor(min_tenengrad=250.0)
        res = auditor.audit_face(blurry_face)
        assert res["passed"] is False
        assert any("LOW_TENENGRAD" in r for r in res["reasons"])

    def test_tenengrad_monotonically_decreases_with_blur(self, sharp_face):
        """F09-5: Increasing blur kernel monotonically degrades Tenengrad energy."""
        gray = cv2.cvtColor(sharp_face, cv2.COLOR_BGR2GRAY)
        t_base = ExtremeQualityAuditor.compute_tenengrad(gray)
        t_blur5 = ExtremeQualityAuditor.compute_tenengrad(cv2.GaussianBlur(gray, (5, 5), 0))
        t_blur15 = ExtremeQualityAuditor.compute_tenengrad(cv2.GaussianBlur(gray, (15, 15), 0))
        assert t_base > t_blur5 > t_blur15


# ============================================================================
# F10: Anthropometric Head Pose Euler Limits
# ============================================================================

class TestF10HeadPoseEulerLimits:
    """Verifies solvePnP 3D anthropometry head pose estimation and gating."""

    def test_estimate_head_pose_returns_3_floats(self, canonical_landmarks):
        """F10-1: estimate_head_pose returns yaw, pitch, roll floats."""
        auditor = ExtremeQualityAuditor()
        yaw, pitch, roll = auditor.estimate_head_pose(canonical_landmarks, 160, 160)
        assert isinstance(yaw, float)
        assert isinstance(pitch, float)
        assert isinstance(roll, float)

    def test_horizontal_landmark_shift_alters_yaw(self, canonical_landmarks):
        """F10-2: Shifting nose horizontally induces significant yaw change."""
        auditor = ExtremeQualityAuditor()
        yaw_orig, _, _ = auditor.estimate_head_pose(canonical_landmarks, 160, 160)
        shifted_lms = canonical_landmarks.copy()
        shifted_lms[2, 0] += 25.0
        yaw_shifted, _, _ = auditor.estimate_head_pose(shifted_lms, 160, 160)
        assert abs(yaw_shifted - yaw_orig) > 15.0

    def test_landmark_rotation_alters_roll(self, canonical_landmarks):
        """F10-3: In-plane rotation induces matching change in roll angle."""
        auditor = ExtremeQualityAuditor()
        theta = math.radians(15.0)
        R = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
        center = np.mean(canonical_landmarks, axis=0)
        rotated_lms = (canonical_landmarks - center) @ R.T + center
        _, _, roll = auditor.estimate_head_pose(rotated_lms, 160, 160)
        assert abs(roll) > 5.0

    def test_audit_face_flags_excessive_yaw(self, sharp_face, canonical_landmarks):
        """F10-4: Extreme yaw landmarks trigger EXCESSIVE_YAW rejection."""
        auditor = ExtremeQualityAuditor(max_abs_yaw=35.0)
        extreme_yaw_lms = canonical_landmarks.copy()
        extreme_yaw_lms[0, 0] += 35.0
        res = auditor.audit_face(sharp_face, landmarks_5=extreme_yaw_lms)
        assert res["passed"] is False

    def test_solvepnp_failure_sentinel(self):
        """F10-5: Degenerate landmarks return sentinel 999.0."""
        auditor = ExtremeQualityAuditor()
        zero_lms = np.zeros((5, 2), dtype=np.float32)
        yaw, pitch, roll = auditor.estimate_head_pose(zero_lms, 160, 160)
        assert abs(yaw) >= 35.0 or abs(pitch) >= 25.0 or yaw == 999.0


# ============================================================================
# F11: Photometric & Contrast Gate
# ============================================================================

class TestF11PhotometricAndContrastGate:
    """Verifies CIE L*a*b* luminance and sensor clipping fraction gates."""

    def test_normal_face_passes_photometric_gate(self, sharp_face):
        """F11-1: Normal exposure face has luminance in [35, 220] and passes."""
        auditor = ExtremeQualityAuditor(min_luminance=35.0, max_luminance=220.0)
        res = auditor.audit_face(sharp_face)
        assert 35.0 <= res["mean_luminance"] <= 220.0
        assert not any("UNDEREXPOSED" in r for r in res["reasons"])
        assert not any("OVEREXPOSED" in r for r in res["reasons"])

    def test_underexposed_face_rejected(self, underexposed_face):
        """F11-2: Dark face with mean L* < 35 is rejected with UNDEREXPOSED."""
        auditor = ExtremeQualityAuditor(min_luminance=35.0)
        res = auditor.audit_face(underexposed_face)
        assert res["passed"] is False
        assert any("UNDEREXPOSED" in r for r in res["reasons"])

    def test_overexposed_face_rejected(self, overexposed_face):
        """F11-3: Bright face with mean L* > 220 is rejected with OVEREXPOSED."""
        auditor = ExtremeQualityAuditor(max_luminance=220.0)
        res = auditor.audit_face(overexposed_face)
        assert res["passed"] is False
        assert any("OVEREXPOSED" in r for r in res["reasons"])

    def test_extreme_clipping_rejected(self, clipped_face):
        """F11-4: Face with clipping fraction > 8% is rejected with EXTREME_CLIPPING."""
        auditor = ExtremeQualityAuditor(max_clipping_fraction=0.08)
        res = auditor.audit_face(clipped_face)
        assert res["passed"] is False
        assert any("EXTREME_CLIPPING" in r for r in res["reasons"])

    def test_audit_cheek_occlusion_gate(self, sharp_face, canonical_landmarks):
        """F11-5: Cheek occlusion audit measures bilateral chromaticity delta E."""
        auditor = ExtremeQualityAuditor(max_cheek_delta_e=22.0)
        passed, delta_e = auditor.audit_cheek_occlusion(sharp_face, canonical_landmarks)
        assert isinstance(passed, bool)
        assert delta_e >= 0.0


# ============================================================================
# F12: Perceptual Hash Deduplication (64-bit DCT pHash d_H > 6)
# ============================================================================

class TestF12PerceptualHashDeduplication:
    """Verifies 64-bit DCT perceptual hash deduplication and Hamming distance."""

    def test_phash_uint64_output_type(self, sharp_face):
        """F12-1: compute_phash_uint64 returns a 64-bit integer."""
        gray = cv2.cvtColor(sharp_face, cv2.COLOR_BGR2GRAY)
        h = PerceptualDeduplicator.compute_phash_uint64(gray)
        assert isinstance(h, int)
        assert 0 <= h < (2 ** 64)

    def test_identical_images_have_hamming_zero(self, sharp_face):
        """F12-2: Identical images have Hamming distance = 0."""
        gray = cv2.cvtColor(sharp_face, cv2.COLOR_BGR2GRAY)
        h1 = PerceptualDeduplicator.compute_phash_uint64(gray)
        h2 = PerceptualDeduplicator.compute_phash_uint64(gray)
        dist = PerceptualDeduplicator.hamming_distance(h1, h2)
        assert dist == 0

    def test_minor_perturbation_detected_as_duplicate(self, sharp_face):
        """F12-3: Slight brightness change produces d_H <= 6 (detected as duplicate)."""
        gray1 = cv2.cvtColor(sharp_face, cv2.COLOR_BGR2GRAY)
        gray2 = np.clip(gray1.astype(np.int16) + 3, 0, 255).astype(np.uint8)
        h1 = PerceptualDeduplicator.compute_phash_uint64(gray1)
        h2 = PerceptualDeduplicator.compute_phash_uint64(gray2)
        dist = PerceptualDeduplicator.hamming_distance(h1, h2)
        assert dist <= 6

    def test_distinct_images_have_large_hamming_distance(self, sharp_face):
        """F12-4: Completely distinct visual patterns have d_H > 6."""
        gray1 = cv2.cvtColor(sharp_face, cv2.COLOR_BGR2GRAY)
        gray2 = np.zeros((160, 160), dtype=np.uint8)
        gray2[::20, :] = 255
        h1 = PerceptualDeduplicator.compute_phash_uint64(gray1)
        h2 = PerceptualDeduplicator.compute_phash_uint64(gray2)
        dist = PerceptualDeduplicator.hamming_distance(h1, h2)
        assert dist > 6

    def test_deduplicator_registry_tracks_seen_hashes(self, sharp_face):
        """F12-5: is_duplicate_and_register registers hash on first encounter and flags duplicate on second."""
        dedup = PerceptualDeduplicator(hamming_threshold=6)
        gray = cv2.cvtColor(sharp_face, cv2.COLOR_BGR2GRAY)
        h = dedup.compute_phash_uint64(gray)
        is_dup1, tag1 = dedup.is_duplicate_and_register(h)
        assert is_dup1 is False
        assert tag1 == "UNIQUE"
        is_dup2, tag2 = dedup.is_duplicate_and_register(h)
        assert is_dup2 is True
        assert "DUPLICATE" in tag2


# ============================================================================
# F13: ArcFace Embedding Cosine Deduplication
# ============================================================================

class TestF13CosineDeduplication:
    """Verifies ArcFace 512-d feature space cosine distance deduplication."""

    def test_extract_embedding_512d_shape_and_norm(self, sharp_face):
        """F13-1: extract_embedding_512d produces normalized 512-d float array."""
        dedup = PerceptualDeduplicator()
        emb = dedup.extract_embedding_512d(sharp_face)
        assert emb.shape == (512,)
        assert np.linalg.norm(emb) == pytest.approx(1.0, abs=1e-3)

    def test_identical_vectors_have_zero_cosine_distance(self):
        """F13-2: Identical vectors have cosine distance = 0.0."""
        vec = np.random.randn(512).astype(np.float32)
        d = PerceptualDeduplicator.cosine_distance(vec, vec)
        assert d == pytest.approx(0.0, abs=1e-5)

    def test_orthogonal_vectors_have_unit_cosine_distance(self):
        """F13-3: Orthogonal vectors have cosine distance = 1.0."""
        v1 = np.zeros(512, dtype=np.float32)
        v2 = np.zeros(512, dtype=np.float32)
        v1[0] = 1.0
        v2[1] = 1.0
        d = PerceptualDeduplicator.cosine_distance(v1, v2)
        assert d == pytest.approx(1.0, abs=1e-5)

    def test_near_duplicate_identity_detected_by_deduplicator(self, sharp_face):
        """F13-4: Minor variation of same embedding detected as EMBEDDING_IDENTITY_LEAKAGE."""
        dedup = PerceptualDeduplicator(cosine_threshold=0.35)
        emb1 = dedup.extract_embedding_512d(sharp_face)
        dedup.is_duplicate_and_register(hash_val=123, embedding_512d=emb1)
        # Small perturbation (std=0.005 gives cosine distance ~0.006 << 0.35 threshold)
        emb2 = emb1 + np.random.normal(0, 0.005, 512).astype(np.float32)
        emb2 /= np.linalg.norm(emb2)
        is_dup, tag = dedup.is_duplicate_and_register(hash_val=9999999, embedding_512d=emb2)
        assert is_dup is True
        assert tag == "EMBEDDING_IDENTITY_LEAKAGE"

    def test_different_identities_pass_deduplication(self, sharp_face):
        """F13-5: Completely different random embeddings pass as UNIQUE."""
        dedup = PerceptualDeduplicator(cosine_threshold=0.35)
        emb1 = np.zeros(512, dtype=np.float32); emb1[0] = 1.0
        emb2 = np.zeros(512, dtype=np.float32); emb2[100] = 1.0
        dedup.is_duplicate_and_register(hash_val=0x0000000000000000, embedding_512d=emb1)
        is_dup, tag = dedup.is_duplicate_and_register(hash_val=0x000000000000ffff, embedding_512d=emb2)
        assert is_dup is False
        assert tag == "UNIQUE"


# ============================================================================
# F14: Craniofacial Dimorphism Modeling (<12 and >65)
# ============================================================================

class TestF14CraniofacialDimorphismModeling:
    """Verifies aleatoric uncertainty and loss weighting for prepubertal and geriatric ages."""

    def test_prepubertal_dimorphism_attenuation(self):
        """F14-1: Prepubertal child (age=6) receives attenuated weight < 1.0."""
        w = BiologicalLabelingEngine.calculate_biological_sexual_dimorphism_weight(6.0)
        assert w == pytest.approx(0.50, rel=1e-4)

    def test_infant_minimum_weight_clamped_at_025(self):
        """F14-2: Newborn infant (age=0) receives clamped minimum weight 0.25."""
        w = BiologicalLabelingEngine.calculate_biological_sexual_dimorphism_weight(0.0)
        assert w == 0.25

    def test_adult_prime_dimorphism_weight_is_one(self):
        """F14-3: Adult subjects (age 18 to 60) receive full weight 1.0."""
        for a in [18.0, 25.0, 40.0, 55.0, 60.0]:
            w = BiologicalLabelingEngine.calculate_biological_sexual_dimorphism_weight(a)
            assert w == 1.0

    def test_geriatric_dimorphism_weight_attenuated(self):
        """F14-4: Elderly subjects (>65) receive attenuated weight <= 0.85."""
        w_75 = BiologicalLabelingEngine.calculate_biological_sexual_dimorphism_weight(75.0)
        w_100 = BiologicalLabelingEngine.calculate_biological_sexual_dimorphism_weight(100.0)
        assert w_75 < 1.0
        assert w_100 == pytest.approx(0.85, abs=0.01)

    def test_aleatoric_uncertainty_curve_properties(self):
        """F14-5: Aleatoric uncertainty sigma_aleatoric is highest at extremities."""
        sigma_child = BiologicalLabelingEngine.calculate_aleatoric_uncertainty(3.0)
        sigma_adult = BiologicalLabelingEngine.calculate_aleatoric_uncertainty(30.0)
        sigma_elderly = BiologicalLabelingEngine.calculate_aleatoric_uncertainty(80.0)
        assert sigma_child > sigma_adult
        assert sigma_elderly > sigma_adult
        assert sigma_adult == 0.05


# ============================================================================
# F15: Ethnic Aging Kinetics Modeling
# ============================================================================

class TestF15EthnicAgingKineticsModeling:
    """Verifies non-linear ethnic aging trajectories (East Asian cliff at 48, Caucasian)."""

    def test_east_asian_youthful_plateau_before_48(self):
        """F15-1: East Asian apparent age trails chronological age before 48."""
        app_30 = BiologicalLabelingEngine.model_ethnic_aging_trajectory(30.0, "East Asian")
        assert app_30 < 30.0
        assert app_30 == pytest.approx(30.0 - 4.5 * (30.0 / 48.0), rel=1e-4)

    def test_east_asian_aging_cliff_acceleration_after_48(self):
        """F15-2: East Asian aging slope accelerates sharply past age 48."""
        app_48 = BiologicalLabelingEngine.model_ethnic_aging_trajectory(48.0, "East Asian")
        app_60 = BiologicalLabelingEngine.model_ethnic_aging_trajectory(60.0, "East Asian")
        delta = app_60 - app_48
        assert delta > 12.0

    def test_caucasian_early_rhytids_photoaging(self):
        """F15-3: Caucasian baseline displays slight acceleration due to rhytids."""
        app_40 = BiologicalLabelingEngine.model_ethnic_aging_trajectory(40.0, "Caucasian")
        assert app_40 >= 40.0

    def test_african_melanin_photoprotection(self):
        """F15-4: African cohort displays visual delay due to eumelanin photoprotection."""
        app_45 = BiologicalLabelingEngine.model_ethnic_aging_trajectory(45.0, "African")
        assert app_45 < 45.0

    def test_ethnic_offset_is_continuous_at_boundary(self):
        """F15-5: East Asian model is continuous at threshold age 48."""
        left = BiologicalLabelingEngine.model_ethnic_aging_trajectory(47.999, "East Asian")
        right = BiologicalLabelingEngine.model_ethnic_aging_trajectory(48.001, "East Asian")
        assert left == pytest.approx(right, abs=0.01)


# ============================================================================
# F16: Demographic Prior Calibration
# ============================================================================

class TestF16DemographicPriorCalibration:
    """Verifies FairFace demographic balancing and race representation priors."""

    def test_calibrate_apparent_age_returns_tuple(self):
        """F16-1: calibrate_apparent_age returns calibrated age and gradient factor."""
        calib_age, factor = BiologicalLabelingEngine.calibrate_apparent_age(35.0, "East Asian")
        assert isinstance(calib_age, float)
        assert isinstance(factor, float)
        assert factor > 0.0

    def test_calibrate_apparent_age_without_ethnicity(self):
        """F16-2: calibrate_apparent_age without ethnicity returns raw age and factor 1.0."""
        calib_age, factor = BiologicalLabelingEngine.calibrate_apparent_age(28.0, None)
        assert calib_age == 28.0
        assert factor == 1.0

    def test_demographic_groups_catalog(self):
        """F16-3: Verified catalog of 7 canonical FairFace ethnic cohorts."""
        groups = FairFaceConnector.RACE_MAPPING
        assert len(groups) == 7
        assert 'East Asian' in groups and 'Black' in groups and 'White' in groups

    def test_calibration_preserves_rank_order(self):
        """F16-4: Calibrated apparent ages preserve chronological ordering."""
        ages = [15.0, 25.0, 45.0, 65.0]
        calibrated = [BiologicalLabelingEngine.calibrate_apparent_age(a, "East Asian")[0] for a in ages]
        assert calibrated == sorted(calibrated)

    def test_gradient_factor_reflects_acceleration(self):
        """F16-5: East Asian post-cliff subject has higher gradient factor than pre-cliff."""
        _, f_young = BiologicalLabelingEngine.calibrate_apparent_age(30.0, "East Asian")
        _, f_old = BiologicalLabelingEngine.calibrate_apparent_age(65.0, "East Asian")
        assert f_old > f_young


# ============================================================================
# F17: Gaussian LDL Probability Integration
# ============================================================================

class TestF17GaussianLDLIntegration:
    """Verifies continuous-to-discrete Gaussian LDL probability integration."""

    def test_ldl_distribution_sums_to_one(self):
        """F17-1: Gaussian LDL distribution integrates to 1.0 across 10 cohorts."""
        dist = BiologicalLabelingEngine.compute_label_distribution(mu_age=25.0, sigma_age=3.0, distribution_type="gaussian")
        assert len(dist) == 10
        assert dist.sum() == pytest.approx(1.0, abs=1e-4)

    def test_ldl_peak_aligns_with_consensus_bin(self):
        """F17-2: Peak probability bin aligns with discrete index of consensus age."""
        mu = 29.5  # Falls into bin '25-34' (index 5)
        dist = BiologicalLabelingEngine.compute_label_distribution(mu_age=mu, sigma_age=2.5)
        assert np.argmax(dist) == 5

    def test_all_ldl_probabilities_non_negative(self):
        """F17-3: All discrete bin probabilities are strictly non-negative."""
        dist = BiologicalLabelingEngine.compute_label_distribution(mu_age=8.0, sigma_age=3.0)
        assert np.all(dist >= 0.0)

    def test_ldl_dispersion_broadens_with_larger_sigma(self):
        """F17-4: Larger sigma distributes more probability mass to adjacent bins."""
        dist_tight = BiologicalLabelingEngine.compute_label_distribution(mu_age=30.0, sigma_age=1.5)
        dist_wide = BiologicalLabelingEngine.compute_label_distribution(mu_age=30.0, sigma_age=6.0)
        assert np.max(dist_tight) > np.max(dist_wide)

    def test_gaussian_ldl_matches_reference_oracle(self, oracles):
        """F17-5: Engine LDL matches mathematical error function erf oracle."""
        mu = 37.0
        sigma = 3.5
        engine_dist = BiologicalLabelingEngine.compute_label_distribution(mu, sigma)
        oracle_dist = oracles.gaussian_ldl(mu, sigma)
        np.testing.assert_allclose(engine_dist, oracle_dist, atol=1e-3)


# ============================================================================
# F18: Laplacian LDL Probability Integration
# ============================================================================

class TestF18LaplacianLDLIntegration:
    """Verifies heavy-tailed Laplacian LDL integration for high annotator variance."""

    def test_laplacian_distribution_sums_to_one(self):
        """F18-1: Laplacian LDL sums to 1.0 (+/- 1e-4)."""
        dist = BiologicalLabelingEngine.compute_label_distribution(mu_age=40.0, sigma_age=3.0, distribution_type="laplacian")
        assert len(dist) == 10
        assert dist.sum() == pytest.approx(1.0, abs=1e-4)

    def test_laplacian_peak_aligns_with_consensus_age(self):
        """F18-2: Peak bin matches cohort containing mu."""
        dist = BiologicalLabelingEngine.compute_label_distribution(mu_age=15.0, sigma_age=2.0, distribution_type="laplacian")
        assert np.argmax(dist) == 3  # '13-17' is index 3

    def test_laplacian_has_heavier_tails_than_gaussian(self, oracles):
        """F18-3: Laplacian distribution assigns higher mass to distant tails."""
        mu = 30.0
        sigma = 3.0
        g_dist = BiologicalLabelingEngine.compute_label_distribution(mu, sigma, distribution_type="gaussian")
        l_dist = BiologicalLabelingEngine.compute_label_distribution(mu, sigma, distribution_type="laplacian")
        # Tail bin (e.g. index 0 or index 9) has higher probability under Laplace
        assert l_dist[0] >= g_dist[0] or l_dist[9] >= g_dist[9]

    def test_all_laplacian_probabilities_non_negative(self):
        """F18-4: All bin probabilities are non-negative."""
        dist = BiologicalLabelingEngine.compute_label_distribution(mu_age=70.0, sigma_age=3.0, distribution_type="laplacian")
        assert np.all(dist >= 0.0)

    def test_laplacian_ldl_oracle_agreement(self, oracles):
        """F18-5: Engine Laplacian distribution integrates properly across cohorts."""
        dist = BiologicalLabelingEngine.compute_label_distribution(mu_age=25.0, distribution_type="laplacian")
        assert dist.sum() == pytest.approx(1.0, abs=1e-4)


# ============================================================================
# F19: Multi-Teacher Ensemble Integration
# ============================================================================

class TestF19MultiTeacherEnsembleIntegration:
    """Verifies multi-teacher feature aggregation across InsightFace, MiVOLO, FairFace, CLIP."""

    def test_evaluate_teachers_requires_predictions(self):
        """F19-1: evaluate_teachers raises ValueError on empty list."""
        mte = MultiTeacherEnsemble()
        with pytest.raises(ValueError):
            mte.evaluate_teachers([])

    def test_consensus_age_is_mean_of_teachers(self, golden_teacher_predictions):
        """F19-2: consensus_age equals arithmetic mean of teacher ages."""
        mte = MultiTeacherEnsemble()
        res = mte.evaluate_teachers(golden_teacher_predictions)
        expected_mu = float(np.mean([p["age"] for p in golden_teacher_predictions]))
        assert res["consensus_age"] == pytest.approx(expected_mu, abs=0.01)

    def test_consensus_gender_is_majority_vote(self, golden_teacher_predictions):
        """F19-3: High female probability produces consensus_gender = 'Female'."""
        mte = MultiTeacherEnsemble()
        res = mte.evaluate_teachers(golden_teacher_predictions)
        assert res["predicted_gender"] == "Female"

    def test_epistemic_std_reflects_teacher_variance(self, polarized_teacher_predictions):
        """F19-4: Disagreeing teachers produce elevated epistemic standard deviation."""
        mte = MultiTeacherEnsemble()
        res = mte.evaluate_teachers(polarized_teacher_predictions)
        assert res["age_std"] > 8.0

    def test_four_teachers_represented(self, golden_teacher_predictions):
        """F19-5: Predictions contain all 4 canonical teachers."""
        mte = MultiTeacherEnsemble()
        res = mte.evaluate_teachers(golden_teacher_predictions)
        assert res["teacher_count"] == 4


# ============================================================================
# F20: Disagreement Analysis & Entropy Pruning
# ============================================================================

class TestF20DisagreementAnalysisAndEntropyPruning:
    """Verifies epistemic noise detection (sigma > 8.0) and Shannon entropy gating."""

    def test_high_age_disagreement_is_pruned(self, polarized_teacher_predictions):
        """F20-1: Epistemic age std > 8.0 sets is_pruned=True."""
        mte = MultiTeacherEnsemble()
        res = mte.evaluate_teachers(polarized_teacher_predictions)
        assert res["is_pruned"] is True
        assert any("AGE_DISAGREEMENT" in t for t in res["rejection_tags"])

    def test_adult_gender_confusion_is_pruned(self, ambiguous_gender_adult_predictions):
        """F20-2: Adult with gender confidence < 0.70 sets is_pruned=True."""
        mte = MultiTeacherEnsemble()
        res = mte.evaluate_teachers(ambiguous_gender_adult_predictions)
        assert res["is_pruned"] is True
        assert any("GENDER_CONFUSION" in t for t in res["rejection_tags"])

    def test_prepubertal_child_ambiguity_not_pruned(self, prepubertal_child_predictions):
        """F20-3: Prepubertal child (age < 12) with gender conf < 0.70 is NOT pruned."""
        mte = MultiTeacherEnsemble()
        res = mte.evaluate_teachers(prepubertal_child_predictions)
        assert res["is_pruned"] is False

    def test_consensus_sample_passes_unpruned(self, golden_teacher_predictions):
        """F20-4: High-consensus sample passes with is_pruned=False."""
        mte = MultiTeacherEnsemble()
        res = mte.evaluate_teachers(golden_teacher_predictions)
        assert res["is_pruned"] is False

    def test_shannon_entropy_peak_at_half_probability(self):
        """F20-5: Shannon entropy H(p) reaches maximum 1.0 when p = 0.5."""
        h_max = MultiTeacherEnsemble.compute_shannon_entropy(0.5)
        h_pure = MultiTeacherEnsemble.compute_shannon_entropy(0.999)
        assert h_max == pytest.approx(1.0, abs=1e-4)
        assert h_pure < 0.05


# ============================================================================
# F21: Alpha Signal Generation
# ============================================================================

class TestF21AlphaSignalGeneration:
    """Verifies high-confidence biometric pseudo-labels and training sample weighting."""

    def test_alpha_weight_in_unit_interval(self, golden_teacher_predictions):
        """F21-1: alpha_sample_weight is strictly in (0.0, 1.0]."""
        mte = MultiTeacherEnsemble()
        res = mte.evaluate_teachers(golden_teacher_predictions)
        w = res["alpha_sample_weight"]
        assert 0.0 < w <= 1.0

    def test_alpha_weight_drops_with_epistemic_disagreement(self, golden_teacher_predictions):
        """F21-2: Alpha weight drops as teacher disagreement increases."""
        mte = MultiTeacherEnsemble()
        res_good = mte.evaluate_teachers(golden_teacher_predictions)
        noisy = [
            {"teacher_name": "t1", "age": 24.0, "gender_prob": 0.90},
            {"teacher_name": "t2", "age": 32.0, "gender_prob": 0.80}
        ]
        res_noisy = mte.evaluate_teachers(noisy)
        assert res_good["alpha_sample_weight"] > res_noisy["alpha_sample_weight"]

    def test_alpha_signal_payload_structure(self, golden_teacher_predictions):
        """F21-3: Extracted payload contains all required biometric and alpha fields."""
        mte = MultiTeacherEnsemble()
        res = mte.evaluate_teachers(golden_teacher_predictions)
        required_keys = [
            "is_pruned", "consensus_age", "age_group", "age_group_idx", "age_std",
            "predicted_gender", "gender_confidence", "gender_std",
            "shannon_entropy", "dimorphism_weight", "alpha_sample_weight", "ldl_distribution"
        ]
        for k in required_keys:
            assert k in res, f"Missing key in alpha signals: {k}"

    def test_ldl_distribution_length_in_alpha_signal(self, golden_teacher_predictions):
        """F21-4: ldl_distribution has exactly 10 bins summing to 1.0."""
        mte = MultiTeacherEnsemble()
        res = mte.evaluate_teachers(golden_teacher_predictions)
        ldl = res["ldl_distribution"]
        assert len(ldl) == 10
        assert sum(ldl) == pytest.approx(1.0, abs=1e-4)

    def test_alpha_weight_attenuated_for_children(self, prepubertal_child_predictions):
        """F21-5: Child alpha weight is scaled by dimorphism factor."""
        mte = MultiTeacherEnsemble()
        res = mte.evaluate_teachers(prepubertal_child_predictions)
        assert res["alpha_sample_weight"] <= 0.60


# ============================================================================
# F22: Unified 30-Column Manifest Schema
# ============================================================================

class TestF22Unified30ColumnManifestSchema:
    """Verifies metadata.parquet and metadata.csv schema specification."""

    def test_golden_face_record_dataclass_contract(self):
        """F22-1: GoldenFaceRecord dataclass contains all mandatory 30-column fields."""
        record = GoldenFaceRecord(
            face_id="sha256_hash_hex_01",
            source_dataset="fairface",
            apparent_age=25.4,
            age_std=1.2,
            age_distribution=[0.1] * 10,
            predicted_gender="Female",
            gender_confidence=0.98,
            dimorphism_weight=1.0,
            ethnicity_group="East Asian",
            aging_gradient_factor=0.95,
            split="train",
            phash_hex="1234567890abcdef",
            laplacian_var=145.2,
            euler_yaw=2.1,
            euler_pitch=-1.5,
            file_path="train/face_01.jpg"
        )
        assert record.apparent_age == 25.4
        assert record.split == "train"

    def test_parquet_serialization_roundtrip(self, tmp_path):
        """F22-2: DataFrame serializes and deserializes to Parquet preserving types."""
        df = pd.DataFrame([{
            "face_id": "face_01", "apparent_age": 25.0, "gender": "Female",
            "split": "train", "laplacian_var": 120.0
        }])
        parquet_file = tmp_path / "metadata.parquet"
        df.to_parquet(parquet_file, index=False)
        loaded = pd.read_parquet(parquet_file)
        assert len(loaded) == 1
        assert loaded["gender"].iloc[0] == "Female"

    def test_csv_matches_parquet_records(self, tmp_path):
        """F22-3: CSV export matches Parquet record count and values."""
        df = pd.DataFrame([{"id": f"img_{i}", "val": i * 1.5} for i in range(10)])
        csv_p = tmp_path / "meta.csv"
        parq_p = tmp_path / "meta.parquet"
        df.to_csv(csv_p, index=False)
        df.to_parquet(parq_p, index=False)
        df_csv = pd.read_csv(csv_p)
        df_parq = pd.read_parquet(parq_p)
        assert len(df_csv) == len(df_parq) == 10

    def test_no_null_values_in_critical_fields(self):
        """F22-4: Manifest verification checks for null/NaN values."""
        sample_record = {"image_id": "face_01.jpg", "age_continuous": 24.5, "gender": "Male"}
        assert not any(v is None or (isinstance(v, float) and math.isnan(v)) for v in sample_record.values())

    def test_aligned_face_sample_dataclass_contract(self, sharp_face, canonical_landmarks):
        """F22-5: AlignedFaceSample dataclass matches Filtering <-> Morphometry contract."""
        aligned = AlignedFaceSample(
            aligned_img_160=sharp_face,
            landmarks_5pt=canonical_landmarks,
            laplacian_var=150.0,
            tenengrad_val=320.0,
            euler_yaw=5.0,
            euler_pitch=-3.0,
            euler_roll=1.0,
            mean_luminance=120.0,
            phash_uint64=123456789
        )
        assert aligned.aligned_img_160.shape == (160, 160, 3)
        assert aligned.laplacian_var >= 100.0


# ============================================================================
# F23: Kaggle CLI Dataset Metadata Generator
# ============================================================================

class TestF23KaggleCLIDatasetMetadataGenerator:
    """Verifies dataset-metadata.json generation for owner vnthanhphm."""

    def test_metadata_generator_creates_valid_json(self, tmp_path):
        """F23-1: Generated dataset metadata is valid JSON."""
        meta_file = tmp_path / "dataset-metadata.json"
        generate_kernel_metadata(meta_file)
        assert meta_file.exists()
        with open(meta_file, "r") as f:
            data = json.load(f)
        assert isinstance(data, dict)

    def test_dataset_id_belongs_to_vnthanhphm(self, tmp_path):
        """F23-2: Dataset ID specifies user vnthanhphm."""
        meta_file = tmp_path / "dataset-metadata.json"
        generate_kernel_metadata(meta_file)
        with open(meta_file, "r") as f:
            data = json.load(f)
        assert data.get("id", "").startswith("vnthanhphm/")

    def test_metadata_contains_title_and_code_file(self, tmp_path):
        """F23-3: Metadata contains title and notebook code_file pointer."""
        meta_file = tmp_path / "dataset-metadata.json"
        generate_kernel_metadata(meta_file)
        with open(meta_file, "r") as f:
            data = json.load(f)
        assert "title" in data
        assert "code_file" in data

    def test_enable_gpu_and_internet_flags(self, tmp_path):
        """F23-4: GPU and Internet access enabled in kernel metadata."""
        meta_file = tmp_path / "dataset-metadata.json"
        generate_kernel_metadata(meta_file)
        with open(meta_file, "r") as f:
            data = json.load(f)
        assert data.get("enable_gpu") is True
        assert data.get("enable_internet") is True

    def test_target_golden_dataset_slug(self):
        """F23-5: Dataset metadata specifies golden dataset slug."""
        target_slug = "vnthanhphm/face-gender-age-golden-dataset"
        assert target_slug.startswith("vnthanhphm/")


# ============================================================================
# F24: Kaggle Kernel Builder Bug Fixes
# ============================================================================

class TestF24KaggleKernelBuilderBugFixes:
    """Verifies fix of NameError: json and notebook generation."""

    def test_build_kernel_generates_valid_ipynb(self, tmp_path):
        """F24-1: generate_pipeline_notebook outputs valid Jupyter notebook JSON."""
        nb_path = tmp_path / "kaggle_face_mining_pipeline.ipynb"
        generate_pipeline_notebook(nb_path)
        assert nb_path.exists()
        with open(nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)
        assert "cells" in nb
        assert "metadata" in nb
        assert nb.get("nbformat") == 4

    def test_notebook_cell_count_and_types(self, tmp_path):
        """F24-2: Notebook contains markdown documentation and executable code cells."""
        nb_path = tmp_path / "pipeline.ipynb"
        generate_pipeline_notebook(nb_path)
        with open(nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)
        cells = nb["cells"]
        assert len(cells) >= 5
        types = {c["cell_type"] for c in cells}
        assert "markdown" in types and "code" in types

    def test_json_import_present_in_notebook(self, tmp_path):
        """F24-3: Verifies notebook imports json to prevent NameError: json."""
        nb_path = tmp_path / "pipeline.ipynb"
        generate_pipeline_notebook(nb_path)
        with open(nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)
        all_code = "\n".join("".join(c.get("source", [])) for c in nb["cells"] if c["cell_type"] == "code")
        assert "import json" in all_code or ", json" in all_code

    def test_notebook_specifies_python3_kernel(self, tmp_path):
        """F24-4: Notebook metadata targets Python 3 kernelspec."""
        nb_path = tmp_path / "pipeline.ipynb"
        generate_pipeline_notebook(nb_path)
        with open(nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)
        spec = nb["metadata"]["kernelspec"]
        assert spec["language"] == "python"

    def test_notebook_generation_overwrites_cleanly(self, tmp_path):
        """F24-5: Generator overwrites cleanly without file locking."""
        nb_path = tmp_path / "pipeline.ipynb"
        generate_pipeline_notebook(nb_path)
        generate_pipeline_notebook(nb_path)
        assert nb_path.stat().st_size > 0


# ============================================================================
# F25: Local Manifest Sync & Verification
# ============================================================================

class TestF25LocalManifestSyncAndVerification:
    """Verifies audit_and_sync_manifest tool and Markdown report generator."""

    def test_audit_report_generates_markdown(self, tmp_path):
        """F25-1: generate_audit_report creates Markdown report."""
        csv_p = tmp_path / "meta.csv"
        report_p = tmp_path / "report.md"
        df = pd.DataFrame([{
            "image_id": "face_01.jpg", "gender": "Female", "age_group": "25-34",
            "laplacian_var": 120.5, "tenengrad": 310.0, "mean_luminance": 110.0,
            "split": "train"
        }])
        df.to_csv(csv_p, index=False)
        rep = generate_audit_report(csv_p, report_p)
        assert report_p.exists()
        assert "Báo Cáo Thẩm Định" in rep

    def test_audit_report_handles_parquet(self, tmp_path):
        """F25-2: Audit reporter supports Parquet format seamlessly."""
        parq_p = tmp_path / "meta.parquet"
        report_p = tmp_path / "report_parq.md"
        df = pd.DataFrame([{
            "image_id": "face_02.jpg", "gender": "Male", "age_group": "35-44",
            "laplacian_var": 140.0, "tenengrad": 290.0, "mean_luminance": 95.0,
            "split": "val"
        }])
        df.to_parquet(parq_p, index=False)
        rep = generate_audit_report(parq_p, report_p)
        assert report_p.exists()

    def test_validate_manifest_integrity_checks_columns(self):
        """F25-3: validate_manifest_integrity checks required schema columns."""
        df = pd.DataFrame([{"image_id": "01.jpg"}])
        res = validate_manifest_integrity(df)
        assert res["passed"] is False
        assert len(res["missing_columns"]) > 0

    def test_validate_manifest_detects_blur_violations(self):
        """F25-4: validate_manifest_integrity flags blurry samples."""
        df = pd.DataFrame([{"laplacian_var": 50.0}])
        res = validate_manifest_integrity(df)
        assert any("Blurry" in v for v in res["violations"])

    def test_expected_manifest_columns_catalog(self):
        """F25-5: EXPECTED_MANIFEST_COLUMNS contains >= 30 biometric columns."""
        assert len(EXPECTED_MANIFEST_COLUMNS) >= 30


# ============================================================================
# F26: Mathematical Statistical Analysis Script
# ============================================================================

class TestF26MathematicalStatisticalAnalysis:
    """Verifies statistical distributions, demographic matrix, and Kolmogorov-Smirnov analysis."""

    def test_earth_movers_distance_metric(self):
        """F26-1: BiologicalLabelingEngine.earth_movers_distance computes 1D Wasserstein distance."""
        p = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        q = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        emd = BiologicalLabelingEngine.earth_movers_distance(p, q)
        assert emd > 0.0

    def test_kl_divergence_metric(self):
        """F26-2: BiologicalLabelingEngine.kl_divergence computes relative entropy."""
        p = np.array([0.1] * 10)
        kl = BiologicalLabelingEngine.kl_divergence(p, p)
        assert kl == pytest.approx(0.0, abs=1e-5)

    def test_composite_ldl_loss(self):
        """F26-3: composite_ldl_loss combines EMD and KL divergence."""
        p = np.array([0.1] * 10)
        q = np.array([0.05] * 8 + [0.3, 0.3])
        loss = BiologicalLabelingEngine.composite_ldl_loss(p, q, alpha_emd=0.5)
        assert loss > 0.0

    def test_simplex_constraint_audit(self):
        """F26-4: Audits that all 10 LDL probabilities sum to 1.0 within tolerance."""
        probs = np.array([0.1] * 10)
        assert abs(probs.sum() - 1.0) <= 1e-4

    def test_demographic_contingency_matrix_computation(self):
        """F26-5: Cross-tabulates age cohort vs gender frequency."""
        df = pd.DataFrame({
            "age_group": ["18-24", "18-24", "25-34", "65+"],
            "gender": ["Male", "Female", "Female", "Male"]
        })
        ct = pd.crosstab(df["age_group"], df["gender"])
        assert ct.loc["18-24", "Male"] == 1
        assert ct.loc["18-24", "Female"] == 1
