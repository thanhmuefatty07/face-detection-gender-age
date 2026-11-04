"""
Comprehensive Unit, Integration, and Edge-Case Tests for:
1. 23 Fine-Grained 3-5 Year Age Intervals (`FINE_AGE_INTERVALS`)
2. Box-Convolved Interval Label Distribution Learning (`compute_interval_box_ldl`)
3. Interval-Censored Ordinal Loss (`interval_censored_ordinal_loss` & `UnifiedBiometricLoss`)
4. Expanded Real-Face Dataset Connectors (`AdienceConnector`, `AxonAgeConnector`,
   `MegaAgeAsianConnector`, `LagendaConnector`, `CACDConnector`, `AllAgeFacesConnector`)
5. Unified Local & Kaggle Dataset Curation (`ExtremeDatasetCurator` & `FaceGoldenDataset`)
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import torch

from src.biometrics.core.types import BiometricPrediction
from src.biometrics.estimation.dataset import FaceGoldenDataset
from src.biometrics.estimation.intervals import (
    FINE_AGE_INTERVALS,
    NUM_FINE_AGE_INTERVALS,
    compute_interval_box_ldl,
    compute_interval_epistemic_weight,
    get_fine_interval_for_age,
    get_overlapping_fine_intervals,
    integrate_101_to_fine_intervals,
    map_bounds_to_fine_interval,
    parse_age_interval_string,
)
from src.biometrics.estimation.ldl_engine import ContinuousLDLEngine, compute_gaussian_ldl
from src.biometrics.estimation.losses import UnifiedBiometricLoss, interval_censored_ordinal_loss
from src.mining import (
    DATASET_SOURCE_REGISTRY,
    AdienceConnector,
    AllAgeFacesConnector,
    AxonAgeConnector,
    CACDConnector,
    ExtremeDatasetCurator,
    ExtremeQualityAuditor,
    InCloudStreamingHarvester,
    LagendaConnector,
    MegaAgeAsianConnector,
    RawImageSample,
)
from scripts.kaggle_pipeline.kaggle_face_engine import BiologicalLabelingEngine


def _make_textured_face_jpeg(seed: int = 42, size: int = 160) -> bytes:
    """Generates a sharp, well-exposed synthetic face-like image with distinct low-frequency DCT structure per seed."""
    rng = np.random.default_rng(seed)
    yy, xx = np.indices((size, size), dtype=np.float32)
    # Seed-specific low-frequency spatial harmonics (symmetric horizontally so bilateral cheek Delta E remains low)
    freq_y = 1.0 + (seed % 7) * 0.6
    phase_y = (seed % 13) * 0.5
    sym_x = np.abs(xx - size / 2.0) / (size / 2.0)
    low_freq = 28.0 * np.cos(2.0 * math.pi * freq_y * (yy / size) + phase_y) * np.cos(math.pi * sym_x * ((seed % 5) + 1))
    # High-frequency checkerboard + texture so Laplacian >= 100 and Tenengrad >= 250
    checker = ((((xx.astype(np.int32) // 2 + yy.astype(np.int32) // 2) % 2) * 34) - 17).astype(np.float32)
    noise = rng.integers(-10, 11, size=(size, size, 3)).astype(np.float32)
    img = np.clip(128.0 + low_freq[:, :, None] + checker[:, :, None] + noise, 25.0, 225.0).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    assert ok
    return buf.tobytes()



class TestFineAgeIntervalsSpec:
    """Verifies the mathematical partition properties of the 23 3-5 year age intervals."""

    def test_contiguous_partition_and_3_to_5_year_spans(self):
        assert NUM_FINE_AGE_INTERVALS == 23
        assert len(FINE_AGE_INTERVALS) == 23
        assert FINE_AGE_INTERVALS[0][1] == 0
        assert FINE_AGE_INTERVALS[-1][2] == 100

        prev_high = -1
        for idx, (name, low, high) in enumerate(FINE_AGE_INTERVALS):
            span = high - low + 1
            assert 3 <= span <= 5, f"Interval {idx} ({name}) has span {span}, expected 3..5 years"
            assert low == prev_high + 1, f"Gap or overlap before interval {idx} ({name})"
            assert name == f"{low}-{high}"
            prev_high = high

    def test_every_integer_age_maps_uniquely(self):
        for age in range(0, 101):
            name, idx, low, high = get_fine_interval_for_age(float(age))
            assert 0 <= idx < 23
            assert low <= age <= high
            assert name == f"{low}-{high}"

    def test_boundary_clamping_and_non_finite_rejection(self):
        assert get_fine_interval_for_age(-10.0) == ("0-2", 0, 0, 2)
        assert get_fine_interval_for_age(0.0) == ("0-2", 0, 0, 2)
        assert get_fine_interval_for_age(100.0) == ("96-100", 22, 96, 100)
        assert get_fine_interval_for_age(135.0) == ("96-100", 22, 96, 100)

        with pytest.raises(ValueError):
            get_fine_interval_for_age(float("nan"))
        with pytest.raises(ValueError):
            get_fine_interval_for_age(float("inf"))

    def test_parse_age_interval_string_formats_and_adience_typos(self):
        # Adience tuple strings + documented benchmark typos
        assert parse_age_interval_string("(25, 32)") == (25.0, 32.0, "interval")
        assert parse_age_interval_string("(8, 23)") == (8.0, 12.0, "interval")
        assert parse_age_interval_string("(38, 42)") == (38.0, 43.0, "interval")
        assert parse_age_interval_string("(38, 48)") == (38.0, 43.0, "interval")

        # Range strings
        assert parse_age_interval_string("26-30") == (26.0, 30.0, "interval")
        assert parse_age_interval_string("[18, 21]") == (18.0, 21.0, "interval")
        assert parse_age_interval_string("31 to 35") == (31.0, 35.0, "interval")

        # Open-ended senior brackets
        assert parse_age_interval_string("more than 70") == (70.0, 100.0, "interval")
        assert parse_age_interval_string("65+") == (65.0, 100.0, "interval")

        # Scalar with snapping to 3-5 year interval
        assert parse_age_interval_string(28, snap_point_to_fine_interval=True) == (26.0, 30.0, "fine_interval")
        assert parse_age_interval_string("28", snap_point_to_fine_interval=False) == (28.0, 28.0, "exact")

        # Invalid inputs
        assert parse_age_interval_string(None) is None
        assert parse_age_interval_string("unknown") is None
        assert parse_age_interval_string(-3) is None

    def test_overlapping_fine_intervals_weights_sum_to_one(self):
        overlaps = get_overlapping_fine_intervals(20.0, 29.0)
        assert len(overlaps) >= 2
        total_w = sum(w for _, _, w in overlaps)
        assert math.isclose(total_w, 1.0, rel_tol=1e-6)


class TestBoxConvolvedLDLandIntervalLoss:
    """Verifies Box-Convolved LDL simplex conservation, integration, and interval loss gradients."""

    @pytest.mark.parametrize(
        "age_min,age_max",
        [(0.0, 2.0), (3.0, 5.0), (18.0, 21.0), (26.0, 30.0), (61.0, 65.0), (96.0, 100.0)],
    )
    def test_box_convolved_ldl_simplex_and_concentration(self, age_min: float, age_max: float):
        dist_101 = compute_interval_box_ldl(age_min, age_max, sigma=1.8, num_bins=101)
        assert dist_101.shape == (101,)
        assert np.all(dist_101 > 0.0)
        assert math.isclose(float(np.sum(dist_101)), 1.0, rel_tol=1e-5)

        dist_23 = integrate_101_to_fine_intervals(dist_101)
        assert dist_23.shape == (23,)
        assert math.isclose(float(np.sum(dist_23)), 1.0, rel_tol=1e-5)

        _, target_idx, _, _ = map_bounds_to_fine_interval(age_min, age_max)
        # Peak interval probability must occur at the target 3-5 year interval
        assert int(np.argmax(dist_23)) == target_idx
        assert float(dist_23[target_idx]) > 0.45

    def test_batch_2d_integration_to_fine_intervals(self):
        d1 = compute_interval_box_ldl(3.0, 5.0)
        d2 = compute_interval_box_ldl(46.0, 50.0)
        batch_101 = np.stack([d1, d2], axis=0)
        batch_23 = integrate_101_to_fine_intervals(batch_101)
        assert batch_23.shape == (2, 23)
        np.testing.assert_allclose(batch_23.sum(axis=1), np.ones(2), atol=1e-5)

    def test_continuous_ldl_engine_fine_interval_prediction(self):
        engine = ContinuousLDLEngine(num_bins=101)
        dist = compute_gaussian_ldl(mean_age=28.0, sigma=2.0)
        name, idx, conf, bounds = engine.predict_fine_interval(dist)
        assert name == "26-30"
        assert idx == 8
        assert bounds == (26, 30)
        assert 0.50 < conf <= 1.0

    def test_interval_censored_ordinal_loss_and_unified_loss(self):
        # Construct two predictions for target interval [26, 30]:
        # pred_good concentrates mass in 26..30; pred_bad concentrates mass in 60..65
        good_np = compute_interval_box_ldl(26.0, 30.0, sigma=1.5)
        bad_np = compute_interval_box_ldl(61.0, 65.0, sigma=1.5)
        pred_good = torch.tensor(good_np, dtype=torch.float32).unsqueeze(0).requires_grad_(True)
        pred_bad = torch.tensor(bad_np, dtype=torch.float32).unsqueeze(0).requires_grad_(True)

        age_min = torch.tensor([26.0], dtype=torch.float32)
        age_max = torch.tensor([30.0], dtype=torch.float32)

        loss_good = interval_censored_ordinal_loss(pred_good, age_min, age_max)
        loss_bad = interval_censored_ordinal_loss(pred_bad, age_min, age_max)
        assert float(loss_good.item()) < float(loss_bad.item())
        loss_good.backward()
        assert pred_good.grad is not None
        assert torch.all(torch.isfinite(pred_good.grad))

        # Test UnifiedBiometricLoss with interval bounds
        criterion = UnifiedBiometricLoss(weight_interval=0.25)
        preds = {
            "age_probs": pred_good.detach().clone().requires_grad_(True),
            "expected_age": torch.tensor([28.0], requires_grad=True),
            "gender_logit": torch.tensor([1.5], requires_grad=True),
            "log_variance": torch.tensor([0.2], requires_grad=True),
        }
        targets = {
            "ldl_101": torch.tensor(good_np).unsqueeze(0),
            "age_continuous": torch.tensor([28.0]),
            "age_min": age_min,
            "age_max": age_max,
            "gender": torch.tensor([1.0]),
            "gender_mask": torch.tensor([1.0]),
            "dimorphism_weight": torch.tensor([1.0]),
        }
        total_loss, metrics = criterion(preds, targets)
        assert "loss_interval" in metrics
        assert metrics["loss_interval"] > 0.0
        total_loss.backward()
        assert preds["age_probs"].grad is not None

    def test_biological_labeling_engine_fine_intervals(self):
        name, idx, low, high = BiologicalLabelingEngine.age_to_fine_interval(19.6)
        assert (name, idx, low, high) == ("18-21", 6, 18, 21)
        dist_23 = BiologicalLabelingEngine.compute_fine_interval_distribution(
            mu_age=19.5, sigma_age=2.0, age_min=18.0, age_max=21.0
        )
        assert len(dist_23) == 23
        assert math.isclose(float(np.sum(dist_23)), 1.0, rel_tol=1e-5)
        assert int(np.argmax(dist_23)) == 6


class TestExpandedDatasetConnectors:
    """Tests all 6 newly added real human face dataset connectors and harvester discovery."""

    def test_registry_has_16_real_face_sources(self):
        assert len(DATASET_SOURCE_REGISTRY) == 16
        assert "utkface" not in DATASET_SOURCE_REGISTRY
        for expected in (
            "wikimedia_commons", "fairface", "adience", "axon_age", "appa_real", "agedb",
            "morph", "fgnet", "afad", "megaage_asian", "lagenda", "cacd", "aaf",
            "b3fd", "imdb_wiki", "celeba",
        ):
            assert expected in DATASET_SOURCE_REGISTRY

    def test_adience_connector_streams_fold_and_repairs_typos(self, tmp_path: Path):
        adience_dir = tmp_path / "adience-dataset"
        faces_dir = adience_dir / "faces" / "12345@N01"
        faces_dir.mkdir(parents=True)
        img_bytes = _make_textured_face_jpeg(seed=101)
        (faces_dir / "coarse_tilt_aligned_face.7.sample1.jpg").write_bytes(img_bytes)
        (faces_dir / "coarse_tilt_aligned_face.8.sample2.jpg").write_bytes(img_bytes)

        fold_txt = (
            "user_id\toriginal_image\tface_id\tage\tgender\n"
            "12345@N01\tsample1.jpg\t7\t(25, 32)\tf\n"
            "12345@N01\tsample2.jpg\t8\t(38, 42)\tm\n"
        )
        (adience_dir / "fold_0_data.txt").write_text(fold_txt, encoding="utf-8")

        conn = AdienceConnector(adience_dir)
        assert conn.is_available()
        samples = list(conn.stream_samples())
        assert len(samples) == 2
        assert samples[0].declared_gender == "Female"
        assert (samples[0].age_min, samples[0].age_max) == (25.0, 32.0)
        assert samples[0].age_interval_name == "26-30"
        # Typo (38, 42) repaired to (38, 43) -> midpoint 40.5 -> '36-40' or '41-45'
        assert samples[1].declared_gender == "Male"
        assert (samples[1].age_min, samples[1].age_max) == (38.0, 43.0)

    def test_axon_age_connector_csv_and_folder_brackets(self, tmp_path: Path):
        axon_dir = tmp_path / "axon-selfie-age"
        bracket_dir = axon_dir / "22-25" / "female"
        bracket_dir.mkdir(parents=True)
        img_bytes = _make_textured_face_jpeg(seed=102)
        (bracket_dir / "selfie_01.jpg").write_bytes(img_bytes)

        conn = AxonAgeConnector(axon_dir)
        assert conn.is_available()
        samples = list(conn.stream_samples())
        assert len(samples) == 1
        assert samples[0].declared_gender == "Female"
        assert (samples[0].age_min, samples[0].age_max) == (22.0, 25.0)
        assert samples[0].age_interval_name == "22-25"

    def test_megaage_asian_connector(self, tmp_path: Path):
        mega_dir = tmp_path / "megaage-asian"
        (mega_dir / "list").mkdir(parents=True)
        (mega_dir / "train").mkdir(parents=True)
        img_bytes = _make_textured_face_jpeg(seed=103)
        (mega_dir / "train" / "0001.jpg").write_bytes(img_bytes)
        (mega_dir / "list" / "train_name.txt").write_text("0001.jpg\n", encoding="utf-8")
        (mega_dir / "list" / "train_age.txt").write_text("27\n", encoding="utf-8")
        (mega_dir / "list" / "train_gender.txt").write_text("0\n", encoding="utf-8")

        conn = MegaAgeAsianConnector(mega_dir)
        assert conn.is_available()
        samples = list(conn.stream_samples())
        assert len(samples) == 1
        assert samples[0].declared_age == 27.0
        assert samples[0].declared_gender == "Female"
        assert samples[0].declared_ethnicity == "East Asian"
        assert samples[0].age_interval_name == "26-30"

    def test_lagenda_cacd_and_aaf_connectors(self, tmp_path: Path):
        img_bytes = _make_textured_face_jpeg(seed=104)

        # 1. Lagenda
        lag_dir = tmp_path / "lagenda"
        lag_dir.mkdir()
        (lag_dir / "person1.jpg").write_bytes(img_bytes)
        pd.DataFrame([{"img_name": "person1.jpg", "age": "31-35", "gender": "M", "person_id": "p1"}]).to_csv(
            lag_dir / "annotations.csv", index=False
        )
        lag_samples = list(LagendaConnector(lag_dir).stream_samples())
        assert len(lag_samples) == 1
        assert lag_samples[0].age_interval_name == "31-35"
        assert lag_samples[0].declared_gender == "Male"

        # 2. CACD
        cacd_dir = tmp_path / "cacd"
        cacd_dir.mkdir()
        (cacd_dir / "28_Emma_Watson_0001.jpg").write_bytes(img_bytes)
        cacd_samples = list(CACDConnector(cacd_dir).stream_samples())
        assert len(cacd_samples) == 1
        assert cacd_samples[0].declared_age == 28.0
        assert cacd_samples[0].age_interval_name == "26-30"
        assert cacd_samples[0].metadata["subject_id"] == "cacd:emma_watson"

        # 3. All-Age-Faces (AAF)
        aaf_dir = tmp_path / "all-age-faces"
        aaf_dir.mkdir()
        (aaf_dir / "00120A19.jpg").write_bytes(img_bytes)
        (aaf_dir / "09200A44.jpg").write_bytes(img_bytes)
        aaf_samples = list(AllAgeFacesConnector(aaf_dir).stream_samples())
        assert len(aaf_samples) == 2
        by_id = {s.raw_id: s for s in aaf_samples}
        assert by_id["00120A19.jpg"].declared_gender == "Female"
        assert by_id["00120A19.jpg"].age_interval_name == "18-21"
        assert by_id["09200A44.jpg"].declared_gender == "Male"
        assert by_id["09200A44.jpg"].age_interval_name == "41-45"

        # 4. InCloudStreamingHarvester auto-discovery
        harvester = InCloudStreamingHarvester(kaggle_input_dir=str(tmp_path), max_samples_per_dataset=10)
        streamed = list(harvester.stream_all())
        assert len(streamed) >= 4


class TestUnifiedCurationAndGoldenDataset:
    """Verifies end-to-end local curation, manifest schema, and PyTorch dataset loading."""

    def test_extreme_dataset_curator_and_golden_dataset_roundtrip(self, tmp_path: Path):
        raw_samples = [
            RawImageSample(
                image_bytes=_make_textured_face_jpeg(seed=201),
                source_dataset="adience",
                raw_id="s1.jpg",
                declared_gender="Female",
                age_min=18.0,
                age_max=21.0,
                age_label_type="interval",
                metadata={"subject_id": "adience:u1"},
            ),
            RawImageSample(
                image_bytes=_make_textured_face_jpeg(seed=302),
                source_dataset="fairface",
                raw_id="s2.jpg",
                declared_age=33.0,
                declared_gender="Male",
                declared_ethnicity="Caucasian",
                metadata={"subject_id": "fairface:u2"},
            ),
            # Exact duplicate of s1 -> should be rejected by DualLayerDeduplicator
            RawImageSample(
                image_bytes=_make_textured_face_jpeg(seed=201),
                source_dataset="adience",
                raw_id="s1_dup.jpg",
                declared_gender="Female",
                age_min=18.0,
                age_max=21.0,
                age_label_type="interval",
                metadata={"subject_id": "adience:u1"},
            ),
            # Banned UTKFace source -> strictly rejected by ExtremeDatasetCurator
            RawImageSample(
                image_bytes=_make_textured_face_jpeg(seed=405),
                source_dataset="utkface",
                raw_id="s_utk.jpg",
                declared_age=27.0,
                declared_gender="Female",
            ),
        ]

        out_dir = tmp_path / "curated_out"
        auditor = ExtremeQualityAuditor(min_laplacian_var=20.0, min_tenengrad=50.0)
        curator = ExtremeDatasetCurator(
            output_dir=out_dir,
            auditor=auditor,
            write_tar_shards=True,
            write_direct_images=True,
        )
        df, summary = curator.curate_samples(raw_samples)

        assert summary.total_ingested == 4
        assert summary.accepted == 2
        assert summary.rejected_duplicate == 1
        assert summary.rejection_breakdown.get("BANNED_SOURCE_UTK") == 1
        assert "utkface" not in set(df["source_dataset"].unique())
        assert summary.manifest_parquet_path is not None and summary.manifest_parquet_path.exists()
        assert len(df) == 2
        assert set(["age_min", "age_max", "age_interval", "age_interval_idx", "shannon_entropy", "michelson_contrast"]).issubset(df.columns)

        # Load into FaceGoldenDataset and verify 3-5 year interval tensors
        ds = FaceGoldenDataset(
            manifest_path=summary.manifest_parquet_path,
            shards_dir=out_dir / "golden_dataset" / "shards",
        )
        assert len(ds) == 2
        item0 = ds[0]
        assert item0["image"].shape == (3, 160, 160)
        assert item0["ldl_101"].shape == (101,)
        assert item0["ldl_fine_23"].shape == (23,)
        assert math.isclose(float(item0["ldl_fine_23"].sum().item()), 1.0, rel_tol=1e-5)
        assert item0["age_interval_name"] in {"18-21", "31-35"}
        ds.close()

    def test_biometric_prediction_auto_populates_fine_interval(self):
        dist = compute_gaussian_ldl(mean_age=42.0, sigma=2.0)
        pred = BiometricPrediction(
            expected_age=42.0,
            age_ldl_distribution=dist,
            age_cohort_name="35-44",
            age_cohort_index=6,
            apparent_age_std=2.0,
            gender="Female",
            female_probability=0.92,
            dimorphism_weight=1.0,
            aleatoric_uncertainty=2.0,
            alpha_sample_weight=0.85,
            quality_score=0.95,
        )
        assert pred.age_interval_name == "41-45"
        assert pred.age_interval_index == 11
        assert pred.age_interval_bounds == (41, 45)
        assert pred.age_interval_probs is not None
        assert pred.age_interval_probs.shape == (23,)
        assert pred.age_interval_confidence > 0.50

    def test_box_convolved_ldl_boundary_reflection_and_nonfinite_guards(self):
        # Full-range [0, 100] with boundary reflection across [-0.5, 100.5] must be flat uniform
        dist_full = compute_interval_box_ldl(0.0, 100.0, sigma=2.5, num_bins=101)
        np.testing.assert_allclose(dist_full, np.full(101, 1.0 / 101.0), atol=1e-4)

        # Infant bracket [0, 2] must not dip at bin 0 due to missing boundary reflection
        dist_infant = compute_interval_box_ldl(0.0, 2.0, sigma=1.5, num_bins=101)
        assert dist_infant[0] >= dist_infant[1] >= dist_infant[2]

        # Non-finite bounds must raise ValueError; non-finite sigma must fall back safely
        with pytest.raises(ValueError):
            compute_interval_box_ldl(float("nan"), 25.0)
        with pytest.raises(ValueError):
            compute_interval_box_ldl(10.0, float("inf"))
        dist_nan_sigma = compute_interval_box_ldl(26.0, 30.0, sigma=float("nan"))
        assert np.all(np.isfinite(dist_nan_sigma))
        assert math.isclose(float(dist_nan_sigma.sum()), 1.0, rel_tol=1e-5)

    def test_half_integer_boundary_consistency_across_all_modules(self):
        from src.biometrics.estimation.dimorphism import get_cohort_for_age
        from scripts.kaggle_pipeline.kaggle_dataset_connectors import _resolve_fine_interval

        for half_age in (2.5, 5.5, 8.5, 11.5, 14.5, 17.5, 21.5, 24.5, 25.5, 30.5, 34.5, 40.5, 44.5, 54.5, 64.5):
            n1, i1, l1, h1 = get_fine_interval_for_age(half_age)
            n2, i2, l2, h2 = _resolve_fine_interval(half_age)
            n3, i3, l3, h3 = BiologicalLabelingEngine.age_to_fine_interval(half_age)
            assert (n1, i1, l1, h1) == (n2, i2, l2, h2) == (n3, i3, l3, h3)

            c1, ci1 = get_cohort_for_age(half_age)
            c2, ci2 = BiologicalLabelingEngine.age_to_group(half_age)
            assert (c1, ci1) == (c2, ci2)

    def test_interval_censored_ordinal_loss_masked_nan_and_1d_safety(self):
        # Sample 0 has valid interval [26, 30]; Sample 1 is masked out (age_mask=0) with NaN bounds
        p0 = torch.tensor(compute_interval_box_ldl(26.0, 30.0), dtype=torch.float32)
        p1 = torch.tensor(compute_interval_box_ldl(41.0, 45.0), dtype=torch.float32)
        probs = torch.stack([p0, p1], dim=0).requires_grad_(True)
        age_min = torch.tensor([26.0, float("nan")], dtype=torch.float32)
        age_max = torch.tensor([30.0, float("inf")], dtype=torch.float32)
        age_mask = torch.tensor([1.0, 0.0], dtype=torch.float32)

        loss = interval_censored_ordinal_loss(probs, age_min, age_max, age_mask=age_mask)
        assert torch.isfinite(loss)
        loss.backward()
        assert probs.grad is not None
        assert torch.all(torch.isfinite(probs.grad))
        assert torch.all(probs.grad[1] == 0.0)

        # 1D tensor input
        loss_1d = interval_censored_ordinal_loss(
            p0.detach().clone().requires_grad_(True),
            torch.tensor([26.0]),
            torch.tensor([30.0]),
        )
        assert torch.isfinite(loss_1d)

    def test_new_connectors_reject_path_traversal_and_stream_archives(self, tmp_path: Path):
        import io
        import tarfile
        import zipfile

        payload = _make_textured_face_jpeg(seed=404)
        outside = tmp_path / "outside.jpg"
        outside.write_bytes(payload)

        # 1. Adience path traversal rejection even when inside fallback exists
        adience_root = tmp_path / "adience_sec"
        adience_root.mkdir()
        (adience_root / "outside.jpg").write_bytes(payload)
        (adience_root / "fold_0_data.txt").write_text(
            "user_id\toriginal_image\tface_id\tage\tgender\n"
            f"u1\t../outside.jpg\t1\t(25, 32)\tf\n"
            f"u2\t{outside}\t2\t(25, 32)\tm\n",
            encoding="utf-8",
        )
        assert list(AdienceConnector(adience_root).stream_samples()) == []

        # 2. AxonAge path traversal rejection
        axon_root = tmp_path / "axon_sec"
        axon_root.mkdir()
        (axon_root / "outside.jpg").write_bytes(payload)
        pd.DataFrame([
            {"file": "../outside.jpg", "age": "22-25", "gender": "F"},
            {"file": str(outside), "age": "22-25", "gender": "M"},
        ]).to_csv(axon_root / "meta.csv", index=False)
        assert list(AxonAgeConnector(axon_root).stream_samples()) == []

        # 3. MegaAgeAsian path traversal rejection
        mega_root = tmp_path / "mega_sec"
        (mega_root / "list").mkdir(parents=True)
        (mega_root / "train").mkdir(parents=True)
        (mega_root / "outside.jpg").write_bytes(payload)
        (mega_root / "list" / "train_name.txt").write_text("../outside.jpg\n", encoding="utf-8")
        (mega_root / "list" / "train_age.txt").write_text("25\n", encoding="utf-8")
        assert list(MegaAgeAsianConnector(mega_root).stream_samples()) == []

        # 4. Lagenda path traversal rejection
        lag_root = tmp_path / "lag_sec"
        (lag_root / "images").mkdir(parents=True)
        (lag_root / "images" / "outside.jpg").write_bytes(payload)
        pd.DataFrame([{"img_name": "../outside.jpg", "age": "26-30", "gender": "F"}]).to_csv(
            lag_root / "labels.csv", index=False
        )
        assert list(LagendaConnector(lag_root).stream_samples()) == []

        # 5. CACD & AAF .tar and .zip archive streaming
        cacd_tar_root = tmp_path / "cacd_archive"
        cacd_tar_root.mkdir()
        with tarfile.open(cacd_tar_root / "cacd.tar", "w") as tf:
            ti = tarfile.TarInfo(name="32_John_Doe_0005.jpg")
            ti.size = len(payload)
            tf.addfile(ti, io.BytesIO(payload))
        cacd_samples = list(CACDConnector(cacd_tar_root).stream_samples())
        assert len(cacd_samples) == 1
        assert cacd_samples[0].age_interval_name == "31-35"

        aaf_zip_root = tmp_path / "aaf_archive"
        aaf_zip_root.mkdir()
        with zipfile.ZipFile(aaf_zip_root / "aaf.zip", "w") as zf:
            zf.writestr("aligned/00200A23.jpg", payload)
        aaf_samples = list(AllAgeFacesConnector(aaf_zip_root).stream_samples())
        assert len(aaf_samples) == 1
        assert aaf_samples[0].age_interval_name == "22-25"
        assert aaf_samples[0].declared_gender == "Female"

    def test_curator_output_passes_manifest_integrity_and_audit_dataset(self, tmp_path: Path):
        import warnings
        from scripts.kaggle_pipeline.audit_and_sync_manifest import (
            audit_dataset,
            validate_manifest_integrity,
        )

        # Include only 4 known subjects + unknown subjects + unlabelled gender + interval-only samples
        samples = []
        for i in range(12):
            subj_meta = {"subject_id": f"subj_{i % 4:03d}"} if i < 9 else {}
            declared_gender = None if i in (2, 7) else ("Female" if i % 2 == 0 else "Male")
            if i % 3 == 0:
                samples.append(
                    RawImageSample(
                        image_bytes=_make_textured_face_jpeg(seed=500 + i * 17),
                        source_dataset="adience",
                        raw_id=f"face_{i:03d}.jpg",
                        declared_age=None,
                        age_min=26.0,
                        age_max=30.0,
                        age_label_type="interval",
                        declared_gender=declared_gender,
                        declared_ethnicity="Caucasian",
                        metadata=subj_meta,
                    )
                )
            else:
                samples.append(
                    RawImageSample(
                        image_bytes=_make_textured_face_jpeg(seed=500 + i * 17),
                        source_dataset="agedb" if i < 9 else "fairface",
                        raw_id=f"{i}_Person{i % 4}_{20 + i}_f.jpg" if i < 9 else f"face_{i:03d}.jpg",
                        declared_age=float(18 + (i * 5) % 60),
                        declared_gender=declared_gender,
                        declared_ethnicity="East Asian" if i % 2 == 0 else "Caucasian",
                        metadata=subj_meta,
                    )
                )

        out_dir = tmp_path / "audit_verified_run"
        curator = ExtremeDatasetCurator(output_dir=out_dir, write_tar_shards=True)
        df, summary = curator.curate_samples(samples)
        assert summary.accepted >= 8

        # Unknown identities must be assigned to train with subject_id=None and identity_id=NaN
        unknown_rows = df[df["subject_id"].isna()]
        assert len(unknown_rows) >= 1
        assert set(unknown_rows["split"].unique()) == {"train"}
        assert unknown_rows["identity_id"].isna().all()

        # Unlabelled gender rows must have gender='unknown' and finite gender_prob=0.5
        unlabelled_rows = df[df["gender"] == "unknown"]
        assert len(unlabelled_rows) >= 1
        assert (unlabelled_rows["gender_prob"] == 0.5).all()

        # Validate manifest integrity using the repository's strict audit_and_sync_manifest gate
        integrity = validate_manifest_integrity(df)
        assert integrity["passed"], f"Manifest integrity failed: {integrity}"
        assert integrity["unlabelled_gender_rows"] >= 1

        # Validate full dataset + TAR shard byte/hash/quality verification with zero FutureWarnings
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            report = audit_dataset(out_dir / "golden_dataset")
        assert report["integrity_passed"], f"Dataset audit failed: {report}"
        assert report["dataset_violations"] == []
        assert report["master_matches_shards"] is True
        assert report["csv_matches_parquet"] is True

    def test_quality_auditor_boundary_rounding_and_clipping_guard(self):
        # Verify IEEE 754 float64 sum of rounded shadow_clip + highlight_clip never exceeds 0.08
        # e.g. 13 pixels < 8 (13 / 25600 = 0.0005078 -> 0.0005) and 2035 pixels > 240 (2035 / 25600 = 0.079492 -> 0.0795)
        # where raw sum is 2048 / 25600 = 0.08 exactly, but float64(0.0005) + float64(0.0795) == 0.08000000000000002 > 0.08!
        raw_bytes = _make_textured_face_jpeg(seed=901, size=160)
        bgr = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)
        assert bgr is not None
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l_flat = np.clip(lab[:, :, 0].reshape(-1), 20, 220)
        l_flat[:13] = 2
        l_flat[13 : 13 + 2035] = 250
        lab[:, :, 0] = l_flat.reshape(160, 160)
        crafted_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        # Force exact L-channel counts in case BGR->LAB roundtrip shifts 1-2 levels
        auditor = ExtremeQualityAuditor(min_laplacian_var=10.0, min_tenengrad=10.0)
        res = auditor.audit_face(crafted_bgr)
        if res.passed:
            assert float(np.float64(res.shadow_clip_fraction) + np.float64(res.highlight_clip_fraction)) <= 0.08

    def test_generated_kaggle_notebook_curation_passes_full_audit(self, tmp_path: Path):
        import json
        from scripts.kaggle_pipeline.audit_and_sync_manifest import (
            audit_dataset,
            validate_manifest_integrity,
        )

        nb_path = Path("notebooks/kaggle_face_mining_pipeline.ipynb")
        assert nb_path.is_file(), "Generated Kaggle notebook is missing"
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        code_cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
        assert len(code_cells) >= 6

        # Build synthetic multi-dataset /kaggle/input layout covering AgeDB, FG-NET, MORPH (CSV), APPA-REAL (_face.jpg), AxonAge
        fake_input = tmp_path / "kaggle_input"
        agedb_dir = fake_input / "agedb-database"
        agedb_dir.mkdir(parents=True)
        for idx in range(6):
            (agedb_dir / f"{idx}_Subject{idx}_{22 + idx * 4}_{'f' if idx % 2 == 0 else 'm'}.jpg").write_bytes(
                _make_textured_face_jpeg(seed=1000 + idx * 19)
            )

        appa_dir = fake_input / "appa-real-age-only-v1" / "train"
        appa_dir.mkdir(parents=True)
        (appa_dir / "000001.jpg_face.jpg").write_bytes(_make_textured_face_jpeg(seed=1201))
        pd.DataFrame([{"file_name": "000001.jpg", "apparent_age_avg": 29.4, "apparent_age_std": 2.1}]).to_csv(
            fake_input / "appa-real-age-only-v1" / "gt_avg_train.csv", index=False
        )

        morph_dir = fake_input / "morph" / "images"
        morph_dir.mkdir(parents=True)
        (morph_dir / "person99_01.jpg").write_bytes(_make_textured_face_jpeg(seed=1301))
        pd.DataFrame([{"id_num": "person99", "picture": "images/person99_01.jpg", "age": 43, "gender": "M", "race": "B"}]).to_csv(
            fake_input / "morph" / "morph_2008_nonCommercial.csv", index=False
        )

        axon_dir = fake_input / "face-recognition-age-estimation-dataset" / "31-35" / "female"
        axon_dir.mkdir(parents=True)
        (axon_dir / "selfie_01.jpg").write_bytes(_make_textured_face_jpeg(seed=1401))

        # Execute notebook cells 2..6 in an isolated namespace redirected to tmp_path
        ns = {
            "__builtins__": __builtins__,
            "os": __import__("os"),
            "sys": __import__("sys"),
            "time": __import__("time"),
            "math": math,
            "json": json,
            "shutil": __import__("shutil"),
            "hashlib": __import__("hashlib"),
            "tarfile": __import__("tarfile"),
            "zipfile": __import__("zipfile"),
            "io": __import__("io"),
            "re": __import__("re"),
            "Path": Path,
            "Dict": dict,
            "List": list,
            "Tuple": tuple,
            "Optional": Optional if "Optional" in globals() else object,
            "Iterator": object,
            "dataclass": __import__("dataclasses").dataclass,
            "np": np,
            "pd": pd,
            "cv2": cv2,
            "Image": __import__("PIL.Image").Image,
            "torch": torch,
        }
        from typing import Dict, List, Tuple, Optional, Iterator
        ns.update({"Dict": Dict, "List": List, "Tuple": Tuple, "Optional": Optional, "Iterator": Iterator})

        out_golden = tmp_path / "kaggle_working" / "golden_dataset"
        hb_file = tmp_path / "kaggle_working" / "telemetry_heartbeat.json"

        cell2_code = (
            code_cells[1]
            .replace("'/kaggle/working/golden_dataset'", repr(str(out_golden)))
            .replace("'/kaggle/input'", repr(str(fake_input)))
            .replace("min_free_gb=2.5", "min_free_gb=0.001")
        )
        cell3_code = (
            code_cells[2]
            .replace("min_lap=100.0, min_ten=250.0", "min_lap=100.0, min_ten=250.0")
        )
        cell4_code = (
            code_cells[3]
            .replace("'/kaggle/working/golden_dataset'", repr(str(out_golden)))
            .replace("'/kaggle/working/telemetry_heartbeat.json'", repr(str(hb_file)))
        )
        cell5_code = code_cells[4]
        cell6_code = code_cells[5]

        exec(compile(cell2_code, "<nb_cell_2>", "exec"), ns)
        exec(compile(cell3_code, "<nb_cell_3>", "exec"), ns)
        exec(compile(cell4_code, "<nb_cell_4>", "exec"), ns)
        exec(compile(cell5_code, "<nb_cell_5>", "exec"), ns)
        exec(compile(cell6_code, "<nb_cell_6>", "exec"), ns)

        df_manifest = ns["df_manifest"]
        assert len(df_manifest) >= 6
        assert set(df_manifest["source_dataset"].unique()).issuperset({"agedb", "appa_real", "morph", "axon_age"})

        integrity = validate_manifest_integrity(df_manifest)
        assert integrity["passed"], f"Notebook manifest integrity failed: {integrity}"

        audit_res = audit_dataset(out_golden)
        assert audit_res["integrity_passed"], f"Notebook dataset audit failed: {audit_res}"
        assert audit_res["dataset_violations"] == []
        assert audit_res["master_matches_shards"] is True
        assert audit_res["csv_matches_parquet"] is True

        # Also test notebook's embedded FaceGoldenDataset
        nb_ds = ns["FaceGoldenDataset"](
            manifest_path=out_golden / "metadata.parquet",
            shards_dir=out_golden / "shards",
            split="train",
        )
        assert len(nb_ds) >= 1
        sample0 = nb_ds[0]
        assert sample0["ldl_fine_23"].shape == (23,)
        nb_ds.close()

    def test_face_golden_dataset_security_and_age_mask(self, tmp_path: Path):
        outside = tmp_path / "secret.jpg"
        outside.write_bytes(_make_textured_face_jpeg(seed=777))

        img_dir = tmp_path / "ds_imgs"
        img_dir.mkdir()
        manifest_path = tmp_path / "manifest.csv"
        pd.DataFrame([
            {
                "image_id": "../secret.jpg",
                "age_continuous": 25.0,
                "age_min": 30.0,
                "age_max": 26.0,  # inverted bounds -> should be auto-ordered to [26, 30]
                "age_label_type": "binary",
                "has_valid_age": False,
                "gender": "Female",
                "split": "train",
            }
        ]).to_csv(manifest_path, index=False)

        ds = FaceGoldenDataset(manifest_path=manifest_path, images_dir=img_dir)
        item = ds[0]
        # Unsafe "../secret.jpg" must be blocked (returns zero placeholder -> -1.0 after normalization)
        assert torch.allclose(item["image"], torch.full((3, 160, 160), -1.0))
        # Binary age sample must have age_mask == 0.0 and ordered bounds [26.0, 30.0]
        assert float(item["age_mask"].item()) == 0.0
        assert (float(item["age_min"].item()), float(item["age_max"].item())) == (26.0, 30.0)
        ds.close()

    def test_nested_kaggle_layouts_fairface_adience_axon_b3fd(self, tmp_path: Path):
        from scripts.kaggle_pipeline.kaggle_dataset_connectors import (
            FairFaceConnector,
            AdienceConnector,
            AxonAgeConnector,
            B3FDConnector,
        )

        # 1. Nested FairFace (ghaidaalatoum/fairface -> fairface/fairface_label_train.csv + fairface/train/1.jpg)
        ff_root = tmp_path / "fairface_slug"
        (ff_root / "fairface" / "train").mkdir(parents=True)
        (ff_root / "fairface" / "train" / "1.jpg").write_bytes(_make_textured_face_jpeg(seed=801))
        pd.DataFrame([
            {"file": "train/1.jpg", "age": "20-29", "gender": "Female", "race": "East Asian"}
        ]).to_csv(ff_root / "fairface" / "fairface_label_train.csv", index=False)

        ff_conn = FairFaceConnector(ff_root, max_samples=10)
        assert ff_conn.is_available()
        ff_samples = list(ff_conn.stream_samples())
        assert len(ff_samples) == 1
        assert ff_samples[0].age_label_type == "interval"
        assert (ff_samples[0].age_min, ff_samples[0].age_max) == (20.0, 29.0)

        # 2. Nested Adience (ttungl/adience-benchmark-gender-and-age-classification -> depth-2 fold_0_data.txt)
        ad_root = tmp_path / "adience_slug"
        ad_inner = ad_root / "AdienceBenchmarkGenderAndAgeClassification" / "AdienceBenchmarkGenderAndAgeClassification"
        (ad_inner / "faces" / "12345@N01").mkdir(parents=True)
        (ad_inner / "faces" / "12345@N01" / "coarse_tilt_aligned_face.99.pic1.jpg").write_bytes(
            _make_textured_face_jpeg(seed=802)
        )
        pd.DataFrame([
            {"user_id": "12345@N01", "original_image": "pic1.jpg", "face_id": "99", "age": "(25, 32)", "gender": "m"}
        ]).to_csv(ad_inner / "fold_0_data.txt", sep="\t", index=False)

        ad_conn = AdienceConnector(ad_root, max_samples=10)
        assert ad_conn.is_available()
        ad_samples = list(ad_conn.stream_samples())
        assert len(ad_samples) == 1
        assert ad_samples[0].age_label_type == "interval"
        assert (ad_samples[0].age_min, ad_samples[0].age_max) == (25.0, 32.0)

        # 3. Nested AxonAge (axondata/face-recognition-age-estimation-dataset -> Face_Age_Estimation_Dataset_10-30/age_10/age_10_0001.jpeg)
        ax_root = tmp_path / "axon_slug"
        (ax_root / "Face_Age_Estimation_Dataset_10-30" / "age_10").mkdir(parents=True)
        (ax_root / "Face_Age_Estimation_Dataset_10-30" / "age_10" / "age_10_0001.jpeg").write_bytes(
            _make_textured_face_jpeg(seed=803)
        )
        ax_conn = AxonAgeConnector(ax_root, max_samples=10)
        assert ax_conn.is_available()
        ax_samples = list(ax_conn.stream_samples())
        assert len(ax_samples) == 1
        assert ax_samples[0].declared_age == 10.0
        assert (ax_samples[0].age_min, ax_samples[0].age_max) == (9.0, 11.0)

        # 4. Nested B3FD (moritzm00/biometrically-filtered-famous-figure-dataset -> b3fd/B3FD/celeb/1001_1990_2020.jpg)
        b3_root = tmp_path / "b3fd_slug"
        (b3_root / "b3fd" / "B3FD" / "celeb_a").mkdir(parents=True)
        (b3_root / "b3fd" / "B3FD" / "celeb_a" / "1001_1990_2020.jpg").write_bytes(
            _make_textured_face_jpeg(seed=804)
        )
        b3_conn = B3FDConnector(b3_root, max_samples=10)
        assert b3_conn.is_available()
        b3_samples = list(b3_conn.stream_samples())
        assert len(b3_samples) == 1
        assert b3_samples[0].declared_age == 30.0


class TestAdversarialFixesAndEndToEndIntervals:
    """Verifies fixes for bounded kernel generation, connector interleaving, cross-dataset celebrity leakage, real_age priority, and inference intervals."""

    def test_bounded_verification_kernel_replaces_max_curated_and_max_samples(self, tmp_path: Path):
        import json
        from scripts.kaggle_pipeline.build_kaggle_kernel import generate_bounded_verification_kernel

        verify_dir = tmp_path / "verify_kernel"
        nb_path = generate_bounded_verification_kernel(
            verify_dir=verify_dir,
            max_samples_per_dataset=123,
            max_curated=321,
        )
        assert nb_path.exists()
        nb_data = json.loads(nb_path.read_text(encoding="utf-8"))
        all_src = "\n".join("".join(c.get("source", [])) for c in nb_data["cells"])
        assert "MAX_CURATED = 321" in all_src
        assert "max_samples_per_dataset=123" in all_src

    def test_adience_axon_afad_interleaved_subject_and_age_streaming(self, tmp_path: Path):
        from scripts.kaggle_pipeline.kaggle_dataset_connectors import (
            AdienceConnector,
            AxonAgeConnector,
            AFADConnector,
        )

        # 1. Adience: 3 user_ids with 5 rows each in fold_0_data.txt -> first 3 samples must come from 3 distinct user_ids
        ad_root = tmp_path / "adience_interleave"
        rows = []
        for u_idx in range(3):
            uid = f"user_{u_idx}@N01"
            udir = ad_root / "faces" / uid
            udir.mkdir(parents=True)
            for f_idx in range(5):
                fname = f"coarse_tilt_aligned_face.{f_idx}.img{f_idx}.jpg"
                (udir / fname).write_bytes(_make_textured_face_jpeg(seed=900 + u_idx * 10 + f_idx))
                rows.append({
                    "user_id": uid,
                    "original_image": f"img{f_idx}.jpg",
                    "face_id": str(f_idx),
                    "age": "(25, 32)",
                    "gender": "f",
                })
        pd.DataFrame(rows).to_csv(ad_root / "fold_0_data.txt", sep="\t", index=False)

        ad_samples = list(AdienceConnector(ad_root, max_samples=6).stream_samples())
        assert len(ad_samples) == 6
        first_three_subjects = {s.metadata["user_id"] for s in ad_samples[:3]}
        assert len(first_three_subjects) == 3

        # 2. AxonAge: 3 age folders with 4 images each -> first 3 samples must come from 3 distinct ages
        ax_root = tmp_path / "axon_interleave"
        for age in (10, 20, 30):
            adir = ax_root / f"age_{age}"
            adir.mkdir(parents=True)
            for i in range(4):
                (adir / f"age_{age}_{i:04d}.jpg").write_bytes(_make_textured_face_jpeg(seed=950 + age + i))
        ax_samples = list(AxonAgeConnector(ax_root, max_samples=6).stream_samples())
        assert len(ax_samples) == 6
        assert {s.declared_age for s in ax_samples[:3]} == {10.0, 20.0, 30.0}

        # 3. AFAD: 3 age folders with 4 images each -> first 3 samples must come from 3 distinct ages
        af_root = tmp_path / "afad_interleave"
        for age in (18, 28, 38):
            gdir = af_root / str(age) / "111"
            gdir.mkdir(parents=True)
            for i in range(4):
                (gdir / f"img_{i:04d}.jpg").write_bytes(_make_textured_face_jpeg(seed=980 + age + i))
        af_samples = list(AFADConnector(af_root, max_samples=6).stream_samples())
        assert len(af_samples) == 6
        assert {s.declared_age for s in af_samples[:3]} == {18.0, 28.0, 38.0}

    def test_cross_dataset_celebrity_identity_split_canonicalization(self):
        from src.mining.curation import canonicalize_split_identity_key, deterministic_identity_split

        keys = [
            "agedb:MariaCallas",
            "b3fd:Maria_Callas",
            "cacd:Maria Callas",
            "imdb_wiki:maria-callas",
        ]
        canon = {canonicalize_split_identity_key(k) for k in keys}
        assert canon == {"celebrity:mariacallas"}
        splits = {deterministic_identity_split(k) for k in keys}
        assert len(splits) == 1

    def test_curator_prioritizes_chronological_real_age_over_apparent_age(self, tmp_path: Path):
        curator = ExtremeDatasetCurator(output_dir=tmp_path / "curated_real_age")
        sample = RawImageSample(
            source_dataset="appa_real",
            raw_id="appa_0001.jpg",
            image_bytes=_make_textured_face_jpeg(seed=1501),
            declared_age=24.0,
            declared_gender="Female",
            age_min=24.0,
            age_max=24.0,
            age_label_type="exact",
            metadata={"real_age": 32.0, "apparent_age_std": 2.0, "subject_id": "appa_subj_1"},
        )
        df, _ = curator.curate_samples([sample])
        assert len(df) == 1
        row = df.iloc[0]
        assert float(row["age_continuous"]) == 32.0
        assert float(row["apparent_age"]) == 24.0
        assert row["age_interval"] == "31-35"
        assert (float(row["age_min"]), float(row["age_max"])) == (31.0, 35.0)

    def test_inference_backbone_target_locker_server_client_and_edge_return_fine_intervals(self):
        from src.biometrics.estimation.backbone import BiometricMultiTaskModel
        from src.biometrics.tracking.target_locker import TargetLocker
        from scripts.kaggle_pipeline.kaggle_inference_server import process_face_crops
        from src.biometrics.inference.cloud_client import CloudInferenceClient

        model = BiometricMultiTaskModel(backbone_name="convnextv2_nano", pretrained=False)
        model.eval()
        x = torch.zeros((2, 3, 160, 160), dtype=torch.float32)
        with torch.no_grad():
            out = model(x)
        assert "age_interval_probs" in out
        assert out["age_interval_probs"].shape == (2, 23)
        assert "age_interval_idx" in out
        assert out["age_interval_idx"].shape == (2,)

        locker = TargetLocker()
        lock_res = locker.update(track_id=7, age_pred=28.4, aleatoric_std=1.2, gender_logit=2.0)
        assert lock_res["locked_age_interval"] == "26-30"
        assert lock_res["locked_age_interval_idx"] == 8
        assert lock_res["locked_age_bounds"] == (26, 30)

        jpeg_bytes = _make_textured_face_jpeg(seed=1601)
        srv_res = process_face_crops(model=model, crop_bytes_list=[jpeg_bytes], device="cpu")
        assert len(srv_res) == 1
        assert "age_interval" in srv_res[0]
        assert "age_interval_idx" in srv_res[0]
        assert "age_interval_conf" in srv_res[0]
        assert "age_min" in srv_res[0] and "age_max" in srv_res[0]

        client = CloudInferenceClient()
        dummy_crop = np.full((160, 160, 3), 120, dtype=np.uint8)
        dets = [{"bbox": (10.0, 10.0, 90.0, 90.0), "score": 0.95, "crop": dummy_crop}]
        tracks = client.update_local_tracks(dets, srv_res)
        assert len(tracks) == 1
        assert "locked_age_interval" in tracks[0]
        assert "locked_age_interval_idx" in tracks[0]

    def test_wikimedia_commons_agedb_classwise_imdb_csv_and_utk_ban(self, tmp_path: Path):
        from src.mining import AgeDBConnector, IMDBWikiConnector, WikimediaCommonsConnector

        payload = _make_textured_face_jpeg(seed=1701)

        # 1. WikimediaCommonsConnector: biographical filename, crawler filename, and CSV manifest
        wm_dir = tmp_path / "wikimedia-commons-portraits"
        wm_dir.mkdir(parents=True)
        (wm_dir / "AdaLovelace_1815_1843_f.jpg").write_bytes(payload)
        (wm_dir / "crawled_q000_001_99999.jpg").write_bytes(payload)
        (wm_dir / "csv_portrait.jpg").write_bytes(payload)
        pd.DataFrame([
            {"file": "csv_portrait.jpg", "age": 44, "gender": "M", "subject_id": "Q42"}
        ]).to_csv(wm_dir / "commons_manifest.csv", index=False)

        wm_conn = WikimediaCommonsConnector(wm_dir)
        assert wm_conn.is_available()
        wm_samples = list(wm_conn.stream_samples())
        assert len(wm_samples) == 3
        by_raw = {s.raw_id: s for s in wm_samples}
        assert by_raw["AdaLovelace_1815_1843_f.jpg"].declared_age == 28.0
        assert by_raw["AdaLovelace_1815_1843_f.jpg"].declared_gender == "Female"
        assert by_raw["AdaLovelace_1815_1843_f.jpg"].age_interval_name == "26-30"
        assert by_raw["csv_portrait.jpg"].declared_age == 44.0
        assert by_raw["csv_portrait.jpg"].declared_gender == "Male"
        assert by_raw["csv_portrait.jpg"].age_interval_name == "41-45"

        # 2. AgeDBConnector classwise age folders (shukdevdatta/agedb-classwise-dataset)
        agedb_cw_dir = tmp_path / "agedb-classwise-dataset" / "AgeDB"
        for age_folder in (18, 25, 32, 45, 60, 75):
            d = agedb_cw_dir / str(age_folder)
            d.mkdir(parents=True)
            (d / f"face_{age_folder}.jpg").write_bytes(_make_textured_face_jpeg(seed=1750 + age_folder))

        agedb_conn = AgeDBConnector(tmp_path / "agedb-classwise-dataset")
        assert agedb_conn.is_available()
        cw_samples = list(agedb_conn.stream_samples())
        assert len(cw_samples) == 6
        assert {s.declared_age for s in cw_samples} == {18.0, 25.0, 32.0, 45.0, 60.0, 75.0}

        # 3. IMDBWikiConnector CSV fallback (yuulind/imdb-clean)
        imdb_dir = tmp_path / "imdb-clean"
        imdb_dir.mkdir(parents=True)
        (imdb_dir / "celeb_01.jpg").write_bytes(payload)
        pd.DataFrame([
            {"filename": "celeb_01.jpg", "age": 37, "gender": "F", "name": "Jane Doe"}
        ]).to_csv(imdb_dir / "imdb_clean_labels.csv", index=False)

        imdb_conn = IMDBWikiConnector(imdb_dir)
        assert imdb_conn.is_available()
        imdb_samples = list(imdb_conn.stream_samples())
        assert len(imdb_samples) == 1
        assert imdb_samples[0].declared_age == 37.0
        assert imdb_samples[0].declared_gender == "Female"
        assert imdb_samples[0].age_interval_name == "36-40"

        # 4. UTKFace directory present on disk must be ignored when exclude_utk=True / curate_directory
        utk_dir = tmp_path / "utkface-new"
        utk_dir.mkdir(parents=True)
        (utk_dir / "25_0_0_20170116174525125.jpg").write_bytes(payload)

        harvester = InCloudStreamingHarvester(kaggle_input_dir=str(tmp_path), exclude_utk=True)
        mounted_names = {c.get_name() for c in harvester.connectors}
        assert "utkface" not in mounted_names
        assert {"wikimedia_commons", "agedb", "imdb_wiki"}.issubset(mounted_names)

    def test_curate_directory_end_to_end_and_rejects_banned_utk_and_invalid_ages(self, tmp_path: Path):
        input_root = tmp_path / "mounted_datasets"
        # 1. Valid AgeDB dataset with centenarian (age=101) and adult (age=30)
        agedb_dir = input_root / "agedb-database"
        agedb_dir.mkdir(parents=True)
        (agedb_dir / "1_Alice_30_f.jpg").write_bytes(_make_textured_face_jpeg(seed=2101))
        (agedb_dir / "2_Bob_101_m.jpg").write_bytes(_make_textured_face_jpeg(seed=2102))

        # 2. Banned UTKFace folder on disk (must be ignored by curate_directory)
        utk_dir = input_root / "utkface-new"
        utk_dir.mkdir(parents=True)
        (utk_dir / "28_0_0_20170116174525125.jpg").write_bytes(_make_textured_face_jpeg(seed=2103))

        curator = ExtremeDatasetCurator(output_dir=tmp_path / "curated_out")
        df, summary = curator.curate_directory(input_dir=input_root, max_samples_per_dataset=50)
        assert len(df) == 2
        assert "utkface" not in set(df["source_dataset"].unique())
        ages = sorted(df["age_continuous"].tolist())
        assert ages == [30.0, 100.0]
        intervals = set(df["age_interval"].tolist())
        assert intervals == {"26-30", "96-100"}
        assert summary.accepted == 2

    def test_csv_connectors_reject_negative_bounds_nan_columns_and_fallback_cleanly(self, tmp_path: Path):
        from src.mining import AxonAgeConnector, IMDBWikiConnector, LagendaConnector, WikimediaCommonsConnector

        wm_dir = tmp_path / "wm_edge"
        wm_dir.mkdir(parents=True)
        for i in range(1, 7):
            (wm_dir / f"img_{i}.jpg").write_bytes(_make_textured_face_jpeg(seed=2200 + i))

        pd.DataFrame([
            # Row 1: negative chronological age (capture_year < birth_year) -> must be REJECTED even if age=28 is present
            {"file": "img_1.jpg", "birth_year": 1990, "capture_year": 1980, "age": 28, "gender": "F"},
            # Row 2: non-numeric birth_year -> must fallback to age=35
            {"file": np.nan, "filename": "img_2.jpg", "birth_year": "unknown", "capture_year": "unknown", "age": 35, "gender": "M"},
            # Row 3: negative explicit age_min/age_max -> must be REJECTED (not clamped to 0.0)
            {"file": "img_3.jpg", "age_min": -10, "age_max": -2, "gender": "F"},
            # Row 4: non-numeric age_min/age_max -> must fallback to age=42 without ValueError
            {"file": "img_4.jpg", "age_min": "bad", "age_max": "bad", "age": 42, "gender": "F"},
            # Row 5: valid chronological birth_year=1980, capture_year=2020 (no age column) -> must be 40.0 (not 28.0)
            {"file": "img_5.jpg", "birth_year": 1980, "capture_year": 2020, "gender": "M"},
            # Row 6: centenarian chronological birth_year=1915, capture_year=2020 -> age 105 clamped to 100.0 (96-100)
            {"file": "img_6.jpg", "birth_year": 1915, "capture_year": 2020, "gender": "F"},
        ]).to_csv(wm_dir / "manifest.csv", index=False)

        for conn_cls in (WikimediaCommonsConnector, IMDBWikiConnector, AxonAgeConnector, LagendaConnector):
            samples = {s.raw_id: s for s in conn_cls(wm_dir).stream_samples()}
            assert "img_1.jpg" not in samples, f"{conn_cls.__name__} failed to reject negative chronological age"
            assert "img_3.jpg" not in samples, f"{conn_cls.__name__} failed to reject negative age_min/age_max"
            assert samples["img_2.jpg"].declared_age == 35.0
            assert samples["img_4.jpg"].declared_age == 42.0
            assert samples["img_5.jpg"].declared_age == 40.0
            assert samples["img_6.jpg"].declared_age == 100.0
            assert samples["img_6.jpg"].age_interval_name == "96-100"

    def test_agedb_classwise_interleaving_duplicate_filenames_and_centenarian_age(self, tmp_path: Path):
        from src.mining import AgeDBConnector

        cw_root = tmp_path / "agedb_cw_3folders" / "AgeDB"
        for idx, age in enumerate((18, 25, 32)):
            adir = cw_root / str(age)
            adir.mkdir(parents=True)
            # Identical filenames 0001.jpg and 0002.jpg across all 3 age folders
            (adir / "0001.jpg").write_bytes(_make_textured_face_jpeg(seed=2300 + idx * 10 + 1))
            (adir / "0002.jpg").write_bytes(_make_textured_face_jpeg(seed=2300 + idx * 10 + 2))

        conn = AgeDBConnector(tmp_path / "agedb_cw_3folders", max_samples=10)
        assert conn.is_available()
        samples = list(conn.stream_samples())
        assert len(samples) == 6
        # First 3 samples must come from 3 distinct age folders via round-robin interleaving
        assert {s.declared_age for s in samples[:3]} == {18.0, 25.0, 32.0}

    def test_celeba_kaggle_csv_and_nested_img_align_celeba(self, tmp_path: Path):
        from src.mining import CelebAConnector

        celeba_root = tmp_path / "celeba-dataset"
        nested_img_dir = celeba_root / "img_align_celeba" / "img_align_celeba"
        nested_img_dir.mkdir(parents=True)
        (nested_img_dir / "000001.jpg").write_bytes(_make_textured_face_jpeg(seed=2401))

        pd.DataFrame([
            {"image_id": "000001.jpg", "Male": -1, "Young": 1}
        ]).to_csv(celeba_root / "list_attr_celeba.csv", index=False)
        pd.DataFrame([
            {
                "image_id": "000001.jpg",
                "lefteye_x": 69, "lefteye_y": 109,
                "righteye_x": 106, "righteye_y": 113,
                "nose_x": 77, "nose_y": 142,
                "leftmouth_x": 73, "leftmouth_y": 152,
                "rightmouth_x": 108, "rightmouth_y": 154,
            }
        ]).to_csv(celeba_root / "list_landmarks_align_celeba.csv", index=False)

        conn = CelebAConnector(celeba_root, max_samples=5)
        assert conn.is_available()
        samples = list(conn.stream_samples())
        assert len(samples) == 1
        assert samples[0].declared_gender == "Female"
        assert samples[0].declared_landmarks is not None
        assert samples[0].declared_landmarks.shape == (5, 2)

    def test_generated_kaggle_notebook_connector_and_harvester_parity(self, tmp_path: Path):
        import csv
        import hashlib
        import io
        import json
        import os
        import re
        import shutil
        import tarfile
        import time
        import zipfile
        from dataclasses import dataclass, field
        from typing import Any, Dict, Generator, List, Optional, Tuple
        from PIL import Image

        nb_path = Path("notebooks/kaggle_face_mining_pipeline.ipynb")
        assert nb_path.exists()
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        code_cells = ["".join(c["source"]) for c in nb["cells"] if c.get("cell_type") == "code"]
        assert len(code_cells) >= 2

        ns: dict = {
            "os": os, "re": re, "io": io, "csv": csv, "math": math, "json": json,
            "time": time, "shutil": shutil, "zipfile": zipfile, "tarfile": tarfile,
            "hashlib": hashlib, "Path": Path, "dataclass": dataclass, "field": field,
            "Optional": Optional, "List": List, "Dict": Dict, "Tuple": Tuple,
            "Any": Any, "Generator": Generator, "np": np, "pd": pd, "cv2": cv2,
            "Image": Image,
        }
        cell2_lines = [
            ln for ln in code_cells[1].splitlines()
            if not ln.strip().startswith("quota_mgr =") and not ln.strip().startswith("harvester =")
            and "harvester.connectors" not in ln
        ]
        exec("\n".join(cell2_lines), ns)

        HarvesterCls = ns["InCloudStreamingHarvester"]
        assert HarvesterCls(kaggle_input_dir=str(tmp_path)).max_samples_per_dataset == 50000
        assert ns["AllAgeFacesConnector"](tmp_path).get_name() == "aaf"
        assert "CelebAConnector" in ns

        # Verify notebook WikimediaCommonsConnector does not fabricate age=28.0 and rejects negative chronological ages
        wm_dir = tmp_path / "wikimedia_nb"
        wm_dir.mkdir(parents=True)
        (wm_dir / "p1.jpg").write_bytes(_make_textured_face_jpeg(seed=2501))
        (wm_dir / "p2.jpg").write_bytes(_make_textured_face_jpeg(seed=2502))
        pd.DataFrame([
            {"file": "p1.jpg", "birth_year": 1980, "capture_year": 2020, "gender": "F"},
            {"file": "p2.jpg", "birth_year": 2000, "capture_year": 1990, "gender": "M"},
        ]).to_csv(wm_dir / "wm.csv", index=False)
        wm_samples = list(ns["WikimediaCommonsConnector"](wm_dir).stream_samples())
        assert len(wm_samples) == 1
        assert wm_samples[0].raw_id == "p1.jpg"
        assert wm_samples[0].declared_age == 40.0

    def test_non_real_sources_and_binary_pseudo_age_rejected_everywhere(self, tmp_path: Path):
        from src.mining import (
            BANNED_NON_REAL_SOURCE_PATTERNS,
            ExtremeDatasetCurator,
            InCloudStreamingHarvester,
            is_banned_non_real_source,
        )

        assert len(BANNED_NON_REAL_SOURCE_PATTERNS) >= 35
        for banned in (
            "utkface",
            "utkface-new",
            "stylegan2-ffhq",
            "ffhq-dataset",
            "digiface-1m",
            "dcface-synthetic",
            "sfhq-part1",
            "thispersondoesnotexist",
            "cartoonset-100k",
            "anime-faces",
            "cuhk-face-sketch",
            "deepfake-detection",
            "faceforensics-pp",
            "facemorpher-attack",
            "300w-lp",
            "synface",
            "idiff-face",
            "generated-photos",
        ):
            is_banned, reason = is_banned_non_real_source(banned)
            assert is_banned is True and reason is not None, f"Failed to ban non-real source {banned}"

        for real_src in (
            "agedb",
            "morph",
            "fgnet",
            "afad",
            "appa_real",
            "fairface",
            "adience",
            "axon_age",
            "megaage_asian",
            "lagenda",
            "cacd",
            "aaf",
            "b3fd",
            "imdb_wiki",
            "wikimedia_commons",
        ):
            is_banned, reason = is_banned_non_real_source(real_src)
            assert is_banned is False and reason is None, f"Falsely banned real source {real_src}"

        # Harvester must skip any directory matching banned non-real patterns even if it contains valid-looking files
        input_root = tmp_path / "kaggle_input_non_real"
        for dname in ("stylegan-agedb", "anime-fairface", "deepfake-morph", "utkface-agedb"):
            d = input_root / dname
            d.mkdir(parents=True)
            (d / "1_Alice_30_f.jpg").write_bytes(_make_textured_face_jpeg(seed=2601))
        real_dir = input_root / "agedb-database"
        real_dir.mkdir(parents=True)
        (real_dir / "1_Alice_30_f.jpg").write_bytes(_make_textured_face_jpeg(seed=2602))

        harvester = InCloudStreamingHarvester(kaggle_input_dir=str(input_root), exclude_utk=True)
        assert [c.get_name() for c in harvester.connectors] == ["agedb"]

        # Curator must reject non-real sources and binary pseudo-age (celeba)
        curator = ExtremeDatasetCurator(output_dir=tmp_path / "curated_non_real_check")
        samples = [
            RawImageSample(
                source_dataset="stylegan_ffhq",
                raw_id="synth_01.jpg",
                image_bytes=_make_textured_face_jpeg(seed=2603),
                declared_age=25.0,
                declared_gender="Female",
            ),
            RawImageSample(
                source_dataset="celeba",
                raw_id="000001.jpg",
                image_bytes=_make_textured_face_jpeg(seed=2604),
                declared_age=24.0,
                age_min=18.0,
                age_max=34.0,
                age_label_type="binary",
                declared_gender="Female",
            ),
            RawImageSample(
                source_dataset="agedb",
                raw_id="1_RealPerson_30_f.jpg",
                image_bytes=_make_textured_face_jpeg(seed=2605),
                declared_age=30.0,
                declared_gender="Female",
            ),
        ]
        df, summary = curator.curate_samples(samples)
        assert len(df) == 1
        assert df.iloc[0]["source_dataset"] == "agedb"
        assert summary.rejection_breakdown.get("BANNED_SOURCE_NON_REAL", 0) == 1
        assert summary.rejection_breakdown.get("REJECTED_BINARY_PSEUDO_AGE", 0) == 1

    def test_quality_auditor_rejects_flat_cartoon_sketch_and_blank_renders(self):
        from scripts.kaggle_pipeline.kaggle_face_engine import ExtremeQualityAuditor as EngineAuditor

        local_auditor = ExtremeQualityAuditor()
        engine_auditor = EngineAuditor()

        # 1. Flat-shaded cartoon / vector avatar (high boundary edges in outer border so Laplacian & Tenengrad pass, but interior is flat)
        cartoon = np.full((160, 160, 3), (150, 180, 210), dtype=np.uint8)
        for r in (62, 66, 70, 74):
            cv2.rectangle(cartoon, (80 - r, 80 - r), (80 + r, 80 + r), (20, 30, 230), 1)
        res_local_cartoon = local_auditor.audit_face(cartoon)
        res_eng_cartoon = engine_auditor.audit_face(cartoon)
        assert not res_local_cartoon.passed
        assert any("SYNTHETIC_FLAT_SHADING_OR_CARTOON" in r for r in res_local_cartoon.rejection_reasons)
        assert not res_eng_cartoon["passed"]
        assert any("SYNTHETIC_FLAT_SHADING_OR_CARTOON" in r for r in res_eng_cartoon["reasons"])

        # 2. Monochromatic pencil/ink sketch (achromatic paper + dense dark ink hatching)
        sketch = np.full((160, 160, 3), 215, dtype=np.uint8)
        for y in range(10, 150, 4):
            cv2.line(sketch, (15, y), (145, y + (y % 3)), (45, 45, 45), 1)
        for x in range(10, 150, 5):
            cv2.line(sketch, (x, 15), (x, 145), (55, 55, 55), 1)
        res_local_sketch = local_auditor.audit_face(sketch)
        res_eng_sketch = engine_auditor.audit_face(sketch)
        assert not res_local_sketch.passed
        assert any(
            ("POSTERIZED_OR_SKETCH_HISTOGRAM" in r or "INSUFFICIENT_TONAL_ENTROPY_NON_REAL" in r)
            for r in res_local_sketch.rejection_reasons
        )
        assert not res_eng_sketch["passed"]

    def test_nested_megaage_lagenda_and_gendered_fgnet_connectors(self, tmp_path: Path):
        from src.mining import FGNetConnector, LagendaConnector, MegaAgeAsianConnector
        from scripts.kaggle_pipeline.kaggle_identity_split import extract_subject_key

        payload1 = _make_textured_face_jpeg(seed=2701)
        payload2 = _make_textured_face_jpeg(seed=2702)

        # 1. Nested MegaAge (baopmessi/megaage -> megaage_asian/megaage_asian/list/{train,test}_name.txt + duplicate 1.jpg in train/ and test/)
        mega_root = tmp_path / "megaage_slug"
        mega_inner = mega_root / "megaage_asian" / "megaage_asian"
        (mega_inner / "list").mkdir(parents=True)
        (mega_inner / "train").mkdir(parents=True)
        (mega_inner / "test").mkdir(parents=True)
        (mega_inner / "train" / "1.jpg").write_bytes(payload1)
        (mega_inner / "test" / "1.jpg").write_bytes(payload2)
        (mega_inner / "list" / "train_name.txt").write_text("1.jpg\n", encoding="utf-8")
        (mega_inner / "list" / "train_age.txt").write_text("23\n", encoding="utf-8")
        (mega_inner / "list" / "test_name.txt").write_text("1.jpg\n", encoding="utf-8")
        (mega_inner / "list" / "test_age.txt").write_text("47\n", encoding="utf-8")

        mega_conn = MegaAgeAsianConnector(mega_root, max_samples=10)
        assert mega_conn.is_available()
        mega_samples = list(mega_conn.stream_samples())
        assert len(mega_samples) == 2
        assert {s.declared_age for s in mega_samples} == {23.0, 47.0}

        # 2. Nested LagendA (maiabboud/lagenda -> subset_1gb/test.csv + subset_1gb/test/lag_benchmark/img_01.jpg)
        lag_root = tmp_path / "lagenda_slug"
        (lag_root / "subset_1gb" / "test" / "lag_benchmark").mkdir(parents=True)
        (lag_root / "subset_1gb" / "test" / "lag_benchmark" / "img_01.jpg").write_bytes(payload1)
        pd.DataFrame([
            {"img_name": "test/lag_benchmark/img_01.jpg", "age": 34, "gender": "F"}
        ]).to_csv(lag_root / "subset_1gb" / "test.csv", index=False)

        lag_conn = LagendaConnector(lag_root, max_samples=10)
        assert lag_conn.is_available()
        lag_samples = list(lag_conn.stream_samples())
        assert len(lag_samples) == 1
        assert lag_samples[0].declared_age == 34.0
        assert lag_samples[0].declared_gender == "Female"

        # 3. Gendered FG-NET with duplicate letter suffix (001A43aM.JPG and 002A12b_f.jpg)
        fg_root = tmp_path / "fgnet_gendered"
        fg_root.mkdir(parents=True)
        (fg_root / "001A43aM.JPG").write_bytes(payload1)
        (fg_root / "002A12b_f.jpg").write_bytes(payload2)
        fg_conn = FGNetConnector(fg_root, max_samples=10)
        fg_samples = {s.raw_id: s for s in fg_conn.stream_samples()}
        assert len(fg_samples) == 2
        assert fg_samples["001A43aM.JPG"].declared_age == 43.0
        assert fg_samples["001A43aM.JPG"].declared_gender == "Male"
        assert fg_samples["001A43aM.JPG"].metadata["subject_id"] == "001"
        assert fg_samples["002A12b_f.jpg"].declared_age == 12.0
        assert fg_samples["002A12b_f.jpg"].declared_gender == "Female"
        assert extract_subject_key("fgnet", "001A43aM.JPG") == "001"

    def test_manifest_integrity_validates_fine_intervals_and_forbids_non_real_sources(self, tmp_path: Path):
        from scripts.kaggle_pipeline.audit_and_sync_manifest import validate_manifest_integrity

        curator = ExtremeDatasetCurator(output_dir=tmp_path / "integrity_interval_check")
        samples = [
            RawImageSample(
                source_dataset="agedb",
                raw_id=f"{i}_Person{i}_{20 + i * 5}_{'f' if i % 2 == 0 else 'm'}.jpg",
                image_bytes=_make_textured_face_jpeg(seed=2800 + i),
                declared_age=float(20 + i * 5),
                declared_gender="Female" if i % 2 == 0 else "Male",
                metadata={"subject_id": f"subj_{i}"},
            )
            for i in range(6)
        ]
        df, _ = curator.curate_samples(samples)
        assert validate_manifest_integrity(df, forbid_non_real_sources=True)["passed"] is True

        # Corrupt age_interval_idx -> must fail
        df_bad_iv = df.copy()
        df_bad_iv.loc[0, "age_interval_idx"] = 22
        res_bad_iv = validate_manifest_integrity(df_bad_iv, forbid_non_real_sources=True)
        assert res_bad_iv["passed"] is False
        assert any("inconsistent 3-5 year interval" in v for v in res_bad_iv["violations"])

        # Corrupt source_dataset with non-real source -> must fail when forbid_non_real_sources=True
        df_bad_src = df.copy()
        df_bad_src.loc[0, "source_dataset"] = "stylegan_ffhq"
        res_bad_src = validate_manifest_integrity(df_bad_src, forbid_non_real_sources=True)
        assert res_bad_src["passed"] is False
        assert any("banned non-real" in v for v in res_bad_src["violations"])

    def test_plastic_skin_microtexture_pose_invariance_and_dedup_embeddings(self):
        from src.biometrics.core.geometry import ARCFACE_CANONICAL_160, estimate_head_pose_pnp
        from src.mining.filtering.quality_auditor import ExtremeQualityAuditor
        from scripts.kaggle_pipeline.kaggle_face_engine import (
            ExtremeQualityAuditor as EngineAuditor,
            PerceptualDeduplicator as EngineDedup,
        )

        # 1. Plastic / CGI skin with sharp drawn eyes/hair (high global Laplacian & Tenengrad, high smooth gradient, zero cheek micro-texture)
        yy, xx = np.mgrid[0:160, 0:160].astype(np.float32)
        b_ch = np.clip(110.0 + 0.45 * xx + 0.35 * yy, 40, 220).astype(np.uint8)
        g_ch = np.clip(125.0 + 0.40 * xx - 0.30 * (yy - 80), 40, 220).astype(np.uint8)
        r_ch = np.clip(145.0 - 0.35 * (xx - 80) + 0.42 * yy, 40, 220).astype(np.uint8)
        cgi_face = np.stack([b_ch, g_ch, r_ch], axis=-1)
        # Add sharp drawn eyes and eyebrows outside cheek ROI so Laplacian >= 100 and Tenengrad >= 250
        for y in range(25, 65, 3):
            cv2.line(cgi_face, (25, y), (135, y), (20, 25, 30), 1)
        local_auditor = ExtremeQualityAuditor()
        eng_auditor = EngineAuditor()
        res_local = local_auditor.audit_face(cgi_face, ARCFACE_CANONICAL_160)
        res_eng = eng_auditor.audit_face(cgi_face, ARCFACE_CANONICAL_160)
        assert not res_local.passed
        assert any("SYNTHETIC_PLASTIC_SKIN_MICROTEXTURE_NON_REAL" in r for r in res_local.rejection_reasons)
        assert not res_eng["passed"]
        assert any("SYNTHETIC_PLASTIC_SKIN_MICROTEXTURE_NON_REAL" in r for r in res_eng["reasons"])

        # 2. Translation & scale invariance of solvePnP head pose on off-center portrait faces
        yaw_c, pitch_c, roll_c = estimate_head_pose_pnp(ARCFACE_CANONICAL_160, 160, 160)
        off_center_lms = ARCFACE_CANONICAL_160 * 0.5 + np.array([50.0, 45.0], dtype=np.float32)
        yaw_o, pitch_o, roll_o = estimate_head_pose_pnp(off_center_lms, 500, 500)
        assert abs(yaw_c - yaw_o) < 0.5
        assert abs(pitch_c - pitch_o) < 0.5
        assert abs(roll_c - roll_o) < 0.5
        assert abs(pitch_o) < 5.0

        # 3. Zero-centered 512d deduplicator embeddings distinguish distinct faces without false positives
        dedup = EngineDedup(hamming_threshold=6, cosine_threshold=0.35)
        f1 = cv2.imdecode(np.frombuffer(_make_textured_face_jpeg(seed=3101), np.uint8), cv2.IMREAD_COLOR)
        f2 = cv2.imdecode(np.frombuffer(_make_textured_face_jpeg(seed=3102), np.uint8), cv2.IMREAD_COLOR)
        h1 = dedup.compute_phash_uint64(cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY))
        h2 = dedup.compute_phash_uint64(cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY))
        emb1 = dedup.extract_embedding_512d(f1)
        emb2 = dedup.extract_embedding_512d(f2)
        assert dedup.cosine_distance(emb1, emb2) >= 0.35
        is_dup1, _ = dedup.is_duplicate_and_register(h1, emb1)
        is_dup2, _ = dedup.is_duplicate_and_register(h2, emb2)
        assert is_dup1 is False
        assert is_dup2 is False

        # 4. Generated Kaggle notebook ExtremeQualityAuditor rejects grain-injected flat-shaded cartoon
        import json
        nb_data = json.loads(Path("notebooks/kaggle_face_mining_pipeline.ipynb").read_text(encoding="utf-8"))
        cell3_src = "".join(nb_data["cells"][3]["source"])
        nb_ns = {"np": np, "cv2": cv2, "math": math}
        # Execute ExtremeQualityAuditor class definition from Cell 3
        auditor_start = cell3_src.index("class ExtremeQualityAuditor:")
        auditor_end = cell3_src.index("class PerceptualDeduplicator:")
        exec(compile(cell3_src[auditor_start:auditor_end], "nb_cell3_auditor.py", "exec"), nb_ns)
        nb_auditor = nb_ns["ExtremeQualityAuditor"]()

        rng = np.random.default_rng(999)
        flat_cartoon = np.full((160, 160, 3), 120, dtype=np.int16)
        # Add sharp outer border lines so global Laplacian >= 100 and Tenengrad >= 250
        for x in range(4, 20, 2):
            flat_cartoon[:, x, :] = 20 if (x % 4 == 0) else 220
        # Inject subtle grain so raw |Gx|+|Gy| > 1.0 on interior pixels, attempting to bypass raw flat_grad_ratio
        grain = rng.integers(-2, 3, size=(160, 160, 3), dtype=np.int16)
        grain_cartoon = np.clip(flat_cartoon + grain, 0, 255).astype(np.uint8)
        nb_passed, nb_metrics = nb_auditor.audit(grain_cartoon, ARCFACE_CANONICAL_160)
        assert nb_passed is False
        assert nb_metrics["rejection_reason"] in ("SYNTHETIC_FLAT_SHADING_OR_CARTOON", "INSUFFICIENT_TONAL_ENTROPY_NON_REAL")

    def test_connector_archive_cache_agedb_fallback_and_notebook_golden_dataset(self, tmp_path: Path):
        import io
        import json
        import re
        import tarfile
        from PIL import Image
        from scripts.kaggle_pipeline.kaggle_dataset_connectors import AgeDBConnector

        img_bytes = _make_textured_face_jpeg(seed=3201)

        # 1. find_archives("*.csv") skips image split directories ('train', 'images') and finds depth-2 manifests
        lag_root = tmp_path / "lagenda_perf"
        (lag_root / "train").mkdir(parents=True)
        (lag_root / "images").mkdir(parents=True)
        (lag_root / "subset_1gb" / "meta").mkdir(parents=True)
        # Put a decoy CSV inside train/ that should be skipped by find_archives
        (lag_root / "train" / "decoy.csv").write_text("img_name,age\ndecoy.jpg,25\n", encoding="utf-8")
        (lag_root / "images" / "real1.jpg").write_bytes(img_bytes)
        pd.DataFrame([{"img_name": "real1.jpg", "age": "26-30", "gender": "F", "person_id": "p42"}]).to_csv(
            lag_root / "subset_1gb" / "meta" / "annotations.csv", index=False
        )
        lag_conn = LagendaConnector(lag_root, max_samples=10)
        found_csvs = lag_conn.find_archives("*.csv")
        assert len(found_csvs) == 1
        assert found_csvs[0].name == "annotations.csv"
        # Cached on second call
        assert lag_conn.find_archives("*.csv") is found_csvs
        lag_samples = list(lag_conn.stream_samples())
        assert len(lag_samples) == 1
        assert lag_samples[0].raw_id == "real1.jpg"

        # 2. AgeDBConnector does not run root _iter_files_lazy fallback when classwise folders already yielded samples
        agedb_root = tmp_path / "agedb_classwise_only"
        (agedb_root / "025").mkdir(parents=True)
        (agedb_root / "025" / "cw_sample.jpg").write_bytes(img_bytes)
        agedb_conn = AgeDBConnector(agedb_root, max_samples=10)
        root_lazy_calls = []
        orig_lazy = agedb_conn._iter_files_lazy

        def tracked_lazy(root=None, max_depth=6):
            if root is None:
                root_lazy_calls.append(True)
            return orig_lazy(root=root, max_depth=max_depth)

        agedb_conn._iter_files_lazy = tracked_lazy  # type: ignore[method-assign]
        assert agedb_conn.is_available() is True
        agedb_samples = list(agedb_conn.stream_samples())
        assert len(agedb_samples) == 1
        assert len(root_lazy_calls) == 0

        # 3. Embedded notebook Cell 4 PreFlight has try/except resilience, Cell 7 has forbid_non_real_sources=True,
        # and Cell 6 FaceGoldenDataset returns identity and age_mask
        nb_data = json.loads(Path("notebooks/kaggle_face_mining_pipeline.ipynb").read_text(encoding="utf-8"))
        assert len(nb_data["cells"]) == 8
        cell4_src = "".join(nb_data["cells"][4]["source"])
        assert "failed during warmup:" in cell4_src
        cell7_src = "".join(nb_data["cells"][7]["source"])
        assert "audit_dataset(OUTPUT_DIR, forbid_non_real_sources=True)" in cell7_src

        cell6_src = "".join(nb_data["cells"][6]["source"])
        nb_ns = {
            "Path": Path,
            "pd": pd,
            "np": np,
            "cv2": cv2,
            "math": math,
            "io": io,
            "re": re,
            "tarfile": tarfile,
            "Image": Image,
            "torch": torch,
            "BiologicalLabelingEngine": BiologicalLabelingEngine,
        }
        exec(compile(cell6_src, "nb_cell6_dataset.py", "exec"), nb_ns)

        NbFaceGoldenDataset = nb_ns["FaceGoldenDataset"]
        ds_dir = tmp_path / "nb_golden_ds"
        img_dir = ds_dir / "images"
        img_dir.mkdir(parents=True)
        (img_dir / "00000001.jpg").write_bytes(img_bytes)
        manifest_path = ds_dir / "manifest.csv"
        pd.DataFrame(
            [
                {
                    "image_id": "00000001.jpg",
                    "split": "train",
                    "age_continuous": 28.0,
                    "age_min": 26.0,
                    "age_max": 30.0,
                    "apparent_age_std": 2.0,
                    "gender": "female",
                    "subject_id": "agedb:person_99",
                    "has_valid_age": True,
                    "age_label_type": "exact",
                }
            ]
        ).to_csv(manifest_path, index=False)

        nb_ds = NbFaceGoldenDataset(manifest_path, images_dir=img_dir, split="train")
        item = nb_ds[0]
        assert "age_mask" in item and float(item["age_mask"].item()) == 1.0
        assert "identity" in item and int(item["identity"].item()) > 0

    def test_high_density_13_source_upgrades_skin_tone_ita_union_find_and_debiasing(self, tmp_path: Path):
        from src.mining.filtering.quality_auditor import ExtremeQualityAuditor
        from src.mining.curation import ExtremeDatasetCurator
        from scripts.kaggle_pipeline.kaggle_dataset_connectors import (
            CACDConnector,
            IMDBWikiConnector,
        )
        from scripts.kaggle_pipeline.audit_and_sync_manifest import audit_dataset

        # 1. Dermatological CIE L*a*b* Individual Typology Angle (ITA°) & 6 Fitzpatrick/Chardon groups
        light_bgr = np.full((160, 160, 3), (225, 232, 242), dtype=np.uint8)
        dark_bgr = np.full((160, 160, 3), (35, 45, 65), dtype=np.uint8)
        light_lab = cv2.cvtColor(light_bgr, cv2.COLOR_BGR2LAB)
        dark_lab = cv2.cvtColor(dark_bgr, cv2.COLOR_BGR2LAB)
        ita_light, grp_light = ExtremeQualityAuditor.compute_skin_tone_ita(light_lab)
        ita_dark, grp_dark = ExtremeQualityAuditor.compute_skin_tone_ita(dark_lab)
        assert ita_light > 41.0 and grp_light in ("Type_I_Very_Light", "Type_II_Light")
        assert ita_dark < 10.0 and grp_dark in ("Type_V_Brown", "Type_VI_Dark")

        # 2. Transitive Union-Find identity + cross-dataset celebrity + pHash (d_H <= 6) disjointness
        df_uf = pd.DataFrame(
            [
                # Cluster A: agedb:BradPitt and b3fd:Brad_Pitt linked by canonical celebrity key, plus row 2 linked by pHash (d_H = 1)
                {"image_id": "00000001.jpg", "sha256": "h1", "subject_id": "agedb:BradPitt", "phash": "0000000000000001", "split": "train", "gender": "Male", "gender_prob": 0.0, "ethnicity": "Caucasian", "age_interval": "31-35", "skin_tone_group": "Type_II_Light", "alpha_sample_weight": 1.0},
                {"image_id": "00000002.jpg", "sha256": "h2", "subject_id": "b3fd:Brad_Pitt", "phash": "0000ffff0000ffff", "split": "val", "gender": "Unknown", "gender_prob": 0.5, "ethnicity": "Unknown", "age_interval": "31-35", "skin_tone_group": "Type_II_Light", "alpha_sample_weight": 1.0},
                {"image_id": "00000003.jpg", "sha256": "h3", "subject_id": "imdb_wiki:nm0000093", "phash": "0000000000000003", "split": "test", "gender": "Unknown", "gender_prob": 0.5, "ethnicity": "Unknown", "age_interval": "31-35", "skin_tone_group": "Type_II_Light", "alpha_sample_weight": 1.0},
                # Cluster B: unknown subject row linked by pHash (d_H = 2) to a known subject -> entire component quarantined to train
                {"image_id": "00000004.jpg", "sha256": "h4", "subject_id": None, "phash": "ffffffff00000000", "split": "val", "gender": "Female", "gender_prob": 1.0, "ethnicity": "Asian", "age_interval": "21-25", "skin_tone_group": "Type_III_Intermediate", "alpha_sample_weight": 1.0},
                {"image_id": "00000005.jpg", "sha256": "h5", "subject_id": "morph:999", "phash": "ffffffff00000003", "split": "test", "gender": "Female", "gender_prob": 1.0, "ethnicity": "Black", "age_interval": "66-70", "skin_tone_group": "Type_VI_Dark", "alpha_sample_weight": 1.0},
                # Cluster C & D: distinct known subjects with distant pHashes
                {"image_id": "00000006.jpg", "sha256": "h6", "subject_id": "fgnet:001", "phash": "aaaaaaaabbbbbbbb", "split": "train", "gender": "Male", "gender_prob": 0.0, "ethnicity": "Caucasian", "age_interval": "31-35", "skin_tone_group": "Type_II_Light", "alpha_sample_weight": 1.0},
                {"image_id": "00000007.jpg", "sha256": "h7", "subject_id": "cacd:Tom_Hanks", "phash": "123456789abcdef0", "split": "train", "gender": "Male", "gender_prob": 0.0, "ethnicity": "Caucasian", "age_interval": "31-35", "skin_tone_group": "Type_II_Light", "alpha_sample_weight": 1.0},
            ]
        )
        df_uf = ExtremeDatasetCurator._propagate_subject_gender(df_uf)
        # b3fd:Brad_Pitt must inherit Male and Caucasian from agedb:BradPitt
        assert df_uf.loc[1, "gender"] == "Male"
        assert float(df_uf.loc[1, "gender_prob"]) == 0.0
        assert df_uf.loc[1, "ethnicity"] == "Caucasian"

        df_uf = ExtremeDatasetCurator._ensure_all_splits_covered(df_uf)
        # Rows 0, 1, 2 must share the exact same split (transitive celebrity + pHash union)
        assert df_uf.loc[0, "split"] == df_uf.loc[1, "split"] == df_uf.loc[2, "split"]
        # Rows 3 & 4 must be in 'train' because row 3 has unknown subject_id
        assert df_uf.loc[3, "split"] == "train" and df_uf.loc[4, "split"] == "train"
        # All 3 splits must be covered (3 pure-known components + 1 unknown-linked component)
        assert set(df_uf["split"].unique()) == {"train", "val", "test"}

        # 3. Demographic sample-weight calibration boosts underrepresented tail stratum (row 4: 66-70, Black, dark, Female)
        df_uf = ExtremeDatasetCurator._calibrate_demographic_sample_weights(df_uf)
        assert float(df_uf.loc[4, "alpha_sample_weight"]) > float(df_uf.loc[0, "alpha_sample_weight"])

        # 4. CACDConnector (CACD_features_sex.csv + cacd_split) & IMDBWikiConnector (imdb-clean CSV + wiki_crop filename)
        img_bytes = _make_textured_face_jpeg(seed=4401)
        cacd_root = tmp_path / "cacd_slug"
        (cacd_root / "cacd_split" / "cacd_train").mkdir(parents=True)
        fname_cacd = "38_Brad_Pitt_0012.jpg"
        (cacd_root / "cacd_split" / "cacd_train" / fname_cacd).write_bytes(img_bytes)
        pd.DataFrame([{"name": fname_cacd, "age": 38, "sex": 1}]).to_csv(
            cacd_root / "CACD_features_sex.csv", index=False
        )
        cacd_conn = CACDConnector(cacd_root, max_samples=5)
        assert cacd_conn.is_available()
        cacd_samples = list(cacd_conn.stream_samples())
        assert len(cacd_samples) == 1
        assert cacd_samples[0].declared_age == 38.0
        assert cacd_samples[0].declared_gender == "Male"
        assert cacd_samples[0].metadata["subject_id"] in ("brad_pitt", "cacd:brad_pitt")

        imdb_root = tmp_path / "imdb_wiki_slug"
        (imdb_root / "imdb-clean-1024" / "imdb-clean-1024" / "00").mkdir(parents=True)
        rel_img = "00/nm0000093_rm123_1963-12-18_2001.jpg"
        (imdb_root / "imdb-clean-1024" / "imdb-clean-1024" / rel_img).write_bytes(img_bytes)
        pd.DataFrame([{"filename": rel_img, "age": 38, "gender": "M"}]).to_csv(
            imdb_root / "imdb-clean-1024" / "imdb_train_new_1024.csv", index=False
        )
        (imdb_root / "wiki_crop" / "01").mkdir(parents=True)
        (imdb_root / "wiki_crop" / "01" / "998877_1980-05-10_2015.jpg").write_bytes(img_bytes)
        imdb_conn = IMDBWikiConnector(imdb_root, max_samples=5)
        assert imdb_conn.is_available()
        imdb_samples = list(imdb_conn.stream_samples())
        assert len(imdb_samples) == 2
        ages_found = {s.declared_age for s in imdb_samples}
        assert ages_found == {38.0, 35.0}

        # 5. Audit dataset computes skin_tone_group_counts and demographic_entropy_bits
        curator = ExtremeDatasetCurator(output_dir=tmp_path / "curated_audit_check")
        cur_samples = [
            RawImageSample(
                source_dataset="cacd",
                raw_id=f"3{i}_Actor_{i}_0001.jpg",
                image_bytes=_make_textured_face_jpeg(seed=4500 + i),
                declared_age=float(20 + i * 10),
                declared_gender="Male" if i % 2 == 0 else "Female",
                age_min=float(20 + i * 10),
                age_max=float(20 + i * 10),
                age_label_type="exact",
                metadata={"subject_id": f"actor_{i}"},
            )
            for i in range(4)
        ]
        curator.curate_samples(cur_samples)
        audit_res = audit_dataset(tmp_path / "curated_audit_check" / "golden_dataset", forbid_non_real_sources=True)
        assert audit_res["integrity_passed"] is True
        assert "skin_tone_group_counts" in audit_res and len(audit_res["skin_tone_group_counts"]) >= 1
        assert "demographic_entropy_bits" in audit_res
        assert {"age_interval", "gender", "ethnicity", "skin_tone_group"}.issubset(
            audit_res["demographic_entropy_bits"].keys()
        )

    def test_demographic_metadata_joins_achromatic_ita_and_order_invariant_union_find(self, tmp_path: Path):
        import scipy.io
        from src.mining.filtering.quality_auditor import ExtremeQualityAuditor
        from src.mining.curation import (
            ExtremeDatasetCurator,
            canonicalize_split_identity_key,
        )
        from scripts.kaggle_pipeline.kaggle_dataset_connectors import (
            _normalize_connector_gender,
            AppaRealConnector,
            B3FDConnector,
            IMDBWikiConnector,
        )
        from scripts.kaggle_pipeline.audit_and_sync_manifest import validate_manifest_integrity

        # 1. Numeric gender normalization guard (allow_numeric=True vs False)
        assert _normalize_connector_gender("1", allow_numeric=False) is None
        assert _normalize_connector_gender("0", allow_numeric=False) is None
        assert _normalize_connector_gender("1", allow_numeric=True) == "Male"
        assert _normalize_connector_gender("0", allow_numeric=True) == "Female"

        # 2. Continuous physiological melanin chrominance regularization for achromatic/grayscale & cool-lit faces
        gray_light_bgr = np.full((160, 160, 3), (205, 205, 205), dtype=np.uint8)
        gray_mid_bgr = np.full((160, 160, 3), (145, 145, 145), dtype=np.uint8)
        gray_dark_bgr = np.full((160, 160, 3), (55, 55, 55), dtype=np.uint8)
        cool_light_bgr = np.full((160, 160, 3), (210, 195, 190), dtype=np.uint8)  # B > R -> b* < 0

        ita_gl, grp_gl = ExtremeQualityAuditor.compute_skin_tone_ita(cv2.cvtColor(gray_light_bgr, cv2.COLOR_BGR2LAB))
        ita_gm, grp_gm = ExtremeQualityAuditor.compute_skin_tone_ita(cv2.cvtColor(gray_mid_bgr, cv2.COLOR_BGR2LAB))
        ita_gd, grp_gd = ExtremeQualityAuditor.compute_skin_tone_ita(cv2.cvtColor(gray_dark_bgr, cv2.COLOR_BGR2LAB))
        ita_cl, _ = ExtremeQualityAuditor.compute_skin_tone_ita(cv2.cvtColor(cool_light_bgr, cv2.COLOR_BGR2LAB))

        assert ita_gl > ita_gm > ita_gd
        assert abs(ita_gl) < 89.0 and abs(ita_gm) < 89.0 and abs(ita_gd) < 89.0
        assert abs(ita_gl - ita_cl) < 15.0  # No +/- 90 deg jump across b* = 0
        assert grp_gl in ("Type_I_Very_Light", "Type_II_Light")
        assert grp_gd in ("Type_V_Brown", "Type_VI_Dark")

        # 3. B3FD numeric shard folder ignored & IMDb nm\d+_rm\d+ suffix stripped for cross-dataset canonicalization
        img_bytes = _make_textured_face_jpeg(seed=4901)
        b3fd_root = tmp_path / "b3fd_shard_test"
        (b3fd_root / "00").mkdir(parents=True)
        (b3fd_root / "00" / "nm0000093_rm12345_1963-12-18_2001.jpg").write_bytes(img_bytes)
        b3fd_samples = list(B3FDConnector(b3fd_root, max_samples=5).stream_samples())
        assert len(b3fd_samples) == 1
        assert b3fd_samples[0].metadata["b3fd_id"] == "nm0000093"
        assert canonicalize_split_identity_key("b3fd:nm0000093_rm12345") == "celebrity:nm0000093"
        assert canonicalize_split_identity_key("imdb_wiki:nm0000093") == "celebrity:nm0000093"

        # 4. APPA-REAL allCategories_train.csv demographic join (gender & race)
        appa_root = tmp_path / "appa_demo_join"
        (appa_root / "train").mkdir(parents=True)
        (appa_root / "train" / "000001.jpg_face.jpg").write_bytes(img_bytes)
        pd.DataFrame([{"file_name": "000001.jpg", "apparent_age_avg": 27.5, "apparent_age_std": 1.8, "real_age": 26.0}]).to_csv(
            appa_root / "gt_avg_train.csv", index=False
        )
        pd.DataFrame([{"file_name": "000001.jpg", "gender": "female", "race": "asian"}]).to_csv(
            appa_root / "allCategories_train.csv", index=False
        )
        appa_samples = list(AppaRealConnector(appa_root, max_samples=5).stream_samples())
        assert len(appa_samples) == 1
        assert appa_samples[0].declared_gender == "Female"
        assert appa_samples[0].declared_ethnicity == "East Asian"

        # 5. Wiki-Crop .mat metadata join (gender & subject name)
        wiki_root = tmp_path / "wiki_mat_join"
        (wiki_root / "wiki_crop" / "01").mkdir(parents=True)
        (wiki_root / "wiki_crop" / "01" / "998877_1980-05-10_2015.jpg").write_bytes(img_bytes)
        scipy.io.savemat(
            str(wiki_root / "wiki_crop" / "wiki.mat"),
            {
                "wiki": {
                    "full_path": np.array(["01/998877_1980-05-10_2015.jpg"], dtype=object),
                    "gender": np.array([0.0]),
                    "name": np.array(["Jane_Doe"], dtype=object),
                }
            },
        )
        wiki_samples = list(IMDBWikiConnector(wiki_root, max_samples=5).stream_samples())
        assert len(wiki_samples) == 1
        assert wiki_samples[0].declared_gender == "Female"
        assert wiki_samples[0].metadata["subject_id"] == "imdb_wiki:jane_doe"

        # 6. Order-invariant Union-Find split assignment & 2-known + 1-unknown split rebalancing
        rows_order_1 = [
            {"image_id": "1.jpg", "sha256": "s1", "subject_id": "agedb:SubjectAlpha", "phash": "0000000000000001", "split": "train", "gender": "Male", "gender_prob": 0.0, "ethnicity": "Caucasian"},
            {"image_id": "2.jpg", "sha256": "s2", "subject_id": "cacd:SubjectBeta", "phash": "0000000000000003", "split": "train", "gender": "Unknown", "gender_prob": 0.5, "ethnicity": "Diverse"},
            {"image_id": "3.jpg", "sha256": "s3", "subject_id": "morph:777", "phash": "ffffffff00000000", "split": "train", "gender": "Female", "gender_prob": 1.0, "ethnicity": "African"},
            {"image_id": "4.jpg", "sha256": "s4", "subject_id": None, "phash": "123456789abcdef0", "split": "train", "gender": "Female", "gender_prob": 1.0, "ethnicity": "Asian"},
        ]
        df_o1 = ExtremeDatasetCurator._ensure_all_splits_covered(
            ExtremeDatasetCurator._propagate_subject_gender(pd.DataFrame(rows_order_1))
        )
        df_o2 = ExtremeDatasetCurator._ensure_all_splits_covered(
            ExtremeDatasetCurator._propagate_subject_gender(pd.DataFrame(list(reversed(rows_order_1))))
        )
        map_o1 = dict(zip(df_o1["image_id"], df_o1["split"]))
        map_o2 = dict(zip(df_o2["image_id"], df_o2["split"]))
        assert map_o1 == map_o2
        assert set(df_o1["split"].unique()) == {"train", "val", "test"}
        # Transitive propagation across pHash-linked rows 1.jpg and 2.jpg
        g_map = dict(zip(df_o1["image_id"], df_o1["gender"]))
        e_map = dict(zip(df_o1["image_id"], df_o1["ethnicity"]))
        assert g_map["2.jpg"] == "Male"
        assert e_map["2.jpg"] == "Caucasian"

        # 7. Cross-split canonical celebrity detection in validate_manifest_integrity
        curator = ExtremeDatasetCurator(output_dir=tmp_path / "celeb_leak_check")
        c_samples = [
            RawImageSample(
                source_dataset="agedb" if i == 0 else ("b3fd" if i == 1 else "cacd"),
                raw_id=f"3{i}_Person_{i}_0001.jpg",
                image_bytes=_make_textured_face_jpeg(seed=5100 + i),
                declared_age=float(25 + i * 5),
                declared_gender="Male",
                metadata={"subject_id": "BradPitt" if i == 0 else ("Brad_Pitt" if i == 1 else f"Other_{i}")},
            )
            for i in range(4)
        ]
        df_c, _ = curator.curate_samples(c_samples)
        assert validate_manifest_integrity(df_c)["passed"] is True
        df_leaked = df_c.copy()
        df_leaked.loc[0, "split"] = "train"
        df_leaked.loc[1, "split"] = "val"
        res_leaked = validate_manifest_integrity(df_leaked)
        assert res_leaked["passed"] is False
        assert any("Cross-split canonical celebrities" in v for v in res_leaked["violations"])

    def test_cross_mount_gender_resolution_23_interval_stratification_and_fine_ldl_columns(self, tmp_path: Path):
        from scripts.kaggle_pipeline.kaggle_dataset_connectors import (
            _interleave_by_fine_interval,
            _first_valid_cell,
            InCloudStreamingHarvester,
        )
        from scripts.kaggle_pipeline.kaggle_face_engine import FacialLandmarkDetector
        from scripts.kaggle_pipeline.audit_and_sync_manifest import (
            FINE_LDL_COLUMNS,
            validate_manifest_integrity,
        )
        from src.mining.curation import ExtremeDatasetCurator

        img_bytes = _make_textured_face_jpeg(seed=5601)

        # 1. Case-insensitive _first_valid_cell lookup
        row_upper = {"FILE_NAME": "sample_01.jpg", "GENDER": "F"}
        assert _first_valid_cell(row_upper, ("file_name",)) == "sample_01.jpg"
        assert _first_valid_cell(row_upper, ("gender",)) == "F"

        # 2. Harvester sibling mount ordering, agedb-classwise skipping, and cross-mount gender resolution
        input_root = tmp_path / "kaggle_input_multi"
        # Mount both agedb-classwise-dataset and agedb-database -> classwise must be skipped
        (input_root / "agedb-classwise-dataset" / "025").mkdir(parents=True)
        (input_root / "agedb-classwise-dataset" / "025" / "cw1.jpg").write_bytes(img_bytes)
        (input_root / "agedb-database").mkdir(parents=True)
        (input_root / "agedb-database" / "1_MerylStreep_62_f.jpg").write_bytes(img_bytes)

        # Mount unlabelled appa-real-face-cropped AND gendered appa-real-dsb05rt -> dsb05rt ordered first, and face-cropped resolves gender from sibling
        (input_root / "appa-real-face-cropped" / "train").mkdir(parents=True)
        (input_root / "appa-real-face-cropped" / "train" / "000010.jpg_face.jpg").write_bytes(img_bytes)
        pd.DataFrame([{"file_name": "000010.jpg", "apparent_age_avg": 33.0, "apparent_age_std": 2.0, "real_age": 32.0}]).to_csv(
            input_root / "appa-real-face-cropped" / "gt_avg_train.csv", index=False
        )
        (input_root / "appa-real-dsb05rt").mkdir(parents=True)
        pd.DataFrame([{"file_name": "000010.jpg", "gender": "female", "race": "caucasian"}]).to_csv(
            input_root / "appa-real-dsb05rt" / "allCategories_train.csv", index=False
        )

        # Mount unlabelled cacd-combined AND gendered cacd-filtered-dataset -> filtered ordered first, and combined resolves gender from sibling
        (input_root / "cacd-combined" / "cacd_split" / "cacd_train").mkdir(parents=True)
        (input_root / "cacd-combined" / "cacd_split" / "cacd_train" / "45_Tom_Hanks_0001.jpg").write_bytes(img_bytes)
        (input_root / "cacd-filtered-dataset").mkdir(parents=True)
        pd.DataFrame([{"name": "45_Tom_Hanks_0001.jpg", "age": 45, "sex": 1}]).to_csv(
            input_root / "cacd-filtered-dataset" / "CACD_features_sex.csv", index=False
        )

        # Mount B3FD with subjects matching AgeDB (Meryl Streep) and CACD (Tom Hanks) -> B3FD resolves gender cross-mount!
        (input_root / "biometrically-filtered-famous-figure-dataset" / "00").mkdir(parents=True)
        (input_root / "biometrically-filtered-famous-figure-dataset" / "00" / "Meryl_Streep_1949-06-22_2011.jpg").write_bytes(img_bytes)
        (input_root / "biometrically-filtered-famous-figure-dataset" / "00" / "Tom_Hanks_1956-07-09_2001.jpg").write_bytes(img_bytes)

        harvester = InCloudStreamingHarvester(kaggle_input_dir=str(input_root), exclude_utk=True)
        agedb_conns = [c for c in harvester.connectors if c.get_name() == "agedb"]
        assert len(agedb_conns) == 1 and agedb_conns[0].root_dir.name == "agedb-database"

        appa_conns = [c for c in harvester.connectors if c.get_name() == "appa_real"]
        assert [c.root_dir.name for c in appa_conns] == ["appa-real-dsb05rt", "appa-real-face-cropped"]
        appa_cropped_samples = list(appa_conns[1].stream_samples())
        assert len(appa_cropped_samples) == 1
        assert appa_cropped_samples[0].declared_gender == "Female"
        assert appa_cropped_samples[0].declared_ethnicity == "Caucasian"

        cacd_conns = [c for c in harvester.connectors if c.get_name() == "cacd"]
        assert [c.root_dir.name for c in cacd_conns] == ["cacd-filtered-dataset", "cacd-combined"]
        cacd_comb_samples = list(cacd_conns[1].stream_samples())
        assert len(cacd_comb_samples) == 1
        assert cacd_comb_samples[0].declared_gender == "Male"

        b3fd_conns = [c for c in harvester.connectors if c.get_name() == "b3fd"]
        b3fd_samples = {s.metadata["b3fd_id"]: s for s in b3fd_conns[0].stream_samples()}
        assert b3fd_samples["Meryl_Streep"].declared_gender == "Female"
        assert b3fd_samples["Tom_Hanks"].declared_gender == "Male"

        # 3. 23-interval stratified interleaving prevents young-adult sequential monopolization
        unbalanced_candidates = (
            [{"id": f"ya_{i}", "age": 25.0, "gender": None} for i in range(40)]
            + [{"id": "infant_1", "age": 1.0, "gender": "Female"}]
            + [{"id": "senior_1", "age": 88.0, "gender": "Male"}]
            + [{"id": "child_1", "age": 7.0, "gender": "Female"}]
        )
        stratified = _interleave_by_fine_interval(
            unbalanced_candidates,
            age_fn=lambda x: x["age"],
            subj_fn=lambda x: x["id"],
            gender_fn=lambda x: x["gender"],
        )
        top_5_ids = {x["id"] for x in stratified[:5]}
        assert {"infant_1", "senior_1", "child_1"}.issubset(top_5_ids)

        # 4. Downscaled YuNet landmark detection on high-res portrait (> 320px) rescales coordinates to original frame
        hi_res = cv2.resize(
            cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR),
            (640, 800),
            interpolation=cv2.INTER_LINEAR,
        )
        det = FacialLandmarkDetector(yunet_model_path="__nonexistent__.onnx")
        seen_input_sizes = []

        class _MockYN:
            def setInputSize(self, sz):
                seen_input_sizes.append(sz)

            def detect(self, img):
                dh, dw = img.shape[:2]
                fake_face = np.array(
                    [[0, 0, dw, dh, dw * 0.35, dh * 0.38, dw * 0.65, dh * 0.38, dw * 0.50, dh * 0.55, dw * 0.38, dh * 0.72, dw * 0.62, dh * 0.72, 0.96]],
                    dtype=np.float32,
                )
                return 1, fake_face

        det.yunet_detector = _MockYN()
        ok_lm, lms_hi, conf_hi = det.detect_landmarks(hi_res)
        assert ok_lm is True and lms_hi.shape == (5, 2) and conf_hi >= 0.90
        assert seen_input_sizes[0] == (256, 320)
        assert float(np.max(lms_hi[:, 0])) > 320.0 and float(np.max(lms_hi[:, 1])) > 400.0

        # 5. Curator & audit validate 23 fine_ldl_* columns and license_provenance
        curator = ExtremeDatasetCurator(output_dir=tmp_path / "fine_ldl_check")
        three_samples = [
            RawImageSample(
                source_dataset="b3fd",
                raw_id=f"Subject_{i}_1970-01-01_{1995 + i * 10}.jpg",
                image_bytes=_make_textured_face_jpeg(seed=5610 + i),
                declared_age=float(25 + i * 10),
                declared_gender="Female" if i % 2 == 0 else "Male",
                metadata={"b3fd_id": f"Subject_{i}"},
            )
            for i in range(4)
        ]
        df_cur, _ = curator.curate_samples(three_samples)
        assert len(df_cur) >= 3
        assert "license_provenance" in df_cur.columns
        assert set(FINE_LDL_COLUMNS).issubset(df_cur.columns)
        assert validate_manifest_integrity(df_cur)["passed"] is True

        df_bad_fine = df_cur.copy()
        df_bad_fine.loc[0, "fine_ldl_0_2"] = 0.95
        res_bad_fine = validate_manifest_integrity(df_bad_fine)
        assert res_bad_fine["passed"] is False
        assert any("Fine 23-interval LDL" in v for v in res_bad_fine["violations"])

    def test_preflight_perf_lossless_metadata_axon_dedup_and_exact_vs_interval_ldl(self, tmp_path: Path):
        from scripts.kaggle_pipeline.kaggle_dataset_connectors import (
            AxonAgeConnector,
            CACDConnector,
            InCloudStreamingHarvester,
            MorphConnector,
        )
        from src.biometrics.estimation.dataset import FaceGoldenDataset
        from src.mining.curation import ExtremeDatasetCurator
        from src.mining.filtering.quality_auditor import ExtremeQualityAuditor

        img_bytes = _make_textured_face_jpeg(seed=6101)

        # 1. AxonAgeConnector: CSV + folder overlap must NOT double-yield, and folder-scanned images must carry subject_id
        axon_root = tmp_path / "axon_dedup_check"
        (axon_root / "test" / "18-20").mkdir(parents=True)
        (axon_root / "Face_Age_Estimation_Dataset" / "age_10").mkdir(parents=True)
        (axon_root / "test" / "18-20" / "25.jpg").write_bytes(img_bytes)
        (axon_root / "Face_Age_Estimation_Dataset" / "age_10" / "age_10_0001.jpeg").write_bytes(img_bytes)
        pd.DataFrame(
            [{"file": "test/18-20/25.jpg", "age": "18-20", "gender": "Female"}]
        ).to_csv(axon_root / "age_detection.csv", index=False)

        axon_samples = list(AxonAgeConnector(axon_root, max_samples=10).stream_samples())
        assert len(axon_samples) == 2  # 25.jpg yielded ONCE (from CSV), age_10_0001.jpeg yielded ONCE (from folder)
        for s in axon_samples:
            assert s.metadata is not None
            assert s.metadata.get("subject_id", "").startswith("axon_age:")

        # 2. Depth-3 CSV discovery & 100% lossless row indexing (> 25,000 rows without stride skipping)
        morph_root = tmp_path / "morph_deep_csv"
        (morph_root / "wrapper" / "Dataset" / "Index").mkdir(parents=True)
        (morph_root / "wrapper" / "Dataset" / "Images").mkdir(parents=True)
        (morph_root / "wrapper" / "Dataset" / "Images" / "99999_01M34.JPG").write_bytes(img_bytes)
        # Put target row at index 15,001 so a stride-skipping subsampler would miss it
        rows_morph = [{"id_num": f"{i:05d}", "race": "W", "gender": "M"} for i in range(15005)]
        rows_morph[15001] = {"id_num": "99999", "race": "B", "gender": "M"}
        pd.DataFrame(rows_morph).to_csv(morph_root / "wrapper" / "Dataset" / "Index" / "Train.csv", index=False)
        morph_samples = list(MorphConnector(morph_root, max_samples=5).stream_samples())
        assert len(morph_samples) == 1
        assert morph_samples[0].declared_ethnicity == "African"

        cacd_root = tmp_path / "cacd_lossless_csv"
        (cacd_root / "cacd_split" / "cacd_train").mkdir(parents=True)
        (cacd_root / "cacd_split" / "cacd_train" / "30_Rare_Celebrity_0001.jpg").write_bytes(img_bytes)
        rows_cacd = [{"identity": f"Common_Celeb_{i % 50}", "sex": 1} for i in range(25005)]
        rows_cacd[25001] = {"identity": "Rare Celebrity", "sex": 0}
        pd.DataFrame(rows_cacd).to_csv(cacd_root / "CACD_features_sex.csv", index=False)
        cacd_samples = list(CACDConnector(cacd_root, max_samples=5).stream_samples())
        assert len(cacd_samples) == 1
        assert cacd_samples[0].declared_gender == "Female"

        # 3. Redundant unlabelled sibling mounts dropped when full/gendered mount is present
        multi_root = tmp_path / "harvester_dedup_mounts"
        (multi_root / "fgnet-gendered").mkdir(parents=True)
        (multi_root / "fgnet-dataset").mkdir(parents=True)
        (multi_root / "appa-real-dsb05rt").mkdir(parents=True)
        (multi_root / "appa-real-age-only-v1").mkdir(parents=True)
        (multi_root / "cacd-filtered-dataset" / "cacd_split").mkdir(parents=True)
        (multi_root / "cacd-combined").mkdir(parents=True)
        harv = InCloudStreamingHarvester(kaggle_input_dir=str(multi_root), exclude_utk=True)
        fg_conns = [c for c in harv.connectors if c.get_name() == "fgnet"]
        assert [c.root_dir.name for c in fg_conns] == ["fgnet-gendered"]
        appa_conns = [c for c in harv.connectors if c.get_name() == "appa_real"]
        assert [c.root_dir.name for c in appa_conns] == ["appa-real-dsb05rt"]
        cacd_conns = [c for c in harv.connectors if c.get_name() == "cacd"]
        assert [c.root_dir.name for c in cacd_conns] == ["cacd-filtered-dataset"]

        # 4. Exact-age vs. interval-age LDL synthesis in ExtremeDatasetCurator & FaceGoldenDataset
        # Age 30.0 is at the upper edge of the '26-30' interval [26, 30].
        # Exact age 30.0 must peak at bin 30 (Gaussian LDL), whereas interval [26, 30] must peak at bin 28 (Box-Convolved LDL).
        s_exact = RawImageSample(
            image_bytes=_make_textured_face_jpeg(seed=6201),
            source_dataset="agedb",
            raw_id="1_ExactPerson_30_f.jpg",
            declared_age=30.0,
            declared_gender="Female",
            age_label_type="exact",
            metadata={"subject_id": "agedb:exactperson"},
        )
        s_interval = RawImageSample(
            image_bytes=_make_textured_face_jpeg(seed=6202),
            source_dataset="adience",
            raw_id="interval_person.jpg",
            declared_age=None,
            age_min=26.0,
            age_max=30.0,
            declared_gender="Male",
            age_label_type="interval",
            metadata={"subject_id": "adience:intervalperson"},
        )
        cur_dir = tmp_path / "exact_vs_interval_out"
        curator = ExtremeDatasetCurator(
            output_dir=cur_dir,
            auditor=ExtremeQualityAuditor(min_laplacian_var=20.0, min_tenengrad=50.0),
            write_tar_shards=True,
            write_direct_images=True,
        )
        df_ei, summary_ei = curator.curate_samples([s_exact, s_interval])
        assert len(df_ei) == 2
        ds_ei = FaceGoldenDataset(
            manifest_path=summary_ei.manifest_parquet_path,
            shards_dir=cur_dir / "golden_dataset" / "shards",
            split=None,
        )
        items_by_src = {
            df_ei.iloc[i]["source_dataset"]: ds_ei[i] for i in range(len(ds_ei))
        }
        ds_ei.close()
        assert int(torch.argmax(items_by_src["agedb"]["ldl_101"]).item()) == 30
        assert int(torch.argmax(items_by_src["adience"]["ldl_101"]).item()) == 28
        assert items_by_src["agedb"]["dataset_source"] == "agedb"
        assert items_by_src["adience"]["dataset_source"] == "adience"
        assert items_by_src["agedb"]["ita_group"] != "nan"
        assert "has_valid_age" in items_by_src["agedb"]
        assert float(items_by_src["agedb"]["has_valid_age"].item()) == 1.0





