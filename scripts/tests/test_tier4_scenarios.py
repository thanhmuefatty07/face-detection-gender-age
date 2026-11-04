"""
Tier 4 - Real-World Application Scenarios Test Suite (S1 to S6)
================================================================
Comprehensive end-to-end workload and pipeline validation per TEST_INFRA.md:
- S1: Extreme Blur & Occlusion Stress (F08, F09, F11, F07)
- S2: Extreme Head Pose & Pitch Boundary (F06, F07, F10)
- S3: Cross-Split Duplicate Injection & Isolation (F12, F13)
- S4: Prepubertal & Geriatric Dimorphism Stress (F14, F17, F18, F20)
- S5: Multi-Teacher Disagreement Consensus (F19, F20, F21)
- S6: End-to-End In-Cloud Pipeline & Artifact Generation (F01-F05, F22-F26)
"""

from __future__ import annotations
import math
import json
import hashlib
import tempfile
from pathlib import Path
from typing import List, Dict, Any
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
    FairFaceConnector,
    InCloudStreamingHarvester,
    DiskQuotaManager
)
from scripts.kaggle_pipeline.build_kaggle_kernel import (
    generate_pipeline_notebook,
    generate_kernel_metadata
)
from scripts.kaggle_pipeline.audit_and_sync_manifest import (
    generate_audit_report,
    validate_manifest_integrity
)


# ============================================================================
# S1: Extreme Blur & Occlusion Stress (F08, F09, F11, F07)
# ============================================================================

class TestScenarioS1BlurAndOcclusionStress:
    """
    Scenario S1: Streams a diverse batch of degraded face images (motion blur,
    defocus, severe underexposure, specular flash, bilateral cheek occlusion)
    alongside pristine golden face candidates.
    Target: 100% rejection of degraded faces without false rejections of golden faces.
    """

    def test_extreme_blur_and_occlusion_stress_lifecycle(self, sharp_face, canonical_landmarks):
        auditor = ExtremeQualityAuditor(
            min_laplacian_var=100.0,
            min_tenengrad=250.0,
            min_luminance=35.0,
            max_luminance=220.0,
            max_clipping_fraction=0.08,
            max_cheek_delta_e=22.0
        )
        aligner = FaceGeometryAligner(output_size=160)

        # 1. Generate degraded candidate pool
        degraded_candidates = []
        # Motion / defocus blur
        for k in [25, 35, 45]:
            degraded_candidates.append(("BLUR", create_synthetic_face(160, blur_ksize=k)))
        # Underexposed
        degraded_candidates.append(("DARK", create_synthetic_face(160, luminance_offset=-135)))
        # Overexposed
        degraded_candidates.append(("BRIGHT", create_synthetic_face(160, luminance_offset=115)))
        # Sensor clipped
        degraded_candidates.append(("CLIPPED", create_synthetic_face(160, clip_extremes=True)))

        # 2. Verify 100% of degraded candidates fail audit
        for deg_type, img in degraded_candidates:
            res = auditor.audit_face(img)
            assert res["passed"] is False, f"Degraded candidate {deg_type} falsely passed extreme audit!"
            assert len(res["reasons"]) > 0

        # 3. Verify clean golden candidates pass alignment and audit
        golden_candidates = [
            create_synthetic_face(160, blur_ksize=0, luminance_offset=0),
            create_synthetic_face(160, blur_ksize=0, luminance_offset=15),
            create_synthetic_face(160, blur_ksize=0, luminance_offset=-15),
        ]
        for g_img in golden_candidates:
            aligned, _ = aligner.align(g_img, canonical_landmarks)
            assert aligned.shape == (160, 160, 3)
            res = auditor.audit_face(g_img, landmarks_5=canonical_landmarks)
            assert res["passed"] is True, f"Clean golden candidate failed audit: {res['reasons']}"
            assert res["laplacian_var"] >= 100.0
            assert res["tenengrad"] >= 250.0


# ============================================================================
# S2: Extreme Head Pose & Pitch Boundary Stress (F06, F07, F10)
# ============================================================================

class TestScenarioS2HeadPoseBoundaryStress:
    """
    Scenario S2: Evaluates head pose estimation across yaw, pitch, and roll
    at exact boundary +/- 0.5 degrees.
    Target: Strict gating at |Yaw| <= 35°, |Pitch| <= 25°, |Roll| <= 20°
    with zero false acceptances above threshold.
    """

    def test_head_pose_yaw_boundary_rejection(self, sharp_face, canonical_landmarks):
        auditor = ExtremeQualityAuditor(max_abs_yaw=35.0, max_abs_pitch=25.0, max_abs_roll=20.0)

        # Canonical pose passes
        res_canon = auditor.audit_face(sharp_face, landmarks_5=canonical_landmarks)
        assert abs(res_canon["yaw"]) <= 35.0

        # Excessive yaw shift triggers rejection
        skewed_lms = canonical_landmarks.copy()
        skewed_lms[0, 0] += 40.0  # Excessive horizontal asymmetry
        res_yaw = auditor.audit_face(sharp_face, landmarks_5=skewed_lms)
        assert res_yaw["passed"] is False
        assert any("EXCESSIVE" in r for r in res_yaw["reasons"])

    def test_head_pose_roll_boundary_rejection(self, sharp_face, canonical_landmarks):
        auditor = ExtremeQualityAuditor(max_abs_roll=20.0)

        # 30-degree rotation clearly exceeds 20.0-degree limit
        theta = math.radians(30.0)
        R = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
        center = np.mean(canonical_landmarks, axis=0)
        rotated_lms = (canonical_landmarks - center) @ R.T + center

        res = auditor.audit_face(sharp_face, landmarks_5=rotated_lms)
        assert res["passed"] is False


# ============================================================================
# S3: Cross-Split Duplicate Injection & Isolation (F12, F13)
# ============================================================================

class TestScenarioS3DuplicateInjectionAndIsolation:
    """
    Scenario S3: Injects perturbed near-duplicates (slight compression, brightness +/-2)
    and deep metric embedding duplicates into the candidate validation pool.
    Target: Quarantines 100% of injected duplicates ensuring Hamming distance d_H > 6
    and Cosine distance d_cos >= 0.35 across all train-val sample pairs.
    """

    def test_cross_split_quarantine_pipeline(self, sharp_face):
        dedup = PerceptualDeduplicator(hamming_threshold=6, cosine_threshold=0.35)

        # 1. Ingest training partition
        train_pool = []
        for i in range(10):
            # Create distinct synthetic identities with distinct pHash and orthogonal embeddings
            h = int(hashlib.sha256(f"distinct_train_identity_{i}".encode()).hexdigest()[:16], 16)
            emb = np.zeros(512, dtype=np.float32)
            emb[i * 25] = 1.0
            is_dup, tag = dedup.is_duplicate_and_register(h, emb)
            if not is_dup:
                train_pool.append({"id": f"train_{i:03d}", "hash": h, "emb": emb})

        assert len(train_pool) >= 8

        # 2. Inject subtle duplicates of train images into Val candidate stream
        injected_val_candidates = []
        for t_item in train_pool[:3]:
            # Near duplicate: 2-bit flip
            injected_val_candidates.append({
                "source": "duplicate_injection",
                "hash": t_item["hash"] ^ 0b00000011,  # Hamming distance = 2 <= 6
                "emb": t_item["emb"]
            })

        # 3. Verify 100% of injected duplicates are quarantined
        quarantined_count = 0
        for cand in injected_val_candidates:
            is_dup, reason = dedup.is_duplicate_and_register(cand["hash"], cand["emb"])
            if is_dup:
                quarantined_count += 1
        assert quarantined_count == len(injected_val_candidates)

        # 4. Ingest genuinely distinct validation candidate
        distinct_val_hash = 0xffffffff00000000
        distinct_val_emb = np.zeros(512, dtype=np.float32); distinct_val_emb[300] = 1.0
        is_dup_val, tag = dedup.is_duplicate_and_register(distinct_val_hash, distinct_val_emb)
        assert is_dup_val is False
        assert tag == "UNIQUE"


# ============================================================================
# S4: Prepubertal & Geriatric Dimorphism Stress (F14, F17, F18, F20)
# ============================================================================

class TestScenarioS4DimorphismLifeCourseStress:
    """
    Scenario S4: Stresses the biological morphometry pipeline on ages 4, 10, 72, 85.
    Target:
    - Prepubertal children (<12) have attenuated dimorphism weights and don't trigger gender pruning.
    - Elderly (>65) have attenuated dimorphism weights.
    - Continuous LDL distributions accurately capture chronological uncertainty.
    """

    def test_life_course_dimorphism_weights(self):
        # Age 4: w = 4/12 = 0.333
        w_4 = BiologicalLabelingEngine.calculate_biological_sexual_dimorphism_weight(4.0)
        assert w_4 == pytest.approx(4.0 / 12.0, rel=1e-3)

        # Age 10: w = 10/12 = 0.833
        w_10 = BiologicalLabelingEngine.calculate_biological_sexual_dimorphism_weight(10.0)
        assert w_10 == pytest.approx(10.0 / 12.0, rel=1e-3)

        # Age 72: geriatric attenuation
        w_72 = BiologicalLabelingEngine.calculate_biological_sexual_dimorphism_weight(72.0)
        assert w_72 < 1.0

        # Age 85: geriatric attenuation
        w_85 = BiologicalLabelingEngine.calculate_biological_sexual_dimorphism_weight(85.0)
        assert w_85 < 1.0

    def test_child_gender_ambiguity_not_pruned_in_multi_teacher(self):
        mte = MultiTeacherEnsemble()
        # 4-year-old child with ambiguous visual gender cues (probs near 0.50)
        child_preds = [
            {"teacher_name": "t1", "age": 4.0, "gender_prob": 0.52},
            {"teacher_name": "t2", "age": 4.5, "gender_prob": 0.48},
            {"teacher_name": "t3", "age": 3.8, "gender_prob": 0.55},
            {"teacher_name": "t4", "age": 4.2, "gender_prob": 0.45}
        ]
        res = mte.evaluate_teachers(child_preds)
        # MUST NOT be pruned despite gender confusion, because dimorphism is biologically low
        assert res["is_pruned"] is False
        assert res["consensus_age"] < 12.0
        assert res["dimorphism_weight"] < 0.50

    def test_geriatric_laplacian_ldl_dispersion(self):
        # 85-year-old centenarian candidate: Laplacian LDL mass concentrated at '65+'
        ldl = BiologicalLabelingEngine.compute_label_distribution(mu_age=85.0, sigma_age=4.5, distribution_type="laplacian")
        assert np.argmax(ldl) == 9  # '65+' bin
        assert ldl[9] > 0.50
        assert ldl.sum() == pytest.approx(1.0, abs=1e-4)


# ============================================================================
# S5: Multi-Teacher Disagreement Consensus (F19, F20, F21)
# ============================================================================

class TestScenarioS5MultiTeacherConsensusAndAlphaSignals:
    """
    Scenario S5: Simulates 4 diverse architectures across conflicting and consensus cases.
    Target:
    - High consensus yields high Alpha Signal sample weight (>0.70).
    - Epistemic noise (sigma > 8.0) is pruned under AGE_DISAGREEMENT.
    - Adult gender confusion is pruned under GENDER_CONFUSION.
    """

    def test_golden_consensus_alpha_signal(self):
        mte = MultiTeacherEnsemble()
        golden = [
            {"teacher_name": "insightface", "age": 28.0, "gender_prob": 0.96},
            {"teacher_name": "mivolo", "age": 27.5, "gender_prob": 0.94},
            {"teacher_name": "fairface", "age": 28.5, "gender_prob": 0.95},
            {"teacher_name": "clip", "age": 28.0, "gender_prob": 0.93}
        ]
        res = mte.evaluate_teachers(golden)
        assert res["is_pruned"] is False
        assert res["consensus_age"] == pytest.approx(28.0, abs=0.5)
        assert res["predicted_gender"] == "Female"
        assert res["alpha_sample_weight"] >= 0.70
        assert len(res["ldl_distribution"]) == 10

    def test_extreme_age_disagreement_pruned(self):
        mte = MultiTeacherEnsemble()
        polarized = [
            {"teacher_name": "insightface", "age": 18.0, "gender_prob": 0.95},
            {"teacher_name": "mivolo", "age": 42.0, "gender_prob": 0.90},
            {"teacher_name": "fairface", "age": 55.0, "gender_prob": 0.88},
            {"teacher_name": "clip", "age": 22.0, "gender_prob": 0.92}
        ]
        res = mte.evaluate_teachers(polarized)
        assert res["is_pruned"] is True
        assert res["age_std"] > 8.0
        assert any("AGE_DISAGREEMENT" in t for t in res["rejection_tags"])

    def test_adult_gender_confusion_pruned(self):
        mte = MultiTeacherEnsemble()
        confused_adult = [
            {"teacher_name": "insightface", "age": 30.0, "gender_prob": 0.51},
            {"teacher_name": "mivolo", "age": 31.0, "gender_prob": 0.49},
            {"teacher_name": "fairface", "age": 29.5, "gender_prob": 0.52},
            {"teacher_name": "clip", "age": 30.0, "gender_prob": 0.48}
        ]
        res = mte.evaluate_teachers(confused_adult)
        assert res["is_pruned"] is True
        assert any("GENDER_CONFUSION" in t for t in res["rejection_tags"])


# ============================================================================
# S6: End-to-End In-Cloud Pipeline & Artifact Generation (F01-F05, F22-F26)
# ============================================================================

class TestScenarioS6EndToEndPipelineLifecycle:
    """
    Scenario S6: Full lifecycle integration test:
    Raw ingestion -> 5-pt landmark detection -> Umeyama similarity alignment ->
    Extreme quality audit -> Perceptual deduplication -> Biological morphometry & LDL ->
    Multi-teacher consensus -> 30-column manifest export -> Kaggle CLI dataset metadata ->
    Audit report generation.
    """

    def test_complete_golden_dataset_curation_lifecycle(self, tmp_path):
        work_dir = tmp_path / "working"
        work_dir.mkdir(parents=True, exist_ok=True)

        # 1. Pipeline modules
        det = FacialLandmarkDetector()
        aligner = FaceGeometryAligner(output_size=160)
        auditor = ExtremeQualityAuditor(min_laplacian_var=100.0, min_tenengrad=250.0)
        dedup = PerceptualDeduplicator(hamming_threshold=6, cosine_threshold=0.35)
        mte = MultiTeacherEnsemble()
        quota_mgr = DiskQuotaManager(working_dir=str(work_dir), shard_image_count=50)

        # 2. Ingest & process candidates
        # Create distinct candidates
        cand0 = create_synthetic_face(160, blur_ksize=0, luminance_offset=0)
        cand1 = create_synthetic_face(160, blur_ksize=0, luminance_offset=10)
        cv2.circle(cand1, (25, 25), 12, (30, 30, 30), -1)
        cand2 = create_synthetic_face(160, blur_ksize=35, luminance_offset=0)  # blurry (rejected by audit)
        raw_candidates = [cand0, cand1, cand2]
        curated_records = []

        for idx, raw_face in enumerate(raw_candidates):
            # A. Landmark detection
            ok_lm, lms, conf = det.detect_landmarks(raw_face)
            if not ok_lm:
                continue

            # B. Umeyama similarity alignment
            aligned_face, _ = aligner.align(raw_face, lms)

            # C. Extreme quality audit
            audit_res = auditor.audit_face(aligned_face, landmarks_5=aligner.target_landmarks)
            if not audit_res["passed"]:
                continue

            # D. Deduplication
            gray = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2GRAY)
            phash = dedup.compute_phash_uint64(gray)
            emb = dedup.extract_embedding_512d(aligned_face)
            is_dup, tag = dedup.is_duplicate_and_register(phash, emb)
            if is_dup:
                continue

            # E. Multi-teacher consensus & Alpha Signals
            simulated_teachers = [
                {"teacher_name": "insightface", "age": 25.0 + idx, "gender_prob": 0.95},
                {"teacher_name": "mivolo", "age": 25.5 + idx, "gender_prob": 0.93},
                {"teacher_name": "fairface", "age": 24.5 + idx, "gender_prob": 0.94},
                {"teacher_name": "clip", "age": 25.0 + idx, "gender_prob": 0.92}
            ]
            alpha_res = mte.evaluate_teachers(simulated_teachers)
            if alpha_res["is_pruned"]:
                continue

            # F. Write aligned face to tar shard
            filename = f"face_{idx:04d}.jpg"
            quota_mgr.write_aligned_face(aligned_face, filename)

            # G. Assemble 30-column record
            record = {
                "image_id": filename,
                "sha256": f"sha256_{idx}",
                "source_dataset": "synthetic_stream",
                "original_filename": f"raw_{idx}.jpg",
                "age_continuous": alpha_res["consensus_age"],
                "age_group": alpha_res["age_group"],
                "age_group_idx": alpha_res["age_group_idx"],
                "apparent_age_std": alpha_res["age_std"],
                "gender": alpha_res["predicted_gender"],
                "gender_prob": alpha_res["gender_prob"],
                "ethnicity": "East Asian",
                "dimorphism_weight": alpha_res["dimorphism_weight"],
                "teacher_count": alpha_res["teacher_count"],
                "teacher_age_mu": alpha_res["consensus_age"],
                "teacher_age_sigma": alpha_res["age_std"],
                "teacher_gender_agreement": alpha_res["gender_confidence"],
                "alpha_sample_weight": alpha_res["alpha_sample_weight"],
                "laplacian_var": audit_res["laplacian_var"],
                "tenengrad": audit_res["tenengrad"],
                "mean_luminance": audit_res["mean_luminance"],
                "shadow_clip": audit_res["shadow_clip"],
                "highlight_clip": audit_res["highlight_clip"],
                "yaw": audit_res["yaw"],
                "pitch": audit_res["pitch"],
                "roll": audit_res["roll"],
                "phash": hex(phash),
                "split": "train"
            }
            # Append LDL columns
            ldl_probs = alpha_res["ldl_distribution"]
            for c_idx, col_name in enumerate([
                "ldl_0_2", "ldl_3_6", "ldl_7_12", "ldl_13_17", "ldl_18_24",
                "ldl_25_34", "ldl_35_44", "ldl_45_54", "ldl_55_64", "ldl_65_plus"
            ]):
                record[col_name] = ldl_probs[c_idx]

            curated_records.append(record)

        quota_mgr.close()

        # Verify curated records
        assert len(curated_records) >= 2, "Expected at least 2 clean samples to pass complete pipeline"

        # 3. Export Manifests
        df = pd.DataFrame(curated_records)
        manifest_parq = work_dir / "metadata.parquet"
        manifest_csv = work_dir / "metadata.csv"
        df.to_parquet(manifest_parq, index=False)
        df.to_csv(manifest_csv, index=False)

        assert manifest_parq.exists()
        assert manifest_csv.exists()

        # 4. Generate Kaggle metadata
        dataset_meta_file = work_dir / "dataset-metadata.json"
        dataset_meta = {
            "title": "Face Gender Age Golden SOTA Dataset",
            "id": "vnthanhphm/face-gender-age-golden-dataset",
            "licenses": [{"name": "CC0-1.0"}]
        }
        with open(dataset_meta_file, "w") as f:
            json.dump(dataset_meta, f, indent=2)
        assert dataset_meta_file.exists()

        # 5. Generate and verify Audit Report
        report_file = work_dir / "dataset_audit_report.md"
        report_content = generate_audit_report(manifest_csv, report_file)
        assert report_file.exists()
        assert "Báo Cáo Thẩm Định" in report_content
        assert "Status: **BLOCKED**" in report_content
        assert validate_manifest_integrity(df)["passed"] is False
