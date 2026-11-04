import hashlib
import io
import runpy
import sys
import tarfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from scripts.kaggle_pipeline import audit_and_sync_manifest as audit


def manifest():
    rows = []
    for split, token, phash in [
        ("train", "a", "0x0000000000000000"),
        ("val", "b", "0xffffffffffffffff"),
        ("test", "c", "0xaaaaaaaaaaaaaaaa"),
    ]:
        row = {column: 0.0 for column in audit.EXPECTED_MANIFEST_COLUMNS}
        row.update(
            image_id=f"face_{token * 16}.jpg", sha256=token * 64,
            source_dataset="utkface", original_filename=f"25_1_0_{token}.jpg",
            age_continuous=25.0, age_group="25-34", age_group_idx=5,
            apparent_age_std=2.0, gender="Female", gender_prob=1.0,
            ethnicity="unknown", dimorphism_weight=1.0,
            teacher_count=0, teacher_age_mu=None, teacher_age_sigma=None,
            teacher_gender_agreement=None, alpha_sample_weight=1.0,
            laplacian_var=200.0, tenengrad=400.0, mean_luminance=120.0,
            phash=phash, split=split,
            subject_id=f"utkface_{token}", identity_id=str(ord(token)),
        )
        row.update({c: 0.1 for c in audit.EXPECTED_MANIFEST_COLUMNS if c.startswith("ldl_")})
        rows.append(row)
    return pd.DataFrame(rows)


def test_valid_manifest_passes_only_manifest_checks():
    result = audit.validate_manifest_integrity(manifest())
    assert result["passed"]


def test_empty_manifest_is_not_success():
    result = audit.validate_manifest_integrity(manifest().iloc[:0])
    assert not result["passed"]


@pytest.mark.parametrize("value", [np.nan, np.inf, -1.0, 101.0, "unknown"])
def test_invalid_age_cannot_pass(value):
    df = manifest().astype({"age_continuous": "object"})
    df.loc[0, "age_continuous"] = value
    assert not audit.validate_manifest_integrity(df)["passed"]


@pytest.mark.parametrize("column", ["laplacian_var", "tenengrad", "yaw", "dimorphism_weight"])
def test_nan_cannot_bypass_numeric_quality_checks(column):
    df = manifest()
    df.loc[0, column] = np.nan
    assert not audit.validate_manifest_integrity(df)["passed"]


def test_negative_probability_is_invalid_even_when_sum_is_one():
    df = manifest()
    df.loc[0, "ldl_0_2"] = -0.1
    df.loc[0, "ldl_3_6"] = 0.3
    assert not audit.validate_manifest_integrity(df)["passed"]


@pytest.mark.parametrize("group,index", [("banana", 5), (None, 5), ("25-34", 0), ("18-24", 4)])
def test_cohort_annotation_must_match_continuous_age(group, index):
    df = manifest()
    df.loc[0, "age_group"] = group
    df.loc[0, "age_group_idx"] = index
    assert not audit.validate_manifest_integrity(df)["passed"]


def test_combined_clipping_cannot_bypass_photometry_gate():
    df = manifest()
    df.loc[0, "shadow_clip"] = 0.05
    df.loc[0, "highlight_clip"] = 0.05
    assert not audit.validate_manifest_integrity(df)["passed"]


@pytest.mark.parametrize("phash", ["garbage", "0x10000000000000000", -1, 1.5])
def test_invalid_phash_is_not_silently_skipped(phash):
    df = manifest()
    df.loc[0, "phash"] = phash
    assert not audit.validate_manifest_integrity(df)["passed"]


def test_train_test_leakage_is_detected():
    df = manifest()
    df.loc[2, "phash"] = "0x000000000000003f"
    assert not audit.validate_manifest_integrity(df)["passed"]


def test_leakage_after_first_hundred_rows_is_detected():
    df = pd.concat([manifest().iloc[[0]]] * 101, ignore_index=True)
    for i in range(101):
        df.loc[i, "image_id"] = f"face_{i:016x}.jpg"
        df.loc[i, "sha256"] = f"{i:064x}"
        df.loc[i, "phash"] = hex(0xFFFFFFFFFFFF0000 + i)
    df = pd.concat([df, manifest().iloc[1:]], ignore_index=True)
    df.loc[100, "phash"] = "0x0"
    df.loc[101, "phash"] = "0x0"
    assert not audit.validate_manifest_integrity(df)["passed"]


@pytest.mark.parametrize("splits", [["train"] * 3, ["train", "val", "unknown"]])
def test_missing_or_unknown_holdout_split_blocks_acceptance(splits):
    df = manifest()
    df["split"] = splits
    assert not audit.validate_manifest_integrity(df)["passed"]


def test_duplicate_content_across_splits_is_detected():
    df = manifest()
    df.loc[2, "sha256"] = df.loc[0, "sha256"]
    assert not audit.validate_manifest_integrity(df)["passed"]


def test_same_subject_cannot_appear_in_two_splits():
    df = manifest()
    df["source_dataset"] = "agedb"
    df["subject_id"] = ["person_1", "person_2", "person_1"]
    assert not audit.validate_manifest_integrity(df)["passed"]


def test_unknown_subjects_cannot_be_assigned_to_holdout():
    df = manifest()
    df["subject_id"] = ["known-a", None, "known-c"]
    df["identity_id"] = ["1", None, "3"]
    assert not audit.validate_manifest_integrity(df)["passed"]


def test_identity_hash_and_compact_id_must_be_one_to_one():
    df = manifest()
    df.loc[1, "identity_id"] = df.loc[0, "identity_id"]
    assert not audit.validate_manifest_integrity(df)["passed"]


def test_teacher_metadata_must_be_absent_without_real_predictions():
    df = manifest()
    df["teacher_count"] = 0
    df[["teacher_age_mu", "teacher_age_sigma", "teacher_gender_agreement"]] = np.nan
    assert audit.validate_manifest_integrity(df)["passed"]


def test_fake_teacher_predictions_block_acceptance():
    df = manifest()
    df["teacher_count"] = 0
    df["teacher_gender_agreement"] = 0.95
    assert not audit.validate_manifest_integrity(df)["passed"]


@pytest.mark.parametrize("suffix", ["B", "_B", "B1"])
def test_morph_identity_overlap_includes_ethnicity_suffix(suffix):
    df = manifest()
    df = df.drop(columns=["subject_id", "identity_id"])
    df["source_dataset"] = "morph"
    df["original_filename"] = [f"00001_01M25{suffix}.jpg", "00002_01M25W.jpg", f"00001_02M30{suffix}.jpg"]
    result = audit.validate_manifest_integrity(df)
    assert result["subject_overlap"]["covered_rows"] == 3
    assert result["subject_overlap"]["cross_split_subjects"] == 1
    assert not result["passed"]


@pytest.mark.parametrize("threshold", [0, 1, 6, 63])
def test_chunk_index_matches_exhaustive_hamming_oracle(threshold):
    hashes = [0, 1, 3, 63, 127, 0xFFFFFFFFFFFFFFFF, 0xAAAAAAAAAAAAAAAA]
    splits = ["train", "val", "test", "train", "test", "val", "test"]
    expected = sum(
        splits[i] != splits[j] and (hashes[i] ^ hashes[j]).bit_count() <= threshold
        for i in range(len(hashes)) for j in range(i)
    )
    result = audit.cross_split_hash_pairs(hashes, splits, threshold)
    assert result["pairs"] == expected
    assert result["rows_checked"] == 7


def test_phash_comparison_budget_marks_coverage_incomplete_and_fails_closed(monkeypatch):
    result = audit.cross_split_hash_pairs(
        [0, 0, 0], ["train", "val", "test"], max_candidate_comparisons=1
    )

    assert not result["coverage_complete"]
    assert result["limit_reached"]
    assert result["candidate_comparisons"] == 1

    df = manifest()
    df["phash"] = ["0", "0", "0"]
    monkeypatch.setattr(audit, "MAX_PHASH_CANDIDATE_COMPARISONS", 1)
    checked = audit.validate_manifest_integrity(df)
    assert not checked["passed"]
    assert not checked["phash_leakage"]["coverage_complete"]


@pytest.mark.parametrize("suffix", [".csv", ".parquet"])
def test_manifest_reader_rejects_row_count_over_budget(tmp_path, monkeypatch, suffix):
    path = tmp_path / f"manifest{suffix}"
    frame = manifest()
    if suffix == ".csv":
        frame.to_csv(path, index=False)
    else:
        frame.to_parquet(path, index=False)
    monkeypatch.setattr(audit, "MAX_MANIFEST_ROWS", 2)

    with pytest.raises(ValueError, match="row limit"):
        audit.read_manifest(path)


def test_manifest_reader_rejects_file_size_over_budget_before_parsing(tmp_path, monkeypatch):
    path = tmp_path / "manifest.csv"
    path.write_text("not parsed", encoding="utf-8")
    monkeypatch.setattr(audit, "MAX_MANIFEST_BYTES", 4)

    with pytest.raises(ValueError, match="byte limit"):
        audit.read_manifest(path)


def test_manifest_reader_rejects_parquet_expansion_over_budget(tmp_path, monkeypatch):
    path = tmp_path / "manifest.parquet"
    manifest().to_parquet(path, index=False)
    monkeypatch.setattr(audit, "MAX_PARQUET_UNCOMPRESSED_BYTES", 0)

    with pytest.raises(ValueError, match="uncompressed Parquet byte limit"):
        audit.read_manifest(path)


def write_archive(root, df, payloads, extra_member=None):
    root.mkdir(exist_ok=True)
    df.to_csv(root / "metadata.csv", index=False)
    with tarfile.open(root / "shard_0000.tar", "w") as archive:
        for image_id, payload in zip(df.image_id, payloads):
            member = tarfile.TarInfo(image_id)
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
        if extra_member:
            member = tarfile.TarInfo(extra_member)
            member.size = 1
            archive.addfile(member, io.BytesIO(b"x"))


def image_payloads():
    rng = np.random.default_rng(718)
    return [cv2.imencode(".png", rng.integers(60, 180, (160, 160, 3), dtype=np.uint8))[1].tobytes()
            for _ in range(3)]


def test_manifest_without_images_cannot_pass_dataset_audit(tmp_path):
    df = manifest()
    df.to_csv(tmp_path / "metadata.csv", index=False)
    result = audit.audit_dataset(tmp_path)
    assert result["images"]["missing_images"] == 3
    assert result["images"]["decoded_images"] == 0
    assert not result["integrity_passed"]


def test_curated_manifest_can_be_audited_against_upstream_image_archives(tmp_path):
    manifest_root = tmp_path / "curated_manifest"
    image_root = tmp_path / "source_images"
    manifest_root.mkdir()
    df, payloads = manifest(), image_payloads()
    df["sha256"] = [hashlib.sha256(payload).hexdigest() for payload in payloads]
    df.to_csv(manifest_root / "metadata.csv", index=False)
    write_archive(image_root, df, payloads)

    result = audit.audit_dataset(manifest_root, image_root=image_root)

    assert result["images"]["decoded_images"] == 3
    assert result["images"]["missing_images"] == 0
    assert result["image_root"] == str(image_root)
    assert result["integrity_passed"]


def test_actual_image_bytes_are_checked_against_manifest_digest(tmp_path):
    df, payloads = manifest(), image_payloads()
    write_archive(tmp_path, df, payloads)
    result = audit.audit_dataset(tmp_path)
    assert result["images"]["sha256_mismatches"] == 3
    assert result["images"]["decoded_images"] == 3
    assert not result["integrity_passed"]


def test_laplacian_measurement_uses_producers_three_pixel_kernel(tmp_path):
    stripe = np.tile(np.array([100, 110, 120, 130], np.uint8), 40)
    image = np.repeat(np.tile(stripe, (160, 1))[:, :, None], 3, axis=2)
    payload = cv2.imencode(".png", image)[1].tobytes()
    df = manifest()
    df["sha256"] = hashlib.sha256(payload).hexdigest()
    write_archive(tmp_path, df, [payload] * 3)
    result = audit.audit_dataset(tmp_path)
    # The ksize=3 stencil gives 78 values of +/-160 and two of +/-80 per row.
    assert result["images"]["measured_quality_quantiles"]["laplacian_var"]["0.5"] == 12560.0


def test_corrupt_image_is_reported_instead_of_substituted(tmp_path):
    df, payloads = manifest(), image_payloads()
    payloads[1] = b"corrupt image"
    df["sha256"] = [hashlib.sha256(b).hexdigest() for b in payloads]
    write_archive(tmp_path, df, payloads)
    result = audit.audit_dataset(tmp_path)
    assert result["images"]["decode_failures"] == 1
    assert result["images"]["decoded_images"] == 2
    assert not result["integrity_passed"]


def test_truncated_jpeg_is_rejected(tmp_path):
    df, payloads = manifest(), image_payloads()
    image = cv2.imdecode(np.frombuffer(payloads[0], np.uint8), cv2.IMREAD_COLOR)
    payloads[0] = cv2.imencode(".jpg", image)[1].tobytes()[:-100]
    df["sha256"] = [hashlib.sha256(b).hexdigest() for b in payloads]
    write_archive(tmp_path, df, payloads)
    result = audit.audit_dataset(tmp_path)
    assert result["images"]["decode_failures"] == 1
    assert result["images"]["decoded_images"] == 2
    assert not result["integrity_passed"]


def test_corrupt_tar_produces_blocking_evidence(tmp_path):
    manifest().to_csv(tmp_path / "metadata.csv", index=False)
    (tmp_path / "shard_0000.tar").write_bytes(b"not a tar archive")
    result = audit.audit_dataset(tmp_path)
    assert result["images"]["archive_errors"] == 1
    assert result["images"]["missing_images"] == 3
    assert not result["integrity_passed"]


def test_unsafe_tar_member_is_never_extracted(tmp_path):
    df, payloads = manifest(), image_payloads()
    df["sha256"] = [hashlib.sha256(b).hexdigest() for b in payloads]
    write_archive(tmp_path, df, payloads, "../escape.jpg")
    result = audit.audit_dataset(tmp_path)
    assert result["images"]["unsafe_members"] == 1
    assert not (tmp_path.parent / "escape.jpg").exists()
    assert not result["integrity_passed"]


def test_equal_decoded_pixels_detected_even_with_different_encoding(tmp_path):
    df, payloads = manifest(), image_payloads()
    image = cv2.imdecode(np.frombuffer(payloads[0], dtype=np.uint8), cv2.IMREAD_COLOR)
    payloads[1] = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 9])[1].tobytes()
    assert payloads[0] != payloads[1]
    df["sha256"] = [hashlib.sha256(b).hexdigest() for b in payloads]
    write_archive(tmp_path, df, payloads)
    result = audit.audit_dataset(tmp_path)
    assert result["images"]["sha256_mismatches"] == 0
    assert result["images"]["cross_split_pixel_duplicate_groups"] == 1
    assert not result["integrity_passed"]


def test_equal_decoded_pixels_within_train_are_counted(tmp_path):
    df, payloads = manifest(), image_payloads()
    image = cv2.imdecode(np.frombuffer(payloads[0], dtype=np.uint8), cv2.IMREAD_COLOR)
    payloads.append(cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 9])[1].tobytes())
    extra = df.iloc[[0]].copy()
    extra["image_id"] = "face_dddddddddddddddd.jpg"
    df = pd.concat([df, extra], ignore_index=True)
    df["sha256"] = [hashlib.sha256(b).hexdigest() for b in payloads]
    write_archive(tmp_path, df, payloads)
    result = audit.audit_dataset(tmp_path)
    assert result["images"].get("pixel_duplicate_groups", 0) == 1
    assert result["images"]["cross_split_pixel_duplicate_groups"] == 0
    assert not result["integrity_passed"]


def test_explicit_manifest_cli_cannot_bypass_kaggle_runtime_guard(tmp_path, monkeypatch, capsys):
    manifest_path = tmp_path / "synthetic.csv"
    manifest_path.write_text("source_dataset\nsynthetic\n", encoding="utf-8")
    report_path = tmp_path / "audit.md"

    actual_is_dir = Path.is_dir

    def without_kaggle_mounts(path):
        if path in {Path("/kaggle/input"), Path("/kaggle/working")}:
            return False
        return actual_is_dir(path)

    monkeypatch.setattr(Path, "is_dir", without_kaggle_mounts)
    monkeypatch.setattr(
        sys,
        "argv",
        ["audit_and_sync_manifest.py", "--manifest", str(manifest_path), "--report", str(report_path)],
    )

    with pytest.raises(SystemExit) as exit_info:
        runpy.run_path(str(Path(audit.__file__)), run_name="__main__")

    assert exit_info.value.code == 2
    assert "Manifest auditing is Kaggle-only" in capsys.readouterr().err
    assert not report_path.exists()


def test_explicit_manifest_cli_runs_inside_kaggle_runtime(tmp_path, monkeypatch):
    if not Path("/kaggle/input").is_dir():
        actual_is_dir = Path.is_dir

        def with_kaggle_mounts(path):
            if path in {Path("/kaggle/input"), Path("/kaggle/working")}:
                return True
            return actual_is_dir(path)

        monkeypatch.setattr(Path, "is_dir", with_kaggle_mounts)
        monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Batch")
        monkeypatch.setenv("KAGGLE_USER_SECRETS_TOKEN", "synthetic-token")
    assert Path("/kaggle/input").is_dir()
    assert Path("/kaggle/working").is_dir()
    manifest_path = tmp_path / "synthetic.csv"
    manifest().to_csv(manifest_path, index=False)
    report_path = tmp_path / "audit.md"
    monkeypatch.setattr(
        sys,
        "argv",
        ["audit_and_sync_manifest.py", "--manifest", str(manifest_path), "--report", str(report_path)],
    )

    with pytest.raises(SystemExit) as exit_info:
        runpy.run_path(str(Path(audit.__file__)), run_name="__main__")

    assert exit_info.value.code == 0
    assert "MANIFEST CHECKS PASSED; DATASET QUALITY NOT VERIFIED" in report_path.read_text(encoding="utf-8")


def test_canonical_224x224_archive_passes_and_mixed_resolutions_rejected(tmp_path):
    rng = np.random.default_rng(2026)
    payloads_224 = [
        cv2.imencode(".png", rng.integers(60, 180, (224, 224, 3), dtype=np.uint8))[1].tobytes()
        for _ in range(3)
    ]
    df = manifest()
    df["sha256"] = [hashlib.sha256(p).hexdigest() for p in payloads_224]
    dir_224 = tmp_path / "golden_224"
    write_archive(dir_224, df, payloads_224)
    res_224 = audit.audit_dataset(dir_224)
    assert res_224["images"]["decoded_images"] == 3
    assert res_224["images"]["wrong_resolution"] == 0
    assert res_224["integrity_passed"]

    # Mixed 224x224 and 160x160 in the same archive must fail wrong_resolution
    payload_160 = cv2.imencode(".png", rng.integers(60, 180, (160, 160, 3), dtype=np.uint8))[1].tobytes()
    mixed_payloads = [payloads_224[0], payload_160, payloads_224[2]]
    df_mixed = manifest()
    df_mixed["sha256"] = [hashlib.sha256(p).hexdigest() for p in mixed_payloads]
    dir_mixed = tmp_path / "golden_mixed"
    write_archive(dir_mixed, df_mixed, mixed_payloads)
    res_mixed = audit.audit_dataset(dir_mixed)
    assert res_mixed["images"]["wrong_resolution"] >= 1
    assert not res_mixed["integrity_passed"]

