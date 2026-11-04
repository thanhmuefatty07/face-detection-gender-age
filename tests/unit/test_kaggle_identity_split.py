import pandas as pd
import pytest

from scripts.kaggle_pipeline.kaggle_identity_split import (
    curate_manifest,
    extract_subject_key,
)


@pytest.mark.parametrize(
    "source, filename, expected",
    [
        ("agedb", "12_MariaCallas_35_f.jpg", "mariacallas"),
        ("agedb", "88_mariacallas_52_f.jpg", "mariacallas"),
        ("morph", "00001_01M25_B.jpg", "00001"),
        ("morph", "00001_02M30B1.JPG", "00001"),
        ("fgnet", "001A02.jpg", "001"),
        ("fgnet", "001A33.JPG", "001"),
        ("b3fd", "Jane_Doe_1974-01-01_2014.jpg", "jane_doe"),
    ],
)
def test_source_filename_identity_formats(source, filename, expected):
    assert extract_subject_key(source, filename) == expected


@pytest.mark.parametrize(
    "source, filename",
    [
        ("utkface", "25_0_2_20170116174525.jpg"),
        ("afad", "25_111_00001.jpg"),
        ("b3fd", "not-a-subject-age.jpg"),
    ],
)
def test_filename_fields_that_do_not_prove_identity_stay_unknown(source, filename):
    assert extract_subject_key(source, filename) is None


def test_curated_manifest_groups_people_and_keeps_unknowns_in_train():
    source = pd.DataFrame(
        {
            "image_id": [f"face_{i:016x}.jpg" for i in range(5)],
            "sha256": [f"{i + 1:064x}" for i in range(5)],
            "source_dataset": ["agedb", "agedb", "morph", "morph", "utkface"],
            "age_continuous": [35.0, 52.0, 25.0, 30.0, 25.0],
            "original_filename": [
                "12_MariaCallas_35_f.jpg",
                "88_MariaCallas_52_f.jpg",
                "00001_01M25_B.jpg",
                "00001_02M30B1.JPG",
                "25_0_2_20170116174525.jpg",
            ],
            "split": ["test", "train", "val", "test", "val"],
            "teacher_count": [4] * 5,
            "teacher_age_mu": [35.0] * 5,
            "teacher_age_sigma": [2.0] * 5,
            "teacher_gender_agreement": [0.95] * 5,
            "apparent_age_std": [2.0] * 5,
            "dimorphism_weight": [0.8] * 5,
            "alpha_sample_weight": [0.3] * 5,
            "ethnicity": ["Diverse"] * 5,
        }
    )

    curated = curate_manifest(source)

    assert curated.loc[0, "subject_id"] == curated.loc[1, "subject_id"]
    assert curated.loc[0, "identity_id"] == curated.loc[1, "identity_id"]
    assert curated.loc[2, "subject_id"] == curated.loc[3, "subject_id"]
    assert curated.loc[2, "identity_id"] == curated.loc[3, "identity_id"]
    assert curated.loc[0, "split"] == curated.loc[1, "split"]
    assert curated.loc[2, "split"] == curated.loc[3, "split"]
    assert curated.loc[4, "subject_id"] is None
    assert curated.loc[4, "split"] == "train"
    assert set(curated["teacher_count"]) == {0}
    assert curated[["teacher_age_mu", "teacher_age_sigma", "teacher_gender_agreement"]].isna().all().all()
    assert curated["apparent_age_std"].isna().all()
    assert set(curated["dimorphism_weight"]) == {1.0}
    assert set(curated["alpha_sample_weight"]) == {1.0}
    assert curated["ethnicity"].isna().all()
    assert source["split"].tolist() == ["test", "train", "val", "test", "val"]


def test_curated_identity_hashes_are_source_namespaced_and_deterministic():
    frame = pd.DataFrame(
        {
            "image_id": ["face_01.jpg", "face_02.jpg", "face_03.jpg"],
            "sha256": [f"{i:064x}" for i in range(1, 4)],
            "source_dataset": ["agedb", "agedb", "morph"],
            "age_continuous": [35.0, 52.0, 35.0],
            "original_filename": [
                "12_MariaCallas_35_f.jpg",
                "88_MariaCallas_52_f.jpg",
                "12_02F35W.jpg",
            ],
            "split": ["train", "val", "test"],
        }
    )

    first = curate_manifest(frame)
    second = curate_manifest(frame.sample(frac=1, random_state=9).reset_index(drop=True))

    agedb_id = first.loc[0, "subject_id"]
    morph_id = first.loc[2, "subject_id"]
    assert isinstance(agedb_id, str) and len(agedb_id) == 64
    assert int(first.loc[0, "identity_id"]) < 2**53
    assert agedb_id == first.loc[1, "subject_id"]
    assert agedb_id != morph_id
    assert dict(zip(first["original_filename"], first["split"])) == dict(
        zip(second["original_filename"], second["split"])
    )
