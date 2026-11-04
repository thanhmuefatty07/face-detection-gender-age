"""Create a Kaggle-side, source-namespaced identity split manifest."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import pandas as pd


def _is_kaggle_runtime() -> bool:
    # Kaggle's own runtime helper uses these environment markers; they are not cryptographic attestation.
    return (
        bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE"))
        and "KAGGLE_USER_SECRETS_TOKEN" in os.environ
        and Path("/kaggle/input").is_dir()
        and Path("/kaggle/working").is_dir()
    )


EXT = r"(?:jpg|jpeg|png|webp|bmp)"
SUBJECT_PATTERNS = {
    "agedb": re.compile(rf"^\d+_(.+)_\d+_[mf]\.({EXT})$", re.IGNORECASE),
    "morph": re.compile(rf"^(\d+)_\d+[mf]\d{{2}}[a-z0-9_]*\.({EXT})$", re.IGNORECASE),
    "fgnet": re.compile(rf"^(\d{{3}})a\d{{1,3}}[a-e]?(?:[_-]?[mf])?\.({EXT})$", re.IGNORECASE),
    "b3fd": re.compile(
        rf"^(.+?)_\d{{4}}(?:-\d{{1,2}}-\d{{1,2}})?_\d{{4}}\.({EXT})$",
        re.IGNORECASE,
    ),
    "cacd": re.compile(rf"^\d{{1,3}}_(.+?)_\d+\.({EXT})$", re.IGNORECASE),
}


def extract_subject_key(source: str, filename: str) -> str | None:
    """Parse only filename formats that encode a documented subject group."""
    pattern = SUBJECT_PATTERNS.get(str(source).strip().lower())
    if not pattern:
        return None
    match = pattern.fullmatch(Path(str(filename)).name)
    if not match:
        return None
    key = match.group(1).strip().casefold()
    return key or None


def curate_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    """Hash source-scoped identities, group-split known people, and quarantine unknown IDs in train."""
    required = {"image_id", "sha256", "source_dataset", "original_filename", "age_continuous"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest missing required columns: {sorted(missing)}")
    if manifest.empty:
        raise ValueError("Cannot curate an empty manifest")
    if manifest.image_id.isna().any() or manifest.image_id.duplicated().any():
        raise ValueError("Image IDs must be present and unique")
    if not manifest.image_id.astype(str).str.fullmatch(
        r"[^/\\:]+\.(?:jpg|jpeg|png|webp|bmp)", case=False
    ).all():
        raise ValueError("Image IDs must be safe image filenames")
    if not manifest.sha256.astype(str).str.fullmatch(r"[0-9a-fA-F]{64}").all():
        raise ValueError("Every image row must have a valid SHA-256 digest")
    if manifest.sha256.astype(str).str.lower().duplicated().any():
        raise ValueError("Duplicate image content must be resolved before curation")
    ages = pd.to_numeric(manifest.get("age_continuous"), errors="coerce")
    if not ages.notna().all() or not ages.between(0, 100).all():
        raise ValueError("Every row needs a finite age label in [0, 100]")

    result = manifest.copy(deep=True)
    subject_ids: list[str | None] = []
    identity_ids: list[str | None] = []
    splits: list[str] = []
    digest_owners: dict[str, tuple[str, str]] = {}
    compact_id_owners: dict[str, tuple[str, str]] = {}
    for source, filename in zip(result.source_dataset, result.original_filename):
        source_key = str(source).strip().lower()
        subject_key = extract_subject_key(source_key, filename)
        if subject_key is None:
            subject_ids.append(None)
            identity_ids.append(None)
            splits.append("train")
            continue
        group = (source_key, subject_key)
        digest = hashlib.sha256(f"{source_key}\0{subject_key}".encode("utf-8")).hexdigest()
        owner = digest_owners.setdefault(digest, group)
        if owner != group:
            raise RuntimeError("SHA-256 identity collision; refusing to merge subject groups")
        compact_id = str(int(digest[:13], 16))
        compact_owner = compact_id_owners.setdefault(compact_id, group)
        if compact_owner != group:
            raise RuntimeError("Compact identity collision; refusing to merge subject groups")
        subject_ids.append(digest)
        identity_ids.append(compact_id)
        bucket = int(digest[:16], 16) % 100
        splits.append("train" if bucket < 80 else ("val" if bucket < 90 else "test"))

    result["subject_id"] = pd.Series(subject_ids, index=result.index, dtype="object")
    result["identity_id"] = pd.Series(identity_ids, index=result.index, dtype="object")
    result["split"] = splits

    result["teacher_count"] = 0
    for column in ["teacher_age_mu", "teacher_age_sigma", "teacher_gender_agreement"]:
        result[column] = pd.Series([None] * len(result), index=result.index, dtype="object")
    if "apparent_age_std" in result:
        has_source_uncertainty = result.source_dataset.astype(str).str.lower().eq("appa_real")
        result.loc[~has_source_uncertainty, "apparent_age_std"] = None
    if "dimorphism_weight" in result:
        result["dimorphism_weight"] = 1.0
    if "alpha_sample_weight" in result:
        result["alpha_sample_weight"] = 1.0
    if "ethnicity" in result:
        result.loc[result.ethnicity.astype(str).str.strip().str.casefold().eq("diverse"), "ethnicity"] = None

    return result


def _find_input_root(input_dir: Path) -> Path:
    candidates = sorted(
        {p.parent for p in input_dir.rglob("metadata.parquet")}
        | {p.parent for p in input_dir.rglob("metadata.csv")}
    )
    if len(candidates) != 1:
        raise RuntimeError(f"Expected exactly one mining manifest root; found {len(candidates)}")
    return candidates[0]


def main() -> None:
    if not _is_kaggle_runtime():
        raise SystemExit("Kaggle-only: identity splitting must run inside a Kaggle notebook/runtime")
    input_root = _find_input_root(Path("/kaggle/input"))
    source_path = next(
        p for p in (input_root / "metadata.parquet", input_root / "metadata.csv") if p.is_file()
    )
    source = pd.read_parquet(source_path) if source_path.suffix == ".parquet" else pd.read_csv(source_path)
    curated = curate_manifest(source)

    output = Path("/kaggle/working/identity_grouped_dataset")
    output.mkdir(parents=True, exist_ok=True)
    curated.to_parquet(output / "metadata.parquet", index=False)
    curated.to_csv(output / "metadata.csv", index=False)

    summary = {
        "schema_version": 1,
        "source_manifest_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "rows": int(len(curated)),
        "split_counts": {str(k): int(v) for k, v in curated.split.value_counts().items()},
        "source_split_counts": pd.crosstab(curated.source_dataset, curated.split).to_dict(),
        "identity_rows": int(curated.subject_id.notna().sum()),
        "unknown_identity_rows": int(curated.subject_id.isna().sum()),
        "identity_groups": int(curated.subject_id.nunique()),
        "unknown_identity_holdout_rows": int(
            (curated.subject_id.isna() & curated.split.ne("train")).sum()
        ),
        "teacher_count_values": {
            str(k): int(v) for k, v in curated.teacher_count.value_counts(dropna=False).items()
        },
        "teacher_evidence_rows": int(curated.teacher_age_mu.notna().sum()),
        "identity_scope": "Source-namespaced; cross-source identity equivalence is unverified.",
        "split_rule": "SHA-256(subject group) mod 100: train <80, val <90, otherwise test; unknown IDs train-only.",
        "images_copied": False,
        "training_executed": False,
    }
    (output / "curation_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "source_split_counts"}, flush=True))


if __name__ == "__main__":
    main()
