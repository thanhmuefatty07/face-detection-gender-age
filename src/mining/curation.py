"""
Unified Local & Kaggle Dataset Curation Pipeline
================================================
Combines multi-source dataset harvesting (16 real human face dataset connectors),
5-point Umeyama similarity alignment (160x160), optical/photometric/pose quality
auditing (`ExtremeQualityAuditor`), dual-layer perceptual + metric deduplication
(`DualLayerDeduplicator`), source-namespaced SHA-256 identity splitting, and
3-5 year interval + 101-bin Box-Convolved LDL manifest generation.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd

from ..biometrics.core.geometry import ARCFACE_CANONICAL_112, UmeyamaAligner
from ..biometrics.core.types import FaceLandmarks
from ..biometrics.estimation.dimorphism import (
    AGE_COHORTS,
    calculate_ethnic_aging_offset,
    calculate_sexual_dimorphism_weight,
    get_cohort_for_age,
)
from ..biometrics.estimation.intervals import (
    FINE_AGE_INTERVALS,
    compute_interval_box_ldl,
    compute_interval_epistemic_weight,
    get_fine_interval_for_age,
    integrate_101_to_fine_intervals,
    map_bounds_to_fine_interval,
)
from ..biometrics.estimation.ldl_engine import ContinuousLDLEngine, compute_gaussian_ldl
from .connectors import (
    BaseFaceConnector,
    DiskQuotaManager,
    InCloudStreamingHarvester,
    RawImageSample,
    SOURCE_LICENSE_PROVENANCE,
    csv_safe_manifest,
    is_banned_non_real_source,
)
from .filtering.deduplicator import DualLayerDeduplicator
from .filtering.quality_auditor import ExtremeQualityAuditor


@dataclass
class CurationSummary:
    """Structured telemetry from a local or Kaggle curation run."""
    total_ingested: int = 0
    accepted: int = 0
    rejected_decode: int = 0
    rejected_quality: int = 0
    rejected_duplicate: int = 0
    rejection_breakdown: Dict[str, int] = field(default_factory=dict)
    interval_distribution: Dict[str, int] = field(default_factory=dict)
    gender_distribution: Dict[str, int] = field(default_factory=dict)
    source_distribution: Dict[str, int] = field(default_factory=dict)
    manifest_parquet_path: Optional[Path] = None
    manifest_csv_path: Optional[Path] = None


def canonicalize_split_identity_key(identity_key: str) -> str:
    """
    Normalizes named celebrity identities across datasets ('agedb', 'b3fd', 'cacd', 'imdb_wiki')
    so the same public figure is deterministically assigned to the same split across sources,
    preventing cross-dataset celebrity leakage while preserving per-source subject_id in the manifest.
    """
    import re as _re

    if ":" in identity_key:
        src, raw_subj = identity_key.split(":", 1)
        src_clean = src.strip().lower()
        if src_clean in {"agedb", "b3fd", "cacd", "imdb_wiki"}:
            cleaned_subj = _re.sub(
                r"^(nm\d+)_rm[a-z0-9]+.*$", r"\1", raw_subj.strip(), flags=_re.IGNORECASE
            )
            norm = _re.sub(r"[^a-z0-9]", "", cleaned_subj.casefold())
            if norm and any(c.isalpha() for c in norm):
                return f"celebrity:{norm}"
    return identity_key


def deterministic_identity_split(
    identity_key: str,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
) -> str:
    """
    Assigns all images of a given `source:subject_id` identity deterministically
    to the exact same split ('train', 'val', or 'test') via SHA-256 hashing,
    preventing cross-split subject leakage (including cross-dataset celebrity overlap).
    """
    canonical_key = canonicalize_split_identity_key(identity_key)
    digest = hashlib.sha256(canonical_key.encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") / float(1 << 64)
    if bucket < train_ratio:
        return "train"
    elif bucket < (train_ratio + val_ratio):
        return "val"
    return "test"



class ExtremeDatasetCurator:
    """
    End-to-end local and Kaggle dataset curator producing 160x160 canonical
    aligned faces and a 3-5 year interval + 101-bin LDL Golden Manifest.
    """

    COHORT_COL_NAMES = [
        "ldl_0_2",
        "ldl_3_6",
        "ldl_7_12",
        "ldl_13_17",
        "ldl_18_24",
        "ldl_25_34",
        "ldl_35_44",
        "ldl_45_54",
        "ldl_55_64",
        "ldl_65_plus",
    ]
    FINE_INTERVAL_COL_NAMES = [
        f"fine_ldl_{(iv[0] if isinstance(iv, tuple) else iv.name).replace('-', '_')}"
        for iv in FINE_AGE_INTERVALS
    ]

    def __init__(
        self,
        output_dir: str | Path,
        auditor: Optional[ExtremeQualityAuditor] = None,
        deduplicator: Optional[DualLayerDeduplicator] = None,
        write_tar_shards: bool = True,
        write_direct_images: bool = False,
        max_storage_gb: float = 18.0,
        shard_image_count: int = 5000,
        min_free_gb: float = 0.05,
        exclude_binary_only_age: bool = True,
        target_size: int = 160,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.target_size = int(target_size)
        self.auditor = auditor or ExtremeQualityAuditor()
        self.deduplicator = deduplicator or DualLayerDeduplicator()
        self.aligner = UmeyamaAligner(target_size=self.target_size)
        self.ldl_engine = ContinuousLDLEngine(num_bins=101)
        self.write_tar_shards = write_tar_shards
        self.write_direct_images = write_direct_images
        self.max_storage_gb = max_storage_gb
        self.shard_image_count = shard_image_count
        self.min_free_gb = min_free_gb
        self.exclude_binary_only_age = exclude_binary_only_age

    @staticmethod
    def _default_canonical_landmarks(width: int, height: int) -> np.ndarray:
        """
        Synthesizes canonical 5-point landmarks scaled to (width, height)
        for pre-cropped dataset images that do not ship explicit 5-point coordinates.
        """
        scale_x = float(width) / 112.0
        scale_y = float(height) / 112.0
        pts = ARCFACE_CANONICAL_112.copy()
        pts[:, 0] *= scale_x
        pts[:, 1] *= scale_y
        return pts.astype(np.float32)


    @staticmethod
    def _extract_subject_identity(sample: RawImageSample) -> Optional[str]:
        """
        Builds a source-namespaced identity key `{source_dataset}:{subject_id}`
        when a subject identifier exists in metadata or canonical filename pattern,
        or returns None for unannotated singletons so unknown identities are never placed in val/test.
        """
        import re as _re

        meta = sample.metadata or {}
        for key in (
            "subject_id",
            "name",
            "morph_subject_id",
            "fgnet_subject_id",
            "b3fd_id",
            "user_id",
            "subject_name",
            "person_id",
            "agedb_id",
        ):
            val = meta.get(key)
            if val is not None and str(val).strip():
                raw_s = str(val).strip()
                if raw_s.lower() in {"none", "nan", "null", "-1", "", "unknown"}:
                    continue
                if ":" in raw_s:
                    src_p, id_p = raw_s.split(":", 1)
                    id_p = _re.sub(
                        r"^(nm\d+)_rm[a-z0-9]+.*$", r"\1", id_p.strip(), flags=_re.IGNORECASE
                    )
                    return f"{src_p}:{id_p}"
                raw_s = _re.sub(
                    r"^(nm\d+)_rm[a-z0-9]+.*$", r"\1", raw_s, flags=_re.IGNORECASE
                )
                return f"{sample.source_dataset}:{raw_s}"

        source = str(sample.source_dataset or "").strip().lower()
        name = str(sample.raw_id or "")
        pattern = {
            "agedb": r"^\d+_(.+)_\d+_[mf]\.",
            "fgnet": r"^(\d{3})[aA]\d{1,3}[a-eA-E]?(?:[_-]?[mMfF])?\.",
            "morph": r"^(\d+)_\d+[MFmf]\d{2}[A-Za-z0-9_]*\.",
            "b3fd": r"^(?:(nm\d+)_rm[A-Za-z0-9]+|(.+?))_\d{4}(?:-\d{1,2}-\d{1,2})?_\d{4}\.",
            "cacd": r"^\d+_([a-z][a-z0-9_'\-]+?)_\d+\.",
            "imdb_wiki": r"^(?:(nm\d+)_[A-Za-z0-9]+|(\d+))_\d{4}-\d{1,2}-\d{1,2}_\d{4}\.",
        }.get(source)
        if pattern:
            match = _re.match(pattern, name, flags=_re.IGNORECASE)
            if match:
                subj_grp = next((g for g in match.groups() if g), None)
                if subj_grp:
                    return f"{sample.source_dataset}:{subj_grp.casefold()}"
        return None

    @staticmethod
    def _sync_shards_with_master(df: pd.DataFrame, shards_dir: Optional[Path]) -> None:
        """Synchronizes post-processed master columns back into individual shard manifests."""
        if df.empty or shards_dir is None or not shards_dir.exists() or "image_id" not in df.columns:
            return
        sync_cols = [
            c for c in (
                "split",
                "subject_id",
                "identity_id",
                "gender",
                "gender_prob",
                "ethnicity",
                "alpha_sample_weight",
                "skin_tone_ita_deg",
                "skin_tone_group",
            )
            if c in df.columns
        ]
        if not sync_cols:
            return
        col_maps = {c: dict(zip(df["image_id"], df[c])) for c in sync_cols}
        for sp in sorted(shards_dir.glob("shard_*_manifest.parquet")):
            try:
                sdf = pd.read_parquet(sp)
                if "image_id" in sdf.columns:
                    for c, cmap in col_maps.items():
                        sdf[c] = sdf["image_id"].map(cmap)
                    sdf.to_parquet(str(sp), index=False)
            except Exception:
                pass

    @staticmethod
    def _build_identity_phash_components(
        df: pd.DataFrame,
        hamming_threshold: int = 6,
    ) -> List[List[int]]:
        """
        Builds connected components over:
          1. Source-scoped subject keys (`subject_id`)
          2. Cross-dataset canonical celebrity keys (`canonicalize_split_identity_key`)
          3. 64-bit DCT perceptual hash near-neighbors (`d_H <= hamming_threshold`)
        """
        n = len(df)
        if n == 0:
            return []
        parent = list(range(n))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(i: int, j: int) -> None:
            ri, rj = find(i), find(j)
            if ri != rj:
                if ri < rj:
                    parent[rj] = ri
                else:
                    parent[ri] = rj

        if "subject_id" in df.columns:
            subj_first: Dict[str, int] = {}
            canon_first: Dict[str, int] = {}
            subjs = df["subject_id"].tolist()
            for idx, raw_s in enumerate(subjs):
                if pd.notna(raw_s) and str(raw_s).strip() and str(raw_s).strip().lower() not in {"none", "nan"}:
                    s_str = str(raw_s).strip()
                    if s_str in subj_first:
                        union(idx, subj_first[s_str])
                    else:
                        subj_first[s_str] = idx
                    ckey = canonicalize_split_identity_key(s_str)
                    if ckey in canon_first:
                        union(idx, canon_first[ckey])
                    else:
                        canon_first[ckey] = idx

        if "phash" in df.columns:
            count = hamming_threshold + 1
            edges = [(64 * i // count, 64 * (i + 1) // count) for i in range(count)]
            buckets: List[Dict[int, List[int]]] = [{} for _ in edges]
            parsed_hashes: List[Optional[int]] = []
            for val in df["phash"].tolist():
                try:
                    h_int = int(str(val).strip(), 16) if isinstance(val, str) else int(val)
                    parsed_hashes.append(h_int if 0 <= h_int < (1 << 64) else None)
                except Exception:
                    parsed_hashes.append(None)

            for i, h_val in enumerate(parsed_hashes):
                if h_val is None:
                    continue
                chunks = [(h_val >> lo) & ((1 << (hi - lo)) - 1) for lo, hi in edges]
                checked: set[int] = set()
                for b_idx, chunk in enumerate(chunks):
                    for j in buckets[b_idx].get(chunk, ()):
                        if j in checked:
                            continue
                        checked.add(j)
                        hj = parsed_hashes[j]
                        if hj is not None and (h_val ^ hj).bit_count() <= hamming_threshold:
                            union(i, j)
                for b_idx, chunk in enumerate(chunks):
                    buckets[b_idx].setdefault(chunk, []).append(i)

        comp_rows: Dict[int, List[int]] = {}
        for idx in range(n):
            comp_rows.setdefault(find(idx), []).append(idx)
        return list(comp_rows.values())

    @classmethod
    def _enforce_transitive_identity_and_phash_disjointness(
        cls,
        df: pd.DataFrame,
        hamming_threshold: int = 6,
    ) -> pd.DataFrame:
        """
        Builds a transitive Union-Find graph over:
          1. Source-scoped subject keys (`subject_id`)
          2. Cross-dataset canonical celebrity keys (`canonicalize_split_identity_key`)
          3. 64-bit DCT perceptual hash near-neighbors (`d_H <= hamming_threshold`)
        Guarantees:
          - Every connected component is assigned atomically and order-invariantly to a single split.
          - Any component containing an unknown-identity row is quarantined to 'train'.
          - Zero cross-split subject leakage and zero cross-split pHash leakage.
        """
        if df.empty or "split" not in df.columns or "subject_id" not in df.columns:
            return df

        components = cls._build_identity_phash_components(df, hamming_threshold=hamming_threshold)
        subjs = df["subject_id"].tolist()
        splits_list = df["split"].astype(str).tolist()
        known_entries: List[Tuple[str, List[int]]] = []
        unknown_comps: List[List[int]] = []

        for comp_indices in components:
            has_unknown = any(
                pd.isna(subjs[i]) or str(subjs[i]).strip().lower() in {"", "none", "nan"}
                for i in comp_indices
            )
            if has_unknown:
                unknown_comps.append(comp_indices)
                for i in comp_indices:
                    splits_list[i] = "train"
            else:
                rep_key = min(
                    canonicalize_split_identity_key(str(subjs[i]).strip()) for i in comp_indices
                )
                known_entries.append((rep_key, comp_indices))

        known_entries.sort(key=lambda item: item[0])
        known_comps: List[List[int]] = []
        comp_split_map: List[str] = []
        for rep_key, comp_indices in known_entries:
            target_sp = deterministic_identity_split(rep_key)
            known_comps.append(comp_indices)
            comp_split_map.append(target_sp)
            for i in comp_indices:
                splits_list[i] = target_sp

        total_distinct_units = len(known_comps) + (1 if unknown_comps else 0)
        if len(known_comps) >= 2 and total_distinct_units >= 3:
            for missing_split in ("train", "val", "test"):
                if missing_split not in set(splits_list):
                    for donor_split in ("train", "val", "test"):
                        donor_k_indices = [
                            k_idx for k_idx, sp in enumerate(comp_split_map) if sp == donor_split
                        ]
                        total_in_donor = len(donor_k_indices) + (
                            len(unknown_comps) if donor_split == "train" else 0
                        )
                        if len(donor_k_indices) >= 1 and total_in_donor >= 2:
                            chosen_k = donor_k_indices[-1]
                            comp_split_map[chosen_k] = missing_split
                            for row_i in known_comps[chosen_k]:
                                splits_list[row_i] = missing_split
                            break

        df["split"] = splits_list
        return df

    @classmethod
    def _ensure_all_splits_covered(cls, df: pd.DataFrame, shards_dir: Optional[Path] = None) -> pd.DataFrame:
        """
        Ensures train, val, and test splits are all non-empty whenever >= 3 distinct
        identity components exist, without splitting any subject, cross-dataset celebrity,
        or pHash near-neighbor cluster across splits.
        """
        if df.empty or "split" not in df.columns or "subject_id" not in df.columns:
            return df
        df = cls._enforce_transitive_identity_and_phash_disjointness(df)
        cls._sync_shards_with_master(df, shards_dir)
        return df

    @classmethod
    def _propagate_subject_gender(cls, df: pd.DataFrame, shards_dir: Optional[Path] = None) -> pd.DataFrame:
        """
        Propagates unanimous known subject gender ('Male'/'Female') and known ethnicity
        across rows sharing the same canonical subject identity or transitive known-identity
        pHash cluster (e.g., filling b3fd/imdb_wiki celebrity gender & demography from agedb/cacd).
        """
        if df.empty or "subject_id" not in df.columns:
            return df
        known_mask = (
            df["subject_id"].notna()
            & (df["subject_id"].astype(str) != "")
            & (df["subject_id"].astype(str) != "None")
        )
        if not known_mask.any():
            return df

        components = cls._build_identity_phash_components(df, hamming_threshold=6)

        if "gender" in df.columns:
            canon_genders: Dict[str, set] = {}
            for sid, g in zip(df.loc[known_mask, "subject_id"], df.loc[known_mask, "gender"]):
                if g in ("Male", "Female"):
                    ckey = canonicalize_split_identity_key(str(sid))
                    canon_genders.setdefault(ckey, set()).add(str(g))

            unanimous_g = {k: next(iter(v)) for k, v in canon_genders.items() if len(v) == 1}
            if unanimous_g:
                for idx in df.index[known_mask & ~df["gender"].isin(["Male", "Female"])]:
                    ckey = canonicalize_split_identity_key(str(df.at[idx, "subject_id"]))
                    if ckey in unanimous_g:
                        resolved_g = unanimous_g[ckey]
                        df.at[idx, "gender"] = resolved_g
                        if "gender_prob" in df.columns:
                            df.at[idx, "gender_prob"] = 1.0 if resolved_g == "Female" else 0.0

            g_col = df["gender"].tolist()
            for comp_indices in components:
                comp_g = {g_col[i] for i in comp_indices if g_col[i] in ("Male", "Female")}
                if len(comp_g) == 1:
                    resolved_g = next(iter(comp_g))
                    for i in comp_indices:
                        if g_col[i] not in ("Male", "Female"):
                            row_label = df.index[i]
                            df.at[row_label, "gender"] = resolved_g
                            if "gender_prob" in df.columns:
                                df.at[row_label, "gender_prob"] = 1.0 if resolved_g == "Female" else 0.0

        if "ethnicity" in df.columns:
            uninformative_eth = {"", "unknown", "diverse", "none", "nan", "null"}
            canon_eths: Dict[str, set] = {}
            for sid, eth in zip(df.loc[known_mask, "subject_id"], df.loc[known_mask, "ethnicity"]):
                eth_s = str(eth or "").strip()
                if eth_s.lower() not in uninformative_eth:
                    ckey = canonicalize_split_identity_key(str(sid))
                    canon_eths.setdefault(ckey, set()).add(eth_s)
            unanimous_eth = {k: next(iter(v)) for k, v in canon_eths.items() if len(v) == 1}
            if unanimous_eth:
                for idx in df.index[known_mask]:
                    cur_e = str(df.at[idx, "ethnicity"] or "").strip().lower()
                    if cur_e in uninformative_eth:
                        ckey = canonicalize_split_identity_key(str(df.at[idx, "subject_id"]))
                        if ckey in unanimous_eth:
                            df.at[idx, "ethnicity"] = unanimous_eth[ckey]

            e_col = df["ethnicity"].tolist()
            for comp_indices in components:
                comp_e = {
                    str(e_col[i] or "").strip()
                    for i in comp_indices
                    if str(e_col[i] or "").strip().lower() not in uninformative_eth
                }
                if len(comp_e) == 1:
                    resolved_e = next(iter(comp_e))
                    for i in comp_indices:
                        if str(e_col[i] or "").strip().lower() in uninformative_eth:
                            row_label = df.index[i]
                            df.at[row_label, "ethnicity"] = resolved_e

        cls._sync_shards_with_master(df, shards_dir)
        return df

    @classmethod
    def _calibrate_demographic_sample_weights(
        cls, df: pd.DataFrame, shards_dir: Optional[Path] = None
    ) -> pd.DataFrame:
        """
        Calibrates `alpha_sample_weight` via joint marginal inverse-frequency debiasing across
        (age_interval, ethnicity, skin_tone_group, gender) combined with biological dimorphism
        and interval epistemic certainty, ensuring tail cohorts and underrepresented skin-tone
        or ethnic strata receive mathematically balanced gradient mass.
        """
        if df.empty or "alpha_sample_weight" not in df.columns:
            return df

        n = float(len(df))
        if n <= 1.0:
            return df

        def _marginal_boost(series: pd.Series, power: float) -> np.ndarray:
            vals = series.fillna("unknown").astype(str)
            counts = vals.value_counts()
            k = max(1.0, float(len(counts)))
            target = n / k
            freq = vals.map(counts).to_numpy(dtype=np.float64)
            return np.power(np.clip(target / np.maximum(freq, 1.0), 0.4, 4.0), power)

        w_age = (
            _marginal_boost(df["age_interval"], 0.30)
            if "age_interval" in df.columns
            else np.ones(len(df), dtype=np.float64)
        )
        w_eth = (
            _marginal_boost(df["ethnicity"], 0.25)
            if "ethnicity" in df.columns
            else np.ones(len(df), dtype=np.float64)
        )
        w_skin = (
            _marginal_boost(df["skin_tone_group"], 0.20)
            if "skin_tone_group" in df.columns
            else np.ones(len(df), dtype=np.float64)
        )
        w_gen = (
            _marginal_boost(df["gender"], 0.15)
            if "gender" in df.columns
            else np.ones(len(df), dtype=np.float64)
        )

        demo_factor = np.clip(w_age * w_eth * w_skin * w_gen, 0.50, 2.50)
        mean_demo = float(np.mean(demo_factor))
        if mean_demo > 1e-6:
            demo_factor = demo_factor / mean_demo

        base_w = pd.to_numeric(df["alpha_sample_weight"], errors="coerce").fillna(0.5).to_numpy(dtype=np.float64)
        calibrated = np.clip(base_w * demo_factor, 0.05, 5.0)
        df["alpha_sample_weight"] = [round(float(x), 4) for x in calibrated]
        cls._sync_shards_with_master(df, shards_dir)
        return df

    @staticmethod
    def _subject_key_to_identity_id(subject_key: Optional[str]) -> float:
        """Maps a known subject_key 1-to-1 to a 52-bit integer float in [0, 2**53), or NaN if unknown."""
        if not subject_key:
            return float("nan")
        digest = hashlib.sha256(subject_key.encode("utf-8")).hexdigest()
        return float(int(digest[:13], 16))

    def curate_samples(
        self,
        samples: Iterable[RawImageSample],
        max_accepted: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, CurationSummary]:
        """
        Processes a stream of RawImageSample candidates through quality auditing,
        deduplication, and 3-5 year interval LDL target synthesis.
        """
        summary = CurationSummary()
        records: List[Dict[str, Any]] = []

        quota_mgr: Optional[DiskQuotaManager] = None
        if self.write_tar_shards:
            quota_mgr = DiskQuotaManager(
                working_dir=str(self.output_dir),
                max_storage_gb=self.max_storage_gb,
                min_free_gb=self.min_free_gb,
                shard_image_count=self.shard_image_count,
            )

        images_dir = self.output_dir / "images"
        if self.write_direct_images:
            images_dir.mkdir(parents=True, exist_ok=True)

        try:
            for sample in samples:
                if max_accepted is not None and summary.accepted >= max_accepted:
                    break

                summary.total_ingested += 1
                is_banned_src, ban_code = is_banned_non_real_source(sample.source_dataset)
                if not is_banned_src:
                    is_banned_src, ban_code = is_banned_non_real_source(sample.raw_id)
                if is_banned_src and ban_code is not None:
                    summary.rejected_quality += 1
                    summary.rejection_breakdown[ban_code] = (
                        summary.rejection_breakdown.get(ban_code, 0) + 1
                    )
                    continue

                meta_has_valid_age = (sample.metadata or {}).get("has_valid_age")
                if self.exclude_binary_only_age and (
                    str(sample.age_label_type or "").strip().lower() == "binary"
                    or str(sample.source_dataset or "").strip().lower() == "celeba"
                    or meta_has_valid_age is False
                ):
                    summary.rejected_quality += 1
                    summary.rejection_breakdown["REJECTED_BINARY_PSEUDO_AGE"] = (
                        summary.rejection_breakdown.get("REJECTED_BINARY_PSEUDO_AGE", 0) + 1
                    )
                    continue

                meta_real_age = (sample.metadata or {}).get("real_age")
                has_real_age = meta_real_age is not None and not math.isnan(float(meta_real_age))
                has_decl_age = sample.declared_age is not None and not math.isnan(float(sample.declared_age))
                has_bounds = (
                    sample.age_min is not None
                    and sample.age_max is not None
                    and not math.isnan(float(sample.age_min))
                    and not math.isnan(float(sample.age_max))
                )
                if not (has_real_age or has_decl_age or has_bounds):
                    summary.rejected_quality += 1
                    summary.rejection_breakdown["NO_AGE"] = (
                        summary.rejection_breakdown.get("NO_AGE", 0) + 1
                    )
                    continue

                bgr = sample.to_bgr_image()
                if bgr is None:
                    summary.rejected_decode += 1
                    continue

                raw_h, raw_w = bgr.shape[:2]
                tsz = self.target_size
                if sample.declared_landmarks is not None and sample.declared_landmarks.shape == (5, 2):
                    lm_pts = sample.declared_landmarks.astype(np.float32)
                    aligned_160 = self.aligner.align(bgr, FaceLandmarks(points=lm_pts))
                else:
                    lm_pts = self._default_canonical_landmarks(raw_w, raw_h)
                    aligned_160 = (
                        cv2.resize(bgr, (tsz, tsz), interpolation=cv2.INTER_CUBIC)
                        if (raw_h, raw_w) != (tsz, tsz)
                        else bgr.copy()
                    )

                # Encode aligned canonical JPEG first and audit/hash the decoded JPEG buffer
                # so manifest quality metrics and pHash match the archived TAR bytes identically.
                ok, enc_buf = cv2.imencode(".jpg", aligned_160, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                if not ok:
                    summary.rejected_decode += 1
                    continue
                encoded_bytes = enc_buf.tobytes()
                decoded_160 = cv2.imdecode(enc_buf, cv2.IMREAD_COLOR)
                if decoded_160 is None or decoded_160.shape != (tsz, tsz, 3):
                    summary.rejected_decode += 1
                    continue

                # Quality audit on the exact stored JPEG representation
                audit = self.auditor.audit_face(
                    decoded_160,
                    landmarks_5=lm_pts if sample.declared_landmarks is not None else None,
                    raw_width=raw_w,
                    raw_height=raw_h,
                )
                if not audit.passed:
                    summary.rejected_quality += 1
                    for reason in audit.rejection_reasons:
                        tag = reason.split(" ")[0]
                        summary.rejection_breakdown[tag] = summary.rejection_breakdown.get(tag, 0) + 1
                    continue

                # Dual-layer deduplication on the exact stored JPEG representation
                is_dup, dup_reason, phash_val = self.deduplicator.check_and_register(decoded_160)
                if is_dup:
                    summary.rejected_duplicate += 1
                    summary.rejection_breakdown[dup_reason] = (
                        summary.rejection_breakdown.get(dup_reason, 0) + 1
                    )
                    continue

                sha256_hex = hashlib.sha256(encoded_bytes).hexdigest()
                image_id = f"{sample.source_dataset}_{sha256_hex[:16]}.jpg"

                # Resolve chronological age & 3-5 year interval bounds
                # When metadata provides explicit chronological 'real_age' (e.g. APPA-REAL),
                # prioritize chronological real_age over perceived apparent_age_avg.
                age_min = float(sample.age_min) if sample.age_min is not None else 0.0
                age_max = float(sample.age_max) if sample.age_max is not None else 0.0
                if has_real_age:
                    raw_real = float(meta_real_age)
                    if not (0.0 <= raw_real <= 120.0):
                        summary.rejected_quality += 1
                        summary.rejection_breakdown["INVALID_AGE"] = (
                            summary.rejection_breakdown.get("INVALID_AGE", 0) + 1
                        )
                        continue
                    age_val = round(max(0.0, min(100.0, raw_real)), 2)
                    fine_name, fine_idx, low_i, high_i = get_fine_interval_for_age(age_val)
                    age_min, age_max = float(low_i), float(high_i)
                else:
                    if has_decl_age and not (0.0 <= float(sample.declared_age) <= 120.0):
                        summary.rejected_quality += 1
                        summary.rejection_breakdown["INVALID_AGE"] = (
                            summary.rejection_breakdown.get("INVALID_AGE", 0) + 1
                        )
                        continue
                    if has_bounds:
                        raw_lo = min(float(sample.age_min), float(sample.age_max))
                        raw_hi = max(float(sample.age_min), float(sample.age_max))
                        if raw_lo < 0.0 or raw_lo > 100.0 or raw_hi > 120.0:
                            summary.rejected_quality += 1
                            summary.rejection_breakdown["INVALID_AGE"] = (
                                summary.rejection_breakdown.get("INVALID_AGE", 0) + 1
                            )
                            continue
                        age_min, age_max = raw_lo, min(100.0, raw_hi)
                    age_val = (
                        float(sample.declared_age)
                        if has_decl_age
                        else round(0.5 * (age_min + age_max), 2)
                    )
                    age_val = round(max(0.0, min(100.0, age_val)), 2)
                    if not has_bounds:
                        fine_name, fine_idx, low_i, high_i = get_fine_interval_for_age(age_val)
                        age_min, age_max = float(low_i), float(high_i)
                    else:
                        fine_name, fine_idx, _, _ = map_bounds_to_fine_interval(age_min, age_max)

                cohort_name, cohort_idx = get_cohort_for_age(age_val)
                ethnicity = sample.declared_ethnicity or "Unknown"
                ethnic_offset = calculate_ethnic_aging_offset(age_val, ethnicity)
                if meta_real_age is not None and sample.declared_age is not None:
                    apparent_age = max(0.0, min(100.0, round(float(sample.declared_age), 2)))
                else:
                    apparent_age = max(0.0, min(100.0, round(age_val + ethnic_offset, 2)))
                apparent_std = float((sample.metadata or {}).get("apparent_age_std", 2.5))
                apparent_std = max(0.05, min(25.0, apparent_std))

                # 101-bin LDL (Gaussian for exact scalar age; Box-Convolved for interval brackets) -> 10 cohorts + 23 fine intervals
                is_interval_only = (
                    has_bounds
                    and not has_real_age
                    and str(getattr(sample, "age_label_type", "exact")).strip().lower() == "interval"
                    and (age_max - age_min) > 1e-3
                )
                if is_interval_only:
                    ldl_101 = compute_interval_box_ldl(
                        age_min=age_min,
                        age_max=age_max,
                        sigma=apparent_std,
                        num_bins=101,
                    )
                else:
                    ldl_101 = compute_gaussian_ldl(
                        mean_age=age_val,
                        sigma=apparent_std,
                        num_bins=101,
                    )
                ldl_10_raw = self.ldl_engine.integrate_to_10_cohorts(ldl_101)
                ldl_10_floats = np.maximum(np.asarray(ldl_10_raw, dtype=np.float64), 1e-5)
                ldl_10_floats /= np.sum(ldl_10_floats)
                ldl_10_rounded = [round(float(v), 5) for v in ldl_10_floats]
                ldl_10_rounded = [max(1e-5, v) for v in ldl_10_rounded]
                max_c_idx = int(np.argmax(ldl_10_rounded))
                diff_sum = round(1.0 - sum(ldl_10_rounded), 5)
                ldl_10_rounded[max_c_idx] = round(max(1e-5, ldl_10_rounded[max_c_idx] + diff_sum), 5)

                ldl_23 = integrate_101_to_fine_intervals(ldl_101)
                ldl_23_floats = np.maximum(np.asarray(ldl_23, dtype=np.float64), 1e-5)
                ldl_23_floats /= np.sum(ldl_23_floats)
                ldl_23_rounded = [round(float(v), 5) for v in ldl_23_floats]
                ldl_23_rounded = [max(1e-5, v) for v in ldl_23_rounded]
                max_f_idx = int(np.argmax(ldl_23_rounded))
                diff_f_sum = round(1.0 - sum(ldl_23_rounded), 5)
                ldl_23_rounded[max_f_idx] = round(max(1e-5, ldl_23_rounded[max_f_idx] + diff_f_sum), 5)

                dimorphism_w = calculate_sexual_dimorphism_weight(age_val)
                epistemic_w = compute_interval_epistemic_weight(
                    age_min=age_min,
                    age_max=age_max,
                    label_type=sample.age_label_type,
                )
                alpha_w = round(float(dimorphism_w * max(0.25, epistemic_w)), 4)

                subject_key = self._extract_subject_identity(sample)
                identity_id = self._subject_key_to_identity_id(subject_key)
                split = deterministic_identity_split(subject_key) if subject_key is not None else "train"

                has_valid_age = sample.age_label_type != "binary"
                gender_str = (
                    sample.declared_gender
                    if sample.declared_gender in ("Male", "Female")
                    else "unknown"
                )
                if gender_str == "Female":
                    gender_prob: float = 1.0
                elif gender_str == "Male":
                    gender_prob = 0.0
                else:
                    gender_prob = 0.5

                record: Dict[str, Any] = {
                    "image_id": image_id,
                    "sha256": sha256_hex,
                    "phash": f"{phash_val:016x}",
                    "source_dataset": sample.source_dataset,
                    "license_provenance": SOURCE_LICENSE_PROVENANCE.get(
                        str(sample.source_dataset).lower(),
                        "Academic Research Only (Non-Commercial)",
                    ),
                    "original_filename": str(sample.raw_id),
                    "raw_id": str(sample.raw_id),
                    "subject_id": subject_key,
                    "identity_id": identity_id,
                    "split": split,
                    "age_continuous": round(age_val, 2),
                    "apparent_age": round(apparent_age, 2),
                    "apparent_age_std": round(apparent_std, 3),
                    "age_min": round(age_min, 2),
                    "age_max": round(age_max, 2),
                    "age_interval": fine_name,
                    "age_interval_idx": int(fine_idx),
                    "age_label_type": sample.age_label_type,
                    "has_valid_age": bool(has_valid_age),
                    "age_group": cohort_name,
                    "age_group_idx": int(cohort_idx),
                    "age_cohort": cohort_name,
                    "age_cohort_idx": int(cohort_idx),
                    "gender": gender_str,
                    "gender_prob": gender_prob,
                    "ethnicity": ethnicity,
                    "dimorphism_weight": round(dimorphism_w, 4),
                    "epistemic_interval_weight": round(epistemic_w, 4),
                    "alpha_sample_weight": alpha_w,
                    "teacher_count": 0,
                    "teacher_age_mu": float("nan"),
                    "teacher_age_sigma": float("nan"),
                    "teacher_gender_agreement": float("nan"),
                    "laplacian_var": audit.laplacian_variance,
                    "laplacian_variance": audit.laplacian_variance,
                    "tenengrad": audit.tenengrad_energy,
                    "tenengrad_energy": audit.tenengrad_energy,
                    "mean_luminance": audit.mean_luminance,
                    "shadow_clip": audit.shadow_clip_fraction,
                    "shadow_clip_fraction": audit.shadow_clip_fraction,
                    "highlight_clip": audit.highlight_clip_fraction,
                    "highlight_clip_fraction": audit.highlight_clip_fraction,
                    "cheek_occlusion_delta_e": audit.cheek_occlusion_delta_e,
                    "shannon_entropy": audit.shannon_entropy,
                    "michelson_contrast": audit.michelson_contrast,
                    "skin_tone_ita_deg": audit.skin_tone_ita_deg,
                    "skin_tone_group": audit.skin_tone_group,
                    "yaw": audit.yaw,
                    "pitch": audit.pitch,
                    "roll": audit.roll,
                    "ldl_fine_max_prob": round(float(np.max(ldl_23)), 5),
                }
                for col_name, prob_val in zip(self.COHORT_COL_NAMES, ldl_10_rounded):
                    record[col_name] = prob_val
                for col_name, prob_val in zip(self.FINE_INTERVAL_COL_NAMES, ldl_23_rounded):
                    record[col_name] = prob_val

                if quota_mgr is not None:
                    wrote = quota_mgr.write_aligned_face(
                        None,
                        filename=image_id,
                        record=record,
                        image_bytes=encoded_bytes,
                    )
                    if not wrote:
                        break

                if self.write_direct_images:
                    (images_dir / image_id).write_bytes(encoded_bytes)

                records.append(record)
                summary.accepted += 1
                summary.interval_distribution[fine_name] = (
                    summary.interval_distribution.get(fine_name, 0) + 1
                )
                g_key = gender_str if gender_str in ("Male", "Female") else "Unlabelled"
                summary.gender_distribution[g_key] = summary.gender_distribution.get(g_key, 0) + 1
                summary.source_distribution[sample.source_dataset] = (
                    summary.source_distribution.get(sample.source_dataset, 0) + 1
                )
        finally:
            if quota_mgr is not None:
                quota_mgr.close()

        df = pd.DataFrame(records)
        golden_dir = self.output_dir / "golden_dataset" if self.write_tar_shards else self.output_dir
        golden_dir.mkdir(parents=True, exist_ok=True)
        df = self._propagate_subject_gender(df, shards_dir=golden_dir / "shards")
        df = self._ensure_all_splits_covered(df, shards_dir=golden_dir / "shards")
        df = self._calibrate_demographic_sample_weights(df, shards_dir=golden_dir / "shards")
        if not df.empty and "gender" in df.columns:
            summary.gender_distribution = {}
            for g_val in df["gender"]:
                g_key = str(g_val) if str(g_val) in ("Male", "Female") else "Unlabelled"
                summary.gender_distribution[g_key] = summary.gender_distribution.get(g_key, 0) + 1
        p_parquet = golden_dir / "metadata.parquet"
        p_csv = golden_dir / "metadata.csv"
        df.to_parquet(str(p_parquet), index=False)
        csv_safe_manifest(df).to_csv(str(p_csv), index=False)
        summary.manifest_parquet_path = p_parquet
        summary.manifest_csv_path = p_csv
        return df, summary

    def curate_directory(
        self,
        input_dir: str | Path = "/kaggle/input",
        max_accepted: Optional[int] = None,
        exclude_utk: bool = True,
        max_samples_per_dataset: int = 50000,
    ) -> Tuple[pd.DataFrame, CurationSummary]:
        """
        Mounts all supported real-face datasets under `input_dir` via
        `InCloudStreamingHarvester` (excluding UTKFace by default) and curates them.
        """
        harvester = InCloudStreamingHarvester(
            kaggle_input_dir=str(input_dir),
            max_samples_per_dataset=max_samples_per_dataset,
            exclude_utk=exclude_utk,
        )
        return self.curate_samples(harvester.stream_all_mounted(), max_accepted=max_accepted)

