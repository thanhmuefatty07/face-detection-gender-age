"""
PyTorch Dataset Helper for Curated Golden Face Biometrics Dataset
==================================================================
Enables seamless downstream model training (MobileNetV4, ConvNeXt-V2)
from curated sharded TAR archives or image directories + metadata.parquet.
Supports 101-bin continuous Label Distribution Learning (LDL), dimorphism
attenuation weights, and multi-task age/gender targets.
"""

from __future__ import annotations
import hashlib
import io
import math
import re
import tarfile
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from .ldl_engine import compute_gaussian_ldl
from .dimorphism import calculate_sexual_dimorphism_weight
from .intervals import (
    FINE_AGE_INTERVALS,
    compute_interval_box_ldl,
    get_fine_interval_for_age,
    integrate_101_to_fine_intervals,
    map_bounds_to_fine_interval,
    parse_age_interval_string,
)


class FaceGoldenDataset(Dataset):
    """
    Production-grade PyTorch Dataset for Face Gender & Continuous / 3-5 Year Interval Age Estimation.
    
    Supports:
    1. Loading directly from sharded TAR archives (`shard_0000.tar`, etc.) without full extraction.
    2. Loading from an extracted image folder.
    3. Seamless Parquet or CSV manifest ingestion (37+ column schema).
    4. Train / Val / Test split filtering.
    5. Continuous 101-bin Gaussian / Box-Convolved LDL and 23-bin (3-5 yr) interval target synthesis.
    """

    _SAFE_IMAGE_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")
    _MAX_IMAGE_BYTES = 16 * 1024 * 1024
    _MAX_IMAGE_PIXELS = 25_000_000

    def __init__(
        self,
        manifest_path: str | Path,
        images_dir: Optional[str | Path] = None,
        shards_dir: Optional[str | Path] = None,
        split: Optional[str] = None,
        num_age_bins: int = 101,
        sigma_ldl: float = 2.5,
        transform: Optional[Callable[[np.ndarray], torch.Tensor]] = None,
        corrupted_transform: Optional[Callable[[np.ndarray], torch.Tensor]] = None,
        image_size: Optional[int] = None,
    ):
        super().__init__()
        self.manifest_path = Path(manifest_path)
        self.images_dir = Path(images_dir) if images_dir else None
        self.shards_dir = Path(shards_dir) if shards_dir else None
        self.split = split
        self.num_age_bins = num_age_bins
        self.sigma_ldl = sigma_ldl
        self.transform = transform
        self.corrupted_transform = corrupted_transform
        self.image_size = int(image_size) if image_size is not None else None

        # Load manifest
        if self.manifest_path.suffix.lower() == ".parquet":
            self.df = pd.read_parquet(self.manifest_path)
        else:
            self.df = pd.read_csv(self.manifest_path)

        # Filter by split if specified
        if self.split is not None and "split" in self.df.columns:
            self.df = self.df[self.df["split"] == self.split].reset_index(drop=True)
        else:
            self.df = self.df.reset_index(drop=True)

        # Index sharded tar files if shards_dir is specified
        self.tar_index: Dict[str, Path] = {}
        self._tar_handles: Dict[Path, tarfile.TarFile] = {}
        if self.shards_dir and self.shards_dir.exists():
            self._index_shards()

    @classmethod
    def _is_safe_image_id(cls, image_id: str) -> bool:
        if not image_id or image_id in {".", ".."}:
            return False
        if "/" in image_id or "\\" in image_id:
            return False
        return bool(cls._SAFE_IMAGE_ID_RE.match(image_id))

    @classmethod
    def _read_bounded_stream(cls, stream: Any, max_bytes: Optional[int] = None) -> Optional[bytes]:
        limit = cls._MAX_IMAGE_BYTES if max_bytes is None else int(max_bytes)
        data = stream.read(limit + 1)
        if not data or len(data) > limit:
            return None
        return data

    @classmethod
    def _decode_safe_image_bgr(cls, raw_bytes: Optional[bytes]) -> Optional[np.ndarray]:
        if not raw_bytes or len(raw_bytes) > cls._MAX_IMAGE_BYTES:
            return None
        try:
            with Image.open(io.BytesIO(raw_bytes)) as pil_img:
                width, height = pil_img.size
                if width <= 0 or height <= 0 or (width * height) > cls._MAX_IMAGE_PIXELS:
                    return None
        except Exception:
            return None
        nparr = np.frombuffer(raw_bytes, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    def _get_tar_handle(self, tar_path: Path) -> Optional[tarfile.TarFile]:
        """Returns or opens a cached TarFile handle for persistent high-throughput reading."""
        if tar_path not in self._tar_handles:
            try:
                self._tar_handles[tar_path] = tarfile.open(tar_path, "r")
            except Exception:
                return None
        return self._tar_handles[tar_path]

    def close(self):
        """Closes all cached TarFile handles."""
        for tf in self._tar_handles.values():
            try:
                tf.close()
            except Exception:
                pass
        self._tar_handles.clear()

    def __del__(self):
        self.close()

    def _index_shards(self):
        """Indexes which shard tar contains which image_id for O(1) shard lookup."""
        for tar_path in sorted(self.shards_dir.glob("*.tar")):
            try:
                with tarfile.open(tar_path, 'r') as tf:
                    for member in tf.getmembers():
                        if (
                            member.isfile()
                            and 0 < member.size <= self._MAX_IMAGE_BYTES
                            and self._is_safe_image_id(member.name)
                        ):
                            self.tar_index[member.name] = tar_path
            except Exception:
                pass

    def __len__(self) -> int:
        return len(self.df)

    def _load_image_bytes(self, image_id: str) -> Optional[bytes]:
        """Loads raw image bytes from directory or tar shard with path-traversal and size guards."""
        if not self._is_safe_image_id(image_id):
            return None

        # 1. From direct image folder
        if self.images_dir:
            try:
                root_dir = self.images_dir.resolve(strict=False)
                candidate = self.images_dir / image_id
                if not candidate.is_symlink():
                    direct_path = candidate.resolve(strict=False)
                    direct_path.relative_to(root_dir)
                    if direct_path.is_file():
                        size = direct_path.stat().st_size
                        if 0 < size <= self._MAX_IMAGE_BYTES:
                            with open(direct_path, "rb") as f:
                                return self._read_bounded_stream(f)
            except Exception:
                pass

        # 2. From sharded tar archive
        if image_id in self.tar_index:
            tar_file = self.tar_index[image_id]
            tf = self._get_tar_handle(tar_file)
            if tf is not None:
                try:
                    member = tf.getmember(image_id)
                    if member.isfile() and 0 < member.size <= self._MAX_IMAGE_BYTES:
                        extracted = tf.extractfile(member)
                        if extracted:
                            return self._read_bounded_stream(extracted)
                except Exception:
                    pass

        # 3. Check shards directory directly if index was empty
        if self.shards_dir and self.shards_dir.exists():
            for tar_path in self.shards_dir.glob("*.tar"):
                tf = self._get_tar_handle(tar_path)
                if tf is not None:
                    try:
                        if image_id in tf.getnames():
                            member = tf.getmember(image_id)
                            if member.isfile() and 0 < member.size <= self._MAX_IMAGE_BYTES:
                                self.tar_index[image_id] = tar_path
                                extracted = tf.extractfile(member)
                                if extracted:
                                    return self._read_bounded_stream(extracted)
                    except Exception:
                        pass

        return None

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        image_id = str(row["image_id"])

        # Decode image safely with decompression-bomb and byte-size checks
        raw_bytes = self._load_image_bytes(image_id)
        img_bgr = self._decode_safe_image_bgr(raw_bytes)

        target_sz = self.image_size or 160
        if img_bgr is None:
            img_bgr = np.zeros((target_sz, target_sz, 3), dtype=np.uint8)
        elif self.image_size is not None and img_bgr.shape[:2] != (self.image_size, self.image_size):
            img_bgr = cv2.resize(img_bgr, (self.image_size, self.image_size), interpolation=cv2.INTER_LANCZOS4)
        elif self.image_size is None and img_bgr.shape[:2] not in {(160, 160), (224, 224)}:
            img_bgr = cv2.resize(img_bgr, (160, 160), interpolation=cv2.INTER_LANCZOS4)

        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            tensor_img = self.transform(img_rgb)
        else:
            # Standard PyTorch normalization to [-1, 1]
            tensor_img = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float() / 127.5 - 1.0

        raw_age_val = row.get("age_continuous", 25.0)
        try:
            age_val = float(raw_age_val) if raw_age_val is not None else 25.0
        except (TypeError, ValueError):
            age_val = 25.0
        if math.isnan(age_val) or math.isinf(age_val):
            age_val = 25.0
        age_val = max(0.0, min(100.0, age_val))

        raw_sigma_val = row.get("apparent_age_std", self.sigma_ldl)
        try:
            sigma_val = float(raw_sigma_val) if raw_sigma_val is not None else self.sigma_ldl
        except (TypeError, ValueError):
            sigma_val = self.sigma_ldl
        if math.isnan(sigma_val) or math.isinf(sigma_val) or sigma_val <= 0:
            sigma_val = self.sigma_ldl

        # Determine whether age supervision is valid (not binary / missing)
        has_valid_age = True
        if "has_valid_age" in row and row.get("has_valid_age") is not None:
            val_flag = row.get("has_valid_age")
            if isinstance(val_flag, (bool, np.bool_)):
                has_valid_age = bool(val_flag)
            elif str(val_flag).strip().lower() in {"false", "0", "0.0", "no", "none", "nan"}:
                has_valid_age = False
        if "age_label_type" in row and row.get("age_label_type") is not None:
            if str(row.get("age_label_type")).strip().lower() in {"binary", "missing", "none", "invalid"}:
                has_valid_age = False

        label_type_str = (
            str(row.get("age_label_type", "")).strip().lower()
            if "age_label_type" in row and row.get("age_label_type") is not None
            else ""
        )
        is_exact_label = label_type_str in {"exact", "fine_interval"}

        # Resolve 3-5 year interval bounds [age_min, age_max]
        raw_min = row.get("age_min", None)
        raw_max = row.get("age_max", None)
        has_explicit_interval = False
        parsed_bounds = False
        if raw_min is not None and raw_max is not None:
            try:
                f_min = float(raw_min)
                f_max = float(raw_max)
                if not math.isnan(f_min) and not math.isinf(f_min) and not math.isnan(f_max) and not math.isinf(f_max):
                    age_min_val = max(0.0, min(100.0, min(f_min, f_max)))
                    age_max_val = max(0.0, min(100.0, max(f_min, f_max)))
                    has_explicit_interval = (age_max_val > age_min_val) and not is_exact_label
                    interval_name, interval_idx, _, _ = map_bounds_to_fine_interval(age_min_val, age_max_val)
                    parsed_bounds = True
            except (TypeError, ValueError):
                parsed_bounds = False

        if not parsed_bounds:
            if "age_interval" in row and row.get("age_interval") is not None:
                parsed = parse_age_interval_string(row.get("age_interval"))
                if parsed is not None:
                    age_min_val, age_max_val, _ = parsed
                    has_explicit_interval = (age_max_val > age_min_val) and not is_exact_label
                    interval_name, interval_idx, _, _ = map_bounds_to_fine_interval(age_min_val, age_max_val)
                else:
                    interval_name, interval_idx, low_i, high_i = get_fine_interval_for_age(age_val)
                    age_min_val, age_max_val = float(low_i), float(high_i)
            else:
                interval_name, interval_idx, low_i, high_i = get_fine_interval_for_age(age_val)
                age_min_val, age_max_val = float(low_i), float(high_i)

        # Compute 101-bin continuous LDL simplex vector (Box-Convolved when explicit interval is given; Gaussian for exact age)
        if has_explicit_interval:
            ldl_101 = compute_interval_box_ldl(
                age_min=age_min_val,
                age_max=age_max_val,
                sigma=sigma_val,
                num_bins=self.num_age_bins,
            )
        else:
            ldl_101 = compute_gaussian_ldl(
                mean_age=age_val,
                sigma=sigma_val,
                num_bins=self.num_age_bins,
            )

        # Compute or load 23-bracket (3-5 year) interval distribution
        fine_cols = [f"fine_ldl_{iv_name.replace('-', '_')}" for iv_name, _, _ in FINE_AGE_INTERVALS]
        if self.num_age_bins == 101 and all(c in row and row.get(c) is not None for c in fine_cols):
            try:
                vals_23 = np.array([float(row[c]) for c in fine_cols], dtype=np.float32)
                s_23 = float(np.sum(vals_23))
                if np.all(np.isfinite(vals_23)) and s_23 > 0.0:
                    ldl_fine_23 = vals_23 / s_23
                else:
                    ldl_fine_23 = integrate_101_to_fine_intervals(ldl_101)
            except (TypeError, ValueError):
                ldl_fine_23 = integrate_101_to_fine_intervals(ldl_101)
        elif self.num_age_bins == 101:
            ldl_fine_23 = integrate_101_to_fine_intervals(ldl_101)
        else:
            ldl_fine_23 = np.full(23, 1.0 / 23.0, dtype=np.float32)

        # Gender target and validity mask:
        # has_gender is True only if a valid gender annotation exists ('female', 'male', 'f', 'm', '1', '0').
        # If unlabelled/missing, gender_mask is 0.0 and gender_target defaults to 0.0 to prevent gradient corruption.
        gender_raw = row.get("gender")
        has_gender = False
        gender_target = 0.0

        if gender_raw is not None and not (isinstance(gender_raw, float) and math.isnan(gender_raw)):
            g_str = str(gender_raw).strip().lower()
            if g_str in ["female", "f", "1", "1.0"]:
                has_gender = True
                gender_target = 1.0
            elif g_str in ["male", "m", "0", "0.0"]:
                has_gender = True
                gender_target = 0.0
            elif g_str in ["none", "nan", "", "-1", "-1.0", "unknown"]:
                has_gender = False
                gender_target = 0.0
            else:
                try:
                    val = float(g_str)
                    if 0.0 <= val <= 1.0:
                        has_gender = True
                        gender_target = 1.0 if val >= 0.5 else 0.0
                except ValueError:
                    has_gender = False
                    gender_target = 0.0

        gender_mask = 1.0 if has_gender else 0.0
        age_mask = 1.0 if has_valid_age else 0.0

        dimorphism_w = float(row.get("dimorphism_weight", calculate_sexual_dimorphism_weight(age_val)))
        if math.isnan(dimorphism_w) or math.isinf(dimorphism_w):
            dimorphism_w = calculate_sexual_dimorphism_weight(age_val)
        alpha_w = float(row.get("alpha_sample_weight", 1.0))
        if math.isnan(alpha_w) or math.isinf(alpha_w):
            alpha_w = 1.0

        # 10-cohort LDL if available in manifest
        ldl_10_cohorts = []
        for col in [
            "ldl_0_2", "ldl_3_6", "ldl_7_12", "ldl_13_17", "ldl_18_24",
            "ldl_25_34", "ldl_35_44", "ldl_45_54", "ldl_55_64", "ldl_65_plus"
        ]:
            if col in row:
                ldl_10_cohorts.append(float(row[col]))
        if len(ldl_10_cohorts) != 10:
            ldl_10_cohorts = [0.1] * 10

        identity_val = self._extract_identity_int(row)

        raw_split = row.get("split", "train")
        split_str = str(raw_split) if (raw_split is not None and not pd.isna(raw_split) and str(raw_split).strip() != "") else "train"

        raw_ita = row.get("ita_group", row.get("skin_tone_group", None))
        ita_str = str(raw_ita) if (raw_ita is not None and not pd.isna(raw_ita) and str(raw_ita).strip() != "") else "unknown"

        raw_src = row.get("dataset_source", row.get("source_dataset", None))
        src_str = str(raw_src) if (raw_src is not None and not pd.isna(raw_src) and str(raw_src).strip() != "") else "unknown"

        age_mask_tensor = torch.tensor(age_mask, dtype=torch.float32)

        sample: Dict[str, Any] = {
            "image": tensor_img,                                    # (3, 160, 160) float tensor
            "age_continuous": torch.tensor(age_val, dtype=torch.float32),
            "age_min": torch.tensor(age_min_val, dtype=torch.float32),
            "age_max": torch.tensor(age_max_val, dtype=torch.float32),
            "age_mask": age_mask_tensor,
            "has_valid_age": age_mask_tensor,
            "age_interval_idx": torch.tensor(interval_idx, dtype=torch.long),
            "age_interval_name": interval_name,
            "ldl_101": torch.from_numpy(ldl_101),                  # (101,) continuous simplex
            "ldl_fine_23": torch.from_numpy(ldl_fine_23),          # (23,) 3-5 yr interval simplex
            "ldl_10": torch.tensor(ldl_10_cohorts, dtype=torch.float32), # (10,) cohort simplex
            "gender": torch.tensor(gender_target, dtype=torch.float32), # 0.0=Male, 1.0=Female
            "gender_mask": torch.tensor(gender_mask, dtype=torch.float32), # 1.0=Valid gender, 0.0=Ignore in BCE
            "dimorphism_weight": torch.tensor(dimorphism_w, dtype=torch.float32),
            "alpha_sample_weight": torch.tensor(alpha_w, dtype=torch.float32),
            "identity": torch.tensor(identity_val, dtype=torch.long),
            "image_id": image_id,
            "split": split_str,
            "ita_group": ita_str,
            "dataset_source": src_str,
        }
        if self.corrupted_transform is not None:
            sample["image_corrupted"] = self.corrupted_transform(img_rgb)
        return sample

    @staticmethod
    def _extract_identity_int(row: pd.Series | Dict[str, Any]) -> int:
        for id_col in ("identity_id", "identity", "subject_id", "person_id"):
            val = row.get(id_col)
            if val is None or (isinstance(val, float) and math.isnan(val)):
                continue
            try:
                if pd.isna(val):
                    continue
            except Exception:
                pass
            if isinstance(val, (int, np.integer)):
                ival = int(val)
                if ival >= 0:
                    return ival
                continue
            if isinstance(val, (float, np.floating)):
                if not math.isnan(float(val)) and not math.isinf(float(val)) and int(val) >= 0:
                    return int(val)
                continue
            sval = str(val).strip()
            sval_lower = sval.lower()
            if not sval or sval_lower in {"-1", "-1.0", "none", "nan", "null", "unknown"}:
                continue
            if "unknown" in sval_lower:
                continue
            try:
                ival = int(float(sval))
                if ival >= 0:
                    return ival
                continue
            except (TypeError, ValueError):
                digest = hashlib.sha256(sval.encode("utf-8")).digest()
                return int.from_bytes(digest[:8], "big") & ((1 << 52) - 1)
        return -1


