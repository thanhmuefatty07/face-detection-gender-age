"""
PyTorch Dataset Helper for Curated Golden Face Biometrics Dataset
==================================================================
Enables seamless downstream model training (MobileNetV4, ConvNeXt-V2)
from curated sharded TAR archives or image directories + metadata.parquet.
Supports 101-bin continuous Label Distribution Learning (LDL), dimorphism
attenuation weights, and multi-task age/gender targets.
"""

from __future__ import annotations
import io
import math
import tarfile
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .ldl_engine import compute_gaussian_ldl
from .dimorphism import calculate_sexual_dimorphism_weight


class FaceGoldenDataset(Dataset):
    """
    Production-grade PyTorch Dataset for Face Gender & Continuous Age Estimation.
    
    Supports:
    1. Loading directly from sharded TAR archives (`shard_0000.tar`, etc.) without full extraction.
    2. Loading from an extracted image folder.
    3. Seamless Parquet or CSV manifest ingestion (37-column schema).
    4. Train / Val / Test split filtering.
    5. Continuous 101-bin Gaussian LDL target synthesis on-the-fly.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        images_dir: Optional[str | Path] = None,
        shards_dir: Optional[str | Path] = None,
        split: Optional[str] = None,
        num_age_bins: int = 101,
        sigma_ldl: float = 2.5,
        transform: Optional[Callable[[np.ndarray], torch.Tensor]] = None,
    ):
        super().__init__()
        self.manifest_path = Path(manifest_path)
        self.images_dir = Path(images_dir) if images_dir else None
        self.shards_dir = Path(shards_dir) if shards_dir else None
        self.split = split
        self.num_age_bins = num_age_bins
        self.sigma_ldl = sigma_ldl
        self.transform = transform

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
                    for name in tf.getnames():
                        self.tar_index[name] = tar_path
            except Exception:
                pass

    def __len__(self) -> int:
        return len(self.df)

    def _load_image_bytes(self, image_id: str) -> Optional[bytes]:
        """Loads raw image bytes from directory or tar shard."""
        # 1. From direct image folder
        if self.images_dir:
            direct_path = self.images_dir / image_id
            if direct_path.exists():
                with open(direct_path, "rb") as f:
                    return f.read()

        # 2. From sharded tar archive
        if image_id in self.tar_index:
            tar_file = self.tar_index[image_id]
            tf = self._get_tar_handle(tar_file)
            if tf is not None:
                try:
                    member = tf.getmember(image_id)
                    extracted = tf.extractfile(member)
                    if extracted:
                        return extracted.read()
                except Exception:
                    pass

        # 3. Check shards directory directly if index was empty
        if self.shards_dir and self.shards_dir.exists():
            for tar_path in self.shards_dir.glob("*.tar"):
                tf = self._get_tar_handle(tar_path)
                if tf is not None:
                    try:
                        if image_id in tf.getnames():
                            self.tar_index[image_id] = tar_path
                            member = tf.getmember(image_id)
                            extracted = tf.extractfile(member)
                            if extracted:
                                return extracted.read()
                    except Exception:
                        pass

        return None

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        image_id = str(row["image_id"])

        # Decode image
        raw_bytes = self._load_image_bytes(image_id)
        if raw_bytes is not None:
            nparr = np.frombuffer(raw_bytes, np.uint8)
            img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        else:
            img_bgr = None

        if img_bgr is None:
            # Fallback placeholder 160x160 neutral image
            img_bgr = np.zeros((160, 160, 3), dtype=np.uint8)
        elif img_bgr.shape[:2] != (160, 160):
            img_bgr = cv2.resize(img_bgr, (160, 160))

        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            tensor_img = self.transform(img_rgb)
        else:
            # Standard PyTorch normalization to [-1, 1]
            tensor_img = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float() / 127.5 - 1.0

        age_val = float(row.get("age_continuous", 25.0))
        if math.isnan(age_val):
            age_val = 25.0
        sigma_val = float(row.get("apparent_age_std", self.sigma_ldl))
        if math.isnan(sigma_val) or sigma_val <= 0:
            sigma_val = self.sigma_ldl

        # Compute 101-bin continuous LDL simplex vector
        ldl_101 = compute_gaussian_ldl(mean_age=age_val, sigma=sigma_val, num_bins=self.num_age_bins)

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

        dimorphism_w = float(row.get("dimorphism_weight", calculate_sexual_dimorphism_weight(age_val)))
        alpha_w = float(row.get("alpha_sample_weight", 1.0))

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

        return {
            "image": tensor_img,                                    # (3, 160, 160) float tensor
            "age_continuous": torch.tensor(age_val, dtype=torch.float32),
            "ldl_101": torch.from_numpy(ldl_101),                  # (101,) continuous simplex
            "ldl_10": torch.tensor(ldl_10_cohorts, dtype=torch.float32), # (10,) cohort simplex
            "gender": torch.tensor(gender_target, dtype=torch.float32), # 0.0=Male, 1.0=Female
            "gender_mask": torch.tensor(gender_mask, dtype=torch.float32), # 1.0=Valid gender, 0.0=Ignore in BCE
            "dimorphism_weight": torch.tensor(dimorphism_w, dtype=torch.float32),
            "alpha_sample_weight": torch.tensor(alpha_w, dtype=torch.float32),
            "image_id": image_id,
            "split": str(row.get("split", "train"))
        }
