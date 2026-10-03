"""
Kaggle In-Cloud Multi-Source Dataset Connectors & Zero-Disk Streaming Harvester
================================================================================
Implements modular connectors for:
1. FairFace (jovian/fairface-dataset): 7 races, 9 age intervals, balanced demographics
2. CelebA (jessicali9530/celeba-dataset): 5 landmarks, 40 attributes, studio quality
3. IMDB-WIKI (dimitrisk/imdb-wiki-dataset): Continuous age, .mat parsing, outlier pruning
4. AFAD / Asian Face (yueyue/afad-lite): 15-40 age brackets, East Asian morphology
5. B3FD (moritzm00/biometrically-filtered-famous-figure-dataset): Cleaned biometric figures
6. APPA-REAL (vnthanhphm/appa-real-age-only-v1): Real vs Apparent age with human uncertainty

All connectors support zero-disk-write in-memory streaming directly from /kaggle/input
mounts and .zip/.tar archives. Includes DiskQuotaManager to guarantee < 20GB disk footprint.
"""

from __future__ import annotations
import os
import sys
import io
import tarfile
import zipfile
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Dict, Any, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd


@dataclass
class RawImageSample:
    """
    In-memory representation of an ingested raw face image.
    Guarantees 0 bytes written to local disk for discarded candidates.
    """
    image_bytes: bytes                  # In-memory JPEG/PNG buffer
    source_dataset: str                 # e.g. "fairface", "celeba", "imdb_wiki", "afad", "b3fd", "appa_real"
    raw_id: str                         # Identifier or filename in source repository
    declared_age: Optional[float] = None
    declared_gender: Optional[str] = None  # "Male", "Female", or None
    declared_ethnicity: Optional[str] = None
    declared_landmarks: Optional[np.ndarray] = None # (5, 2) if available
    metadata: Optional[Dict[str, Any]] = None

    def to_bgr_image(self) -> Optional[np.ndarray]:
        """Decodes image_bytes into an OpenCV BGR numpy array in RAM."""
        if not self.image_bytes:
            return None
        nparr = np.frombuffer(self.image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img


class BaseFaceConnector(ABC):
    """Abstract Base Class for all In-Cloud Dataset Connectors."""

    SUPPORTED_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}

    def __init__(self, root_dir: Path | str, max_samples: int = 50000):
        self.root_dir = Path(root_dir)
        self.max_samples = max_samples

    @abstractmethod
    def get_name(self) -> str:
        """Returns unique identifier for dataset connector."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Checks if dataset exists at root_dir or in /kaggle/input."""
        pass

    @abstractmethod
    def stream_samples(self) -> Generator[RawImageSample, None, None]:
        """Yields RawImageSample objects without intermediate disk extraction."""
        pass


class FairFaceConnector(BaseFaceConnector):
    """
    Connector for FairFace (CVPR 2019 / TPAMI 2021).
    Demographically balanced across 7 racial classes and 9 age bins.
    """

    AGE_BIN_MIDPOINTS = {
        '0-2': 1.0,
        '3-9': 6.0,
        '10-19': 14.5,
        '20-29': 24.5,
        '30-39': 34.5,
        '40-49': 44.5,
        '50-59': 54.5,
        '60-69': 64.5,
        'more than 70': 75.0
    }

    RACE_MAPPING = {
        'East Asian': 'East Asian',
        'Southeast Asian': 'Southeast Asian',
        'White': 'Caucasian',
        'Black': 'African',
        'Indian': 'South Asian',
        'Middle Eastern': 'Middle Eastern',
        'Latino_Hispanic': 'Latino'
    }

    def get_name(self) -> str:
        return "fairface"

    def is_available(self) -> bool:
        if not self.root_dir.exists():
            return False
        # Check for label files or folders
        has_csv = any((self.root_dir / f).exists() for f in [
            "fairface_label_train.csv", "fairface_label_val.csv",
            "fairface_labels_train.csv", "fairface_labels_val.csv"
        ])
        has_imgs = (self.root_dir / "train").exists() or (self.root_dir / "val").exists()
        has_zip = any(self.root_dir.glob("fairface*.zip"))
        return has_csv or has_imgs or has_zip

    def stream_samples(self) -> Generator[RawImageSample, None, None]:
        if not self.is_available():
            return

        yielded = 0
        csv_candidates = [
            self.root_dir / "fairface_label_train.csv",
            self.root_dir / "fairface_label_val.csv",
            self.root_dir / "fairface_labels_train.csv"
        ]

        # Check for zip archives
        zip_candidates = list(self.root_dir.glob("fairface*.zip"))
        if zip_candidates and not any(p.exists() for p in csv_candidates):
            for zpath in zip_candidates:
                try:
                    with zipfile.ZipFile(zpath, 'r') as zf:
                        # Find CSV inside zip
                        csv_members = [m for m in zf.namelist() if m.endswith('.csv')]
                        if csv_members:
                            df = pd.read_csv(zf.open(csv_members[0]))
                            for _, row in df.iterrows():
                                if yielded >= self.max_samples:
                                    return
                                rel_path = str(row.get('file', ''))
                                if rel_path in zf.namelist():
                                    b = zf.read(rel_path)
                                    age_str = str(row.get('age', ''))
                                    age_mid = self.AGE_BIN_MIDPOINTS.get(age_str, 25.0)
                                    gender = str(row.get('gender', ''))
                                    race = self.RACE_MAPPING.get(str(row.get('race', '')), str(row.get('race', '')))
                                    yield RawImageSample(
                                        image_bytes=b,
                                        source_dataset=self.get_name(),
                                        raw_id=Path(rel_path).name,
                                        declared_age=age_mid,
                                        declared_gender=gender,
                                        declared_ethnicity=race,
                                        metadata={"fairface_age_bin": age_str, "service_test": row.get('service_test', False)}
                                    )
                                    yielded += 1
                except Exception:
                    continue

        for csv_path in csv_candidates:
            if not csv_path.exists():
                continue
            try:
                df = pd.read_csv(csv_path)
            except Exception:
                continue

            for _, row in df.iterrows():
                if yielded >= self.max_samples:
                    return
                rel_path = str(row.get('file', ''))
                img_path = self.root_dir / rel_path
                if not img_path.exists():
                    # Attempt alternative path
                    img_path = self.root_dir / Path(rel_path).name
                if not img_path.exists():
                    continue

                try:
                    with open(img_path, 'rb') as f:
                        b = f.read()
                except Exception:
                    continue

                age_str = str(row.get('age', ''))
                age_mid = self.AGE_BIN_MIDPOINTS.get(age_str, 25.0)
                gender = str(row.get('gender', ''))
                race = self.RACE_MAPPING.get(str(row.get('race', '')), str(row.get('race', '')))

                yield RawImageSample(
                    image_bytes=b,
                    source_dataset=self.get_name(),
                    raw_id=Path(rel_path).name,
                    declared_age=age_mid,
                    declared_gender=gender,
                    declared_ethnicity=race,
                    metadata={"fairface_age_bin": age_str, "service_test": row.get('service_test', False)}
                )
                yielded += 1


class CelebAConnector(BaseFaceConnector):
    """
    Connector for CelebA (ICCV 2015).
    Contains 202,599 celebrity images with 5 ground-truth landmarks & 40 binary attributes.
    """

    def get_name(self) -> str:
        return "celeba"

    def is_available(self) -> bool:
        if not self.root_dir.exists():
            return False
        has_zip = (self.root_dir / "img_align_celeba.zip").exists()
        has_dir = (self.root_dir / "img_align_celeba").exists()
        has_lm = (self.root_dir / "list_landmarks_align_celeba.txt").exists()
        return has_zip or has_dir or has_lm

    def stream_samples(self) -> Generator[RawImageSample, None, None]:
        if not self.is_available():
            return

        yielded = 0
        landmarks_map: Dict[str, np.ndarray] = {}
        lm_file = self.root_dir / "list_landmarks_align_celeba.txt"
        if lm_file.exists():
            try:
                with open(lm_file, 'r') as f:
                    lines = [ln.strip() for ln in f if ln.strip()]
                # Skip first two header lines
                for ln in lines[2:self.max_samples + 2000]:
                    parts = ln.split()
                    if len(parts) >= 11:
                        fname = parts[0]
                        coords = [float(x) for x in parts[1:11]]
                        # 5 points: lefteye, righteye, nose, leftmouth, rightmouth
                        pts = np.array([
                            [coords[0], coords[1]],
                            [coords[2], coords[3]],
                            [coords[4], coords[5]],
                            [coords[6], coords[7]],
                            [coords[8], coords[9]]
                        ], dtype=np.float32)
                        landmarks_map[fname] = pts
            except Exception:
                pass

        # Parse attribute file for Male & Young
        attr_map: Dict[str, Tuple[str, bool]] = {}
        attr_file = self.root_dir / "list_attr_celeba.txt"
        if attr_file.exists():
            try:
                with open(attr_file, 'r') as f:
                    lines = [ln.strip() for ln in f if ln.strip()]
                if len(lines) >= 3:
                    header = lines[1].split()
                    male_idx = header.index("Male") + 1 if "Male" in header else -1
                    young_idx = header.index("Young") + 1 if "Young" in header else -1
                    for ln in lines[2:self.max_samples + 2000]:
                        parts = ln.split()
                        fname = parts[0]
                        gender = "Male" if male_idx > 0 and int(parts[male_idx]) == 1 else "Female"
                        young = True if young_idx > 0 and int(parts[young_idx]) == 1 else False
                        attr_map[fname] = (gender, young)
            except Exception:
                pass

        # Stream directly from zip if present
        zip_path = self.root_dir / "img_align_celeba.zip"
        if zip_path.exists():
            try:
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    for name in zf.namelist():
                        if yielded >= self.max_samples:
                            return
                        if not any(name.lower().endswith(ext) for ext in self.SUPPORTED_EXTS):
                            continue
                        fname = Path(name).name
                        b = zf.read(name)
                        gender_info = attr_map.get(fname, ("Unknown", True))
                        declared_gender = gender_info[0] if gender_info[0] != "Unknown" else None
                        yield RawImageSample(
                            image_bytes=b,
                            source_dataset=self.get_name(),
                            raw_id=fname,
                            declared_age=25.0 if gender_info[1] else 45.0,
                            declared_gender=declared_gender,
                            declared_ethnicity=None,
                            declared_landmarks=landmarks_map.get(fname),
                            metadata={"celeba_young": gender_info[1]}
                        )
                        yielded += 1
                return
            except Exception:
                pass

        # Stream from directory
        img_dir = self.root_dir / "img_align_celeba"
        if not img_dir.exists():
            img_dir = self.root_dir

        for entry in img_dir.glob("*.*"):
            if yielded >= self.max_samples:
                return
            if entry.suffix.lower() not in self.SUPPORTED_EXTS:
                continue
            fname = entry.name
            try:
                with open(entry, 'rb') as f:
                    b = f.read()
            except Exception:
                continue
            gender_info = attr_map.get(fname, ("Unknown", True))
            declared_gender = gender_info[0] if gender_info[0] != "Unknown" else None
            yield RawImageSample(
                image_bytes=b,
                source_dataset=self.get_name(),
                raw_id=fname,
                declared_age=25.0 if gender_info[1] else 45.0,
                declared_gender=declared_gender,
                declared_ethnicity=None,
                declared_landmarks=landmarks_map.get(fname),
                metadata={"celeba_young": gender_info[1]}
            )
            yielded += 1


class IMDBWikiConnector(BaseFaceConnector):
    """
    Connector for IMDB-WIKI (IJCV 2018).
    Continuous ages computed from DOB and photo_taken, with rigorous noise pruning:
    - Filters: face_score > 1.0, second_face_score nan/<0.5, age in [0, 100].
    """

    def get_name(self) -> str:
        return "imdb_wiki"

    def is_available(self) -> bool:
        if not self.root_dir.exists():
            return False
        has_mat = any((self.root_dir / m).exists() for m in ["imdb.mat", "wiki.mat"])
        has_tar = any((self.root_dir / t).exists() for t in ["imdb_crop.tar", "wiki_crop.tar"])
        has_folders = (self.root_dir / "imdb_crop").exists() or (self.root_dir / "wiki_crop").exists()
        return has_mat or has_tar or has_folders

    def stream_samples(self) -> Generator[RawImageSample, None, None]:
        if not self.is_available():
            return

        yielded = 0
        try:
            from scipy.io import loadmat
        except ImportError:
            loadmat = None

        mat_candidates = [
            (self.root_dir / "wiki.mat", self.root_dir / "wiki_crop.tar", self.root_dir / "wiki_crop"),
            (self.root_dir / "imdb.mat", self.root_dir / "imdb_crop.tar", self.root_dir / "imdb_crop")
        ]

        for mat_file, tar_file, crop_dir in mat_candidates:
            if not mat_file.exists() or loadmat is None:
                continue
            try:
                data = loadmat(str(mat_file))
                key = "wiki" if "wiki" in mat_file.stem else "imdb"
                if key not in data:
                    continue
                db = data[key][0, 0]
                dob = db['dob'][0]
                photo_taken = db['photo_taken'][0]
                full_path = db['full_path'][0]
                gender = db['gender'][0]
                face_score = db['face_score'][0]
                second_face_score = db['second_face_score'][0]
            except Exception:
                continue

            # Stream through indices
            num_samples = len(dob)
            tar_obj = None
            if tar_file.exists():
                try:
                    tar_obj = tarfile.open(tar_file, 'r')
                except Exception:
                    tar_obj = None

            for i in range(num_samples):
                if yielded >= self.max_samples:
                    if tar_obj:
                        tar_obj.close()
                    return

                # Noise pruning
                f_score = face_score[i]
                if np.isnan(f_score) or f_score < 1.0:
                    continue
                s_score = second_face_score[i]
                if not np.isnan(s_score) and s_score >= 0.5:
                    continue
                g_val = gender[i]
                if np.isnan(g_val):
                    continue

                # Compute age
                p_year = int(photo_taken[i])
                # Matlab serial date conversion: days from year 0000
                dob_serial = dob[i]
                try:
                    dob_year = int(dob_serial / 365.25)
                    calc_age = float(p_year - dob_year)
                except Exception:
                    continue

                if not (0.0 <= calc_age <= 100.0):
                    continue

                rel_p = str(full_path[i][0])
                img_bytes = None

                # Extract image bytes
                if tar_obj:
                    try:
                        member = tar_obj.getmember(rel_p)
                        f = tar_obj.extractfile(member)
                        if f:
                            img_bytes = f.read()
                    except Exception:
                        pass

                if not img_bytes and crop_dir.exists():
                    p_path = crop_dir / rel_p
                    if p_path.exists():
                        try:
                            with open(p_path, 'rb') as fp:
                                img_bytes = fp.read()
                        except Exception:
                            pass

                if not img_bytes:
                    continue

                declared_gender = "Female" if g_val == 0.0 else "Male"
                yield RawImageSample(
                    image_bytes=img_bytes,
                    source_dataset=self.get_name(),
                    raw_id=Path(rel_p).name,
                    declared_age=calc_age,
                    declared_gender=declared_gender,
                    declared_ethnicity=None,
                    metadata={"face_score": float(f_score), "photo_year": p_year}
                )
                yielded += 1

            if tar_obj:
                tar_obj.close()


class AFADConnector(BaseFaceConnector):
    """
    Connector for AFAD (Asian Face Age Dataset, PAKDD 2016).
    164,432 East Asian faces, ages 15-40.
    Directory layout: AFAD-Full/{age}/{gender}/{id}.jpg (111: Male, 112: Female).
    """

    def get_name(self) -> str:
        return "afad"

    def is_available(self) -> bool:
        if not self.root_dir.exists():
            return False
        # Check for AFAD-Full or afad-lite folder or direct age folders
        candidates = [self.root_dir / "AFAD-Full", self.root_dir / "afad-lite", self.root_dir]
        for c in candidates:
            if c.exists():
                subdirs = [d for d in c.iterdir() if d.is_dir() and d.name.isdigit()]
                if len(subdirs) >= 3:
                    return True
        return False

    def stream_samples(self) -> Generator[RawImageSample, None, None]:
        if not self.is_available():
            return

        yielded = 0
        base = self.root_dir
        for sub in ["AFAD-Full", "afad-lite"]:
            if (self.root_dir / sub).exists():
                base = self.root_dir / sub
                break

        for age_dir in sorted(base.iterdir()):
            if yielded >= self.max_samples:
                return
            if not age_dir.is_dir() or not age_dir.name.isdigit():
                continue
            age_val = float(age_dir.name)

            for g_dir in age_dir.iterdir():
                if not g_dir.is_dir():
                    continue
                gender = "Male" if g_dir.name == "111" else ("Female" if g_dir.name == "112" else None)

                for img_file in g_dir.glob("*.*"):
                    if yielded >= self.max_samples:
                        return
                    if img_file.suffix.lower() not in self.SUPPORTED_EXTS:
                        continue
                    try:
                        with open(img_file, 'rb') as f:
                            b = f.read()
                    except Exception:
                        continue

                    yield RawImageSample(
                        image_bytes=b,
                        source_dataset=self.get_name(),
                        raw_id=img_file.name,
                        declared_age=age_val,
                        declared_gender=gender,
                        declared_ethnicity="East Asian",
                        metadata={"afad_age": int(age_val)}
                    )
                    yielded += 1


class B3FDConnector(BaseFaceConnector):
    """
    Connector for B3FD (Biometrically Filtered Famous Figure Dataset).
    Cleaned biometric ages 1 to 101.
    """

    def get_name(self) -> str:
        return "b3fd"

    def is_available(self) -> bool:
        if not self.root_dir.exists():
            return False
        return any(self.root_dir.glob("*b3fd*.csv")) or (self.root_dir / "metadata.csv").exists()

    def stream_samples(self) -> Generator[RawImageSample, None, None]:
        if not self.is_available():
            return

        csv_file = None
        for cand in self.root_dir.glob("*.csv"):
            csv_file = cand
            break
        if not csv_file or not csv_file.exists():
            return

        try:
            df = pd.read_csv(csv_file)
        except Exception:
            return

        yielded = 0
        for _, row in df.iterrows():
            if yielded >= self.max_samples:
                return
            img_name = str(row.get('image_name', row.get('file', row.get('filename', ''))))
            img_path = self.root_dir / img_name
            if not img_path.exists():
                img_path = self.root_dir / "images" / img_name
            if not img_path.exists():
                continue

            try:
                with open(img_path, 'rb') as f:
                    b = f.read()
            except Exception:
                continue

            age_val = float(row.get('age', 30.0))
            gender_val = str(row.get('gender', ''))
            yield RawImageSample(
                image_bytes=b,
                source_dataset=self.get_name(),
                raw_id=Path(img_name).name,
                declared_age=age_val,
                declared_gender=gender_val if gender_val in ["Male", "Female"] else None,
                declared_ethnicity=None,
                metadata={"quality_score": row.get('quality_score', 1.0)}
            )
            yielded += 1


class AppaRealConnector(BaseFaceConnector):
    """
    Connector for APPA-REAL (FG 2017 / vnthanhphm/appa-real-age-only-v1).
    Ground-truth biological age and apparent age distribution with annotator standard deviation.
    """

    def get_name(self) -> str:
        return "appa_real"

    def is_available(self) -> bool:
        if not self.root_dir.exists():
            return False
        return any((self.root_dir / f).exists() for f in [
            "gt_avg_train.csv", "gt_avg_valid.csv", "gt_avg_test.csv"
        ])

    def stream_samples(self) -> Generator[RawImageSample, None, None]:
        if not self.is_available():
            return

        yielded = 0
        for split_csv in ["gt_avg_train.csv", "gt_avg_valid.csv", "gt_avg_test.csv"]:
            csv_path = self.root_dir / split_csv
            if not csv_path.exists():
                continue
            try:
                df = pd.read_csv(csv_path)
            except Exception:
                continue

            for _, row in df.iterrows():
                if yielded >= self.max_samples:
                    return
                fname = str(row.get('file_name', ''))
                # Path resolution
                img_path = self.root_dir / fname
                if not img_path.exists():
                    img_path = self.root_dir / split_csv.replace('gt_avg_', '').replace('.csv', '') / fname
                if not img_path.exists():
                    continue

                try:
                    with open(img_path, 'rb') as f:
                        b = f.read()
                except Exception:
                    continue

                real_age = float(row.get('real_age', row.get('apparent_age_avg', 25.0)))
                apparent_avg = float(row.get('apparent_age_avg', real_age))
                apparent_std = float(row.get('apparent_age_std', 2.5))

                yield RawImageSample(
                    image_bytes=b,
                    source_dataset=self.get_name(),
                    raw_id=fname,
                    declared_age=apparent_avg,
                    declared_gender=None,
                    declared_ethnicity=None,
                    metadata={
                        "real_age": real_age,
                        "apparent_age_avg": apparent_avg,
                        "apparent_age_std": apparent_std
                    }
                )
                yielded += 1


class InCloudStreamingHarvester:
    """
    Unified In-Cloud Zero-Disk Streaming Harvester.
    Automatically detects mounted datasets in /kaggle/input and orchestrates streaming.
    """

    def __init__(self, kaggle_input_dir: str = "/kaggle/input", max_samples_per_dataset: int = 50000):
        self.input_dir = Path(kaggle_input_dir)
        self.max_samples_per_dataset = max_samples_per_dataset
        self.connectors: List[BaseFaceConnector] = []
        self._initialize_connectors()

    def _initialize_connectors(self):
        """Discovers and registers connectors matching mounted folders."""
        if not self.input_dir.exists():
            return

        for dataset_dir in self.input_dir.iterdir():
            if not dataset_dir.is_dir():
                continue
            dname = dataset_dir.name.lower()
            if "fairface" in dname:
                self.connectors.append(FairFaceConnector(dataset_dir, self.max_samples_per_dataset))
            elif "celeba" in dname:
                self.connectors.append(CelebAConnector(dataset_dir, self.max_samples_per_dataset))
            elif "imdb" in dname or "wiki" in dname:
                self.connectors.append(IMDBWikiConnector(dataset_dir, self.max_samples_per_dataset))
            elif "afad" in dname or "asian" in dname:
                self.connectors.append(AFADConnector(dataset_dir, self.max_samples_per_dataset))
            elif "b3fd" in dname or "famous" in dname:
                self.connectors.append(B3FDConnector(dataset_dir, self.max_samples_per_dataset))
            elif "appa" in dname:
                self.connectors.append(AppaRealConnector(dataset_dir, self.max_samples_per_dataset))

    def register_connector(self, connector: BaseFaceConnector):
        """Manually registers an additional connector."""
        self.connectors.append(connector)

    def stream_all(self) -> Generator[RawImageSample, None, None]:
        """
        Interleaves and streams samples from all available connectors with 0 local disk footprint.
        """
        for conn in self.connectors:
            if conn.is_available():
                print(f"[Harvester] Streaming from connector: {conn.get_name()} ({conn.root_dir})")
                count = 0
                for sample in conn.stream_samples():
                    yield sample
                    count += 1
                print(f"[Harvester] Yielded {count} samples from {conn.get_name()}")


class DiskQuotaManager:
    """
    Enforces working storage quota (< 20GB on Kaggle) and manages shard packaging
    to prevent inode depletion and out-of-disk crashes.
    """

    def __init__(
        self,
        working_dir: str = "/kaggle/working",
        max_storage_gb: float = 18.0,
        min_free_gb: float = 2.0,
        shard_image_count: int = 5000,
        archive_format: str = "tar"
    ):
        self.working_dir = Path(working_dir)
        self.max_storage_bytes = int(max_storage_gb * (1024 ** 3))
        self.min_free_bytes = int(min_free_gb * (1024 ** 3))
        self.shard_image_count = shard_image_count
        self.archive_format = archive_format

        self.shards_dir = self.working_dir / "golden_dataset" / "shards"
        self.shards_dir.mkdir(parents=True, exist_ok=True)

        self.current_shard_idx = 0
        self.current_shard_count = 0
        self.current_tar_obj: Optional[tarfile.TarFile] = None
        self._open_new_shard()

    def _open_new_shard(self):
        if self.current_tar_obj:
            self.current_tar_obj.close()
        shard_file = self.shards_dir / f"shard_{self.current_shard_idx:04d}.tar"
        self.current_tar_obj = tarfile.open(shard_file, 'w')
        self.current_shard_count = 0

    def check_quota(self) -> bool:
        """Checks if current disk usage is safely below quota ceiling."""
        try:
            total, used, free = shutil.disk_usage(self.working_dir)
            if free < self.min_free_bytes:
                print(f"[QuotaManager] CRITICAL WARNING: Free disk space {free / (1024**3):.2f} GB < min {self.min_free_bytes / (1024**3):.2f} GB")
                return False
            # Also calculate size of working dir
            working_size = sum(f.stat().st_size for f in self.working_dir.rglob('*') if f.is_file())
            if working_size > self.max_storage_bytes:
                print(f"[QuotaManager] QUOTA EXCEEDED: Working dir size {working_size / (1024**3):.2f} GB > limit {self.max_storage_bytes / (1024**3):.2f} GB")
                return False
            return True
        except Exception:
            return True

    def write_aligned_face(self, face_bgr: np.ndarray, filename: str) -> bool:
        """
        Compresses aligned face to high-quality JPEG and appends to current tar shard.
        Avoids creating thousands of separate disk inodes.
        """
        if not self.check_quota():
            return False

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
        success, enc_buf = cv2.imencode('.jpg', face_bgr, encode_param)
        if not success:
            return False

        img_bytes = enc_buf.tobytes()
        tar_info = tarfile.TarInfo(name=filename)
        tar_info.size = len(img_bytes)

        if self.current_tar_obj:
            self.current_tar_obj.addfile(tar_info, io.BytesIO(img_bytes))
            self.current_shard_count += 1

            if self.current_shard_count >= self.shard_image_count:
                self.current_shard_idx += 1
                self._open_new_shard()

        return True

    def close(self):
        """Finalizes and closes all open archive handles."""
        if self.current_tar_obj:
            self.current_tar_obj.close()
            self.current_tar_obj = None
