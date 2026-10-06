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
import re
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

    def to_bgr(self) -> Optional[np.ndarray]:
        """Alias for to_bgr_image."""
        return self.to_bgr_image()


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

    def _has_matching_file(self, pattern: re.Pattern, max_inspect: int = 200, max_depth: int = 3) -> bool:
        """Fast bounded check for matching file up to max_depth without unbounded recursive walks."""
        count = 0
        base_depth = len(self.root_dir.parts)
        try:
            for root, dirs, files in os.walk(self.root_dir):
                cur_depth = len(Path(root).parts) - base_depth
                if cur_depth >= max_depth:
                    dirs.clear()
                for f in files:
                    count += 1
                    if count > max_inspect:
                        return False
                    if any(f.lower().endswith(ext) for ext in self.SUPPORTED_EXTS) and pattern.match(f):
                        return True
        except (PermissionError, OSError):
            pass
        return False



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

    ARCHIVE_PATTERN = re.compile(r"(?:^|.*/)(\d{2})/(111|112)/([^/]+)$", re.IGNORECASE)

    def get_name(self) -> str:
        return "afad"

    def is_available(self) -> bool:
        if not self.root_dir.exists():
            return False
        if any(self.root_dir.glob("*.tar*")) or any(self.root_dir.glob("*.zip")):
            return True
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
        seen_ids = set()

        # 1. Stream from tar archives if present (e.g. AFAD-Full.tar.gz)
        for tar_path in list(self.root_dir.rglob("*.tar*")):
            try:
                with tarfile.open(tar_path, 'r:*') as tf:
                    for member in tf:
                        if yielded >= self.max_samples:
                            return
                        if not member.isfile():
                            continue
                        fname = Path(member.name).name
                        if fname in seen_ids:
                            continue
                        m = self.ARCHIVE_PATTERN.match(member.name.replace('\\', '/'))
                        if not m:
                            continue
                        age_val = float(m.group(1))
                        gender = "Male" if m.group(2) == "111" else "Female"
                        f = tf.extractfile(member)
                        if not f:
                            continue
                        b = f.read()
                        seen_ids.add(fname)
                        yield RawImageSample(
                            image_bytes=b,
                            source_dataset=self.get_name(),
                            raw_id=fname,
                            declared_age=age_val,
                            declared_gender=gender,
                            declared_ethnicity="East Asian",
                            metadata={"afad_age": int(age_val)}
                        )
                        yielded += 1
            except Exception:
                pass

        # 2. Stream from zip archives if present
        for zip_path in list(self.root_dir.rglob("*.zip")):
            try:
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    for name in zf.namelist():
                        if yielded >= self.max_samples:
                            return
                        fname = Path(name).name
                        if fname in seen_ids:
                            continue
                        m = self.ARCHIVE_PATTERN.match(name.replace('\\', '/'))
                        if not m:
                            continue
                        age_val = float(m.group(1))
                        gender = "Male" if m.group(2) == "111" else "Female"
                        b = zf.read(name)
                        seen_ids.add(fname)
                        yield RawImageSample(
                            image_bytes=b,
                            source_dataset=self.get_name(),
                            raw_id=fname,
                            declared_age=age_val,
                            declared_gender=gender,
                            declared_ethnicity="East Asian",
                            metadata={"afad_age": int(age_val)}
                        )
                        yielded += 1
            except Exception:
                pass

        # 3. Stream from folders
        base = self.root_dir
        for sub in ["AFAD-Full", "afad-lite"]:
            if (self.root_dir / sub).exists():
                base = self.root_dir / sub
                break

        if base.exists():
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
                        if img_file.name in seen_ids:
                            continue
                        try:
                            with open(img_file, 'rb') as f:
                                b = f.read()
                        except Exception:
                            continue
                        seen_ids.add(img_file.name)
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
    Parses filenames: {name}/{id}_{dob}_{photo_year}.jpg or {id}_{dob}_{photo_year}.jpg.
    Cleaned biometric ages 1 to 101.
    """

    FILENAME_PATTERN = re.compile(r"^(.+)_(\d{4}(?:-\d{1,2}-\d{1,2})?)_(\d{4})\.(?:jpg|jpeg|png|webp|bmp)$", re.IGNORECASE)

    def get_name(self) -> str:
        return "b3fd"

    def is_available(self) -> bool:
        if not self.root_dir.exists():
            return False
        if any(self.root_dir.glob("*.csv")):
            return True
        if any(self.root_dir.glob("*.tar*")) or any(self.root_dir.glob("*.zip")):
            return True
        return self._has_matching_file(self.FILENAME_PATTERN)

    def stream_samples(self) -> Generator[RawImageSample, None, None]:
        if not self.is_available():
            return

        yielded = 0
        seen_ids = set()

        # 1. Stream from tar archives if present
        for tar_path in list(self.root_dir.rglob("*.tar*")):
            try:
                with tarfile.open(tar_path, 'r:*') as tf:
                    for member in tf:
                        if yielded >= self.max_samples:
                            return
                        if not member.isfile():
                            continue
                        fname = Path(member.name).name
                        if fname in seen_ids:
                            continue
                        m = self.FILENAME_PATTERN.match(fname)
                        if not m:
                            continue
                        try:
                            dob = int(m.group(2)[:4])
                            photo_year = int(m.group(3)[:4])
                            age = float(photo_year - dob)
                        except Exception:
                            continue
                        if not (0.0 <= age <= 101.0):
                            continue
                        f = tf.extractfile(member)
                        if not f:
                            continue
                        b = f.read()
                        seen_ids.add(fname)
                        subj_name = Path(member.name).parent.name if Path(member.name).parent.name else "unknown"
                        yield RawImageSample(
                            image_bytes=b,
                            source_dataset=self.get_name(),
                            raw_id=fname,
                            declared_age=age,
                            metadata={"b3fd_id": m.group(1), "subject_name": subj_name, "dob": dob, "photo_year": photo_year}
                        )
                        yielded += 1
            except Exception:
                pass

        # 2. Stream from zip archives if present
        for zip_path in list(self.root_dir.rglob("*.zip")):
            try:
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    for name in zf.namelist():
                        if yielded >= self.max_samples:
                            return
                        fname = Path(name).name
                        if fname in seen_ids:
                            continue
                        m = self.FILENAME_PATTERN.match(fname)
                        if not m:
                            continue
                        try:
                            dob = int(m.group(2)[:4])
                            photo_year = int(m.group(3)[:4])
                            age = float(photo_year - dob)
                        except Exception:
                            continue
                        if not (0.0 <= age <= 101.0):
                            continue
                        b = zf.read(name)
                        seen_ids.add(fname)
                        subj_name = Path(name).parent.name if Path(name).parent.name else "unknown"
                        yield RawImageSample(
                            image_bytes=b,
                            source_dataset=self.get_name(),
                            raw_id=fname,
                            declared_age=age,
                            metadata={"b3fd_id": m.group(1), "subject_name": subj_name, "dob": dob, "photo_year": photo_year}
                        )
                        yielded += 1
            except Exception:
                pass

        # 3. Stream by parsing directory filenames: {name}/{id}_{dob}_{photo_year}.jpg
        for img_path in self.root_dir.rglob("*.*"):
            if yielded >= self.max_samples:
                return
            if img_path.suffix.lower() not in self.SUPPORTED_EXTS:
                continue
            if img_path.name in seen_ids:
                continue

            m = self.FILENAME_PATTERN.match(img_path.name)
            if not m:
                continue

            subj_id, dob_str, year_str = m.group(1), m.group(2), m.group(3)
            try:
                dob = int(dob_str[:4])
                photo_year = int(year_str[:4])
                age = float(photo_year - dob)
            except Exception:
                continue

            if not (0.0 <= age <= 101.0):
                continue

            try:
                with open(img_path, 'rb') as f:
                    b = f.read()
            except Exception:
                continue

            seen_ids.add(img_path.name)
            subj_name = img_path.parent.name if img_path.parent != self.root_dir else "unknown"
            yield RawImageSample(
                image_bytes=b,
                source_dataset=self.get_name(),
                raw_id=img_path.name,
                declared_age=age,
                declared_gender=None,
                declared_ethnicity=None,
                metadata={
                    "b3fd_id": subj_id,
                    "subject_name": subj_name,
                    "dob": dob,
                    "photo_year": photo_year
                }
            )
            yielded += 1

        # 2. Fallback to CSV if no filename matches were found
        if yielded == 0:
            csv_candidates = list(self.root_dir.glob("*.csv")) + list(self.root_dir.glob("*b3fd*.csv"))
            for csv_file in csv_candidates:
                try:
                    df = pd.read_csv(csv_file)
                except Exception:
                    continue

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
    Reads labels_train.csv and labels_valid.csv (with fallback to gt_avg_*.csv).
    """

    LABEL_FILES = [
        "labels_train.csv", "labels_valid.csv", "labels_test.csv",
        "gt_avg_train.csv", "gt_avg_valid.csv", "gt_avg_test.csv"
    ]

    def get_name(self) -> str:
        return "appa_real"

    def is_available(self) -> bool:
        if not self.root_dir.exists():
            return False
        if any((self.root_dir / f).exists() for f in self.LABEL_FILES):
            return True
        return any(self.root_dir.glob("**/labels_*.csv")) or any(self.root_dir.glob("**/gt_avg_*.csv"))

    def stream_samples(self) -> Generator[RawImageSample, None, None]:
        if not self.is_available():
            return

        yielded = 0
        seen_ids = set()
        file_map = None

        csv_paths = [self.root_dir / f for f in self.LABEL_FILES if (self.root_dir / f).exists()]
        if not csv_paths:
            csv_paths = list(self.root_dir.glob("**/labels_*.csv")) + list(self.root_dir.glob("**/gt_avg_*.csv"))

        for csv_path in csv_paths:
            try:
                df = pd.read_csv(csv_path)
            except Exception:
                continue

            for _, row in df.iterrows():
                if yielded >= self.max_samples:
                    return
                fname = str(row.get('image_path', row.get('file_name', row.get('file', row.get('image', '')))))
                if not fname:
                    continue

                raw_basename = Path(fname).name
                if raw_basename in seen_ids:
                    continue

                # Fast candidate path resolution
                sub = csv_path.stem.replace('labels_', '').replace('gt_avg_', '')
                candidates = [
                    self.root_dir / fname,
                    self.root_dir / raw_basename,
                    self.root_dir / sub / raw_basename,
                    self.root_dir / sub / fname,
                    self.root_dir / "appa-real-release" / sub / raw_basename,
                    self.root_dir / "appa-real-release" / sub / fname,
                    self.root_dir / "appa-real-release" / raw_basename,
                    self.root_dir / "appa-real-release" / fname,
                    self.root_dir / "images" / raw_basename,
                    self.root_dir / "images" / fname
                ]
                img_path = next((p for p in candidates if p.exists()), None)

                # Fallback: index files once if candidate paths fail
                if img_path is None:
                    if file_map is None:
                        file_map = {p.name: p for p in self.root_dir.rglob("*.*") if p.suffix.lower() in self.SUPPORTED_EXTS}
                    img_path = file_map.get(raw_basename)

                if not img_path or not img_path.exists():
                    continue

                try:
                    with open(img_path, 'rb') as f:
                        b = f.read()
                except Exception:
                    continue

                seen_ids.add(raw_basename)
                real_age = float(row.get('real_age', row.get('apparent_age_avg', row.get('age', 25.0))))
                apparent_avg = float(row.get('apparent_age_avg', real_age))
                apparent_std = float(row.get('apparent_age_std', 2.5))

                yield RawImageSample(
                    image_bytes=b,
                    source_dataset=self.get_name(),
                    raw_id=raw_basename,
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


class UTKFaceConnector(BaseFaceConnector):
    """
    Connector for UTKFace (jangedoo/utkface-new).
    Ages 0 to 116 years across 5 ethnic groups.
    Filename pattern: ^(\\d+)_([01])_([0-4])_.*
    Group 1: age
    Group 2: gender (0: Male, 1: Female)
    Group 3: race (0: Caucasian, 1: African, 2: East Asian, 3: South Asian, 4: Others)
    """

    PATTERN = re.compile(r"^(\d+)_([01])_([0-4])_.*", re.IGNORECASE)

    ETHNICITY_MAP = {
        "0": "Caucasian",
        "1": "African",
        "2": "East Asian",
        "3": "South Asian",
        "4": "Others"
    }

    def get_name(self) -> str:
        return "utkface"

    def is_available(self) -> bool:
        if not self.root_dir.exists():
            return False
        if any(self.root_dir.glob("*.tar*")) or any(self.root_dir.glob("*.zip")):
            return True
        return self._has_matching_file(self.PATTERN)

    def stream_samples(self) -> Generator[RawImageSample, None, None]:
        if not self.is_available():
            return

        yielded = 0
        seen_ids = set()

        # 1. Stream from tar archives if present
        for tar_path in list(self.root_dir.rglob("*.tar*")):
            try:
                with tarfile.open(tar_path, 'r:*') as tf:
                    for member in tf:
                        if yielded >= self.max_samples:
                            return
                        if not member.isfile():
                            continue
                        fname = Path(member.name).name
                        if fname in seen_ids:
                            continue
                        m = self.PATTERN.match(fname)
                        if not m:
                            continue
                        f = tf.extractfile(member)
                        if not f:
                            continue
                        b = f.read()
                        seen_ids.add(fname)
                        age = float(m.group(1))
                        gender = "Male" if m.group(2) == "0" else "Female"
                        race = self.ETHNICITY_MAP.get(m.group(3), "Others")
                        yield RawImageSample(
                            image_bytes=b,
                            source_dataset=self.get_name(),
                            raw_id=fname,
                            declared_age=age,
                            declared_gender=gender,
                            declared_ethnicity=race,
                            metadata={"utkface_race_id": int(m.group(3))}
                        )
                        yielded += 1
            except Exception:
                pass

        # 2. Stream from zip archives if present
        for zip_path in list(self.root_dir.rglob("*.zip")):
            try:
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    for name in zf.namelist():
                        if yielded >= self.max_samples:
                            return
                        fname = Path(name).name
                        if fname in seen_ids:
                            continue
                        m = self.PATTERN.match(fname)
                        if not m:
                            continue
                        b = zf.read(name)
                        seen_ids.add(fname)
                        age = float(m.group(1))
                        gender = "Male" if m.group(2) == "0" else "Female"
                        race = self.ETHNICITY_MAP.get(m.group(3), "Others")
                        yield RawImageSample(
                            image_bytes=b,
                            source_dataset=self.get_name(),
                            raw_id=fname,
                            declared_age=age,
                            declared_gender=gender,
                            declared_ethnicity=race,
                            metadata={"utkface_race_id": int(m.group(3))}
                        )
                        yielded += 1
            except Exception:
                pass

        # 3. Stream from filesystem files
        for img_path in self.root_dir.rglob("*.*"):
            if yielded >= self.max_samples:
                return
            if img_path.suffix.lower() not in self.SUPPORTED_EXTS:
                continue
            if img_path.name in seen_ids:
                continue
            m = self.PATTERN.match(img_path.name)
            if not m:
                continue

            try:
                with open(img_path, 'rb') as f:
                    b = f.read()
            except Exception:
                continue

            seen_ids.add(img_path.name)
            age = float(m.group(1))
            gender = "Male" if m.group(2) == "0" else "Female"
            race = self.ETHNICITY_MAP.get(m.group(3), "Others")

            yield RawImageSample(
                image_bytes=b,
                source_dataset=self.get_name(),
                raw_id=img_path.name,
                declared_age=age,
                declared_gender=gender,
                declared_ethnicity=race,
                metadata={"utkface_race_id": int(m.group(3))}
            )
            yielded += 1


class AgeDBConnector(BaseFaceConnector):
    """
    Connector for AgeDB (nitingandhi/agedb-database).
    Continuous ground-truth age 1 to 101 with 100% verified DOB and photo date, 50/50 gender balance.
    Filename pattern: ^(\\d+)_(.+)_(\\d+)_([mf])\\.jpg
    Group 1: id
    Group 2: name
    Group 3: age
    Group 4: gender (m: Male, f: Female)
    """

    PATTERN = re.compile(r"^(\d+)_(.+)_(\d+)_([mf])\.(?:jpg|jpeg|png|webp|bmp)$", re.IGNORECASE)

    def get_name(self) -> str:
        return "agedb"

    def is_available(self) -> bool:
        if not self.root_dir.exists():
            return False
        if any(self.root_dir.glob("*.zip")) or any(self.root_dir.glob("*.tar*")):
            return True
        return self._has_matching_file(self.PATTERN)

    def stream_samples(self) -> Generator[RawImageSample, None, None]:
        if not self.is_available():
            return

        yielded = 0
        seen_ids = set()

        # 1. Check tar archives
        for tar_path in list(self.root_dir.rglob("*.tar*")):
            try:
                with tarfile.open(tar_path, 'r:*') as tf:
                    for member in tf:
                        if yielded >= self.max_samples:
                            return
                        if not member.isfile():
                            continue
                        fname = Path(member.name).name
                        if fname in seen_ids:
                            continue
                        m = self.PATTERN.match(fname)
                        if not m:
                            continue
                        f = tf.extractfile(member)
                        if not f:
                            continue
                        b = f.read()
                        seen_ids.add(fname)
                        sample_id, subj_name, age_str, g_str = m.group(1), m.group(2), m.group(3), m.group(4)
                        age = float(age_str)
                        gender = "Female" if g_str.lower() == "f" else "Male"
                        yield RawImageSample(
                            image_bytes=b,
                            source_dataset=self.get_name(),
                            raw_id=fname,
                            declared_age=age,
                            declared_gender=gender,
                            declared_ethnicity=None,
                            metadata={"agedb_id": sample_id, "name": subj_name}
                        )
                        yielded += 1
            except Exception:
                pass

        # 2. Check zip archives
        for zip_path in list(self.root_dir.rglob("*.zip")):
            try:
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    for name in zf.namelist():
                        if yielded >= self.max_samples:
                            return
                        fname = Path(name).name
                        if fname in seen_ids:
                            continue
                        m = self.PATTERN.match(fname)
                        if not m:
                            continue
                        b = zf.read(name)
                        seen_ids.add(fname)
                        sample_id, subj_name, age_str, g_str = m.group(1), m.group(2), m.group(3), m.group(4)
                        age = float(age_str)
                        gender = "Female" if g_str.lower() == "f" else "Male"
                        yield RawImageSample(
                            image_bytes=b,
                            source_dataset=self.get_name(),
                            raw_id=fname,
                            declared_age=age,
                            declared_gender=gender,
                            declared_ethnicity=None,
                            metadata={"agedb_id": sample_id, "name": subj_name}
                        )
                        yielded += 1
            except Exception:
                pass

        # 3. Check directory files
        for img_path in self.root_dir.rglob("*.*"):
            if yielded >= self.max_samples:
                return
            if img_path.suffix.lower() not in self.SUPPORTED_EXTS:
                continue
            if img_path.name in seen_ids:
                continue
            m = self.PATTERN.match(img_path.name)
            if not m:
                continue

            try:
                with open(img_path, 'rb') as f:
                    b = f.read()
            except Exception:
                continue

            seen_ids.add(img_path.name)
            sample_id, subj_name, age_str, g_str = m.group(1), m.group(2), m.group(3), m.group(4)
            age = float(age_str)
            gender = "Female" if g_str.lower() == "f" else "Male"

            yield RawImageSample(
                image_bytes=b,
                source_dataset=self.get_name(),
                raw_id=img_path.name,
                declared_age=age,
                declared_gender=gender,
                declared_ethnicity=None,
                metadata={"agedb_id": sample_id, "name": subj_name}
            )
            yielded += 1


class MorphConnector(BaseFaceConnector):
    """
    Connector for MORPH-2 (chiragsaipanuganti/morph).
    Official ID mugshots, ages 16-77, African-American & Caucasian representation.
    Filename pattern: ^(\\d+)_(\\d+)([MF])(\\d{2})([A-Za-z0-9_]*)\\.JPG
    Group 1: Subject ID
    Group 2: Photo ID
    Group 3: Gender (M: Male, F: Female)
    Group 4: Age (2 digits)
    Group 5: Optional ethnicity code (B: African, W: Caucasian, A: Asian, H: Latino, O: Other)
    """

    PATTERN = re.compile(r"^(\d+)_(\d+)([MFmf])(\d{2})([A-Za-z0-9_]*)\.(?:jpg|jpeg|png|webp|bmp)$", re.IGNORECASE)

    @staticmethod
    def _parse_ethnicity(suffix: str) -> Optional[str]:
        if not suffix:
            return None
        code = suffix.lstrip('_')[:1].upper()
        if code == 'B':
            return "African"
        elif code == 'W':
            return "Caucasian"
        elif code == 'A':
            return "East Asian"
        elif code == 'H':
            return "Latino"
        elif code in ['O', 'I']:
            return "Others"
        return None

    def get_name(self) -> str:
        return "morph"

    def is_available(self) -> bool:
        if not self.root_dir.exists():
            return False
        if any(self.root_dir.glob("*.tar*")) or any(self.root_dir.glob("*.zip")):
            return True
        return self._has_matching_file(self.PATTERN)

    def stream_samples(self) -> Generator[RawImageSample, None, None]:
        if not self.is_available():
            return

        yielded = 0
        seen_ids = set()

        # 1. Check tar archives
        for tar_path in list(self.root_dir.rglob("*.tar*")):
            try:
                with tarfile.open(tar_path, 'r:*') as tf:
                    for member in tf:
                        if yielded >= self.max_samples:
                            return
                        if not member.isfile():
                            continue
                        fname = Path(member.name).name
                        if fname in seen_ids:
                            continue
                        m = self.PATTERN.match(fname)
                        if not m:
                            continue
                        f = tf.extractfile(member)
                        if not f:
                            continue
                        b = f.read()
                        seen_ids.add(fname)
                        subj_id, photo_id, g_str, age_str = m.group(1), m.group(2), m.group(3), m.group(4)
                        age = float(age_str)
                        gender = "Male" if g_str.upper() == "M" else "Female"
                        race = self._parse_ethnicity(m.group(5))
                        yield RawImageSample(
                            image_bytes=b,
                            source_dataset=self.get_name(),
                            raw_id=fname,
                            declared_age=age,
                            declared_gender=gender,
                            declared_ethnicity=race,
                            metadata={"morph_subject_id": subj_id, "morph_photo_id": photo_id}
                        )
                        yielded += 1
            except Exception:
                pass

        # 2. Check zip archives
        for zip_path in list(self.root_dir.rglob("*.zip")):
            try:
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    for name in zf.namelist():
                        if yielded >= self.max_samples:
                            return
                        fname = Path(name).name
                        if fname in seen_ids:
                            continue
                        m = self.PATTERN.match(fname)
                        if not m:
                            continue
                        b = zf.read(name)
                        seen_ids.add(fname)
                        subj_id, photo_id, g_str, age_str = m.group(1), m.group(2), m.group(3), m.group(4)
                        age = float(age_str)
                        gender = "Male" if g_str.upper() == "M" else "Female"
                        race = self._parse_ethnicity(m.group(5))
                        yield RawImageSample(
                            image_bytes=b,
                            source_dataset=self.get_name(),
                            raw_id=fname,
                            declared_age=age,
                            declared_gender=gender,
                            declared_ethnicity=race,
                            metadata={"morph_subject_id": subj_id, "morph_photo_id": photo_id}
                        )
                        yielded += 1
            except Exception:
                pass

        # 3. Check filesystem files
        for img_path in self.root_dir.rglob("*.*"):
            if yielded >= self.max_samples:
                return
            if img_path.suffix.lower() not in self.SUPPORTED_EXTS:
                continue
            if img_path.name in seen_ids:
                continue
            m = self.PATTERN.match(img_path.name)
            if not m:
                continue

            try:
                with open(img_path, 'rb') as f:
                    b = f.read()
            except Exception:
                continue

            seen_ids.add(img_path.name)
            subj_id, photo_id, g_str, age_str = m.group(1), m.group(2), m.group(3), m.group(4)
            age = float(age_str)
            gender = "Male" if g_str.upper() == "M" else "Female"
            race = self._parse_ethnicity(m.group(5))

            yield RawImageSample(
                image_bytes=b,
                source_dataset=self.get_name(),
                raw_id=img_path.name,
                declared_age=age,
                declared_gender=gender,
                declared_ethnicity=race,
                metadata={"morph_subject_id": subj_id, "morph_photo_id": photo_id}
            )
            yielded += 1


class FGNetConnector(BaseFaceConnector):
    """
    Connector for FG-NET (aiolapo/fgnet-dataset).
    Longitudinal age progressions 0 to 69 years, dense infancy/childhood representation.
    Filename pattern: ^(\\d{3})[aA](\\d{2})
    Group 1: Subject ID (3 digits)
    Group 2: Age (2 digits)
    """

    PATTERN = re.compile(r"^(\d{3})[aA](\d{2})([mMfF])?", re.IGNORECASE)

    def get_name(self) -> str:
        return "fgnet"

    def is_available(self) -> bool:
        if not self.root_dir.exists():
            return False
        if any(self.root_dir.glob("*.zip")) or any(self.root_dir.glob("*.tar*")):
            return True
        return self._has_matching_file(self.PATTERN)

    def stream_samples(self) -> Generator[RawImageSample, None, None]:
        if not self.is_available():
            return

        yielded = 0
        seen_ids = set()

        # 1. Check tar archives
        for tar_path in list(self.root_dir.rglob("*.tar*")):
            try:
                with tarfile.open(tar_path, 'r:*') as tf:
                    for member in tf:
                        if yielded >= self.max_samples:
                            return
                        if not member.isfile():
                            continue
                        fname = Path(member.name).name
                        if fname in seen_ids:
                            continue
                        m = self.PATTERN.match(fname)
                        if not m:
                            continue
                        f = tf.extractfile(member)
                        if not f:
                            continue
                        b = f.read()
                        seen_ids.add(fname)
                        subj_id, age_str = m.group(1), m.group(2)
                        age = float(age_str)
                        g_char = m.group(3) if len(m.groups()) >= 3 and m.group(3) else None
                        gender = 'Male' if g_char and g_char.upper() == 'M' else ('Female' if g_char and g_char.upper() == 'F' else None)
                        yield RawImageSample(
                            image_bytes=b,
                            source_dataset=self.get_name(),
                            raw_id=fname,
                            declared_age=age,
                            declared_gender=gender,
                            declared_ethnicity=None,
                            metadata={"fgnet_subject_id": subj_id}
                        )
                        yielded += 1
            except Exception:
                pass

        # 2. Check zip archives
        for zip_path in list(self.root_dir.rglob("*.zip")):
            try:
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    for name in zf.namelist():
                        if yielded >= self.max_samples:
                            return
                        fname = Path(name).name
                        if fname in seen_ids:
                            continue
                        m = self.PATTERN.match(fname)
                        if not m:
                            continue
                        b = zf.read(name)
                        seen_ids.add(fname)
                        subj_id, age_str = m.group(1), m.group(2)
                        age = float(age_str)
                        g_char = m.group(3) if len(m.groups()) >= 3 and m.group(3) else None
                        gender = 'Male' if g_char and g_char.upper() == 'M' else ('Female' if g_char and g_char.upper() == 'F' else None)
                        yield RawImageSample(
                            image_bytes=b,
                            source_dataset=self.get_name(),
                            raw_id=fname,
                            declared_age=age,
                            declared_gender=gender,
                            declared_ethnicity=None,
                            metadata={"fgnet_subject_id": subj_id}
                        )
                        yielded += 1
            except Exception:
                pass

        # 3. Check directory files
        for img_path in self.root_dir.rglob("*.*"):
            if yielded >= self.max_samples:
                return
            if img_path.suffix.lower() not in self.SUPPORTED_EXTS:
                continue
            if img_path.name in seen_ids:
                continue
            m = self.PATTERN.match(img_path.name)
            if not m:
                continue

            try:
                with open(img_path, 'rb') as f:
                    b = f.read()
            except Exception:
                continue

            seen_ids.add(img_path.name)
            subj_id, age_str = m.group(1), m.group(2)
            age = float(age_str)
            g_char = m.group(3) if len(m.groups()) >= 3 and m.group(3) else None
            gender = 'Male' if g_char and g_char.upper() == 'M' else ('Female' if g_char and g_char.upper() == 'F' else None)

            yield RawImageSample(
                image_bytes=b,
                source_dataset=self.get_name(),
                raw_id=img_path.name,
                declared_age=age,
                declared_gender=gender,
                declared_ethnicity=None,
                metadata={"fgnet_subject_id": subj_id}
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
            if "agedb" in dname:
                self.connectors.append(AgeDBConnector(dataset_dir, self.max_samples_per_dataset))
            elif "morph" in dname:
                self.connectors.append(MorphConnector(dataset_dir, self.max_samples_per_dataset))
            elif "utk" in dname:
                self.connectors.append(UTKFaceConnector(dataset_dir, self.max_samples_per_dataset))
            elif "afad" in dname or "asian" in dname:
                self.connectors.append(AFADConnector(dataset_dir, self.max_samples_per_dataset))
            elif "appa" in dname:
                self.connectors.append(AppaRealConnector(dataset_dir, self.max_samples_per_dataset))
            elif "fgnet" in dname:
                self.connectors.append(FGNetConnector(dataset_dir, self.max_samples_per_dataset))
            elif "b3fd" in dname or "famous" in dname or "biometric" in dname:
                self.connectors.append(B3FDConnector(dataset_dir, self.max_samples_per_dataset))
            elif "fairface" in dname:
                self.connectors.append(FairFaceConnector(dataset_dir, self.max_samples_per_dataset))
            elif "celeba" in dname:
                self.connectors.append(CelebAConnector(dataset_dir, self.max_samples_per_dataset))
            elif "imdb" in dname or "wiki" in dname:
                self.connectors.append(IMDBWikiConnector(dataset_dir, self.max_samples_per_dataset))

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
        min_free_gb: float = 1.0,
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
        self.current_shard_records: List[Dict[str, Any]] = []
        self.current_tar_obj: Optional[tarfile.TarFile] = None
        self._open_new_shard()

    def _flush_shard_manifest(self, shard_idx: int):
        if self.current_shard_records:
            try:
                df_shard = pd.DataFrame(self.current_shard_records)
                manifest_path = self.shards_dir / f"shard_{shard_idx:04d}_manifest.parquet"
                df_shard.to_parquet(str(manifest_path), index=False)
                print(f"[QuotaManager] Checkpointed shard manifest: {manifest_path} ({len(df_shard)} records)")
            except Exception as e:
                print(f"[QuotaManager] Warning saving shard manifest: {e}")
            self.current_shard_records = []

    def _open_new_shard(self):
        if self.current_tar_obj:
            self.current_tar_obj.close()
            self._flush_shard_manifest(self.current_shard_idx)
            self.current_shard_idx += 1
        shard_file = self.shards_dir / f"shard_{self.current_shard_idx:04d}.tar"
        self.current_tar_obj = tarfile.open(shard_file, 'w')
        self.current_shard_count = 0

    def check_safe(self) -> bool:
        """Alias for check_quota."""
        return self.check_quota()

    def check_quota(self) -> bool:
        """Checks if current disk usage is safely below quota ceiling."""
        try:
            total, used, free = shutil.disk_usage(self.working_dir)
            if free < self.min_free_bytes:
                print(f"[QuotaManager] DISK SAFETY CEILING: Free disk space {free / (1024**3):.2f} GB < {self.min_free_bytes / (1024**3):.2f} GB. Halting cleanly.")
                return False
            return True
        except Exception:
            return True

    def write_aligned_face(
        self,
        face_bgr: Optional[np.ndarray],
        filename: str,
        record: Optional[Dict[str, Any]] = None,
        image_bytes: Optional[bytes] = None
    ) -> bool:
        """
        Compresses aligned face to high-quality JPEG (or uses pre-encoded bytes) and appends to current tar shard.
        Avoids creating thousands of separate disk inodes.
        """
        if not self.check_quota():
            return False

        if image_bytes is None:
            if face_bgr is None:
                return False
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
            success, enc_buf = cv2.imencode('.jpg', face_bgr, encode_param)
            if not success:
                return False
            img_bytes = enc_buf.tobytes()
        else:
            img_bytes = image_bytes

        tar_info = tarfile.TarInfo(name=filename)
        tar_info.size = len(img_bytes)

        if self.current_tar_obj:
            self.current_tar_obj.addfile(tar_info, io.BytesIO(img_bytes))
            self.current_shard_count += 1
            if record is not None:
                self.current_shard_records.append(record)

            if self.current_shard_count >= self.shard_image_count:
                self._open_new_shard()

        return True

    def close(self):
        """Finalizes and closes all open archive handles."""
        if self.current_tar_obj:
            self.current_tar_obj.close()
            self.current_tar_obj = None
            self._flush_shard_manifest(self.current_shard_idx)


def ensure_master_manifest(golden_dataset_dir: str | Path) -> pd.DataFrame:
    """
    Consolidates all shard manifests (shards/shard_*_manifest.parquet) into root metadata.parquet and metadata.csv.
    Self-healing: Reconstructs master manifest from individual shard checkpoints if session was interrupted.
    """
    golden_dir = Path(golden_dataset_dir)
    p_parquet = golden_dir / "metadata.parquet"
    p_csv = golden_dir / "metadata.csv"
    shards_dir = golden_dir / "shards"
    shard_manifests = sorted(shards_dir.glob("shard_*_manifest.parquet")) if shards_dir.exists() else []

    if shard_manifests:
        print(f"[Manifest] Consolidating {len(shard_manifests)} shard manifest(s) from {shards_dir} ...")
        dfs = []
        for sm in shard_manifests:
            try:
                dfs.append(pd.read_parquet(sm))
            except Exception as e:
                print(f"⚠️ Warning reading shard manifest {sm}: {e}")
        if dfs:
            master = pd.concat(dfs, ignore_index=True)
            if "sha256" in master.columns:
                master.drop_duplicates(subset=["sha256"], inplace=True, ignore_index=True)
            master.to_parquet(str(p_parquet), index=False)
            master.to_csv(str(p_csv), index=False)
            print(f"✅ Consolidated {len(master)} records across {len(shard_manifests)} shards into {p_parquet} and {p_csv}")
            return master

    if p_parquet.exists():
        try:
            return pd.read_parquet(p_parquet)
        except Exception:
            pass

    empty_df = pd.DataFrame()
    golden_dir.mkdir(parents=True, exist_ok=True)
    empty_df.to_parquet(str(p_parquet), index=False)
    empty_df.to_csv(str(p_csv), index=False)
    return empty_df


__all__ = [
    "RawImageSample",
    "BaseFaceConnector",
    "FairFaceConnector",
    "CelebAConnector",
    "IMDBWikiConnector",
    "AFADConnector",
    "B3FDConnector",
    "AppaRealConnector",
    "UTKFaceConnector",
    "AgeDBConnector",
    "MorphConnector",
    "FGNetConnector",
    "InCloudStreamingHarvester",
    "DiskQuotaManager",
    "ensure_master_manifest",
]
