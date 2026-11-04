"""Fail-closed manifest audit. A manifest check never certifies image or label quality."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import tarfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def _is_kaggle_runtime() -> bool:
    # Kaggle's own runtime helper uses these environment markers; they are not cryptographic attestation.
    return (
        bool(os.environ.get('KAGGLE_KERNEL_RUN_TYPE'))
        and 'KAGGLE_USER_SECRETS_TOKEN' in os.environ
        and Path('/kaggle/input').is_dir()
        and Path('/kaggle/working').is_dir()
    )


EXPECTED_MANIFEST_COLUMNS = [
    'image_id', 'sha256', 'source_dataset', 'original_filename',
    'age_continuous', 'age_group', 'age_group_idx', 'apparent_age_std',
    'gender', 'gender_prob', 'ethnicity', 'dimorphism_weight',
    'ldl_0_2', 'ldl_3_6', 'ldl_7_12', 'ldl_13_17', 'ldl_18_24',
    'ldl_25_34', 'ldl_35_44', 'ldl_45_54', 'ldl_55_64', 'ldl_65_plus',
    'teacher_count', 'teacher_age_mu', 'teacher_age_sigma',
    'teacher_gender_agreement', 'alpha_sample_weight',
    'laplacian_var', 'tenengrad', 'mean_luminance', 'subject_id', 'identity_id',
    'shadow_clip', 'highlight_clip', 'yaw', 'pitch', 'roll', 'phash', 'split',
]
LDL_COLUMNS = [c for c in EXPECTED_MANIFEST_COLUMNS if c.startswith('ldl_')]
SPLITS = {'train', 'val', 'test'}
AGE_GROUPS = ['0-2', '3-6', '7-12', '13-17', '18-24', '25-34', '35-44', '45-54', '55-64', '65+']
AGE_CUTS = [2.5, 6.5, 12.5, 17.5, 24.5, 34.5, 44.5, 54.5, 64.5]
FINE_AGE_INTERVALS = [
    ('0-2', 0, 2), ('3-5', 3, 5), ('6-8', 6, 8), ('9-11', 9, 11),
    ('12-14', 12, 14), ('15-17', 15, 17), ('18-21', 18, 21), ('22-25', 22, 25),
    ('26-30', 26, 30), ('31-35', 31, 35), ('36-40', 36, 40), ('41-45', 41, 45),
    ('46-50', 46, 50), ('51-55', 51, 55), ('56-60', 56, 60), ('61-65', 61, 65),
    ('66-70', 66, 70), ('71-75', 71, 75), ('76-80', 76, 80), ('81-85', 81, 85),
    ('86-90', 86, 90), ('91-95', 91, 95), ('96-100', 96, 100),
]
FINE_INTERVAL_NAMES = {name: idx for idx, (name, _, _) in enumerate(FINE_AGE_INTERVALS)}
FINE_LDL_COLUMNS = [f"fine_ldl_{name.replace('-', '_')}" for name, _, _ in FINE_AGE_INTERVALS]
BANNED_NON_REAL_SOURCE_PATTERNS = (
    'utk', 'utkface', 'stylegan', 'thispersondoesnotexist', 'tpne',
    'ffhq', 'flickrfaceshq', 'sfhq', 'digiface', 'dcface', 'idiff',
    'synface', 'synthetics', 'synthetic', 'gansynth', 'generated_photos', 'generated-photos',
    'ai_generated', 'ai-generated', 'aigenerated', 'gravex', 'midjourney',
    'stable_diffusion', 'stablediffusion', 'dall_e', 'dalle',
    'metahuman', 'makehuman', 'renderme', '3d_face', '3d-face', '300w_lp', '300w-lp', '300wlp',
    'cgi_face', 'cgiface', 'deepfake', 'dfdc', 'faceforensics', 'celeb_df', 'celeb-df',
    'faceswap', 'deepfacelab', 'facemorph', 'face_morph', 'face-morph', 'cartoon', 'anime',
    'danbooru', 'waifu', 'sketch', 'caricature', 'iiit_cfw', 'iiit-cfw', 'webcaricature',
    'painting', 'metfaces', 'avatar', 'bitmoji', 'emoji', 'statue', 'sculpture', 'mannequin', 'doll',
)


def _is_non_real_source(source_name: str) -> bool:
    s = str(source_name or '').strip().lower()
    return any(p in s for p in BANNED_NON_REAL_SOURCE_PATTERNS)
MAX_MANIFEST_BYTES = 128 * 1024 * 1024
MAX_PARQUET_UNCOMPRESSED_BYTES = 512 * 1024 * 1024
MAX_MANIFEST_ROWS = 500_000
MAX_PHASH_CANDIDATE_COMPARISONS = 1_000_000


def parse_phash(value):
    if isinstance(value, (bool, np.bool_, float, np.floating)):
        raise ValueError('pHash must be an integer or a hexadecimal string')
    if isinstance(value, (int, np.integer)):
        parsed = int(value)
    elif isinstance(value, str) and re.fullmatch(r'(?:0x)?[0-9a-fA-F]{1,16}', value.strip()):
        parsed = int(value, 16)
    else:
        raise ValueError('Invalid 64-bit pHash')
    if not 0 <= parsed < 2**64:
        raise ValueError('pHash outside unsigned 64-bit range')
    return parsed


def cross_split_hash_pairs(hashes, splits, threshold=6, *, max_candidate_comparisons=None):
    if max_candidate_comparisons is None:
        max_candidate_comparisons = MAX_PHASH_CANDIDATE_COMPARISONS
    if (not 0 <= threshold < 64 or len(hashes) != len(splits)
            or max_candidate_comparisons < 0):
        raise ValueError('Invalid Hamming threshold, comparison budget, or mismatched inputs')
    # With threshold+1 disjoint chunks, every pair within threshold shares a chunk.
    count = threshold + 1
    edges = [(64 * i // count, 64 * (i + 1) // count) for i in range(count)]
    indexes = [defaultdict(lambda: defaultdict(list)) for _ in edges]
    counts, affected, examples = Counter(), set(), []
    comparisons = 0
    for i, (value, split) in enumerate(zip(hashes, splits)):
        chunks = [(value >> lo) & ((1 << (hi - lo)) - 1) for lo, hi in edges]
        candidates = set()
        for index, chunk in zip(indexes, chunks):
            for indexed_split, rows in index[chunk].items():
                if indexed_split == split:
                    continue
                for j in rows:
                    if j in candidates:
                        continue
                    if comparisons >= max_candidate_comparisons:
                        return {'pairs': sum(counts.values()), 'by_split_pair': dict(counts),
                                'affected_rows': len(affected), 'examples': examples,
                                'candidate_comparisons': comparisons,
                                'rows_checked': i, 'total_rows': len(hashes), 'threshold': threshold,
                                'coverage_complete': False, 'limit_reached': True,
                                'scope': 'incomplete: candidate comparison budget reached'}
                    candidates.add(j)
                    comparisons += 1
                    distance = (value ^ hashes[j]).bit_count()
                    if distance <= threshold:
                        key = '/'.join(sorted([str(split), str(splits[j])]))
                        counts[key] += 1
                        affected.update([i, j])
                        if len(examples) < 20:
                            examples.append({'row_a': j, 'row_b': i, 'distance': distance})
        for index, chunk in zip(indexes, chunks):
            index[chunk][split].append(i)
    return {'pairs': sum(counts.values()), 'by_split_pair': dict(counts),
            'affected_rows': len(affected), 'examples': examples,
            'candidate_comparisons': comparisons, 'rows_checked': len(hashes),
            'total_rows': len(hashes), 'threshold': threshold,
            'coverage_complete': True, 'limit_reached': False,
            'scope': 'all rows, all distinct splits'}


def _canonical_celebrity_key(source: str, subject: str) -> str | None:
    s = str(subject or '').strip().lower()
    if ':' in s:
        _, s = s.split(':', 1)
    s = re.sub(r'^(nm\d+)_rm[a-z0-9]+$', r'\1', s.strip())
    norm = re.sub(r'[^a-z0-9]', '', s)
    if not norm:
        return None
    if re.match(r'^nm\d+$', norm):
        return f'imdb:{norm}'
    if source in {'agedb', 'cacd', 'b3fd', 'imdb_wiki', 'imdb_clean'} and any(c.isalpha() for c in norm):
        return f'celeb:{norm}'
    return None


def subject_overlap(df):
    groups = defaultdict(set)
    celeb_groups = defaultdict(set)
    covered, coverage, origins, unknown_holdout = 0, Counter(), Counter(), 0
    for _, row in df.iterrows():
        source = str(row.get('source_dataset', '')).strip().lower()
        subject, origin = None, None
        for column in ['subject_id', 'identity', 'person_id', 'identity_id']:
            value = row.get(column)
            if pd.notna(value) and str(value).strip().lower() not in {'', '-1', 'unknown', 'none', 'nan'}:
                subject, origin = str(value).strip(), column
                break
        if subject is None:
            name = str(row.get('original_filename', ''))
            pattern = {'agedb': r'^\d+_(.+)_\d+_[mf]\.',
                       'fgnet': r'^(\d{3})[aA]\d{2}',
                       'morph': r'^(\d+)_\d+[MFmf]\d{2}[A-Za-z0-9_]*\.'}.get(source)
            match = re.match(pattern, name, flags=re.IGNORECASE) if pattern else None
            if match:
                subject, origin = match.group(1).casefold(), 'filename_derived'
        if subject is not None:
            covered += 1
            coverage[source] += 1
            origins[origin] += 1
            split_val = str(row.get('split', ''))
            groups[(source, subject)].add(split_val)
            ckey = _canonical_celebrity_key(source, subject)
            if ckey is not None:
                celeb_groups[ckey].add(split_val)
        elif str(row.get('split', '')) != 'train':
            unknown_holdout += 1
    leaking = Counter(source for (source, _), splits in groups.items() if len(splits) > 1)
    cross_source_celeb_leaks = sum(1 for _, splits in celeb_groups.items() if len(splits) > 1)
    return {'covered_rows': covered, 'unknown_rows': len(df) - covered,
            'coverage_by_source': dict(coverage), 'identity_origins': dict(origins),
            'subjects_checked': len(groups), 'cross_split_subjects': sum(leaking.values()),
            'cross_split_subjects_by_source': dict(leaking),
            'cross_split_celebrities': cross_source_celeb_leaks,
            'unknown_identity_holdout_rows': unknown_holdout,
            'limit': 'Source-namespaced IDs plus canonical celebrity/IMDB cross-source check.'}


def validate_manifest_integrity(df: pd.DataFrame, *, forbid_non_real_sources: bool = False) -> dict:
    result = {'total_records': len(df), 'columns_present': list(df.columns),
              'missing_columns': [c for c in EXPECTED_MANIFEST_COLUMNS if c not in df],
              'null_counts': {c: int(n) for c, n in df.isna().sum().items()},
              'violations': [], 'passed': False, 'scope': 'manifest only'}
    errors = result['violations']
    if result['missing_columns']:
        errors.append(f"Missing expected columns: {result['missing_columns']}")
    if df.empty:
        errors.append('Empty manifest')
    for c in ['image_id', 'sha256', 'source_dataset', 'original_filename', 'split']:
        if c in df:
            bad = df[c].isna() | df[c].astype(str).str.strip().eq('')
            if bad.any():
                errors.append(f'{c}: {int(bad.sum())} missing values')
    if 'image_id' in df:
        bad = ~df.image_id.astype(str).str.fullmatch(r'[^/\\:]+\.(?:jpg|jpeg|png|webp|bmp)', case=False)
        if bad.any():
            errors.append(f'image_id: {int(bad.sum())} unsafe or invalid filenames')
        if df.image_id.duplicated().any():
            errors.append(f'image_id: {int(df.image_id.duplicated().sum())} duplicate rows')
    if 'sha256' in df:
        invalid = ~df.sha256.astype(str).str.fullmatch(r'[0-9a-fA-F]{64}')
        if invalid.any():
            errors.append(f'sha256: {int(invalid.sum())} invalid digests')
        if df.sha256.duplicated().any():
            errors.append(f'sha256: {int(df.sha256.duplicated().sum())} duplicate content rows')
    if 'split' in df:
        result['split_counts'] = {str(k): int(v) for k, v in df.split.value_counts(dropna=False).items()}
        if set(df.split.dropna()) != SPLITS:
            errors.append('Require nonempty train, val, test splits with no unknown split values')
    bounds = {'age_continuous': (0, 100), 'age_group_idx': (0, 9),
              'gender_prob': (0, 1),
              'dimorphism_weight': (0.25, 1), 'alpha_sample_weight': (0, None),
              'teacher_count': (0, None),
              'laplacian_var': (100, None), 'tenengrad': (250, None),
              'mean_luminance': (35, 220), 'shadow_clip': (0, 0.08),
              'highlight_clip': (0, 0.08), 'yaw': (-35, 35), 'pitch': (-25, 25), 'roll': (-20, 20)}
    for column, (lo, hi) in bounds.items():
        if column not in df:
            continue
        values = pd.to_numeric(df[column], errors='coerce').to_numpy(dtype=float)
        bad = ~np.isfinite(values) | (values < lo)
        if hi is not None:
            bad |= values > hi
        if column in {'age_group_idx', 'teacher_count'}:
            bad |= values != np.floor(values)
        if column == 'teacher_count':
            bad |= values > 0
        if bad.any():
            prefix = 'Blurry-sample gate: ' if column == 'laplacian_var' else ''
            errors.append(f'{prefix}{column}: {int(bad.sum())} invalid or out-of-policy values')
    if forbid_non_real_sources and 'source_dataset' in df:
        bad_src = df.source_dataset.astype(str).map(_is_non_real_source)
        if bad_src.any():
            errors.append(f'source_dataset: {int(bad_src.sum())} rows from banned non-real or pseudo-labeled sources')
    if {'age_min', 'age_max'}.issubset(df.columns):
        amin = pd.to_numeric(df.age_min, errors='coerce').to_numpy(dtype=float)
        amax = pd.to_numeric(df.age_max, errors='coerce').to_numpy(dtype=float)
        bad_bounds = ~np.isfinite(amin) | ~np.isfinite(amax) | (amin < 0) | (amax > 100) | (amin > amax)
        if bad_bounds.any():
            errors.append(f'Age intervals: {int(bad_bounds.sum())} invalid [age_min, age_max] bounds')
    if {'age_interval', 'age_interval_idx'}.issubset(df.columns):
        iv_idx = pd.to_numeric(df.age_interval_idx, errors='coerce').to_numpy(dtype=float)
        expected_iv_idx = np.array([FINE_INTERVAL_NAMES.get(str(x), -1) for x in df.age_interval.fillna('')], dtype=float)
        bad_iv = (expected_iv_idx < 0) | (iv_idx != expected_iv_idx)
        if bad_iv.any():
            errors.append(f'Age intervals: {int(bad_iv.sum())} inconsistent 3-5 year interval annotations')
    if 'teacher_count' in df:
        teacher_columns = ['teacher_age_mu', 'teacher_age_sigma', 'teacher_gender_agreement']
        if set(teacher_columns).issubset(df.columns):
            has_teacher_evidence = df[teacher_columns].notna().any(axis=1)
            if has_teacher_evidence.any():
                errors.append(f'Teacher metadata: {int(has_teacher_evidence.sum())} rows claim unverified teacher outputs')
    if 'apparent_age_std' in df:
        values = pd.to_numeric(df.apparent_age_std, errors='coerce').to_numpy(dtype=float)
        bad = np.isfinite(values) & (values < 0)
        if bad.any():
            errors.append(f'apparent_age_std: {int(bad.sum())} negative values')
    if {'subject_id', 'identity_id'}.issubset(df):
        known = df.subject_id.notna() & df.subject_id.astype(str).str.strip().ne('')
        ids = pd.to_numeric(df.identity_id, errors='coerce').to_numpy(dtype=float)
        bad = known.to_numpy() != np.isfinite(ids)
        bad |= np.isfinite(ids) & ((ids < 0) | (ids >= 2**53) | (ids != np.floor(ids)))
        if bad.any():
            errors.append(f'Identity IDs: {int(bad.sum())} missing or invalid paired IDs')
        pairs = df.loc[known, ['subject_id', 'identity_id']].drop_duplicates()
        if pairs.subject_id.duplicated().any() or pairs.identity_id.duplicated().any():
            errors.append('Identity IDs: subject and compact ID mapping is not one-to-one')
    if {'age_continuous', 'age_group', 'age_group_idx'}.issubset(df):
        ages = pd.to_numeric(df.age_continuous, errors='coerce').to_numpy(dtype=float)
        indices = pd.to_numeric(df.age_group_idx, errors='coerce').to_numpy(dtype=float)
        expected = np.searchsorted(AGE_CUTS, ages, side='right')
        names = np.asarray(AGE_GROUPS)[expected]
        bad = (indices != expected) | (df.age_group.fillna('').to_numpy() != names)
        if bad.any():
            errors.append(f'Age cohorts: {int(bad.sum())} missing or inconsistent annotations')
    if {'shadow_clip', 'highlight_clip'}.issubset(df):
        clipped = pd.to_numeric(df.shadow_clip, errors='coerce') + pd.to_numeric(df.highlight_clip, errors='coerce')
        if (clipped > 0.08).any():
            errors.append(f'Combined clipping: {int((clipped > 0.08).sum())} samples exceed 0.08')
    if set(LDL_COLUMNS).issubset(df.columns):
        probs = df[LDL_COLUMNS].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float)
        bad = ~np.isfinite(probs).all(axis=1) | (probs < 0).any(axis=1) | (probs > 1).any(axis=1)
        bad |= np.abs(probs.sum(axis=1) - 1) > 1e-3
        if bad.any():
            errors.append(f'LDL: {int(bad.sum())} invalid probability vectors')
    if set(FINE_LDL_COLUMNS).issubset(df.columns):
        fine_probs = df[FINE_LDL_COLUMNS].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float)
        bad_fine = ~np.isfinite(fine_probs).all(axis=1) | (fine_probs < 0).any(axis=1) | (fine_probs > 1).any(axis=1)
        bad_fine |= np.abs(fine_probs.sum(axis=1) - 1) > 1e-3
        if bad_fine.any():
            errors.append(f'Fine 23-interval LDL: {int(bad_fine.sum())} invalid probability vectors')
    if 'gender' in df:
        values = df.gender.fillna('unknown').astype(str).str.strip().str.lower()
        allowed = {'male', 'female', 'm', 'f', '0', '1', '0.0', '1.0', '', 'unknown', 'none', 'nan', '-1', '-1.0'}
        if (~values.isin(allowed)).any():
            errors.append(f'gender: {int((~values.isin(allowed)).sum())} invalid annotations')
        result['unlabelled_gender_rows'] = int(values.isin({'', 'unknown', 'none', 'nan', '-1', '-1.0'}).sum())
    if {'phash', 'split'}.issubset(df.columns):
        hashes, splits, invalid = [], [], 0
        for value, split in zip(df.phash, df.split):
            try:
                hashes.append(parse_phash(value))
                splits.append(split)
            except ValueError:
                invalid += 1
        if invalid:
            errors.append(f'phash: {invalid} invalid hashes; coverage incomplete')
        result['phash_leakage'] = cross_split_hash_pairs(hashes, splits)
        if result['phash_leakage']['pairs']:
            errors.append(f"Cross-split pHash candidates: {result['phash_leakage']['pairs']}")
        if not result['phash_leakage']['coverage_complete']:
            errors.append('pHash candidate comparison budget reached; leakage coverage incomplete')
    result['subject_overlap'] = subject_overlap(df)
    if result['subject_overlap']['cross_split_subjects']:
        errors.append(f"Cross-split subjects: {result['subject_overlap']['cross_split_subjects']}")
    if result['subject_overlap'].get('cross_split_celebrities', 0):
        errors.append(f"Cross-split canonical celebrities: {result['subject_overlap']['cross_split_celebrities']}")
    if result['subject_overlap']['unknown_identity_holdout_rows']:
        errors.append(f"Unknown identities in holdout: {result['subject_overlap']['unknown_identity_holdout_rows']} rows")
    result['passed'] = not errors
    return result


def read_manifest(path):
    path = Path(path)
    if not path.is_file():
        raise ValueError('Manifest path is not a regular file')
    if path.stat().st_size > MAX_MANIFEST_BYTES:
        raise ValueError(f'Manifest exceeds byte limit ({MAX_MANIFEST_BYTES})')
    if path.suffix.lower() == '.parquet':
        try:
            import pyarrow.parquet as pq
        except ImportError as error:
            raise RuntimeError('pyarrow is required for bounded Parquet metadata inspection') from error
        metadata = pq.ParquetFile(path).metadata
        if metadata.num_rows > MAX_MANIFEST_ROWS:
            raise ValueError(f'Manifest exceeds row limit ({MAX_MANIFEST_ROWS})')
        uncompressed_bytes = sum(metadata.row_group(i).total_byte_size for i in range(metadata.num_row_groups))
        if uncompressed_bytes > MAX_PARQUET_UNCOMPRESSED_BYTES:
            raise ValueError(f'Manifest exceeds uncompressed Parquet byte limit ({MAX_PARQUET_UNCOMPRESSED_BYTES})')
        return pd.read_parquet(path)
    df = pd.read_csv(path, dtype={'image_id': str, 'sha256': str, 'phash': str},
                     nrows=MAX_MANIFEST_ROWS + 1)
    if len(df) > MAX_MANIFEST_ROWS:
        raise ValueError(f'Manifest exceeds row limit ({MAX_MANIFEST_ROWS})')
    return df


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    total_bytes = 0
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            total_bytes += len(chunk)
            if total_bytes > MAX_MANIFEST_BYTES:
                raise ValueError(f'Manifest exceeds byte limit ({MAX_MANIFEST_BYTES}) while hashing')
            digest.update(chunk)
    return digest.hexdigest()


def audit_dataset(root: Path, *, image_root: Path | None = None, forbid_non_real_sources: bool = False) -> dict:
    import cv2
    from PIL import Image

    cv2.setNumThreads(1)
    root = Path(root)
    image_root = Path(image_root) if image_root is not None else root
    manifest_path = next((root / n for n in ['metadata.parquet', 'metadata.csv'] if (root / n).is_file()), None)
    if manifest_path is None:
        raise FileNotFoundError(f'No manifest in {root}')
    df = read_manifest(manifest_path)
    result = {'root': str(root), 'image_root': str(image_root),
              'manifest_sha256': _sha256_file(manifest_path),
              'manifest': validate_manifest_integrity(df, forbid_non_real_sources=forbid_non_real_sources), 'integrity_passed': False,
              'dataset_violations': [],
              'training_ready': False, 'readiness': 'NOT VERIFIED',
              'unverified': ['Independent label adjudication', 'Cross-source identity equivalence',
                             'Official holdout preservation', 'License and consent provenance',
                             'Deployment coverage and demographic performance']}
    for col in ['source_dataset', 'age_group', 'age_interval', 'gender', 'ethnicity', 'skin_tone_group']:
        if col in df:
            result[col + '_counts'] = {str(k): int(v) for k, v in df[col].fillna('unknown').value_counts().items()}
    entropy_stats = {}
    for col in ['source_dataset', 'age_group', 'age_interval', 'gender', 'ethnicity', 'skin_tone_group']:
        if col in df and len(df) > 0:
            probs = df[col].fillna('unknown').value_counts(normalize=True).to_numpy(dtype=float)
            probs = probs[probs > 0]
            entropy_stats[col] = round(float(-np.sum(probs * np.log2(probs))), 4)
    if entropy_stats:
        result['demographic_entropy_bits'] = entropy_stats
    if {'source_dataset', 'split'}.issubset(df):
        result['source_split_counts'] = pd.crosstab(df.source_dataset, df.split).to_dict()
    if 'age_continuous' in df:
        ages = pd.to_numeric(df.age_continuous, errors='coerce')
        finite = ages[np.isfinite(ages)]
        result['age_quantiles'] = {str(q): float(finite.quantile(q)) for q in [0, .01, .1, .5, .9, .99, 1]} if len(finite) else {}
    result['unique_values'] = {c: int(df[c].nunique()) for c in ['teacher_count', 'teacher_gender_agreement', 'teacher_age_mu',
                               'laplacian_var', 'tenengrad', 'mean_luminance'] if c in df}
    result['images'] = images = {k: 0 for k in ['archive_images', 'decoded_images', 'missing_images',
        'sha256_mismatches', 'decode_failures', 'unsafe_members', 'oversized_members',
        'unexpected_members', 'duplicate_members', 'wrong_resolution', 'constant_images',
        'cross_split_pixel_duplicate_groups', 'pixel_duplicate_groups', 'pixel_duplicate_excess_rows', 'archive_errors']}
    if not {'image_id', 'sha256', 'split'}.issubset(df):
        result['readiness'] = 'BLOCKED'
        return result
    records = {str(row.image_id): row for row in df.itertuples(index=False)}
    seen, pixel_splits, pixel_counts = Counter(), defaultdict(set), Counter()
    seen_shapes = set()
    decoded_hashes, decoded_splits = [], []
    metrics, quality_fails, phash_changes = defaultdict(list), Counter(), Counter()

    def consume(name, payload):
        seen[name] += 1
        if seen[name] > 1:
            images['duplicate_members'] += 1
        row = records[name]
        if hashlib.sha256(payload).hexdigest() != str(row.sha256).lower():
            images['sha256_mismatches'] += 1
        try:
            with Image.open(io.BytesIO(payload)) as header:
                if header.width * header.height > 1024 * 1024:
                    images['oversized_members'] += 1
                    return
                header.load()
            bgr = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
        except (OSError, ValueError, SyntaxError, Image.DecompressionBombError, cv2.error):
            bgr = None
        if bgr is None:
            images['decode_failures'] += 1
            return
        images['decoded_images'] += 1
        if bgr.shape not in {(160, 160, 3), (224, 224, 3)}:
            images['wrong_resolution'] += 1
        elif seen_shapes and bgr.shape not in seen_shapes:
            images['wrong_resolution'] += 1
        else:
            seen_shapes.add(bgr.shape)
        if float(bgr.std()) < 1e-6:
            images['constant_images'] += 1
        pixel_digest = hashlib.sha256(str(bgr.shape).encode() + bgr.tobytes()).hexdigest()
        pixel_splits[pixel_digest].add(str(row.split))
        pixel_counts[pixel_digest] += 1
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        lap = float(cv2.Laplacian(gray, cv2.CV_64F, ksize=3).var())
        gx, gy = cv2.Sobel(gray, cv2.CV_64F, 1, 0), cv2.Sobel(gray, cv2.CV_64F, 0, 1)
        ten = float(np.mean(gx * gx + gy * gy))
        l_channel = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[:, :, 0]
        luminance = float(l_channel.mean())
        clipped = float(np.mean(l_channel < 8) + np.mean(l_channel > 240))
        for key, value in [('laplacian_var', lap), ('tenengrad', ten), ('mean_luminance', luminance)]:
            metrics[key].append(value)
        quality_fails['laplacian_below_100'] += int(lap < 100)
        quality_fails['tenengrad_below_250'] += int(ten < 250)
        quality_fails['luminance_outside_35_220'] += int(not 35 <= luminance <= 220)
        quality_fails['combined_clipping_above_0_08'] += int(clipped > 0.08)
        dct = cv2.dct(np.float32(cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)))[:8, :8]
        bits = (dct > np.median(dct[1:, 1:])).flatten()
        phash = int.from_bytes(np.packbits(bits).tobytes(), 'big')
        decoded_hashes.append(phash)
        decoded_splits.append(str(row.split))
        try:
            phash_changes[str((phash ^ parse_phash(getattr(row, 'phash', None))).bit_count())] += 1
        except ValueError:
            phash_changes['invalid_recorded_hash'] += 1

    archives = sorted(image_root.rglob('*.tar'))
    images['archive_count'] = len(archives)
    for path in archives:
        try:
            with tarfile.open(path, 'r:*') as archive:
                for member in archive:
                    if not member.isfile():
                        if not member.isdir():
                            images['unsafe_members'] += 1
                        continue
                    images['archive_images'] += 1
                    if not re.fullmatch(r'[^/\\:]+\.(?:jpg|jpeg|png|webp|bmp)', member.name, flags=re.IGNORECASE):
                        images['unsafe_members'] += 1
                        continue
                    if member.name not in records:
                        images['unexpected_members'] += 1
                        continue
                    if not 0 < member.size <= 16 * 1024 * 1024:
                        images['oversized_members'] += 1
                        continue
                    with archive.extractfile(member) as stream:
                        consume(member.name, stream.read(member.size + 1))
        except (tarfile.TarError, OSError) as error:
            images['archive_errors'] += 1
            result['dataset_violations'].append(f'{path.name}: {type(error).__name__}; archive coverage incomplete')
        print(json.dumps({'audited_archive': path.name, 'decoded': images['decoded_images']}), flush=True)
    for name in records.keys() - seen.keys():
        if not re.fullmatch(r'[^/\\:]+\.(?:jpg|jpeg|png|webp|bmp)', name, flags=re.IGNORECASE):
            continue
        path = next((p for p in [image_root / 'images' / name, image_root / name] if p.is_file()), None)
        if path is not None:
            if path.stat().st_size <= 16 * 1024 * 1024:
                consume(name, path.read_bytes())
            else:
                images['oversized_members'] += 1
    images['missing_images'] = len(records.keys() - seen.keys())
    images['cross_split_pixel_duplicate_groups'] = sum(len(v) > 1 for v in pixel_splits.values())
    images['pixel_duplicate_groups'] = sum(n > 1 for n in pixel_counts.values())
    images['pixel_duplicate_excess_rows'] = sum(n - 1 for n in pixel_counts.values() if n > 1)
    images['decoded_phash_leakage'] = cross_split_hash_pairs(decoded_hashes, decoded_splits)
    images['stored_vs_decoded_phash_hamming'] = dict(phash_changes)
    images['measured_quality_failures'] = dict(quality_fails)
    images['measured_quality_quantiles'] = {key: {str(q): float(np.quantile(values, q)) for q in [0, .01, .5, .99, 1]}
                                             for key, values in metrics.items()}
    images['phash_limit'] = 'Similarity candidates require adjudication; pHash cannot establish identity disjointness.'
    if images['decoded_images'] != len(df):
        result['dataset_violations'].append('Image decode coverage does not equal manifest row count')
    def _canonicalize_manifest_nulls(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        for col in out.columns:
            if out[col].isna().any() and out[col].dtype == object:
                out[col] = [np.nan if pd.isna(x) else x for x in out[col]]
        return out

    for path in sorted((root / 'shards').glob('*_manifest.parquet')):
        result.setdefault('shard_manifest_rows', {})[path.name] = len(pd.read_parquet(path))
    if 'shard_manifest_rows' in result:
        shard_df = pd.concat([pd.read_parquet(p) for p in sorted((root / 'shards').glob('*_manifest.parquet'))], ignore_index=True)
        try:
            pd.testing.assert_frame_equal(
                _canonicalize_manifest_nulls(df.sort_values('image_id').reset_index(drop=True)),
                _canonicalize_manifest_nulls(shard_df.sort_values('image_id').reset_index(drop=True)),
                check_dtype=False,
            )
            result['master_matches_shards'] = True
        except AssertionError:
            result['master_matches_shards'] = False
            result['dataset_violations'].append('Master manifest differs from concatenated shard manifests')
    if (root / 'metadata.csv').exists() and manifest_path.suffix == '.parquet':
        try:
            pd.testing.assert_frame_equal(
                _canonicalize_manifest_nulls(df.reset_index(drop=True)),
                _canonicalize_manifest_nulls(read_manifest(root / 'metadata.csv')),
                check_dtype=False,
            )
            result['csv_matches_parquet'] = True
        except AssertionError:
            result['csv_matches_parquet'] = False
            result['dataset_violations'].append('CSV and Parquet manifest disagree')
    failures = [images[k] for k in ['missing_images', 'sha256_mismatches', 'decode_failures', 'unsafe_members',
        'oversized_members', 'unexpected_members', 'duplicate_members', 'wrong_resolution', 'constant_images',
        'cross_split_pixel_duplicate_groups', 'pixel_duplicate_groups', 'archive_errors']]
    result['integrity_passed'] = (result['manifest']['passed'] and not result['dataset_violations']
                                  and not any(failures) and not any(quality_fails.values())
                                  and images['decoded_phash_leakage']['pairs'] == 0)
    result['readiness'] = 'NOT VERIFIED' if result['integrity_passed'] else 'BLOCKED'
    return result


def generate_audit_report(manifest_path: Path, output_report_path: Path, *, result=None) -> str:
    if result is None:
        result = validate_manifest_integrity(read_manifest(manifest_path))
    status = 'BLOCKED' if not result['passed'] else 'MANIFEST CHECKS PASSED; DATASET QUALITY NOT VERIFIED'
    report = (f'# Báo Cáo Thẩm Định Manifest Dataset\n\nStatus: **{status}**\n\n'
              f"Rows: {result['total_records']:,}\n\n"
              'Scope: all manifest rows and all three splits. Image availability, ground-truth '
              'correctness, cross-source identities, licensing, and real-world bias require separate evidence.\n\n')
    report += '\n'.join(f'- {e}' for e in result['violations']) + '\n'
    report += f"\nRecorded pHash candidates: {json.dumps(result.get('phash_leakage', {}))}\n"
    report += f"\nIdentity coverage: {json.dumps(result['subject_overlap'])}\n"
    output_report_path = Path(output_report_path)
    output_report_path.parent.mkdir(parents=True, exist_ok=True)
    output_report_path.write_text(report, encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--manifest-root', type=Path)
    parser.add_argument('--images-root', type=Path)
    parser.add_argument('--report', type=Path, default=Path('reports/dataset_audit_report.md'))
    args = parser.parse_args()
    if not _is_kaggle_runtime():
        parser.error('Manifest auditing is Kaggle-only; run this command inside a Kaggle notebook/runtime')
    if args.manifest is not None:
        if args.manifest_root is not None or args.images_root is not None:
            parser.error('--manifest cannot be combined with --manifest-root or --images-root')
        result = validate_manifest_integrity(read_manifest(args.manifest))
        generate_audit_report(args.manifest, args.report, result=result)
        raise SystemExit(0 if result['passed'] else 1)
    import platform
    import time
    import importlib.metadata
    import cv2

    started = time.time()
    if args.manifest_root is not None:
        roots = [args.manifest_root]
    else:
        roots = sorted({p.parent for p in Path('/kaggle/input').rglob('metadata.parquet')}
                       | {p.parent for p in Path('/kaggle/input').rglob('metadata.csv')})
    if not roots:
        raise FileNotFoundError('No dataset manifests mounted on Kaggle')
    datasets = []
    for root in roots:
        try:
            datasets.append(audit_dataset(root, image_root=args.images_root))
        except (OSError, ValueError) as error:
            datasets.append({'root': str(root), 'readiness': 'BLOCKED', 'integrity_passed': False,
                             'training_ready': False, 'error': f'{type(error).__name__}: incomplete audit'})
    summary = {'schema_version': 1, 'training_executed': False, 'dataset_created': False,
               'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'python': platform.python_version(),
               'versions': {**{n: importlib.metadata.version(n) for n in ['numpy', 'pandas', 'Pillow']},
                            'opencv': cv2.__version__},
               'datasets': datasets}
    summary['elapsed_seconds'] = time.time() - started
    output = Path('/kaggle/working/dataset_audit.json')
    output.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'audit_output': str(output), 'datasets': len(roots), 'training_executed': False,
                      'readiness': [r['readiness'] for r in summary['datasets']]}), flush=True)
