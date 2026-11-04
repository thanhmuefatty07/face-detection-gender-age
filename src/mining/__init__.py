"""
Face Mining, Multi-Source Dataset Ingestion, Extreme Quality Auditing,
Dual-Layer Deduplication, and 3-5 Year Age Interval Curation Module
===================================================================
"""

from __future__ import annotations

from .connectors import (
    AFADConnector,
    AdienceConnector,
    AgeDBConnector,
    AllAgeFacesConnector,
    AppaRealConnector,
    AxonAgeConnector,
    B3FDConnector,
    BANNED_NON_REAL_SOURCE_PATTERNS,
    BaseFaceConnector,
    CACDConnector,
    CelebAConnector,
    DATASET_SOURCE_REGISTRY,
    DiskQuotaManager,
    FGNetConnector,
    FairFaceConnector,
    IMDBWikiConnector,
    InCloudStreamingHarvester,
    LagendaConnector,
    MegaAgeAsianConnector,
    MorphConnector,
    RawImageSample,
    UTKFaceConnector,
    WikimediaCommonsConnector,
    ensure_master_manifest,
    is_banned_non_real_source,
)
from .curation import CurationSummary, ExtremeDatasetCurator, deterministic_identity_split
from .filtering.deduplicator import DualLayerDeduplicator, compute_64bit_dct_phash, hamming_distance_64bit
from .filtering.quality_auditor import ExtremeQualityAuditor
from .intervals import (
    FINE_AGE_INTERVALS,
    NUM_FINE_AGE_INTERVALS,
    compute_interval_box_ldl,
    compute_interval_epistemic_weight,
    get_fine_interval_for_age,
    integrate_101_to_fine_intervals,
    map_bounds_to_fine_interval,
    parse_age_interval_string,
)

__version__ = "3.0.0"

__all__ = [
    "__version__",
    "FINE_AGE_INTERVALS",
    "NUM_FINE_AGE_INTERVALS",
    "BANNED_NON_REAL_SOURCE_PATTERNS",
    "is_banned_non_real_source",
    "compute_interval_box_ldl",
    "compute_interval_epistemic_weight",
    "get_fine_interval_for_age",
    "integrate_101_to_fine_intervals",
    "map_bounds_to_fine_interval",
    "parse_age_interval_string",
    "DATASET_SOURCE_REGISTRY",
    "RawImageSample",
    "BaseFaceConnector",
    "FairFaceConnector",
    "CelebAConnector",
    "IMDBWikiConnector",
    "AFADConnector",
    "B3FDConnector",
    "AppaRealConnector",
    "UTKFaceConnector",
    "WikimediaCommonsConnector",
    "AgeDBConnector",
    "MorphConnector",
    "FGNetConnector",
    "AdienceConnector",
    "AxonAgeConnector",
    "MegaAgeAsianConnector",
    "LagendaConnector",
    "CACDConnector",
    "AllAgeFacesConnector",
    "InCloudStreamingHarvester",
    "DiskQuotaManager",
    "ensure_master_manifest",
    "ExtremeQualityAuditor",
    "DualLayerDeduplicator",
    "compute_64bit_dct_phash",
    "hamming_distance_64bit",
    "ExtremeDatasetCurator",
    "CurationSummary",
    "deterministic_identity_split",
]

