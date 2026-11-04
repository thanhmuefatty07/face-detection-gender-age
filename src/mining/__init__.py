"""
Face Quality Auditing, Perceptual Deduplication, and Age Interval Utilities
===========================================================================
"""

from __future__ import annotations

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
    "compute_interval_box_ldl",
    "compute_interval_epistemic_weight",
    "get_fine_interval_for_age",
    "integrate_101_to_fine_intervals",
    "map_bounds_to_fine_interval",
    "parse_age_interval_string",
    "ExtremeQualityAuditor",
    "DualLayerDeduplicator",
    "compute_64bit_dct_phash",
    "hamming_distance_64bit",
]
