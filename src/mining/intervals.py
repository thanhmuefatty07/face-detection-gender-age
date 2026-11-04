"""
Data Mining & Curation Age Interval Interface
=============================================
Exposes the 23 fine-grained 3-5 year age intervals and Box-Convolved LDL utilities
to all dataset connectors, manifest auditors, and curation pipelines.
"""

from __future__ import annotations

from ..biometrics.estimation.intervals import (
    ADIENCE_TYPO_MAP,
    FINE_AGE_INTERVALS,
    FINE_INTERVAL_CENTERS,
    FINE_INTERVAL_NAME_TO_IDX,
    NUM_FINE_AGE_INTERVALS,
    AgeIntervalSpec,
    compute_interval_box_ldl,
    compute_interval_epistemic_weight,
    get_fine_interval_by_index,
    get_fine_interval_for_age,
    get_overlapping_fine_intervals,
    integrate_101_to_fine_intervals,
    map_bounds_to_fine_interval,
    parse_age_interval_string,
)

__all__ = [
    "ADIENCE_TYPO_MAP",
    "FINE_AGE_INTERVALS",
    "FINE_INTERVAL_CENTERS",
    "FINE_INTERVAL_NAME_TO_IDX",
    "NUM_FINE_AGE_INTERVALS",
    "AgeIntervalSpec",
    "compute_interval_box_ldl",
    "compute_interval_epistemic_weight",
    "get_fine_interval_by_index",
    "get_fine_interval_for_age",
    "get_overlapping_fine_intervals",
    "integrate_101_to_fine_intervals",
    "map_bounds_to_fine_interval",
    "parse_age_interval_string",
]
