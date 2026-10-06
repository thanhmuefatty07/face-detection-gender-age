"""
Biometric Multi-Target Tracking & Target Locking
================================================
ByteTrack association and epistemic target locking modules.
"""

from .target_locker import (
    EpistemicAgeKalmanFilter,
    BoundedLeakyGenderAccumulator,
    TargetLocker,
)
from .byte_tracker import (
    ByteTracker,
    Detection,
    STrack,
    TrackState,
)

__all__ = [
    "EpistemicAgeKalmanFilter",
    "BoundedLeakyGenderAccumulator",
    "TargetLocker",
    "ByteTracker",
    "Detection",
    "STrack",
    "TrackState",
]
