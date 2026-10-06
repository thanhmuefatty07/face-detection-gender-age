"""
Unit Tests for 2-Tier ByteTrack with 64-D Re-ID Appearance Recovery
===================================================================
Tests:
1. Continuous tracking across video sequence maintaining unique track IDs.
2. Multi-target handling with max_targets capacity (e.g. 5 targets).
3. 2-tier association preserving tracks across low-confidence frames (0.2 <= conf < 0.6).
4. Re-ID cosine matching recovering lost tracks after complete occlusion / turns.
"""

from __future__ import annotations
import numpy as np
import pytest

from src.biometrics.tracking.byte_tracker import (
    ByteTracker,
    Detection,
    STrack,
)


class TestByteTracker:
    def test_single_target_persistence(self):
        tracker = ByteTracker(track_thresh=0.6, low_thresh=0.2, max_time_lost=30)

        # Target moves smoothly to the right across 10 frames
        track_ids = []
        for f in range(10):
            bbox = (50.0 + f * 5.0, 50.0, 100.0 + f * 5.0, 100.0)
            det = Detection(bbox=bbox, score=0.9)
            online_tracks = tracker.update([det])
            assert len(online_tracks) == 1
            track_ids.append(online_tracks[0].track_id)

        # Must maintain the exact same track ID across all 10 frames
        assert len(set(track_ids)) == 1

    def test_multi_target_capacity_up_to_5(self):
        tracker = ByteTracker(max_targets=5)

        # 5 distinct subjects spread across the camera frame
        base_coords = [
            (20.0, 50.0, 70.0, 120.0),
            (100.0, 50.0, 150.0, 120.0),
            (180.0, 50.0, 230.0, 120.0),
            (260.0, 50.0, 310.0, 120.0),
            (340.0, 50.0, 390.0, 120.0),
        ]

        # Frame 1: all 5 appear
        dets = [Detection(bbox=b, score=0.85) for b in base_coords]
        tracks = tracker.update(dets)
        assert len(tracks) == 5
        assigned_ids = {t.track_id for t in tracks}
        assert len(assigned_ids) == 5

        # Frame 2: slight movement
        dets_f2 = [
            Detection(bbox=(b[0] + 1.0, b[1], b[2] + 1.0, b[3]), score=0.85)
            for b in base_coords
        ]
        tracks_f2 = tracker.update(dets_f2)
        assert len(tracks_f2) == 5
        assert {t.track_id for t in tracks_f2} == assigned_ids

    def test_tier2_low_confidence_retention(self):
        """Detections with conf between 0.2 and 0.6 must sustain existing tracks (Tier 2)."""
        tracker = ByteTracker(track_thresh=0.6, low_thresh=0.2)

        # Frame 1: high confidence detection initializes track
        det1 = Detection(bbox=(100.0, 100.0, 150.0, 150.0), score=0.9)
        tracks1 = tracker.update([det1])
        assert len(tracks1) == 1
        tid = tracks1[0].track_id

        # Frame 2: heavy motion blur causes confidence to drop to 0.35 (Tier 2)
        det2 = Detection(bbox=(102.0, 101.0, 152.0, 151.0), score=0.35)
        tracks2 = tracker.update([det2])

        # Track should be preserved rather than dropped!
        assert len(tracks2) == 1
        assert tracks2[0].track_id == tid

    def test_reid_recovery_after_full_occlusion(self):
        """Track lost for 5 frames is recovered via 64-D Re-ID appearance embedding."""
        tracker = ByteTracker(track_thresh=0.6, low_thresh=0.2, reid_thresh=0.35, max_time_lost=20)

        rng = np.random.RandomState(42)
        # 64-D normalized feature for Person A
        emb_a = rng.randn(64).astype(np.float32)
        emb_a /= np.linalg.norm(emb_a)

        # 1. Person A appears for 3 frames
        for _ in range(3):
            det = Detection(bbox=(100.0, 100.0, 160.0, 160.0), score=0.9, reid_feat=emb_a)
            t = tracker.update([det])
            person_a_id = t[0].track_id

        # 2. Person A is completely occluded for 5 frames (no detections)
        for _ in range(5):
            t = tracker.update([])
            assert len(t) == 0  # No active tracks on screen

        # 3. Person A reappears slightly shifted, with matching Re-ID embedding
        # Due to occlusion gap, IoU with Kalman might be low, but Re-ID matches!
        emb_reappear = emb_a + rng.normal(0, 0.05, 64).astype(np.float32)
        emb_reappear /= np.linalg.norm(emb_reappear)

        det_reappear = Detection(
            bbox=(125.0, 105.0, 185.0, 165.0),
            score=0.92,
            reid_feat=emb_reappear
        )
        tracks_reappear = tracker.update([det_reappear])

        assert len(tracks_reappear) == 1
        # Crucial: ID must be recovered Person A ID, NOT a new ID!
        assert tracks_reappear[0].track_id == person_a_id

    def test_no_duplicate_tracks_on_same_frame_reid_recovery(self):
        """Validates that a track lost and recovered via Re-ID in the exact same frame NEVER duplicates."""
        tracker = ByteTracker(track_thresh=0.6, low_thresh=0.2, reid_thresh=0.4)
        rng = np.random.RandomState(42)
        emb = rng.randn(64).astype(np.float32)
        emb /= np.linalg.norm(emb)

        # Frame 1: Person appears at (10, 10, 50, 50)
        d1 = Detection(bbox=(10.0, 10.0, 50.0, 50.0), score=0.9, reid_feat=emb)
        tracks1 = tracker.update([d1])
        assert len(tracks1) == 1
        orig_id = tracks1[0].track_id

        # Frame 2: Person moves far away to (200, 200, 240, 240) so IoU is 0, but Re-ID matches
        d2 = Detection(bbox=(200.0, 200.0, 240.0, 240.0), score=0.9, reid_feat=emb)
        tracks2 = tracker.update([d2])
        assert len(tracks2) == 1, f"Expected exactly 1 track, got {len(tracks2)} (duplicate bug!)"
        assert tracks2[0].track_id == orig_id

    def test_detection_metadata_association(self):
        """Verifies that det_id is stored on active track and cleared upon prediction."""
        tracker = ByteTracker()
        det = Detection(bbox=(10.0, 10.0, 50.0, 50.0), score=0.9, det_id=42)
        tracks = tracker.update([det])
        assert len(tracks) == 1
        assert tracks[0].last_det_id == 42
        assert tracks[0].last_detection is not None

        # Next frame with no detection: predict clears last_detection
        tracks_empty = tracker.update([])
        # Track is now lost
        assert len(tracks_empty) == 0
