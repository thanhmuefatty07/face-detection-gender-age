"""
2-Tier ByteTrack with 64-D Re-ID Appearance Recovery
====================================================
Production-grade multi-object tracker tailored for face biometrics:
1. Two-Tier Association:
   - Tier 1: High-confidence detections (conf >= 0.6) matched with active tracks via IoU.
   - Tier 2: Low-confidence detections (0.2 <= conf < 0.6) matched with unmatched tracks to
     sustain tracking through motion blur and transient occlusion.
2. 64-D Re-ID Cosine Matching:
   - Recovers lost tracks after full occlusion or subject turning back around.
   - Preserves unique persistent track_id without ID switches.
3. Linear Kalman Box Filtering:
   - Predicts and smooths spatial bounding box motion states.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from scipy.optimize import linear_sum_assignment


class TrackState(IntEnum):
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


@dataclass
class Detection:
    """Bounding box detection with score and optional 64-D Re-ID feature."""
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    score: float
    reid_feat: Optional[np.ndarray] = None   # (64,) normalized float32
    landmarks: Optional[np.ndarray] = None
    det_id: Optional[int] = None


class KalmanBoxTracker:
    """
    Constant velocity bounding box Kalman filter:
    State: [x1, y1, x2, y2, vx1, vy1, vx2, vy2]
    """

    def __init__(self, bbox: Tuple[float, float, float, float]):
        self.x = np.array([bbox[0], bbox[1], bbox[2], bbox[3], 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        # Covariance
        self.P = np.eye(8, dtype=np.float32) * 10.0
        self.P[4:, 4:] *= 100.0

        # Motion matrix F
        self.F = np.eye(8, dtype=np.float32)
        for i in range(4):
            self.F[i, i + 4] = 1.0

        # Measurement matrix H
        self.H = np.zeros((4, 8), dtype=np.float32)
        self.H[:4, :4] = np.eye(4, dtype=np.float32)

        # Process noise Q
        self.Q = np.eye(8, dtype=np.float32) * 1.0
        self.Q[4:, 4:] *= 0.1

        # Measurement noise R
        self.R = np.eye(4, dtype=np.float32) * 5.0

    def predict(self) -> Tuple[float, float, float, float]:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return float(self.x[0]), float(self.x[1]), float(self.x[2]), float(self.x[3])

    def update(self, bbox: Tuple[float, float, float, float]):
        z = np.array(bbox, dtype=np.float32)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        try:
            K = self.P @ self.H.T @ np.linalg.pinv(S)
        except Exception:
            K = np.zeros((8, 4), dtype=np.float32)
        self.x = self.x + K @ y
        I_KH = np.eye(8, dtype=np.float32) - K @ self.H
        # Joseph stabilized form: preserves positive-definiteness and symmetry
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
        self.P = 0.5 * (self.P + self.P.T)

    @property
    def current_bbox(self) -> Tuple[float, float, float, float]:
        x1, y1, x2, y2 = self.x[:4]
        # Ensure x2 > x1, y2 > y1
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        return float(x1), float(y1), float(x2), float(y2)


class STrack:
    """Individual tracked object representation."""

    _count = 0

    def __init__(
        self,
        bbox: Tuple[float, float, float, float],
        score: float,
        reid_feat: Optional[np.ndarray] = None,
        det_id: Optional[int] = None,
    ):
        STrack._count += 1
        self.track_id: int = STrack._count
        self.bbox: Tuple[float, float, float, float] = bbox
        self.score: float = float(score)
        self.reid_feat: Optional[np.ndarray] = None
        if reid_feat is not None:
            feat = np.asarray(reid_feat, dtype=np.float32).flatten()
            norm = np.linalg.norm(feat)
            self.reid_feat = feat / (norm + 1e-8) if norm > 0 else feat

        self.kalman = KalmanBoxTracker(bbox)
        self.state = TrackState.Tracked
        self.is_activated = True
        self.tracklet_len = 0
        self.time_since_update = 0
        self.last_detection: Optional[Detection] = None
        self.last_det_id: Optional[int] = det_id

    @classmethod
    def reset_counter(cls):
        cls._count = 0

    def predict(self):
        pred_box = self.kalman.predict()
        self.bbox = pred_box
        self.time_since_update += 1
        self.last_detection = None
        self.last_det_id = None

    def update(self, det: Detection):
        self.kalman.update(det.bbox)
        self.bbox = self.kalman.current_bbox
        self.score = float(det.score)
        self.state = TrackState.Tracked
        self.is_activated = True
        self.tracklet_len += 1
        self.time_since_update = 0
        self.last_detection = det
        self.last_det_id = det.det_id

        # EMA update appearance embedding
        if det.reid_feat is not None:
            new_feat = np.asarray(det.reid_feat, dtype=np.float32).flatten()
            norm = np.linalg.norm(new_feat)
            if norm > 0:
                new_feat /= norm
                if self.reid_feat is None:
                    self.reid_feat = new_feat
                else:
                    smoothed = 0.9 * self.reid_feat + 0.1 * new_feat
                    s_norm = np.linalg.norm(smoothed)
                    self.reid_feat = smoothed / (s_norm + 1e-8) if s_norm > 0 else smoothed

    def mark_lost(self):
        self.state = TrackState.Lost
        self.last_detection = None
        self.last_det_id = None

    def mark_removed(self):
        self.state = TrackState.Removed


def compute_iou_matrix(boxes_a: List[Tuple[float, float, float, float]], boxes_b: List[Tuple[float, float, float, float]]) -> np.ndarray:
    """Computes pairwise 1 - IoU distance matrix between two lists of bounding boxes."""
    N = len(boxes_a)
    M = len(boxes_b)
    if N == 0 or M == 0:
        return np.empty((N, M), dtype=np.float32)

    cost_matrix = np.ones((N, M), dtype=np.float32)
    for i, a in enumerate(boxes_a):
        area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
        for j, b in enumerate(boxes_b):
            area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
            xx1 = max(a[0], b[0])
            yy1 = max(a[1], b[1])
            xx2 = min(a[2], b[2])
            yy2 = min(a[3], b[3])
            w = max(0.0, xx2 - xx1)
            h = max(0.0, yy2 - yy1)
            inter = w * h
            union = area_a + area_b - inter
            iou = inter / (union + 1e-8) if union > 0 else 0.0
            cost_matrix[i, j] = 1.0 - float(iou)
    return cost_matrix


def compute_cosine_distance_matrix(feats_a: List[np.ndarray], feats_b: List[np.ndarray]) -> np.ndarray:
    """Computes pairwise cosine distance (1 - cos) between two lists of Re-ID vectors."""
    N = len(feats_a)
    M = len(feats_b)
    if N == 0 or M == 0:
        return np.empty((N, M), dtype=np.float32)

    mat = np.ones((N, M), dtype=np.float32)
    for i, fa in enumerate(feats_a):
        if fa is None:
            continue
        for j, fb in enumerate(feats_b):
            if fb is None:
                continue
            cos_sim = float(np.dot(fa, fb))
            mat[i, j] = max(0.0, 1.0 - cos_sim)
    return mat


class ByteTracker:
    """
    2-Tier ByteTrack multi-face tracker with 64-D Re-ID appearance matching.
    """

    def __init__(
        self,
        track_thresh: float = 0.6,
        low_thresh: float = 0.2,
        reid_thresh: float = 0.35,
        max_time_lost: int = 30,
        max_targets: int = 5,
    ):
        self.track_thresh = track_thresh
        self.low_thresh = low_thresh
        self.reid_thresh = reid_thresh
        self.max_time_lost = max_time_lost
        self.max_targets = max_targets

        self.tracked_tracks: List[STrack] = []
        self.lost_tracks: List[STrack] = []
        self.frame_id = 0

    def update(self, detections: List[Detection]) -> List[STrack]:
        """
        Updates tracker state for the current frame.
        
        Args:
            detections: list of Detection objects
        Returns:
            List of active STrack objects currently tracked
        """
        self.frame_id += 1

        # 1. Split detections into high and low confidence pools
        dets_high: List[Detection] = []
        dets_low: List[Detection] = []
        for d in detections:
            if d.score >= self.track_thresh:
                dets_high.append(d)
            elif d.score >= self.low_thresh:
                dets_low.append(d)

        # 2. Predict locations of all current tracks
        for t in self.tracked_tracks:
            t.predict()
        for t in self.lost_tracks:
            t.predict()

        # 3. Tier 1: Associate active tracked tracks with high-confidence detections
        track_boxes = [t.bbox for t in self.tracked_tracks]
        det_high_boxes = [d.bbox for d in dets_high]
        cost_m1 = compute_iou_matrix(track_boxes, det_high_boxes)

        matched_tracks_1 = set()
        matched_dets_high_1 = set()

        if cost_m1.size > 0:
            row_ind, col_ind = linear_sum_assignment(cost_m1)
            for r, c in zip(row_ind, col_ind):
                # IoU threshold = 0.2 -> distance <= 0.8
                if cost_m1[r, c] <= 0.8:
                    self.tracked_tracks[r].update(dets_high[c])
                    matched_tracks_1.add(r)
                    matched_dets_high_1.add(c)

        tracked_stracks = [self.tracked_tracks[i] for i in matched_tracks_1]
        unmatched_tracks_1 = [
            t for i, t in enumerate(self.tracked_tracks) if i not in matched_tracks_1
        ]
        unmatched_dets_high = [
            d for j, d in enumerate(dets_high) if j not in matched_dets_high_1
        ]

        # 4. Tier 2: Associate unmatched active tracks with low-confidence detections
        det_low_boxes = [d.bbox for d in dets_low]
        u_track_boxes = [t.bbox for t in unmatched_tracks_1]
        cost_m2 = compute_iou_matrix(u_track_boxes, det_low_boxes)

        matched_tracks_2 = set()
        matched_dets_low_2 = set()

        if cost_m2.size > 0:
            row_ind, col_ind = linear_sum_assignment(cost_m2)
            for r, c in zip(row_ind, col_ind):
                if cost_m2[r, c] <= 0.7:  # IoU >= 0.3
                    unmatched_tracks_1[r].update(dets_low[c])
                    matched_tracks_2.add(r)
                    matched_dets_low_2.add(c)

        for i in matched_tracks_2:
            tracked_stracks.append(unmatched_tracks_1[i])

        # Tracks still unmatched after Tier 2 become lost
        for i, t in enumerate(unmatched_tracks_1):
            if i not in matched_tracks_2:
                t.mark_lost()
                if t not in self.lost_tracks:
                    self.lost_tracks.append(t)

        # 5. Appearance Re-ID & Spatial Matching: Associate lost tracks with unmatched high-confidence detections
        recovered_stracks = []
        unmatched_dets_high_remaining: List[Detection] = []
        if len(self.lost_tracks) > 0 and len(unmatched_dets_high) > 0:
            lost_feats = [t.reid_feat for t in self.lost_tracks]
            det_feats = [d.reid_feat for d in unmatched_dets_high]
            cost_reid = compute_cosine_distance_matrix(lost_feats, det_feats)

            lost_boxes = [t.bbox for t in self.lost_tracks]
            det_h_boxes = [d.bbox for d in unmatched_dets_high]
            cost_iou = compute_iou_matrix(lost_boxes, det_h_boxes)

            matched_lost = set()
            matched_dets_reid = set()

            if cost_reid.size > 0:
                row_ind, col_ind = linear_sum_assignment(cost_reid)
                for r, c in zip(row_ind, col_ind):
                    has_reid = (
                        self.lost_tracks[r].reid_feat is not None
                        and unmatched_dets_high[c].reid_feat is not None
                    )
                    is_iou_match = (cost_iou.size > 0 and cost_iou[r, c] <= 0.7)

                    if has_reid:
                        # Strict Gated Matching: appearance cosine distance must be <= threshold (0.35).
                        # Overlapping bounding boxes of distinct individuals (cost_reid > 0.35) are strictly rejected.
                        is_reid_match = (cost_reid[r, c] <= self.reid_thresh)
                        is_gated_match = is_reid_match
                    else:
                        # Fallback to spatial IoU proximity only when no Re-ID features are registered
                        is_gated_match = is_iou_match

                    if is_gated_match:
                        rec_t = self.lost_tracks[r]
                        rec_t.update(unmatched_dets_high[c])
                        recovered_stracks.append(rec_t)
                        matched_lost.add(r)
                        matched_dets_reid.add(c)

            # Keep remaining unmatched lost tracks
            self.lost_tracks = [
                t for i, t in enumerate(self.lost_tracks) if i not in matched_lost
            ]
            unmatched_dets_high_remaining = [
                d for j, d in enumerate(unmatched_dets_high) if j not in matched_dets_reid
            ]
        else:
            unmatched_dets_high_remaining = unmatched_dets_high

        # 6. Initialize new tracks for remaining high-confidence detections up to max_targets
        new_stracks = []
        active_ids = {t.track_id for t in tracked_stracks} | {t.track_id for t in recovered_stracks}
        for d in unmatched_dets_high_remaining:
            if len(active_ids) < self.max_targets:
                new_track = STrack(bbox=d.bbox, score=d.score, reid_feat=d.reid_feat, det_id=d.det_id)
                new_track.last_detection = d
                new_stracks.append(new_track)
                active_ids.add(new_track.track_id)

        # 7. Deduplicate and finalize active tracks (guaranteed unique by track_id)
        all_active: Dict[int, STrack] = {}
        for t in tracked_stracks + recovered_stracks + new_stracks:
            if t.state == TrackState.Tracked:
                all_active[t.track_id] = t

        self.tracked_tracks = list(all_active.values())

        # Prune lost tracks
        cleaned_lost = []
        for t in self.lost_tracks:
            if t.track_id in all_active:
                continue
            if t.time_since_update > self.max_time_lost:
                t.mark_removed()
            else:
                cleaned_lost.append(t)
        self.lost_tracks = cleaned_lost

        return list(self.tracked_tracks)
