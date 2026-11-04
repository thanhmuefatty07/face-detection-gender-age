"""
2-Tier ByteTrack with Vectorized 64-D Re-ID Appearance Recovery & Multi-Target Anti-Swap Gating
===============================================================================================
Production-grade multi-object tracker tailored for high-density face biometrics:
1. Two-Tier Association:
   - Tier 1: High-confidence detections (conf >= 0.6) matched with active tracks via fused
     IoU + 64-D Re-ID appearance cost (preventing ID swaps when multiple targets cross paths).
   - Tier 2: Low-confidence detections (0.2 <= conf < 0.6) matched with unmatched tracks to
     sustain tracking through motion blur and transient occlusion.
2. 64-D Re-ID Cosine Matching:
   - Recovers lost tracks after full occlusion or subject turning back around.
   - Preserves unique persistent track_id without ID switches.
3. Linear Kalman Box Filtering (Joseph Stabilized Form):
   - Predicts and smooths spatial bounding box motion states.
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import IntEnum
from typing import List, Tuple, Optional, Dict
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
        self.P = np.eye(8, dtype=np.float32) * 2.0
        self.P[4:, 4:] = np.eye(4, dtype=np.float32) * 10.0

        self.F = np.eye(8, dtype=np.float32)
        for i in range(4):
            self.F[i, i + 4] = 1.0

        self.H = np.zeros((4, 8), dtype=np.float32)
        self.H[:4, :4] = np.eye(4, dtype=np.float32)

        self.Q = np.eye(8, dtype=np.float32) * 0.5
        self.Q[4:, 4:] = np.eye(4, dtype=np.float32) * 0.05

        self.R = np.eye(4, dtype=np.float32) * 3.0

    def predict(self) -> Tuple[float, float, float, float]:
        self.x[4:] *= 0.70
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return float(self.x[0]), float(self.x[1]), float(self.x[2]), float(self.x[3])

    def update(self, bbox: Tuple[float, float, float, float]):
        z = np.array(bbox, dtype=np.float32)
        y = z - self.H @ self.x
        if float(np.max(np.abs(y))) < 1.5:
            y = y * 0.25
            self.x[4:] *= 0.25
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


def compute_iou_matrix(
    boxes_a: List[Tuple[float, float, float, float]],
    boxes_b: List[Tuple[float, float, float, float]],
) -> np.ndarray:
    """Computes vectorized pairwise 1 - IoU distance matrix between two lists of bounding boxes."""
    N = len(boxes_a)
    M = len(boxes_b)
    if N == 0 or M == 0:
        return np.empty((N, M), dtype=np.float32)

    a = np.asarray(boxes_a, dtype=np.float32).reshape(N, 4)
    b = np.asarray(boxes_b, dtype=np.float32).reshape(M, 4)

    area_a = np.maximum(0.0, a[:, 2] - a[:, 0]) * np.maximum(0.0, a[:, 3] - a[:, 1])  # (N,)
    area_b = np.maximum(0.0, b[:, 2] - b[:, 0]) * np.maximum(0.0, b[:, 3] - b[:, 1])  # (M,)

    xx1 = np.maximum(a[:, None, 0], b[None, :, 0])
    yy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    xx2 = np.minimum(a[:, None, 2], b[None, :, 2])
    yy2 = np.minimum(a[:, None, 3], b[None, :, 3])

    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    inter = w * h
    union = area_a[:, None] + area_b[None, :] - inter
    iou = np.where(union > 0.0, inter / (union + 1e-8), 0.0)
    return (1.0 - iou).astype(np.float32)


def compute_cosine_distance_matrix(
    feats_a: List[Optional[np.ndarray]],
    feats_b: List[Optional[np.ndarray]],
) -> np.ndarray:
    """Computes vectorized pairwise cosine distance (1 - cos) between two lists of Re-ID vectors."""
    N = len(feats_a)
    M = len(feats_b)
    if N == 0 or M == 0:
        return np.empty((N, M), dtype=np.float32)

    mat = np.ones((N, M), dtype=np.float32)
    valid_a_idx: List[int] = []
    valid_a_vecs: List[np.ndarray] = []
    dim: Optional[int] = None

    for i, fa in enumerate(feats_a):
        if fa is None:
            continue
        vec = np.asarray(fa, dtype=np.float32).ravel()
        if vec.size == 0:
            continue
        if dim is None:
            dim = vec.size
        if vec.size == dim:
            norm = float(np.linalg.norm(vec))
            valid_a_idx.append(i)
            valid_a_vecs.append(vec / norm if norm > 1e-8 else vec)

    if not valid_a_idx or dim is None:
        return mat

    valid_b_idx: List[int] = []
    valid_b_vecs: List[np.ndarray] = []
    for j, fb in enumerate(feats_b):
        if fb is None:
            continue
        vec = np.asarray(fb, dtype=np.float32).ravel()
        if vec.size == dim:
            norm = float(np.linalg.norm(vec))
            valid_b_idx.append(j)
            valid_b_vecs.append(vec / norm if norm > 1e-8 else vec)

    if not valid_b_idx:
        return mat

    A = np.stack(valid_a_vecs, axis=0)  # (Na, D)
    B = np.stack(valid_b_vecs, axis=0)  # (Mb, D)
    cos_sim = A @ B.T
    dist_sub = np.clip(1.0 - cos_sim, 0.0, 2.0).astype(np.float32)
    mat[np.ix_(valid_a_idx, valid_b_idx)] = dist_sub
    return mat


class ByteTracker:
    """
    2-Tier ByteTrack multi-face tracker with 64-D Re-ID appearance matching.
    """

    def __init__(
        self,
        track_thresh: float = 0.5,
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
            # Fuse Re-ID cosine distance into Tier-1 assignment matrix when both sides have embeddings
            # to disambiguate crossing trajectories or partial occlusions of multiple targets
            assign_cost_1 = cost_m1.copy()
            any_reid_t = any(t.reid_feat is not None for t in self.tracked_tracks)
            any_reid_d = any(d.reid_feat is not None for d in dets_high)
            reid_m1 = None
            if any_reid_t and any_reid_d and (len(self.tracked_tracks) > 1 or len(dets_high) > 1):
                reid_m1 = compute_cosine_distance_matrix(
                    [t.reid_feat for t in self.tracked_tracks],
                    [d.reid_feat for d in dets_high],
                )
                for r_i, trk in enumerate(self.tracked_tracks):
                    if trk.reid_feat is None:
                        continue
                    for c_i, dt in enumerate(dets_high):
                        if dt.reid_feat is None:
                            continue
                        if cost_m1[r_i, c_i] <= 0.8:
                            assign_cost_1[r_i, c_i] = 0.75 * cost_m1[r_i, c_i] + 0.25 * reid_m1[r_i, c_i]

            row_ind, col_ind = linear_sum_assignment(assign_cost_1)
            for r, c in zip(row_ind, col_ind):
                # IoU threshold = 0.2 -> distance <= 0.8
                if cost_m1[r, c] <= 0.8:
                    # Reject severe appearance conflict when spatial overlap is ambiguous (e.g. occlusion crossing)
                    if (
                        reid_m1 is not None
                        and self.tracked_tracks[r].reid_feat is not None
                        and dets_high[c].reid_feat is not None
                        and cost_m1[r, c] > 0.30
                        and reid_m1[r, c] > 0.65
                    ):
                        continue
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
                # Build unified assignment cost matrix: use Re-ID cost when both track and detection
                # have Re-ID embeddings, otherwise fall back to spatial IoU distance so lost tracks
                # without Re-ID features are still optimally matched by Hungarian assignment.
                lost_assign_cost = np.ones_like(cost_reid, dtype=np.float32)
                for r_i, lt in enumerate(self.lost_tracks):
                    for c_i, ud in enumerate(unmatched_dets_high):
                        if lt.reid_feat is not None and ud.reid_feat is not None:
                            lost_assign_cost[r_i, c_i] = cost_reid[r_i, c_i]
                        elif cost_iou.size > 0:
                            lost_assign_cost[r_i, c_i] = cost_iou[r_i, c_i]

                row_ind, col_ind = linear_sum_assignment(lost_assign_cost)
                for r, c in zip(row_ind, col_ind):
                    has_reid = (
                        self.lost_tracks[r].reid_feat is not None
                        and unmatched_dets_high[c].reid_feat is not None
                    )
                    is_iou_match = (cost_iou.size > 0 and cost_iou[r, c] <= 0.7)

                    if has_reid:
                        is_reid_match = (cost_reid[r, c] <= self.reid_thresh)
                        is_gated_match = is_reid_match
                    else:
                        is_gated_match = is_iou_match

                    if is_gated_match:
                        rec_t = self.lost_tracks[r]
                        rec_t.update(unmatched_dets_high[c])
                        recovered_stracks.append(rec_t)
                        matched_lost.add(r)
                        matched_dets_reid.add(c)

            self.lost_tracks = [
                t for i, t in enumerate(self.lost_tracks) if i not in matched_lost
            ]
            unmatched_dets_high_remaining = [
                d for j, d in enumerate(unmatched_dets_high) if j not in matched_dets_reid
            ]
        else:
            unmatched_dets_high_remaining = unmatched_dets_high

        # 6. Initialize new tracks for remaining high-confidence detections up to max_targets
        # Sort by prominence (score * area^0.15) so primary subjects take priority when near capacity
        if len(unmatched_dets_high_remaining) > 1:
            unmatched_dets_high_remaining = sorted(
                unmatched_dets_high_remaining,
                key=lambda d: float(d.score) * (max(1.0, (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1])) ** 0.15),
                reverse=True,
            )

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
