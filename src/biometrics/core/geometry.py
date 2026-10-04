"""
Geometric Alignment & Anthropometric Pose Estimation
=====================================================
Implements:
1. Umeyama (1991) closed-form least-squares similarity transformation to ArcFace canonical coordinates.
2. Anthropometric 3D solvePnP Head Pose Euler Angle estimation with frontal pitch normalization.
"""

from __future__ import annotations
import math
from typing import Tuple, Optional
import cv2
import numpy as np

from .types import FaceLandmarks


# ArcFace 112x112 canonical reference points (InsightFace standard)
ARCFACE_CANONICAL_112 = np.array([
    [38.2946, 51.6963],   # Left eye
    [73.5318, 51.5014],   # Right eye
    [56.0252, 71.7366],   # Nose tip
    [41.5493, 92.3655],   # Left mouth corner
    [70.7299, 92.2041]    # Right mouth corner
], dtype=np.float32)

# ArcFace 160x160 scaled reference points
ARCFACE_CANONICAL_160 = ARCFACE_CANONICAL_112 * (160.0 / 112.0)

# Anthropometric 3D Head Model (in mm) - canonical coordinates
ANTHROPOMETRIC_HEAD_3D = np.array([
    [-25.0, -30.0, -15.0],  # Left eye pupil (above nose)
    [ 25.0, -30.0, -15.0],  # Right eye pupil (above nose)
    [  0.0,   0.0,   0.0],  # Pronasale (nose tip origin)
    [-21.0,  30.0, -15.0],  # Left cheilion / mouth corner (below nose)
    [ 21.0,  30.0, -15.0]   # Right cheilion / mouth corner (below nose)
], dtype=np.float64)


def umeyama_similarity_transform(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """
    Computes optimal 2D similarity transform matrix M [2x3] such that dst ~ M * src.
    Solves min || dst - (c * R * src + t) ||_F using SVD.
    """
    assert src.shape == dst.shape and src.shape[1] == 2, "src and dst must be (N, 2)"
    n, m = src.shape

    src_mean = np.mean(src, axis=0)
    dst_mean = np.mean(dst, axis=0)

    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    var_src = np.mean(np.sum(src_demean**2, axis=1))
    if var_src < 1e-8:
        # Collinear or degenerate points, return identity affine
        M = np.eye(2, 3, dtype=np.float32)
        M[:, 2] = dst_mean - src_mean
        return M

    # Covariance matrix
    sigma = np.dot(dst_demean.T, src_demean) / n

    # SVD
    U, D, Vt = np.linalg.svd(sigma)
    S = np.eye(m)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[m - 1, m - 1] = -1

    R = np.dot(U, np.dot(S, Vt))
    c = (1.0 / var_src) * np.trace(np.dot(np.diag(D), S))
    t = dst_mean - c * np.dot(R, src_mean)

    M = np.zeros((2, 3), dtype=np.float32)
    M[:2, :2] = c * R
    M[:2, 2] = t
    return M


def normalize_angle_to_frontal(angle: float) -> float:
    """
    Normalizes Euler angles to [-90.0, 90.0] for frontal cameras.
    Resolves the 180° solvePnP inversion artifact when the face normal faces the camera.
    """
    while angle > 90.0:
        angle -= 180.0
    while angle < -90.0:
        angle += 180.0
    return angle


def estimate_head_pose_pnp(
    landmarks_5: np.ndarray,
    image_width: int,
    image_height: int
) -> Tuple[float, float, float]:
    """
    Estimates 3D Head Pose Euler angles (Yaw, Pitch, Roll in degrees) using solvePnP.
    Includes frontal angle normalization to prevent 180° camera coordinate inversion.
    """
    if landmarks_5 is None or landmarks_5.shape != (5, 2):
        return (0.0, 0.0, 0.0)

    focal_length = float(image_width)
    center = (image_width / 2.0, image_height / 2.0)
    camera_matrix = np.array([
        [focal_length, 0.0, center[0]],
        [0.0, focal_length, center[1]],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    dist_coeffs = np.zeros((4, 1), dtype=np.float64)
    image_points = landmarks_5.astype(np.float64)

    try:
        rvec = np.zeros((3, 1), dtype=np.float64)
        tvec = np.array([[0.0], [0.0], [float(focal_length)]], dtype=np.float64)
        success, rot_vec, trans_vec = cv2.solvePnP(
            ANTHROPOMETRIC_HEAD_3D,
            image_points,
            camera_matrix,
            dist_coeffs,
            rvec=rvec,
            tvec=tvec,
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not success:
            return (0.0, 0.0, 0.0)

        rot_mat, _ = cv2.Rodrigues(rot_vec)

        # Extract Tait-Bryan Euler angles (ZYX convention)
        sy = math.sqrt(rot_mat[0, 0]**2 + rot_mat[1, 0]**2)
        singular = sy < 1e-6

        if not singular:
            pitch = math.degrees(math.atan2(rot_mat[2, 1], rot_mat[2, 2]))
            yaw = math.degrees(math.atan2(-rot_mat[2, 0], sy))
            roll = math.degrees(math.atan2(rot_mat[1, 0], rot_mat[0, 0]))
        else:
            pitch = math.degrees(math.atan2(-rot_mat[1, 2], rot_mat[1, 1]))
            yaw = math.degrees(math.atan2(-rot_mat[2, 0], sy))
            roll = 0.0

        pitch = normalize_angle_to_frontal(pitch)
        yaw = normalize_angle_to_frontal(yaw)
        roll = normalize_angle_to_frontal(roll)

        return (round(yaw, 2), round(pitch, 2), round(roll, 2))
    except Exception:
        return (0.0, 0.0, 0.0)


class UmeyamaAligner:
    """
    Canonical facial geometry normalizer using Umeyama similarity transform.
    Standardizes faces to ArcFace 160x160 (or custom size) without non-rigid shear.
    """

    def __init__(self, target_size: int = 160):
        self.target_size = target_size
        if target_size == 160:
            self.canonical_points = ARCFACE_CANONICAL_160
        elif target_size == 112:
            self.canonical_points = ARCFACE_CANONICAL_112
        else:
            self.canonical_points = ARCFACE_CANONICAL_112 * (target_size / 112.0)

    def align(self, image: np.ndarray, landmarks: FaceLandmarks | np.ndarray) -> np.ndarray:
        """
        Warps and aligns face image to canonical reference coordinates.
        """
        if isinstance(landmarks, FaceLandmarks):
            src_pts = landmarks.points
        else:
            src_pts = np.asarray(landmarks, dtype=np.float32)

        M = umeyama_similarity_transform(src_pts, self.canonical_points)
        aligned = cv2.warpAffine(
            image,
            M,
            (self.target_size, self.target_size),
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_REFLECT_101
        )
        return aligned
