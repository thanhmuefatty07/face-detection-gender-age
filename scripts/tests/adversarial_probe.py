"""
Adversarial Probe Script for Milestone 3 Extreme Filtering and Geometric Alignment.
Probes:
1. Sharpness (Laplacian and Tenengrad) exact numerical boundaries and edge cases.
2. solvePnP head pose Euler angles at exact boundaries (+/- 35.0/35.1 yaw, +/- 25.0/25.1 pitch, +/- 20.0/20.1 roll)
   and mathematical consistency of Euler angle extraction.
3. Landmark similarity transform stability under extreme affine transforms, collinearity, degeneracy, occlusions.
4. Photometric and sensor clipping gates (L* boundaries, clip fraction 7.9% vs 8.1%).
"""

import math
import numpy as np
import cv2

import sys
import os
sys.path.insert(0, os.path.abspath("."))
from scripts.kaggle_pipeline.kaggle_face_engine import (
    ExtremeQualityAuditor,
    FaceGeometryAligner,
    FacialLandmarkDetector
)

def probe_sharpness():
    print("=== PROBE 1: SHARPNESS BOUNDARIES ===")
    auditor = ExtremeQualityAuditor()
    
    # Check 1.1: Image synthesis with controlled Laplacian variance
    # A flat image has lap_var = 0.0
    img_flat = np.full((160, 160, 3), 128, dtype=np.uint8)
    res_flat = auditor.audit_face(img_flat)
    print(f"Flat image: passed={res_flat['passed']}, lap_var={res_flat['laplacian_var']}, reasons={res_flat['reasons']}")
    
    # Check 1.2: Boundary exact values
    print("Checking exact Laplacian boundary:")
    for v in [99.9, 100.0, 100.1]:
        # Auditor threshold test
        rejected = v < auditor.min_laplacian_var
        print(f"  Laplacian var {v}: rejected={rejected}")
        
    print("Checking exact Tenengrad boundary:")
    for v in [249.9, 250.0, 250.1]:
        rejected = v < auditor.min_tenengrad
        print(f"  Tenengrad {v}: rejected={rejected}")

    # Check 1.3: Extreme image dimensions (e.g. 1x1, 2x2, non-square)
    for dims in [(1, 1), (2, 2), (3, 3), (160, 1), (1, 160)]:
        try:
            small_img = np.full((dims[0], dims[1], 3), 128, dtype=np.uint8)
            res = auditor.audit_face(small_img)
            print(f"  Dims {dims}: success, passed={res['passed']}")
        except Exception as e:
            print(f"  Dims {dims}: EXCEPTION {type(e).__name__}: {e}")

def probe_solvepnp_euler():
    print("\n=== PROBE 2: SOLVEPNP EULER ANGLE BOUNDARIES ===")
    auditor = ExtremeQualityAuditor()
    w, h = 160, 160
    focal = 160.0
    cam = np.array([[focal, 0, 80.0], [0, focal, 80.0], [0, 0, 1]], dtype=np.float64)
    dist = np.zeros((4, 1), dtype=np.float64)
    tvec_true = np.array([[0.0], [0.0], [focal]], dtype=np.float64)

    # Test projection of 3D model points with given rotations
    def test_pose(yaw_deg, pitch_deg, roll_deg):
        # Convert Euler to rotation matrix
        # Note convention in auditor:
        # pitch = atan2(R[2,1], R[2,2])
        # yaw = atan2(-R[2,0], sy)
        # roll = atan2(R[1,0], R[0,0])
        # This corresponds to R = Rz(roll) * Ry(yaw) * Rx(pitch)
        rad_y = math.radians(yaw_deg)
        rad_p = math.radians(pitch_deg)
        rad_r = math.radians(roll_deg)
        
        Ry = np.array([
            [math.cos(rad_y), 0, math.sin(rad_y)],
            [0, 1, 0],
            [-math.sin(rad_y), 0, math.cos(rad_y)]
        ], dtype=np.float64)
        
        Rx = np.array([
            [1, 0, 0],
            [0, math.cos(rad_p), -math.sin(rad_p)],
            [0, math.sin(rad_p), math.cos(rad_p)]
        ], dtype=np.float64)
        
        Rz = np.array([
            [math.cos(rad_r), -math.sin(rad_r), 0],
            [math.sin(rad_r), math.cos(rad_r), 0],
            [0, 0, 1]
        ], dtype=np.float64)
        
        R = Rz @ Ry @ Rx
        rvec_true, _ = cv2.Rodrigues(R)
        
        # Project 3D points to 2D
        pts_2d, _ = cv2.projectPoints(auditor.MODEL_3D_POINTS, rvec_true, tvec_true, cam, dist)
        pts_2d = pts_2d.reshape(-1, 2)
        
        est_yaw, est_pitch, est_roll = auditor.estimate_head_pose(pts_2d, w, h)
        return est_yaw, est_pitch, est_roll

    # Probe exact angles
    test_cases = [
        (0.0, 0.0, 0.0),
        (35.0, 0.0, 0.0),
        (35.1, 0.0, 0.0),
        (-35.0, 0.0, 0.0),
        (-35.1, 0.0, 0.0),
        (0.0, 25.0, 0.0),
        (0.0, 25.1, 0.0),
        (0.0, -25.0, 0.0),
        (0.0, -25.1, 0.0),
        (0.0, 0.0, 20.0),
        (0.0, 0.0, 20.1),
        (0.0, 0.0, -20.0),
        (0.0, 0.0, -20.1),
    ]

    for y, p, r in test_cases:
        ey, ep, er = test_pose(y, p, r)
        # Check audit
        img = np.full((160, 160, 3), 128, dtype=np.uint8)
        # Add high frequency noise so blur doesn't fail
        img[::2, ::2] = 200
        img[1::2, 1::2] = 50
        
        # We need the 2d points again
        rad_y, rad_p, rad_r = math.radians(y), math.radians(p), math.radians(r)
        Ry = np.array([[math.cos(rad_y), 0, math.sin(rad_y)], [0, 1, 0], [-math.sin(rad_y), 0, math.cos(rad_y)]], dtype=np.float64)
        Rx = np.array([[1, 0, 0], [0, math.cos(rad_p), -math.sin(rad_p)], [0, math.sin(rad_p), math.cos(rad_p)]], dtype=np.float64)
        Rz = np.array([[math.cos(rad_r), -math.sin(rad_r), 0], [math.sin(rad_r), math.cos(rad_r), 0], [0, 0, 1]], dtype=np.float64)
        R = Rz @ Ry @ Rx
        rvec_true, _ = cv2.Rodrigues(R)
        pts_2d, _ = cv2.projectPoints(auditor.MODEL_3D_POINTS, rvec_true, tvec_true, cam, dist)
        pts_2d = pts_2d.reshape(-1, 2)
        
        audit_res = auditor.audit_face(img, landmarks_5=pts_2d)
        print(f"Target Pose ({y:5.1f}, {p:5.1f}, {r:5.1f}) -> Est ({ey:5.1f}, {ep:5.1f}, {er:5.1f}) | Passed: {audit_res['passed']} | Reasons: {audit_res['reasons']}")

def probe_aligner_stability():
    print("\n=== PROBE 3: ALIGNER STABILITY UNDER EXTREME CONDITIONS ===")
    aligner = FaceGeometryAligner(output_size=160)
    base_lms = aligner.REFERENCE_LANDMARKS_112 * (160.0 / 112.0)
    img = np.full((160, 160, 3), 128, dtype=np.uint8)

    # 3.1 Normal alignment
    out, mat = aligner.align(img, base_lms)
    print(f"Normal align: out shape={out.shape}, mat={mat.shape if mat is not None else None}")

    # 3.2 Swapped / Mirrored landmarks (Left eye and right eye inverted)
    inverted_lms = base_lms.copy()
    inverted_lms[0], inverted_lms[1] = base_lms[1], base_lms[0] # Eye swap
    try:
        out_inv, mat_inv = aligner.align(img, inverted_lms)
        print(f"Inverted eyes align: out shape={out_inv.shape}, mat={mat_inv.shape if mat_inv is not None else None}")
    except Exception as e:
        print(f"Inverted eyes align: EXCEPTION {type(e).__name__}: {e}")

    # 3.3 Collinear landmarks (All 5 points on a horizontal line y=80)
    collinear_lms = np.array([
        [20.0, 80.0],
        [50.0, 80.0],
        [80.0, 80.0],
        [110.0, 80.0],
        [140.0, 80.0]
    ], dtype=np.float32)
    try:
        out_col, mat_col = aligner.align(img, collinear_lms)
        print(f"Collinear lms align: out shape={out_col.shape}, mat={mat_col.shape if mat_col is not None else None}")
    except Exception as e:
        print(f"Collinear lms align: EXCEPTION {type(e).__name__}: {e}")

    # 3.4 Degenerate (all 5 points identical)
    degen_lms = np.full((5, 2), 80.0, dtype=np.float32)
    try:
        out_deg, mat_deg = aligner.align(img, degen_lms)
        print(f"Degenerate lms align: out shape={out_deg.shape}, mat={mat_deg.shape if mat_deg is not None else None}")
    except Exception as e:
        print(f"Degenerate lms align: EXCEPTION {type(e).__name__}: {e}")

    # 3.5 Extreme scale / coordinates (large numbers, negative coordinates)
    extreme_lms = base_lms * 1000.0
    try:
        out_ext, mat_ext = aligner.align(img, extreme_lms)
        print(f"Extreme scale lms align: out shape={out_ext.shape}, mat={mat_ext.shape if mat_ext is not None else None}")
    except Exception as e:
        print(f"Extreme scale lms align: EXCEPTION {type(e).__name__}: {e}")

    # 3.6 NaN / Inf coordinates
    nan_lms = base_lms.copy()
    nan_lms[0, 0] = np.nan
    try:
        out_nan, mat_nan = aligner.align(img, nan_lms)
        print(f"NaN lms align: out shape={out_nan.shape}, mat={mat_nan.shape if mat_nan is not None else None}")
    except Exception as e:
        print(f"NaN lms align: EXCEPTION {type(e).__name__}: {e}")

def probe_photometric_clipping():
    print("\n=== PROBE 4: PHOTOMETRIC AND CLIPPING GATES ===")
    auditor = ExtremeQualityAuditor()

    # Create image with exact clipping fraction
    total_pixels = 160 * 160 # 25600
    
    # 4.1 Clipping at 7.9% vs 8.1%
    for frac in [0.079, 0.080, 0.081]:
        clip_pixels = int(total_pixels * frac)
        # Create base image with L around 128
        # In Lab, L=128 corresponds to gray RGB ~ 118
        img = np.full((160, 160, 3), 118, dtype=np.uint8)
        # Make clip_pixels black (0, 0, 0) which gives L = 0 < 8
        flat_view = img.reshape(-1, 3)
        flat_view[:clip_pixels] = 0
        
        audit = auditor.audit_face(img)
        print(f"Clip frac {frac*100:.1f}% ({clip_pixels} px): passed={audit['passed']}, reasons={audit['reasons']}, shadow_clip={audit['shadow_clip']}")

    # 4.2 Luminance boundary around 35.0 and 220.0
    # In BGR, BGR=(x, x, x) converts to LAB where L is proportional to lightness
    for bgr_val in [25, 30, 35, 40, 210, 220, 230, 240]:
        img = np.full((160, 160, 3), bgr_val, dtype=np.uint8)
        # Add high freq noise so blur does not mask photometry
        img[::2, ::2] = min(255, bgr_val + 10)
        img[1::2, 1::2] = max(0, bgr_val - 10)
        audit = auditor.audit_face(img)
        print(f"BGR {bgr_val}: mean_L={audit['mean_luminance']}, passed={audit['passed']}, reasons={audit['reasons']}")

if __name__ == "__main__":
    probe_sharpness()
    probe_solvepnp_euler()
    probe_aligner_stability()
    probe_photometric_clipping()
