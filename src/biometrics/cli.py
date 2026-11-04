"""
Command Line Interface for SOTA Biometric Inference
===================================================
Usage:
    python -m src.biometrics.cli --image path/to/image.jpg --engine yunet
    python -m src.biometrics.cli --webcam auto --engine yunet
    python -m src.biometrics.cli --list-cameras
"""

from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENCV_LOG_LEVEL", "OFF")
os.environ.setdefault("OPENCV_VIDEOIO_DEBUG", "0")

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import cv2

from .pipeline import BiometricPipeline
from .runtime import CameraPortScanner, ThreadedCameraStream, HardwareResourceManager


def main():
    parser = argparse.ArgumentParser(description="SOTA Decoupled Biometric Inference (Age LDL, Gender, Dimorphism)")
    parser.add_argument("--image", type=str, help="Path to input image file")
    parser.add_argument("--webcam", type=str, default=None, help="Webcam device ID ('auto', '0', '1'...) or stream")
    parser.add_argument("--list-cameras", action="store_true", help="Scan and list available camera ports")
    parser.add_argument("--engine", type=str, choices=["yunet", "scrfd"], default="yunet", help="Detection engine")
    parser.add_argument("--checkpoint", type=str, default="models/convnextv2_tiny_biometrics.pt", help="Path to ConvNeXt-V2 checkpoint")
    parser.add_argument("--device", type=str, default="auto", help="Execution device ('auto', 'cpu', 'cuda')")
    parser.add_argument("--target-size", type=int, default=224, help="Canonical Umeyama alignment resolution (default: 224)")
    parser.add_argument("--no-audit", action="store_true", help="Disable extreme quality auditing")

    args = parser.parse_args()
    HardwareResourceManager.optimize_process_resources()

    if args.list_cameras:
        ports = CameraPortScanner.scan_available_ports(max_index=6)
        print(f"Found {len(ports)} camera port(s):")
        for p in ports:
            print(f"  Port {p.port} [{p.backend_name}]: {p.device_name} ({p.width}x{p.height} @ {p.fps:.0f}fps, score={p.score:.1f})")
        return

    default_yunet = "models/face_detection_yunet_2023mar.onnx"
    det_model = default_yunet if (args.engine == "yunet" and Path(default_yunet).exists()) else None

    pipeline = BiometricPipeline(
        detector_engine=args.engine,
        detector_model_path=det_model,
        enable_quality_audit=not args.no_audit,
        checkpoint_path=args.checkpoint,
        target_size=args.target_size,
        device=args.device,
    )

    if args.image:
        img_path = Path(args.image)
        if not img_path.exists():
            print(f"❌ Error: Image file not found: {img_path}")
            sys.exit(1)

        bgr = cv2.imread(str(img_path))
        results = pipeline.process_frame(bgr)
        print(f"\n[Biometrics] Processed {img_path.name} — Detected {len(results)} face(s):")
        for idx, (det, audit, pred) in enumerate(results):
            print(f"\n--- Face #{idx + 1} ---")
            print(f"  Detector: {det.detector_name} | BBox: {det.bbox.to_xywh()}")
            if audit:
                print(f"  Quality Audit: {'PASSED' if audit.passed else 'REJECTED'}")
                print(f"  Laplacian Sharpness: {audit.laplacian_variance:.1f} | Tenengrad: {audit.tenengrad_energy:.1f}")
                print(f"  Pose: Yaw={audit.yaw:.1f}°, Pitch={audit.pitch:.1f}°, Roll={audit.roll:.1f}°")
                if not audit.passed:
                    print(f"  Rejection reasons: {audit.rejection_reasons}")
            if pred:
                print(
                    f"  Age: {pred.age_interval_name} years "
                    f"(Expected: {pred.expected_age:.1f}y, Conf: {pred.age_interval_confidence:.0%}, "
                    f"Cohort: {pred.age_cohort_name}, σ={pred.apparent_age_std:.2f})"
                )
                print(f"  Gender: {pred.gender} (P(Female)={pred.female_probability:.2f}, w_dim={pred.dimorphism_weight:.2f})")
                print(f"  Alpha Sample Weight: {pred.alpha_sample_weight:.4f}")

    elif args.webcam is not None:
        cam_src = "auto" if args.webcam.strip().lower() == "auto" else args.webcam
        try:
            cam_src = int(cam_src)
        except ValueError:
            pass
        print(f"🎥 Opening camera source '{cam_src}'... Press 'q' to exit, 'c' to switch camera.")
        stream = ThreadedCameraStream(camera_source=cam_src, use_threading=True)
        if not stream.isOpened():
            print(f"❌ Error: Cannot open camera '{args.webcam}'")
            sys.exit(1)

        win_name = "SOTA Biometrics (Decoupled Dual-Engine)"
        try:
            while True:
                ret, frame = stream.read()
                if not ret or frame is None:
                    break

                img_h, img_w = frame.shape[:2]
                results = pipeline.process_frame(frame)
                for det, audit, pred in results:
                    x, y, w, h = [int(v) for v in det.bbox.to_xywh()]
                    if pred:
                        color = (200, 50, 220) if pred.gender == "Female" else (240, 180, 40)
                    else:
                        color = (0, 255, 0) if (audit is None or audit.passed) else (0, 0, 255)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    if pred:
                        label = f"{pred.gender} | {pred.age_interval_name}"
                        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                        bx1 = max(0, min(x, max(0, img_w - tw - 10)))
                        bx2 = min(img_w, bx1 + tw + 10)
                        if y >= th + 10:
                            by1, by2, ty = y - th - 10, y, y - 5
                        else:
                            by1 = max(0, y)
                            by2 = min(img_h, by1 + th + 10)
                            ty = min(img_h - 4, by1 + th + 4)
                        cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, -1)
                        cv2.putText(frame, label, (bx1 + 5, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

                cv2.imshow(win_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
                if key in (ord("c"), ord("C")):
                    stream.switch_to_next_port()
                try:
                    if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
                        break
                except Exception:
                    pass
        finally:
            stream.release()
            cv2.destroyAllWindows()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
