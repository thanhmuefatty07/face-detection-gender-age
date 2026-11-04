"""
Command Line Interface for SOTA Biometric Inference
===================================================
Usage:
    python -m src.biometrics.cli --image path/to/image.jpg --engine yunet
    python -m src.biometrics.cli --webcam 0 --engine yunet
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
import cv2

from .pipeline import BiometricPipeline


def main():
    parser = argparse.ArgumentParser(description="SOTA Decoupled Biometric Inference (Age LDL, Gender, Dimorphism)")
    parser.add_argument("--image", type=str, help="Path to input image file")
    parser.add_argument("--webcam", type=int, default=None, help="Webcam device ID (e.g. 0)")
    parser.add_argument("--engine", type=str, choices=["yunet", "scrfd"], default="yunet", help="Detection engine")
    parser.add_argument("--checkpoint", type=str, default="models/convnextv2_tiny_biometrics.pt", help="Path to ConvNeXt-V2 checkpoint")
    parser.add_argument("--target-size", type=int, default=224, help="Canonical Umeyama alignment resolution (default: 224)")
    parser.add_argument("--no-audit", action="store_true", help="Disable extreme quality auditing")

    args = parser.parse_args()

    pipeline = BiometricPipeline(
        detector_engine=args.engine,
        enable_quality_audit=not args.no_audit,
        checkpoint_path=args.checkpoint,
        target_size=args.target_size,
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
        print(f"🎥 Opening webcam device {args.webcam}... Press 'q' to exit.")
        cap = cv2.VideoCapture(args.webcam)
        if not cap.isOpened():
            print(f"❌ Error: Cannot open webcam {args.webcam}")
            sys.exit(1)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = pipeline.process_frame(frame)
            for det, audit, pred in results:
                x, y, w, h = [int(v) for v in det.bbox.to_xywh()]
                color = (0, 255, 0) if (audit is None or audit.passed) else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                if pred:
                    label = f"{pred.gender}, {pred.age_interval_name}y ({pred.expected_age:.0f}y)"
                    cv2.putText(frame, label, (x, max(20, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            cv2.imshow("SOTA Biometrics (Decoupled Dual-Engine)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
