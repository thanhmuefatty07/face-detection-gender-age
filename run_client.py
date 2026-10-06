"""
iVCam Biometric Target Locker Client
====================================
Desktop launcher for real-time video biometrics via Kaggle Cloud GPU Inference Tunnel.

Usage:
  # Connect to Kaggle Cloud Tunnel (Cloudflare or ngrok WebSocket):
  python run_client.py --server wss://<tunnel-url>/ws/infer --camera 0

  # Test locally with local inference server:
  python run_client.py --server ws://localhost:8000/ws/infer --camera 0

Controls:
  Press [ESC] on the HUD camera window to exit cleanly.
"""

import argparse
import sys
from pathlib import Path

# Ensure root directory in python path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.biometrics.inference.cloud_client import CloudInferenceClient


def main():
    parser = argparse.ArgumentParser(description="iVCam Cloud Biometric HUD Client")
    parser.add_argument(
        "--server",
        type=str,
        default="ws://localhost:8000/ws/infer",
        help="Kaggle WebSocket or REST inference URL (e.g., wss://xyz.trycloudflare.com/ws/infer)",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default="0",
        help="Camera device index (default: 0 for iVCam DirectShow) or video file path",
    )
    parser.add_argument(
        "--max-targets",
        type=int,
        default=5,
        help="Maximum simultaneous locked face targets (default: 5)",
    )
    parser.add_argument(
        "--no-cloud",
        action="store_true",
        help="Disable cloud server communication and run in local coasting mode",
    )
    args = parser.parse_args()

    # Parse camera index if numeric
    try:
        cam_source = int(args.camera)
    except ValueError:
        cam_source = args.camera

    print("=" * 70)
    print("🚀 iVCam Biometric Target Locker [ConvNeXt-V2 Kaggle Cloud]")
    print(f"📡 Cloud Endpoint : {args.server}")
    print(f"📷 Camera Source  : {cam_source} (DirectShow low-latency)")
    print(f"🎯 Target Density : Max {args.max_targets} subjects")
    print("=" * 70)
    print("💡 Nhấn phím [ESC] trên cửa sổ video để dừng chương trình.")

    client = CloudInferenceClient(
        server_url=args.server,
        max_targets=args.max_targets,
    )

    try:
        # Run live stream generator with GUI display
        for _ in client.run_live_stream(
            camera_source=cam_source,
            display=True,
            use_cloud=not args.no_cloud,
        ):
            pass
    except KeyboardInterrupt:
        print("\n🛑 Đã nhận tín hiệu dừng từ người dùng.")
    finally:
        print("✅ Đã giải phóng luồng camera và đóng kết nối an toàn.")


if __name__ == "__main__":
    main()
