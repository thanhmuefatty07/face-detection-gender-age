"""
Biometric Target Locker Launcher (Local Edge OpenVINO/CUDA/CPU + Cloud Hybrid)
==============================================================================
Universal launcher with:
1. Automatic Multi-Backend Camera Port Discovery (DirectShow, MSMF, V4L2, AVFoundation)
2. Deep Hardware Resource Optimization (CPU Affinity, Thread Tuning, OpenVINO/CUDA/MPS)
3. Multi-Environment Photometric Normalization & Multi-Target Tracking (ByteTrack + TargetLocker)
4. Seamless Local Standalone Inference (`models/convnextv2_tiny_biometrics.pt`) + Optional Cloud Tunnel
"""

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENCV_LOG_LEVEL", "OFF")
os.environ.setdefault("OPENCV_VIDEOIO_DEBUG", "0")

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from src.biometrics.inference.cloud_client import CloudInferenceClient
from src.biometrics.runtime import CameraPortScanner, HardwareResourceManager


def print_camera_scan_report(max_index: int = 6):
    print("=" * 72)
    print("🔍 Đang quét các cổng Camera khả dụng (DirectShow / MSMF / V4L2)...")
    os_names = CameraPortScanner.enumerate_os_camera_names()
    if os_names:
        print(f"📋 Thiết bị phần cứng phát hiện từ HĐH: {', '.join(os_names)}")
    ports = CameraPortScanner.scan_available_ports(max_index=max_index)
    if not ports:
        print("⚠️ Không tìm thấy cổng camera vật lý/iVCam nào đang phát tín hiệu.")
        print("   -> Hãy cắm USB Webcam, bật ứng dụng iVCam/DroidCam, hoặc chạy với --demo.")
    else:
        for p in ports:
            status = "LIVE RGB" if (p.is_live_signal and p.is_color) else ("LIVE" if p.is_live_signal else "STANDBY/BLANK")
            print(
                f"  ✅ Cổng [{p.port}] ({p.backend_name}): {p.device_name} | "
                f"{p.width}x{p.height} @ {p.fps:.0f}fps | {status} | Score={p.score:.1f}"
            )
    print("=" * 72)
    return ports


def main():
    parser = argparse.ArgumentParser(description="Biometric Target Locker HUD (Local Edge + Cloud Hybrid)")
    parser.add_argument(
        "--server",
        type=str,
        default="ws://localhost:8000/ws/infer",
        help="Optional Kaggle WebSocket/REST URL (auto-falls back to local ConvNeXt-V2 model when offline)",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default="auto",
        help="Camera port index ('auto', 0, 1, 2...) or video file / RTSP / HTTP stream URL",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Execution hardware ('auto', 'openvino', 'cuda', 'cpu', 'ov_gpu')",
    )
    parser.add_argument(
        "--max-targets",
        type=int,
        default=16,
        help="Maximum simultaneous locked face targets (default: 16)",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=224,
        help="Canonical Umeyama alignment resolution (default: 224)",
    )
    parser.add_argument(
        "--no-cloud",
        action="store_true",
        help="Force 100% local offline inference using models/convnextv2_tiny_biometrics.pt",
    )
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="Scan and list all available camera ports and exit",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Use synthetic multi-target camera stream if no physical camera is found",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional maximum number of frames to process before exiting",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Run headless without opening OpenCV GUI window",
    )
    args = parser.parse_args()

    hw_profile = HardwareResourceManager.optimize_process_resources()
    hw_backend = HardwareResourceManager.resolve_device_and_backend(args.device)

    if args.list_cameras:
        print_camera_scan_report()
        return

    if args.camera.strip().lower() in ("auto", "scan", "default", ""):
        cam_source = "auto"
    else:
        try:
            cam_source = int(args.camera)
        except ValueError:
            cam_source = args.camera

    best_cam = CameraPortScanner.find_best_camera_port(cam_source)
    if not best_cam.can_read and not args.demo:
        print_camera_scan_report()
        print("⏳ Chờ kết nối camera (cắm Webcam hoặc mở iVCam trong 6 giây, hoặc tự động chuyển chế độ Demo)...")
        for _ in range(3):
            time.sleep(2.0)
            best_cam = CameraPortScanner.find_best_camera_port("auto")
            if best_cam.can_read:
                cam_source = best_cam.port
                break
        if not best_cam.can_read:
            print("💡 Chưa phát hiện camera vật lý -> Tự động kích hoạt luồng giả lập đa mục tiêu (Demo Mode).")
            args.demo = True
            cam_source = "demo"
    elif not best_cam.can_read and args.demo:
        cam_source = "demo"
    elif best_cam.can_read:
        cam_source = best_cam.port

    use_cloud = (not args.no_cloud) and (args.server != "ws://localhost:8000/ws/infer")

    print("=" * 72)
    print("🚀 Biometric Multi-Target Locker [ConvNeXt-V2 224x224]")
    print(
        f"⚙️  Tài nguyên máy : {hw_profile['logical_cores']} luồng CPU "
        f"(OpenCV SIMD={hw_profile['opencv_simd_optimized']}) | "
        f"Backend={hw_backend['ov_device'] if hw_backend['use_openvino'] else hw_backend['torch_device'].upper()}"
    )
    print(
        f"📷 Cổng Camera    : {best_cam.port if best_cam.can_read else 'DEMO'} "
        f"[{best_cam.backend_name}] ({best_cam.device_name})"
    )
    print(f"🎯 Đa mục tiêu    : Tối đa {args.max_targets} khuôn mặt @ {args.target_size}x{args.target_size}")
    print("💡 Phím tắt HUD   : [ESC]/[Q] Thoát | [C] Chuyển đổi cổng Camera")
    print("=" * 72)

    client = CloudInferenceClient(
        server_url=args.server,
        max_targets=args.max_targets,
        target_size=args.target_size,
        enable_local_fallback=True,
        device=args.device,
    )

    try:
        for _ in client.run_live_stream(
            camera_source=cam_source,
            max_frames=args.max_frames,
            display=not args.no_display,
            use_cloud=use_cloud,
            allow_demo_fallback=args.demo,
        ):
            pass
    except KeyboardInterrupt:
        print("\n🛑 Đã nhận tín hiệu dừng từ người dùng.")
    finally:
        HardwareResourceManager.release_unused_memory(force_gc=True)
        print("✅ Đã giải phóng luồng camera và tài nguyên phần cứng an toàn.")


if __name__ == "__main__":
    main()
