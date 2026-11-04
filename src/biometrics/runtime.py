"""
Hardware Resource Manager, Multi-Environment Photometric Normalizer & Smart Camera Port Scanner
===============================================================================================
Provides deep machine resource optimization (CPU thread/affinity/priority tuning, OpenVINO disk-cached
compilation, CUDA AMP / channels_last execution, adaptive load pacing), multi-environment illumination
and blur normalization, and multi-backend camera port auto-discovery with zero-lag threaded capture.
"""

from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass, field
import gc
import math
import os
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

os.environ.setdefault("OPENCV_LOG_LEVEL", "OFF")
os.environ.setdefault("OPENCV_VIDEOIO_DEBUG", "0")

import cv2
import numpy as np
import torch


@contextmanager
def suppress_opencv_logs():
    """Temporarily silences OpenCV C++ videoio stderr spam during camera port scanning."""
    prev_level = None
    try:
        if hasattr(cv2, "getLogLevel") and hasattr(cv2, "setLogLevel"):
            prev_level = cv2.getLogLevel()
            cv2.setLogLevel(0)
        elif hasattr(cv2, "utils") and hasattr(cv2.utils, "logging"):
            prev_level = cv2.utils.logging.getLogLevel()
            cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
    except Exception:
        prev_level = None
    try:
        yield
    finally:
        if prev_level is not None:
            try:
                if hasattr(cv2, "setLogLevel"):
                    cv2.setLogLevel(prev_level)
                elif hasattr(cv2, "utils") and hasattr(cv2.utils, "logging"):
                    cv2.utils.logging.setLogLevel(prev_level)
            except Exception:
                pass


class HardwareResourceManager:
    """
    Deep machine resource inspector, OS process tuner, and multi-backend accelerator selector.
    Maximizes throughput and minimizes frame latency across Intel/AMD CPUs, Intel iGPU/OpenVINO,
    NVIDIA CUDA GPUs, AMD DirectML, and Apple Silicon MPS.
    """

    _initialized: bool = False
    _cached_profile: Optional[Dict[str, Any]] = None

    @classmethod
    def detect_cpu_topology(cls) -> Dict[str, int]:
        """Detects logical and physical CPU cores across Windows, Linux, and macOS."""
        logical = max(1, int(os.cpu_count() or 4))
        physical = max(1, logical // 2) if logical >= 4 else logical

        if sys.platform == "win32":
            try:
                import ctypes
                kernel32 = ctypes.windll.kernel32
                cnt = kernel32.GetActiveProcessorCount(0xFFFF)
                if cnt and cnt > 0:
                    logical = int(cnt)
                    physical = max(1, logical // 2) if logical >= 4 else logical
            except Exception:
                pass
        elif sys.platform.startswith("linux"):
            try:
                cores_set = set()
                with open("/proc/cpuinfo", "r", encoding="utf-8", errors="ignore") as f:
                    phys_id = None
                    core_id = None
                    for line in f:
                        if line.startswith("physical id"):
                            phys_id = line.split(":")[-1].strip()
                        elif line.startswith("core id"):
                            core_id = line.split(":")[-1].strip()
                            cores_set.add((phys_id, core_id))
                if cores_set:
                    physical = max(1, len(cores_set))
            except Exception:
                pass
        elif sys.platform == "darwin":
            try:
                res_phys = subprocess.run(
                    ["sysctl", "-n", "hw.physicalcpu"],
                    capture_output=True,
                    text=True,
                    timeout=1.5,
                )
                if res_phys.returncode == 0 and res_phys.stdout.strip().isdigit():
                    physical = max(1, int(res_phys.stdout.strip()))
                res_log = subprocess.run(
                    ["sysctl", "-n", "hw.logicalcpu"],
                    capture_output=True,
                    text=True,
                    timeout=1.5,
                )
                if res_log.returncode == 0 and res_log.stdout.strip().isdigit():
                    logical = max(1, int(res_log.stdout.strip()))
            except Exception:
                pass

        return {
            "logical_cores": logical,
            "physical_cores": physical,
            "optimal_intra_threads": logical,
            "optimal_opencv_threads": max(1, min(4, physical)),
            "optimal_interop_threads": max(1, min(4, physical)),
        }

    @classmethod
    def detect_memory_resources(cls) -> Dict[str, Any]:
        """Queries physical system RAM and GPU VRAM capacity/availability without external dependencies."""
        total_ram_mb = 8192
        avail_ram_mb = 4096
        mem_load_pct = 50

        if sys.platform == "win32":
            try:
                import ctypes

                class _MEMORYSTATUSEX(ctypes.Structure):
                    _fields_ = [
                        ("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                    ]

                stat = _MEMORYSTATUSEX()
                stat.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
                if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                    total_ram_mb = max(256, int(stat.ullTotalPhys // (1024 * 1024)))
                    avail_ram_mb = max(128, int(stat.ullAvailPhys // (1024 * 1024)))
                    mem_load_pct = int(stat.dwMemoryLoad)
            except Exception:
                pass
        elif sys.platform.startswith("linux"):
            try:
                with open("/proc/meminfo", "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        if line.startswith("MemTotal:"):
                            total_ram_mb = max(256, int(line.split()[1]) // 1024)
                        elif line.startswith("MemAvailable:"):
                            avail_ram_mb = max(128, int(line.split()[1]) // 1024)
                mem_load_pct = int(max(0.0, min(100.0, (1.0 - avail_ram_mb / float(max(1, total_ram_mb))) * 100.0)))
            except Exception:
                pass
        elif sys.platform == "darwin":
            try:
                res_mem = subprocess.run(
                    ["sysctl", "-n", "hw.memsize"],
                    capture_output=True,
                    text=True,
                    timeout=1.5,
                )
                if res_mem.returncode == 0 and res_mem.stdout.strip().isdigit():
                    total_ram_mb = max(256, int(res_mem.stdout.strip()) // (1024 * 1024))
                    avail_ram_mb = max(128, int(total_ram_mb * 0.6))
                res_vm = subprocess.run(
                    ["vm_stat"],
                    capture_output=True,
                    text=True,
                    timeout=1.5,
                )
                if res_vm.returncode == 0 and res_vm.stdout:
                    page_size = 16384
                    free_pages = 0
                    for line in res_vm.stdout.splitlines():
                        if "page size of" in line:
                            for tok in line.split():
                                if tok.isdigit():
                                    page_size = int(tok)
                                    break
                        elif any(
                            line.startswith(k)
                            for k in ("Pages free:", "Pages inactive:", "Pages speculative:")
                        ):
                            val_str = line.split(":")[-1].strip().rstrip(".")
                            if val_str.isdigit():
                                free_pages += int(val_str)
                    if free_pages > 0:
                        avail_ram_mb = max(128, min(total_ram_mb, (free_pages * page_size) // (1024 * 1024)))
                mem_load_pct = int(max(0.0, min(100.0, (1.0 - avail_ram_mb / float(max(1, total_ram_mb))) * 100.0)))
            except Exception:
                pass

        vram_total_mb = 0
        gpu_name = "None"
        if torch.cuda.is_available():
            try:
                props = torch.cuda.get_device_properties(0)
                vram_total_mb = int(props.total_memory // (1024 * 1024))
                gpu_name = str(props.name)
            except Exception:
                pass
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            gpu_name = "Apple Silicon MPS"
            vram_total_mb = total_ram_mb
            if sys.platform == "darwin":
                try:
                    res_cpu = subprocess.run(
                        ["sysctl", "-n", "machdep.cpu.brand_string"],
                        capture_output=True,
                        text=True,
                        timeout=1.5,
                    )
                    if res_cpu.returncode == 0 and res_cpu.stdout.strip():
                        gpu_name = f"{res_cpu.stdout.strip()} (MPS)"
                except Exception:
                    pass

        if vram_total_mb >= 4096 or avail_ram_mb >= 4096:
            optimal_batch = 16
        elif avail_ram_mb >= 2048:
            optimal_batch = 8
        else:
            optimal_batch = 4

        return {
            "total_ram_mb": total_ram_mb,
            "avail_ram_mb": avail_ram_mb,
            "total_ram_gb": round(total_ram_mb / 1024.0, 2),
            "available_ram_gb": round(avail_ram_mb / 1024.0, 2),
            "memory_load_pct": mem_load_pct,
            "vram_total_mb": vram_total_mb,
            "gpu_name": gpu_name,
            "optimal_batch_size": optimal_batch,
        }

    @classmethod
    def elevate_process_priority_and_affinity(
        cls,
        elevate_priority: bool = True,
        bind_all_cores: bool = True,
    ) -> Dict[str, Any]:
        """
        Configures OS process priority (ABOVE_NORMAL on Windows, nice on POSIX),
        unparks all logical CPU cores via process affinity mask, disables Windows
        EcoQoS power throttling, and sets 1ms multimedia timer precision.
        """
        status: Dict[str, Any] = {
            "priority_elevated": False,
            "affinity_bound": False,
            "timer_1ms": False,
            "power_unthrottled": False,
        }
        if sys.platform == "win32":
            try:
                import ctypes
                kernel32 = ctypes.windll.kernel32
                handle = kernel32.GetCurrentProcess()
                if elevate_priority:
                    # ABOVE_NORMAL_PRIORITY_CLASS = 0x00008000
                    ok = kernel32.SetPriorityClass(handle, 0x00008000)
                    status["priority_elevated"] = bool(ok)
                if bind_all_cores:
                    logical = max(1, int(os.cpu_count() or 4))
                    mask = (1 << min(logical, 64)) - 1
                    ok_aff = kernel32.SetProcessAffinityMask(handle, ctypes.c_size_t(mask))
                    status["affinity_bound"] = bool(ok_aff)

                # Disable Windows EcoQoS / Process Power Throttling (ProcessPowerThrottling = 4)
                try:
                    class _PROCESS_POWER_THROTTLING_STATE(ctypes.Structure):
                        _fields_ = [
                            ("Version", ctypes.c_ulong),
                            ("ControlMask", ctypes.c_ulong),
                            ("StateMask", ctypes.c_ulong),
                        ]

                    pwr = _PROCESS_POWER_THROTTLING_STATE(1, 0x1, 0x0)
                    ok_pwr = kernel32.SetProcessInformation(handle, 4, ctypes.byref(pwr), ctypes.sizeof(pwr))
                    status["power_unthrottled"] = bool(ok_pwr)
                except Exception:
                    pass

                # Request 1ms high-resolution OS multimedia scheduler timer (prevents 15.6ms sleep jitter)
                try:
                    ctypes.windll.winmm.timeBeginPeriod(1)
                    status["timer_1ms"] = True
                except Exception:
                    pass
            except Exception:
                pass
        else:
            if elevate_priority:
                try:
                    os.nice(-5)
                    status["priority_elevated"] = True
                except Exception:
                    pass
            if bind_all_cores and hasattr(os, "sched_setaffinity"):
                try:
                    logical = max(1, int(os.cpu_count() or 4))
                    os.sched_setaffinity(0, set(range(logical)))
                    status["affinity_bound"] = True
                except Exception:
                    pass
        return status

    @classmethod
    def optimize_process_resources(
        cls,
        elevate_priority: bool = True,
        bind_affinity: bool = True,
    ) -> Dict[str, Any]:
        """
        Applies deep process-level thread, SIMD, OpenCV, MKL/OpenMP, and PyTorch optimizations.
        Prevents thread oversubscription between OpenCV and PyTorch/OpenVINO thread pools.
        """
        topo = cls.detect_cpu_topology()
        mem_info = cls.detect_memory_resources()
        intra = topo["optimal_intra_threads"]
        cv_threads = topo["optimal_opencv_threads"]

        os.environ.setdefault("OMP_NUM_THREADS", str(intra))
        os.environ.setdefault("MKL_NUM_THREADS", str(intra))
        os.environ.setdefault("OPENBLAS_NUM_THREADS", str(intra))
        os.environ.setdefault("NUMEXPR_NUM_THREADS", str(intra))
        os.environ.setdefault("KMP_BLOCKTIME", "0")

        try:
            cv2.setUseOptimized(True)
            # Bound OpenCV threads to physical cores to prevent contention with OpenVINO/PyTorch worker threads
            cv2.setNumThreads(cv_threads)
        except Exception:
            pass

        try:
            torch.set_num_threads(intra)
        except Exception:
            pass

        try:
            if hasattr(torch.backends, "mkldnn"):
                torch.backends.mkldnn.enabled = True
            if hasattr(torch.backends, "cudnn") and torch.cuda.is_available():
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.allow_tf32 = True
            if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
                torch.backends.cuda.matmul.allow_tf32 = True
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")
        except Exception:
            pass

        os_status = cls.elevate_process_priority_and_affinity(
            elevate_priority=elevate_priority,
            bind_all_cores=bind_affinity,
        )

        profile = {
            **topo,
            **mem_info,
            **os_status,
            "torch_threads": int(torch.get_num_threads()),
            "opencv_threads": int(cv2.getNumThreads()),
            "opencv_simd_optimized": bool(cv2.useOptimized()),
            "cuda_available": bool(torch.cuda.is_available()),
            "mps_available": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
        }
        cls._initialized = True
        cls._cached_profile = profile
        return profile

    @classmethod
    def resolve_device_and_backend(
        cls,
        requested_device: str = "auto",
        use_openvino: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Resolves any user or internal device string ('auto', 'cpu', 'CPU', 'gpu', 'GPU',
        'cuda', 'mps', 'openvino', 'hybrid', 'dml') into a safe, non-conflicting configuration:
        - `torch_device`: strictly valid PyTorch device string ('cuda', 'mps', or 'cpu')
        - `ov_device`: OpenVINO target device ('CPU' or 'GPU')
        - `use_openvino`: whether OpenVINO compilation should be used
        - `use_fp16`: whether FP16 autocast should be used on PyTorch GPU
        - `use_channels_last`: whether NHWC memory layout should be used
        """
        req = str(requested_device or "auto").strip().lower()

        has_cuda = bool(torch.cuda.is_available())
        has_mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
        ov_devices: List[str] = []
        try:
            import openvino as ov
            ov_devices = list(ov.Core().available_devices)
        except Exception:
            ov_devices = []

        has_ov = len(ov_devices) > 0

        # Determine valid PyTorch device (always lowercase!)
        if req.startswith("cuda") or (req in ("auto", "gpu") and has_cuda):
            torch_device = req if req.startswith("cuda") and has_cuda else ("cuda" if has_cuda else "cpu")
        elif req == "mps" or (req in ("auto", "gpu") and has_mps):
            torch_device = "mps" if has_mps else "cpu"
        else:
            torch_device = "cpu"

        # Determine OpenVINO usage & target device
        if use_openvino is None:
            use_ov = has_ov and (torch_device == "cpu") and (
                req in ("auto", "openvino", "ov", "hybrid", "ov_cpu", "ov_gpu", "intel_gpu")
            )
        else:
            use_ov = bool(use_openvino) and has_ov

        if req in ("ov_gpu", "intel_gpu") and "GPU" in ov_devices:
            ov_device = "GPU"
        else:
            ov_device = "CPU" if "CPU" in ov_devices else (ov_devices[0] if ov_devices else "CPU")

        return {
            "requested": requested_device,
            "torch_device": torch_device,
            "use_openvino": use_ov,
            "ov_device": ov_device,
            "ov_available_devices": ov_devices,
            "use_fp16": (torch_device.startswith("cuda")),
            "use_channels_last": True,
        }

    @staticmethod
    def release_unused_memory(force_gc: bool = False) -> None:
        """Frees unused accelerator cache and optionally runs garbage collection."""
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        if force_gc:
            gc.collect()


class EnvironmentNormalizer:
    """
    Real-time Multi-Environment Photometric & Optical Restorer.
    Detects and compensates for:
      1. Extreme Low-Light / Night-Time capture (bilateral read-noise suppression + adaptive gamma + CLAHE)
      2. Strong Backlighting & Directional Raking Shadows (hemiface illumination equalization + CLAHE)
      3. Harsh Overexposure / Specular Glare (highlight S-curve compression + bloom attenuation)
      4. Artificial Illuminant Color Casts (skin-locus-aware Von Kries / LAB gray-world balance)
      5. Motion Blur / Optical Defocus (frequency-selective unsharp contour restoration)
      6. Empty / Blank / Uniform frames (identifies non-signal frames to prevent ghost detections)
    """

    def __init__(
        self,
        low_light_thresh: float = 68.0,
        overexposed_thresh: float = 195.0,
        color_cast_thresh: float = 14.0,
        blur_laplacian_thresh: float = 75.0,
        blank_std_thresh: float = 2.0,
    ):
        self.low_light_thresh = float(low_light_thresh)
        self.overexposed_thresh = float(overexposed_thresh)
        self.color_cast_thresh = float(color_cast_thresh)
        self.blur_laplacian_thresh = float(blur_laplacian_thresh)
        self.blank_std_thresh = float(blank_std_thresh)
        self._clahe_soft = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(4, 4))
        self._clahe_strong = cv2.createCLAHE(clipLimit=2.6, tileGridSize=(6, 6))

    @staticmethod
    def _ensure_3ch_u8(image: np.ndarray, is_bgr: bool = True) -> Optional[np.ndarray]:
        if image is None or image.size == 0:
            return None
        img = image
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR if is_bgr else cv2.COLOR_GRAY2RGB)
        if img.ndim == 3:
            if img.shape[2] == 3:
                return img
            if img.shape[2] == 1:
                return cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2BGR if is_bgr else cv2.COLOR_GRAY2RGB)
            if img.shape[2] == 4:
                return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR if is_bgr else cv2.COLOR_RGBA2RGB)
        return None

    @staticmethod
    def _compute_skin_safe_cast_excess(a_shift: float, b_shift: float) -> Tuple[float, float, float]:
        """
        Computes chrominance deviation outside the natural human skin / neutral locus in CIE L*a*b*.
        Natural human skin in 8-bit LAB has a_shift in [0, 20] and b_shift in [0, 26] (positive warm
        hemoglobin + melanin locus). Only deviations outside this physiological band are treated as
        artificial illuminant color casts (e.g. blue/cyan/green casts or extreme neon/amber casts).
        """
        if a_shift < 0.0:
            excess_a = a_shift
        elif a_shift > 18.0:
            excess_a = a_shift - 18.0
        else:
            excess_a = 0.0

        if b_shift < 0.0:
            excess_b = b_shift
        elif b_shift > 24.0:
            excess_b = b_shift - 24.0
        else:
            excess_b = 0.0

        # Also catch unnatural magenta/purple or green-yellow diagonal casts where a and b have opposite signs
        if (a_shift > 8.0 and b_shift < -3.0) or (a_shift < -3.0 and b_shift > 8.0):
            excess_a = a_shift
            excess_b = b_shift

        return excess_a, excess_b, math.hypot(excess_a, excess_b)

    def analyze(self, image: np.ndarray, is_bgr: bool = True) -> Dict[str, Any]:
        """Analyzes photometric & optical sharpness profile of an image in < 1 ms."""
        img3 = self._ensure_3ch_u8(image, is_bgr=is_bgr)
        if img3 is None:
            return {
                "regime": "blank",
                "is_blank": True,
                "is_low_light": False,
                "is_backlight": False,
                "is_overexposed": False,
                "is_color_cast": False,
                "is_blurry": False,
                "mean_l": 0.0,
                "std_l": 0.0,
                "hemi_asym": 0.0,
                "laplacian_var": 0.0,
                "uncertainty_multiplier": 1.5,
            }

        h, w = img3.shape[:2]
        if max(h, w) > 320:
            scale = 240.0 / float(max(h, w))
            probe = cv2.resize(
                img3,
                (max(16, int(w * scale)), max(16, int(h * scale))),
                interpolation=cv2.INTER_AREA,
            )
        else:
            probe = img3

        lab_code = cv2.COLOR_BGR2LAB if is_bgr else cv2.COLOR_RGB2LAB
        lab = cv2.cvtColor(probe, lab_code)
        l_ch = lab[:, :, 0].astype(np.float32)
        a_ch = lab[:, :, 1].astype(np.float32)
        b_ch = lab[:, :, 2].astype(np.float32)

        mean_l = float(np.mean(l_ch))
        std_l = float(np.std(l_ch))
        dyn_range = float(np.max(l_ch) - np.min(l_ch))

        if std_l < self.blank_std_thresh or dyn_range < 6.0:
            return {
                "regime": "blank",
                "is_blank": True,
                "is_low_light": mean_l < self.low_light_thresh,
                "is_backlight": False,
                "is_overexposed": mean_l > self.overexposed_thresh,
                "is_color_cast": False,
                "is_blurry": True,
                "mean_l": mean_l,
                "std_l": std_l,
                "hemi_asym": 0.0,
                "laplacian_var": 0.0,
                "uncertainty_multiplier": 1.5,
            }

        shadow_ratio = float(np.mean(l_ch < 58.0))
        highlight_ratio = float(np.mean(l_ch > 230.0))
        mid_w = max(1, l_ch.shape[1] // 2)
        mid_h = max(1, l_ch.shape[0] // 2)
        hemi_asym = float(abs(np.mean(l_ch[:, :mid_w]) - np.mean(l_ch[:, mid_w:])))
        vert_asym = float(abs(np.mean(l_ch[:mid_h, :]) - np.mean(l_ch[mid_h:, :])))

        a_shift = float(np.mean(a_ch) - 128.0)
        b_shift = float(np.mean(b_ch) - 128.0)
        excess_a, excess_b, excess_cast_mag = self._compute_skin_safe_cast_excess(a_shift, b_shift)
        raw_cast_mag = math.hypot(a_shift, b_shift)

        gray_u8 = lab[:, :, 0]
        lap_var = float(cv2.Laplacian(gray_u8, cv2.CV_64F).var())

        is_low_light = (mean_l < self.low_light_thresh) or (mean_l < 85.0 and shadow_ratio > 0.45)
        is_backlight = (
            (shadow_ratio > 0.25 and highlight_ratio > 0.06)
            or (hemi_asym > 26.0 and shadow_ratio > 0.10)
            or (vert_asym > 32.0 and shadow_ratio > 0.10)
            or (hemi_asym > 38.0)
            or (vert_asym > 44.0)
        )
        is_overexposed = (mean_l > self.overexposed_thresh) or (highlight_ratio > 0.14)
        # Trigger color cast when excess chrominance outside the physiological skin locus exceeds threshold
        # OR when an image has strong blue/cyan/green negative shift (excess_cast_mag > 8.5)
        is_color_cast = (excess_cast_mag > self.color_cast_thresh) or (
            (a_shift < -6.0 or b_shift < -6.0) and raw_cast_mag > 10.0
        )
        is_blurry = (lap_var < self.blur_laplacian_thresh)

        active_flags = sum([is_low_light, is_backlight, is_overexposed, is_color_cast, is_blurry])
        if active_flags >= 2:
            regime = "compound_degraded"
        elif is_low_light:
            regime = "low_light"
        elif is_backlight:
            regime = "backlight"
        elif is_overexposed:
            regime = "overexposed_glare"
        elif is_color_cast:
            regime = "color_cast"
        elif is_blurry:
            regime = "motion_blur"
        else:
            regime = "clean"

        unc_mult = 1.0
        if is_low_light or is_backlight:
            unc_mult += 0.15
        if is_overexposed:
            unc_mult += 0.12
        if is_blurry:
            unc_mult += 0.15
        if is_color_cast:
            unc_mult += 0.08

        return {
            "regime": regime,
            "is_blank": False,
            "is_low_light": is_low_light,
            "is_backlight": is_backlight,
            "is_overexposed": is_overexposed,
            "is_color_cast": is_color_cast,
            "is_blurry": is_blurry,
            "mean_l": mean_l,
            "std_l": std_l,
            "shadow_ratio": shadow_ratio,
            "highlight_ratio": highlight_ratio,
            "hemi_asym": hemi_asym,
            "vert_asym": vert_asym,
            "a_shift": a_shift,
            "b_shift": b_shift,
            "excess_a": excess_a,
            "excess_b": excess_b,
            "laplacian_var": lap_var,
            "uncertainty_multiplier": min(1.75, unc_mult),
        }

    def normalize_frame_for_detection(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Enhances a full camera frame for robust face detection across extreme lighting
        (dark rooms, strong backlighting, directional shadows, glare, color casts).
        """
        img3 = self._ensure_3ch_u8(frame_bgr, is_bgr=True)
        if img3 is None:
            return frame_bgr, self.analyze(frame_bgr, is_bgr=True)

        info = self.analyze(img3, is_bgr=True)
        if info["is_blank"] or info["regime"] == "clean":
            return img3, info

        lab = cv2.cvtColor(img3, cv2.COLOR_BGR2LAB)
        l_ch = lab[:, :, 0]

        if info["is_low_light"] or info["is_backlight"]:
            if info["is_low_light"]:
                l_ch = cv2.bilateralFilter(l_ch, d=5, sigmaColor=18, sigmaSpace=18)
            mean_norm = max(0.05, min(0.95, info["mean_l"] / 255.0))
            gamma = float(np.clip(math.log(0.50) / math.log(mean_norm), 0.42, 0.85))
            lut = np.array([((i / 255.0) ** gamma) * 255.0 for i in range(256)], dtype=np.uint8)
            l_ch = cv2.LUT(l_ch, lut)
            l_ch = self._clahe_strong.apply(l_ch)
        elif info["is_overexposed"]:
            gamma = 1.45
            lut = np.array([((i / 255.0) ** gamma) * 255.0 for i in range(256)], dtype=np.uint8)
            l_ch = cv2.LUT(l_ch, lut)
            l_ch = self._clahe_soft.apply(l_ch)

        lab[:, :, 0] = l_ch

        if info["is_color_cast"]:
            da = info["excess_a"] if abs(info["excess_a"]) > 1.0 else info["a_shift"] * 0.5
            db = info["excess_b"] if abs(info["excess_b"]) > 1.0 else info["b_shift"] * 0.5
            lab[:, :, 1] = np.clip(lab[:, :, 1].astype(np.float32) - 0.70 * da, 0, 255).astype(np.uint8)
            lab[:, :, 2] = np.clip(lab[:, :, 2].astype(np.float32) - 0.70 * db, 0, 255).astype(np.uint8)

        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return enhanced, info

    def normalize_crop_for_inference(
        self,
        crop: np.ndarray,
        is_bgr: bool = False,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Adaptively normalizes an aligned face crop ONLY when environmental degradation
        is detected, preserving clean human skin crops untouched while restoring degraded regimes.
        """
        img3 = self._ensure_3ch_u8(crop, is_bgr=is_bgr)
        if img3 is None:
            return crop, self.analyze(crop, is_bgr=is_bgr)

        info = self.analyze(img3, is_bgr=is_bgr)
        if info["is_blank"] or info["regime"] == "clean":
            return img3, info

        to_lab = cv2.COLOR_BGR2LAB if is_bgr else cv2.COLOR_RGB2LAB
        from_lab = cv2.COLOR_LAB2BGR if is_bgr else cv2.COLOR_LAB2RGB
        lab = cv2.cvtColor(img3, to_lab)
        l_ch = lab[:, :, 0]

        if info["is_low_light"] or info["is_backlight"]:
            if info["is_low_light"]:
                # Suppress dark sensor shot/read noise before gamma & CLAHE so noise isn't amplified into fake wrinkles
                l_ch = cv2.bilateralFilter(l_ch, d=5, sigmaColor=16, sigmaSpace=16)

            if info.get("hemi_asym", 0.0) > 22.0 or info.get("vert_asym", 0.0) > 28.0:
                # Compensate directional raking shadow via smooth illumination envelope flattening
                l_f = l_ch.astype(np.float32)
                illum = cv2.GaussianBlur(l_f, (0, 0), sigmaX=24.0)
                mean_target = max(95.0, min(145.0, float(np.mean(l_f)) * 1.25))
                l_flat = l_f * (mean_target / np.maximum(illum, 22.0))
                l_ch = np.clip(0.55 * l_flat + 0.45 * l_f, 0.0, 255.0).astype(np.uint8)

            mean_norm = max(0.05, min(0.95, float(np.mean(l_ch)) / 255.0))
            gamma = float(np.clip(math.log(0.48) / math.log(mean_norm), 0.48, 0.92))
            lut = np.array([((i / 255.0) ** gamma) * 255.0 for i in range(256)], dtype=np.uint8)
            l_ch = self._clahe_soft.apply(cv2.LUT(l_ch, lut))
        elif info["is_overexposed"]:
            gamma = 1.35
            lut = np.array([((i / 255.0) ** gamma) * 255.0 for i in range(256)], dtype=np.uint8)
            l_ch = self._clahe_soft.apply(cv2.LUT(l_ch, lut))

        lab[:, :, 0] = l_ch

        if info["is_color_cast"]:
            da = info["excess_a"] if abs(info["excess_a"]) > 1.0 else info["a_shift"] * 0.45
            db = info["excess_b"] if abs(info["excess_b"]) > 1.0 else info["b_shift"] * 0.45
            lab[:, :, 1] = np.clip(lab[:, :, 1].astype(np.float32) - 0.65 * da, 0, 255).astype(np.uint8)
            lab[:, :, 2] = np.clip(lab[:, :, 2].astype(np.float32) - 0.65 * db, 0, 255).astype(np.uint8)

        out = cv2.cvtColor(lab, from_lab)

        if info["is_blurry"] and not info["is_low_light"]:
            gaussian = cv2.GaussianBlur(out, (3, 3), 1.0)
            out = cv2.addWeighted(out, 1.30, gaussian, -0.30, 0)

        return out, info


@dataclass
class CameraPortInfo:
    """Structured diagnostic descriptor for a scanned camera port."""
    port: Union[int, str]
    backend_name: str
    backend_id: int
    device_name: str = "Camera Device"
    is_opened: bool = False
    can_read: bool = False
    is_live_signal: bool = False
    is_color: bool = False
    width: int = 0
    height: int = 0
    fps: float = 0.0
    luminance_mean: float = 0.0
    luminance_std: float = 0.0
    temporal_diff: float = 0.0
    score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "port": self.port,
            "backend_name": self.backend_name,
            "backend_id": self.backend_id,
            "device_name": self.device_name,
            "is_opened": self.is_opened,
            "can_read": self.can_read,
            "is_live_signal": self.is_live_signal,
            "is_color": self.is_color,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "luminance_mean": round(self.luminance_mean, 2),
            "luminance_std": round(self.luminance_std, 2),
            "temporal_diff": round(self.temporal_diff, 3),
            "score": round(self.score, 2),
        }


class CameraPortScanner:
    """
    Multi-Backend Camera Port Auto-Discovery & Signal Verification Engine.
    Scans indices across DirectShow (CAP_DSHOW), Media Foundation (CAP_MSMF), V4L2,
    AVFoundation, and CAP_ANY, evaluating spatial and temporal signal vitality to select
    the optimal live camera.
    """

    @staticmethod
    def get_platform_backends() -> List[Tuple[str, int]]:
        """Returns prioritized OpenCV VideoCapture backends for the current OS."""
        if sys.platform == "win32":
            return [
                ("DSHOW", getattr(cv2, "CAP_DSHOW", 700)),
                ("MSMF", getattr(cv2, "CAP_MSMF", 1400)),
                ("ANY", getattr(cv2, "CAP_ANY", 0)),
            ]
        if sys.platform.startswith("linux"):
            return [
                ("V4L2", getattr(cv2, "CAP_V4L2", 200)),
                ("ANY", getattr(cv2, "CAP_ANY", 0)),
            ]
        if sys.platform == "darwin":
            return [
                ("AVFOUNDATION", getattr(cv2, "CAP_AVFOUNDATION", 1200)),
                ("ANY", getattr(cv2, "CAP_ANY", 0)),
            ]
        return [("ANY", getattr(cv2, "CAP_ANY", 0))]

    @staticmethod
    def enumerate_os_camera_names() -> List[str]:
        """Queries the OS for human-readable camera/webcam/iVCam device names."""
        names: List[str] = []
        if sys.platform == "win32":
            try:
                cmd = [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    (
                        "Get-CimInstance Win32_PnPEntity | "
                        "Where-Object { ($_.PNPClass -match 'Camera|Image') -or "
                        "($_.Name -match 'Camera|Webcam|iVCam|DroidCam|OBS|UVC|Video') } | "
                        "Select-Object -ExpandProperty Name"
                    ),
                ]
                res = subprocess.run(cmd, capture_output=True, text=True, timeout=2.5)
                if res.returncode == 0 and res.stdout:
                    for line in res.stdout.splitlines():
                        clean = line.strip()
                        if clean and clean not in names and "Audio" not in clean and "Host Controller" not in clean:
                            names.append(clean)
            except Exception:
                pass
        elif sys.platform.startswith("linux"):
            try:
                v4l_root = Path("/sys/class/video4linux")
                if v4l_root.exists():
                    for dev_dir in sorted(v4l_root.glob("video*")):
                        name_file = dev_dir / "name"
                        if name_file.exists():
                            n = name_file.read_text(encoding="utf-8", errors="ignore").strip()
                            if n:
                                names.append(f"{dev_dir.name}: {n}")
            except Exception:
                pass
        elif sys.platform == "darwin":
            try:
                res = subprocess.run(
                    ["system_profiler", "SPCameraDataType"],
                    capture_output=True,
                    text=True,
                    timeout=3.0,
                )
                if res.returncode == 0 and res.stdout:
                    for line in res.stdout.splitlines():
                        stripped = line.strip()
                        if (
                            stripped.endswith(":")
                            and not stripped.startswith("Camera:")
                            and not stripped.startswith("Model ID:")
                            and not stripped.startswith("Unique ID:")
                        ):
                            cam_name = stripped[:-1].strip()
                            if cam_name and cam_name not in names:
                                names.append(cam_name)
            except Exception:
                pass
        return names

    @classmethod
    def probe_port(
        cls,
        port: Union[int, str],
        backends: Optional[List[Tuple[str, int]]] = None,
        warmup_frames: int = 3,
        device_name: str = "Camera Device",
    ) -> CameraPortInfo:
        """
        Probes a single camera port or video path/URL across candidate backends
        and scores its optical signal health.
        """
        if isinstance(port, str):
            if port.strip().lower() == "demo":
                return CameraPortInfo(
                    port="DEMO",
                    backend_name="SYNTHETIC",
                    backend_id=0,
                    device_name="Synthetic Multi-Target Demo Stream",
                    is_opened=True,
                    can_read=True,
                    is_live_signal=True,
                    is_color=True,
                    width=640,
                    height=480,
                    fps=60.0,
                    score=50.0,
                )
            try:
                port = int(port)
            except ValueError:
                pass

        if isinstance(port, str):
            with suppress_opencv_logs():
                cap = cv2.VideoCapture(port)
                info = cls._evaluate_capture(
                    cap,
                    port=port,
                    backend_name="STREAM",
                    backend_id=0,
                    warmup_frames=warmup_frames,
                    device_name=port,
                )
                cap.release()
                return info

        candidate_backends = backends or cls.get_platform_backends()
        best_info = CameraPortInfo(port=port, backend_name="NONE", backend_id=0, device_name=device_name)

        with suppress_opencv_logs():
            for b_name, b_id in candidate_backends:
                cap = None
                try:
                    cap = cv2.VideoCapture(int(port), b_id) if b_id != 0 else cv2.VideoCapture(int(port))
                    if not cap.isOpened():
                        cap.release()
                        continue
                    info = cls._evaluate_capture(
                        cap,
                        port=port,
                        backend_name=b_name,
                        backend_id=b_id,
                        warmup_frames=warmup_frames,
                        device_name=device_name,
                    )
                    cap.release()
                    if info.score > best_info.score:
                        best_info = info
                    if info.is_live_signal and info.is_color:
                        break
                except Exception:
                    if cap is not None:
                        try:
                            cap.release()
                        except Exception:
                            pass

        return best_info

    @staticmethod
    def _evaluate_capture(
        cap: cv2.VideoCapture,
        port: Union[int, str],
        backend_name: str,
        backend_id: int,
        warmup_frames: int = 3,
        device_name: str = "Camera Device",
    ) -> CameraPortInfo:
        if cap is None or not cap.isOpened():
            return CameraPortInfo(port=port, backend_name=backend_name, backend_id=backend_id, device_name=device_name)

        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        first_frame: Optional[np.ndarray] = None
        last_frame: Optional[np.ndarray] = None
        read_ok = False
        max_reads = max(1, warmup_frames)
        if backend_name == "AVFOUNDATION":
            max_reads = max(max_reads, 8)
        for i in range(max_reads):
            ret, frm = cap.read()
            if ret and frm is not None and frm.size > 0 and frm.ndim == 3:
                read_ok = True
                if first_frame is None:
                    first_frame = frm.copy()
                last_frame = frm
                if i >= max(1, warmup_frames) - 1 and float(np.std(frm)) > 2.0:
                    break

        if not read_ok or last_frame is None:
            return CameraPortInfo(
                port=port,
                backend_name=backend_name,
                backend_id=backend_id,
                device_name=device_name,
                is_opened=True,
                can_read=False,
                score=10.0,
            )

        h, w = last_frame.shape[:2]
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        if not math.isfinite(fps) or fps <= 0.0 or fps > 240.0:
            fps = 30.0

        gray = cv2.cvtColor(last_frame, cv2.COLOR_BGR2GRAY)
        lum_mean = float(np.mean(gray))
        lum_std = float(np.std(gray))

        temporal_diff = 0.0
        if first_frame is not None and first_frame.shape == last_frame.shape:
            temporal_diff = float(np.mean(np.abs(last_frame.astype(np.float32) - first_frame.astype(np.float32))))

        b_ch = last_frame[:, :, 0].astype(np.float32)
        g_ch = last_frame[:, :, 1].astype(np.float32)
        r_ch = last_frame[:, :, 2].astype(np.float32)
        chroma_diff = float(np.mean(np.abs(r_ch - g_ch)) + np.mean(np.abs(b_ch - g_ch)))
        is_color = chroma_diff > 0.5

        is_live = (lum_std > 1.8) and (3.0 <= lum_mean <= 252.0)

        score = 35.0
        if is_live:
            score += 40.0
            # Reward live temporal sensor vitality (real camera shot noise / motion vs frozen splash image)
            if temporal_diff > 0.02:
                score += 5.0
        if is_color:
            score += 12.0
        if w >= 640 and h >= 480:
            score += 5.0
        if backend_name in ("DSHOW", "AVFOUNDATION", "V4L2"):
            score += 3.0

        return CameraPortInfo(
            port=port,
            backend_name=backend_name,
            backend_id=backend_id,
            device_name=device_name,
            is_opened=True,
            can_read=True,
            is_live_signal=is_live,
            is_color=is_color,
            width=w,
            height=h,
            fps=fps,
            luminance_mean=lum_mean,
            luminance_std=lum_std,
            temporal_diff=temporal_diff,
            score=score,
        )

    @classmethod
    def scan_available_ports(
        cls,
        max_index: int = 5,
        backends: Optional[List[Tuple[str, int]]] = None,
        active_port_info: Optional[CameraPortInfo] = None,
    ) -> List[CameraPortInfo]:
        """
        Scans camera port indices [0 .. max_index - 1] and returns all responsive ports
        sorted from highest optical signal quality to lowest.
        If `active_port_info` is provided, reuses its descriptor for the currently open port
        to avoid exclusive device handle contention on Windows DirectShow/MSMF.
        """
        os_names = cls.enumerate_os_camera_names()
        discovered: List[CameraPortInfo] = []
        consecutive_dead = 0

        for idx in range(max(1, int(max_index))):
            if active_port_info is not None and active_port_info.port == idx and active_port_info.can_read:
                discovered.append(active_port_info)
                consecutive_dead = 0
                continue
            dev_label = os_names[idx] if idx < len(os_names) else f"Camera Port #{idx}"
            info = cls.probe_port(port=idx, backends=backends, warmup_frames=2, device_name=dev_label)
            if info.is_opened or info.can_read:
                discovered.append(info)
                consecutive_dead = 0
            else:
                consecutive_dead += 1
                if idx >= 2 and consecutive_dead >= 3 and len(os_names) <= idx:
                    break

        discovered.sort(key=lambda x: x.score, reverse=True)
        return discovered

    @classmethod
    def find_best_camera_port(
        cls,
        preferred_source: Union[int, str] = "auto",
        max_index: int = 5,
    ) -> CameraPortInfo:
        """
        Resolves the best available camera port:
        - If `preferred_source` is a specific working port or file/URL, returns it (unless it is dead/unopened,
          in which case it automatically scans all ports to find a working fallback).
        - If `preferred_source` is 'auto', scans all ports and selects the highest-scoring live camera.
        """
        if isinstance(preferred_source, str) and preferred_source.strip().lower() not in ("auto", "scan", "default", ""):
            if preferred_source.strip().lower() == "demo":
                return cls.probe_port("demo")
            try:
                pref_idx = int(preferred_source)
                preferred_source = pref_idx
            except ValueError:
                return cls.probe_port(preferred_source)

        if isinstance(preferred_source, int):
            direct_info = cls.probe_port(preferred_source)
            if direct_info.can_read and direct_info.is_live_signal:
                return direct_info
            all_ports = cls.scan_available_ports(max_index=max(max_index, preferred_source + 1))
            if all_ports:
                return all_ports[0]
            return direct_info

        all_ports = cls.scan_available_ports(max_index=max_index)
        if all_ports:
            return all_ports[0]
        return CameraPortInfo(port=0, backend_name="NONE", backend_id=0, is_opened=False, can_read=False, score=0.0)


class SyntheticMultiTargetStream:
    """
    Built-in synthetic multi-target camera generator used when `--demo` is enabled
    or when testing multi-target tracking and HUD rendering without a physical webcam.
    Renders realistic shaded craniofacial structures so native ONNX YuNet detects
    every target reliably on 100% of frames.
    """

    def __init__(self, width: int = 640, height: int = 480, num_targets: int = 3):
        self.width = int(width)
        self.height = int(height)
        self.num_targets = max(1, int(num_targets))
        self.frame_idx = 0
        self._opened = True

    def isOpened(self) -> bool:
        return self._opened

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if not self._opened:
            return False, None
        self.frame_idx += 1
        frame = np.full((self.height, self.width, 3), 48, dtype=np.uint8)
        for i in range(self.num_targets):
            phase = self.frame_idx * 0.04 + i * 2.1
            cx = int(self.width * (0.20 + 0.30 * i) + 24.0 * math.sin(phase))
            cy = int(self.height * 0.50 + 14.0 * math.cos(phase * 0.8))
            cx = max(75, min(self.width - 75, cx))
            cy = max(90, min(self.height - 90, cy))

            # Neck & shoulders silhouette
            cv2.rectangle(frame, (cx - 22, cy + 48), (cx + 22, cy + 88), (130, 150, 180), -1)
            cv2.ellipse(frame, (cx, cy + 95), (68, 28), 0, 180, 360, (70, 85, 105), -1)

            # Hair & cranial crown
            cv2.ellipse(frame, (cx, cy - 10), (54, 68), 0, 0, 360, (32, 36, 44), -1)

            # Shaded facial oval + cheek highlight
            skin_color = (155, 180, 215) if (i % 2 == 0) else (140, 168, 202)
            cv2.ellipse(frame, (cx, cy + 4), (48, 60), 0, 0, 360, skin_color, -1)
            cv2.ellipse(frame, (cx - 18, cy + 10), (14, 18), 0, 0, 360, (165, 190, 224), -1)
            cv2.ellipse(frame, (cx + 18, cy + 10), (14, 18), 0, 0, 360, (165, 190, 224), -1)

            # Eyebrows
            cv2.ellipse(frame, (cx - 19, cy - 18), (13, 4), -6, 180, 360, (35, 40, 50), 3)
            cv2.ellipse(frame, (cx + 19, cy - 18), (13, 4), 6, 180, 360, (35, 40, 50), 3)

            # Ocular sockets, sclera, iris & pupil
            cv2.ellipse(frame, (cx - 19, cy - 8), (12, 7), 0, 0, 360, (125, 145, 175), -1)
            cv2.ellipse(frame, (cx + 19, cy - 8), (12, 7), 0, 0, 360, (125, 145, 175), -1)
            cv2.ellipse(frame, (cx - 19, cy - 8), (10, 5), 0, 0, 360, (238, 240, 242), -1)
            cv2.ellipse(frame, (cx + 19, cy - 8), (10, 5), 0, 0, 360, (238, 240, 242), -1)
            cv2.circle(frame, (cx - 19, cy - 8), 4, (45, 55, 70), -1)
            cv2.circle(frame, (cx + 19, cy - 8), 4, (45, 55, 70), -1)
            cv2.circle(frame, (cx - 19, cy - 8), 2, (15, 15, 20), -1)
            cv2.circle(frame, (cx + 19, cy - 8), 2, (15, 15, 20), -1)

            # Nasal bridge, tip & nostrils
            cv2.line(frame, (cx - 3, cy - 8), (cx - 5, cy + 14), (120, 142, 172), 2)
            cv2.line(frame, (cx + 3, cy - 8), (cx + 5, cy + 14), (120, 142, 172), 2)
            cv2.ellipse(frame, (cx, cy + 14), (9, 6), 0, 0, 360, (135, 160, 192), -1)
            cv2.circle(frame, (cx - 5, cy + 17), 2, (55, 65, 80), -1)
            cv2.circle(frame, (cx + 5, cy + 17), 2, (55, 65, 80), -1)

            # Upper/lower lips & oral commissure
            cv2.ellipse(frame, (cx, cy + 32), (17, 6), 0, 0, 360, (105, 110, 185), -1)
            cv2.line(frame, (cx - 16, cy + 32), (cx + 16, cy + 32), (65, 60, 120), 2)

        # Smooth slight optical sensor PSF so YuNet anchors see natural edge gradients
        frame = cv2.GaussianBlur(frame, (3, 3), 0.7)
        return True, frame

    def release(self) -> None:
        self._opened = False

    def set(self, prop_id: int, value: float) -> bool:
        return True

    def get(self, prop_id: int) -> float:
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self.width)
        if prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self.height)
        if prop_id == cv2.CAP_PROP_FPS:
            return 60.0
        return 0.0


class ThreadedCameraStream:
    """
    Zero-Lag Asynchronous Camera Capture Stream with Automatic Port Discovery & Failover.
    Continuously reads frames on a dedicated background daemon thread so the inference pipeline
    always receives the latest frame with zero I/O blocking.
    """

    def __init__(
        self,
        camera_source: Union[int, str] = "auto",
        max_scan_index: int = 5,
        use_threading: bool = True,
        allow_demo_fallback: bool = False,
    ):
        self.requested_source = camera_source
        self.max_scan_index = max_scan_index
        self.use_threading = use_threading
        self.allow_demo_fallback = allow_demo_fallback
        self._cap_lock = threading.Lock()
        self._lock = threading.Lock()

        if isinstance(camera_source, str) and camera_source.strip().lower() == "demo":
            self.cap: Any = SyntheticMultiTargetStream()
            self.port_info = CameraPortInfo(
                port="DEMO",
                backend_name="SYNTHETIC",
                backend_id=0,
                device_name="Synthetic Multi-Target Demo Stream",
                is_opened=True,
                can_read=True,
                is_live_signal=True,
                is_color=True,
                width=640,
                height=480,
                fps=60.0,
                score=50.0,
            )
        else:
            self.port_info = CameraPortScanner.find_best_camera_port(
                preferred_source=camera_source,
                max_index=max_scan_index,
            )
            self.cap = self._open_from_info(self.port_info)
            if (self.cap is None or not self.cap.isOpened()) and allow_demo_fallback:
                self.cap = SyntheticMultiTargetStream()
                self.port_info = CameraPortInfo(
                    port="DEMO",
                    backend_name="SYNTHETIC",
                    backend_id=0,
                    device_name="Synthetic Multi-Target Demo Stream",
                    is_opened=True,
                    can_read=True,
                    is_live_signal=True,
                    is_color=True,
                    width=640,
                    height=480,
                    fps=60.0,
                    score=50.0,
                )

        self._latest_frame: Optional[np.ndarray] = None
        self._latest_ret: bool = False
        self._frame_seq: int = 0
        self._last_read_seq: int = -1
        self._running: bool = False
        self._thread: Optional[threading.Thread] = None

        if self.cap is not None and self.cap.isOpened():
            with self._cap_lock:
                ret, frm = self.cap.read()
            if ret and frm is not None:
                self._latest_ret = True
                self._latest_frame = frm
                self._frame_seq = 1

            is_live_device = isinstance(self.port_info.port, int)
            if self.use_threading and is_live_device:
                self._running = True
                self._thread = threading.Thread(target=self._reader_loop, daemon=True)
                self._thread.start()

    @staticmethod
    def _open_from_info(info: CameraPortInfo) -> Optional[Any]:
        if isinstance(info.port, str) and info.port.strip().upper() == "DEMO":
            return SyntheticMultiTargetStream()
        with suppress_opencv_logs():
            if isinstance(info.port, int):
                if info.backend_id != 0:
                    cap = cv2.VideoCapture(info.port, info.backend_id)
                else:
                    cap = cv2.VideoCapture(info.port)
                if not cap.isOpened():
                    cap = cv2.VideoCapture(info.port)
            else:
                cap = cv2.VideoCapture(str(info.port))

            if cap is not None and cap.isOpened():
                try:
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    cap.set(cv2.CAP_PROP_FPS, 60)
                except Exception:
                    pass
            return cap

    def _reader_loop(self) -> None:
        consecutive_failures = 0
        while self._running:
            with self._cap_lock:
                if self.cap is None or not self.cap.isOpened():
                    ret, frm = False, None
                else:
                    ret, frm = self.cap.read()

            if ret and frm is not None and frm.size > 0:
                with self._lock:
                    self._latest_ret = True
                    self._latest_frame = frm
                    self._frame_seq += 1
                consecutive_failures = 0
            else:
                consecutive_failures += 1
                if consecutive_failures >= 25:
                    new_info = CameraPortScanner.find_best_camera_port("auto", max_index=self.max_scan_index)
                    if new_info.can_read:
                        with self._cap_lock:
                            old_cap = self.cap
                            self.port_info = new_info
                            self.cap = self._open_from_info(new_info)
                            if old_cap is not None:
                                try:
                                    old_cap.release()
                                except Exception:
                                    pass
                    consecutive_failures = 0
                time.sleep(0.01)

    def isOpened(self) -> bool:
        with self._cap_lock:
            return self.cap is not None and self.cap.isOpened()

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._running and self._thread is not None:
            for _ in range(15):
                with self._lock:
                    if self._frame_seq != self._last_read_seq and self._latest_frame is not None:
                        self._last_read_seq = self._frame_seq
                        return self._latest_ret, self._latest_frame.copy()
                time.sleep(0.002)
            with self._lock:
                if self._latest_frame is not None:
                    return self._latest_ret, self._latest_frame.copy()
                return False, None

        with self._cap_lock:
            if self.cap is None or not self.cap.isOpened():
                return False, None
            return self.cap.read()

    def switch_to_next_port(self) -> CameraPortInfo:
        """Cycles to the next available working camera port at runtime with thread-safe handle handoff."""
        ports = CameraPortScanner.scan_available_ports(
            max_index=self.max_scan_index,
            active_port_info=self.port_info,
        )
        if not ports:
            return self.port_info

        current_port = self.port_info.port
        next_info = ports[0]
        for idx, p in enumerate(ports):
            if p.port == current_port:
                next_info = ports[(idx + 1) % len(ports)]
                break

        if next_info.port != current_port:
            with self._cap_lock:
                old_cap = self.cap
                if old_cap is not None:
                    try:
                        old_cap.release()
                    except Exception:
                        pass
                self.port_info = next_info
                self.cap = self._open_from_info(next_info)
                if self.cap is not None and self.cap.isOpened():
                    ret, frm = self.cap.read()
                    if ret and frm is not None:
                        with self._lock:
                            self._latest_ret = True
                            self._latest_frame = frm
                            self._frame_seq += 1
        return self.port_info

    def release(self) -> None:
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=0.5)
        with self._cap_lock:
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None

