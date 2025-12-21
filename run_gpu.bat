@echo off
title Face Detection (Auto-detect GPU)
cd /d "%~dp0"

echo ========================================
echo   Face Detection with GPU Acceleration
echo ========================================
echo.
echo Detecting GPU type...

:: Check for AMD GPU
wmic path win32_videocontroller get name | findstr /i "AMD Radeon" >nul
if %errorlevel%==0 (
    echo [AMD GPU Detected] Using DirectML acceleration
    set GPU_BACKEND=AMD
) else (
    :: Check for Intel GPU
    wmic path win32_videocontroller get name | findstr /i "Intel" >nul
    if %errorlevel%==0 (
        echo [Intel GPU Detected] Using OpenVINO HYBRID mode
        set GPU_BACKEND=INTEL
    ) else (
        echo [No compatible GPU] Using CPU mode
        set GPU_BACKEND=CPU
    )
)

echo.
echo Controls: Q = Quit, L = Lock Face, U = Unlock
echo.

:: Run based on detected GPU
if "%GPU_BACKEND%"=="AMD" (
    python scripts/main_gpu.py --source 0 --track --device CPU --conf 0.7
    echo.
    echo Note: AMD GPUs use CPU mode with optimized settings.
    echo For better AMD support, install: pip install onnxruntime-directml
) else if "%GPU_BACKEND%"=="INTEL" (
    python scripts/main_gpu.py --source 0 --track --device HYBRID --conf 0.7
) else (
    python scripts/main_gpu.py --source 0 --track --device CPU --conf 0.7
)

pause
