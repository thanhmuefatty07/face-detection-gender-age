@echo off
title Face Detection (Auto-detect GPU)
cd /d "%~dp0"

echo ========================================
echo   Face Detection with GPU Acceleration
echo ========================================
echo.
echo Detecting GPU type...

:: Detect GPU type
set GPU_TYPE=NONE
set DEVICE=CPU

:: Check for AMD GPU
wmic path win32_videocontroller get name 2>nul | findstr /i "AMD Radeon" >nul
if %errorlevel%==0 (
    set GPU_TYPE=AMD
    set DEVICE=AMD
    echo [AMD GPU Detected]
    goto :run
)

:: Check for Intel GPU  
wmic path win32_videocontroller get name 2>nul | findstr /i "Intel" >nul
if %errorlevel%==0 (
    set GPU_TYPE=INTEL
    set DEVICE=HYBRID
    echo [Intel GPU Detected]
    goto :run
)

:: No GPU found
echo [No compatible GPU] Using CPU mode
set GPU_TYPE=CPU
set DEVICE=CPU

:run
echo.
echo ========================================
echo   Configuration:
echo   - GPU Type: %GPU_TYPE%
echo   - Device Mode: %DEVICE%
echo   - Confidence: 0.6 (optimal)
echo ========================================
echo.
echo Controls:
echo   Q = Quit
echo   L = Lock Face (tracking)
echo   U = Unlock
echo   T = Toggle tracking
echo.

:: Run the application
if "%GPU_TYPE%"=="AMD" (
    echo Starting with AMD DirectML acceleration...
    echo Note: Requires onnxruntime-directml
    echo       Install: pip install onnxruntime-directml
    echo.
    python scripts/main_gpu.py --source 0 --track --device AMD --conf 0.6
) else if "%GPU_TYPE%"=="INTEL" (
    echo Starting with Intel OpenVINO HYBRID mode...
    python scripts/main_gpu.py --source 0 --track --device HYBRID --conf 0.6
) else (
    echo Starting with CPU mode...
    python scripts/main_gpu.py --source 0 --track --device CPU --conf 0.6
)

echo.
if %errorlevel% neq 0 (
    echo ========================================
    echo   ERROR: Application crashed!
    echo ========================================
    echo.
    if "%GPU_TYPE%"=="AMD" (
        echo For AMD GPU support, install DirectML:
        echo   pip install onnxruntime-directml
        echo.
    )
)

pause
