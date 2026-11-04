@echo off
setlocal EnableDelayedExpansion
chcp 65001 >nul 2>&1
title Biometric Multi-Target Locker [Auto-Bootstrap + Run]
cd /d "%~dp0"

echo ========================================================================
echo   BIOMETRIC MULTI-TARGET LOCKER (ConvNeXt-V2 + YuNet + OpenVINO/CUDA)
echo   One-Click Universal Launcher - Auto Environment + Hardware Setup
echo ========================================================================

:: ------------------------------------------------------------------------
:: 1. Locate or Auto-Install Python Runtime (>= 3.9)
:: ------------------------------------------------------------------------
set "PY_EXE="

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 9) else 1)" >nul 2>&1
    if !errorlevel! equ 0 (
        set "PY_EXE=.venv\Scripts\python.exe"
    )
)

if "%PY_EXE%"=="" (
    for %%P in (python py) do (
        if "!PY_EXE!"=="" (
            %%P -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 9) else 1)" >nul 2>&1
            if !errorlevel! equ 0 (
                for /f "delims=" %%X in ('%%P -c "import sys; print(sys.executable)" 2^>nul') do (
                    if "!PY_EXE!"=="" set "PY_EXE=%%X"
                )
            )
        )
    )
)

if "%PY_EXE%"=="" (
    echo [1/4] Chua tim thay Python ^>= 3.9 tren thiet bi. Dang tu dong tai Python 3.11 qua winget...
    winget install -e --id Python.Python.3.11 --scope user --accept-package-agreements --accept-source-agreements
    set "PATH=%LOCALAPPDATA%\Programs\Python\Python311;%LOCALAPPDATA%\Programs\Python\Python311\Scripts;%PATH%"
    if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" (
        set "PY_EXE=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
    ) else (
        echo [LOI] Khong the tu dong cai dat Python. Vui long cai Python 3.11 tu https://www.python.org/
        pause
        exit /b 1
    )
)

echo [1/4] Python Runtime : "%PY_EXE%"

:: ------------------------------------------------------------------------
:: 2. Fast Dependency Check & Automatic Environment Bootstrap
:: ------------------------------------------------------------------------
"%PY_EXE%" -c "import torch, timm, cv2, numpy, scipy, PIL, yaml, openvino" >nul 2>&1
if %errorlevel% neq 0 (
    echo [2/4] Phat hien thiet bi chua du thu vien moi truong. Dang khoi tao .venv va tu dong tai tai nguyen...
    if not exist ".venv\Scripts\python.exe" (
        "%PY_EXE%" -m venv .venv
        if !errorlevel! neq 0 (
            echo [CANH BAO] Khong tao duoc .venv, chuyen sang cai dat truc tiep tren Python hien tai...
        ) else (
            set "PY_EXE=.venv\Scripts\python.exe"
        )
    ) else (
        set "PY_EXE=.venv\Scripts\python.exe"
    )

    "%PY_EXE%" -m pip install --upgrade pip wheel setuptools --quiet

    :: Detect NVIDIA CUDA GPU to install CUDA-accelerated PyTorch when available
    powershell -NoProfile -Command "if (Get-CimInstance Win32_VideoController | Where-Object { $_.Name -match 'NVIDIA' }) { exit 0 } else { exit 1 }" >nul 2>&1
    if !errorlevel! equ 0 (
        echo       - Phat hien GPU NVIDIA: Dang tai PyTorch CUDA + cac thu vien...
        "%PY_EXE%" -m pip install torch --index-url https://download.pytorch.org/whl/cu121
    )

    echo       - Dang cai dat cac goi trong requirements.txt...
    "%PY_EXE%" -m pip install -r requirements.txt
    if !errorlevel! neq 0 (
        echo [LOI] Cai dat thu vien that bai. Vui long kiem tra ket noi Internet.
        pause
        exit /b 1
    )
    echo [2/4] Hoan tat cai dat moi truong tu dong!
) else (
    echo [2/4] Moi truong thu vien da san sang.
)

:: ------------------------------------------------------------------------
:: 3. Verify & Auto-Fetch Required Model Weights
:: ------------------------------------------------------------------------
if not exist "models" mkdir "models"

if not exist "models\face_detection_yunet_2023mar.onnx" (
    echo [3/4] Dang tu dong tai model phat hien khuon mat YuNet ONNX...
    "%PY_EXE%" -c "import urllib.request; urllib.request.urlretrieve('https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx', 'models/face_detection_yunet_2023mar.onnx')"
)

if not exist "models\convnextv2_tiny_biometrics.pt" (
    echo [3/4] Dang tu dong tai model ConvNeXt-V2 Biometrics tu kho luu tru...
    "%PY_EXE%" -c "import urllib.request; urllib.request.urlretrieve('https://github.com/thanhmuefatty07/face-detection-gender-age/raw/main/models/convnextv2_tiny_biometrics.pt', 'models/convnextv2_tiny_biometrics.pt')"
)

echo [3/4] Model weights da san sang.

:: ------------------------------------------------------------------------
:: 4. Launch Biometric Multi-Target Locker (Auto Camera Port & Hardware)
:: ------------------------------------------------------------------------
echo [4/4] Dang khoi dong he thong (Tu dong do tim cong Camera va toi uu phan cung)...
echo.
"%PY_EXE%" run_client.py --camera auto --device auto %*

if %errorlevel% neq 0 (
    echo.
    echo [!] Chuong trinh ket thuc voi ma loi %errorlevel%.
    pause
)
endlocal
