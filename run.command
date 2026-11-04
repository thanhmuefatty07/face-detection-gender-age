#!/usr/bin/env bash
# ==============================================================================
# One-Click Biometric Target Locker Launcher (macOS Finder .command & Linux)
# ==============================================================================
# Automatically locates/installs Python 3, sets up .venv, installs dependencies,
# verifies/downloads model weights, scans cameras (AVFoundation / Continuity
# Camera / USB / V4L2), and launches the real-time HUD.
# ==============================================================================

set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
cd "$ROOT_DIR" || exit 1

# Ensure Homebrew, Python.framework, pyenv, conda & local binaries are in PATH when launched from macOS Finder GUI
export PATH="/opt/homebrew/bin:/usr/local/bin:/Library/Frameworks/Python.framework/Versions/3.12/bin:/Library/Frameworks/Python.framework/Versions/3.11/bin:/Library/Frameworks/Python.framework/Versions/3.10/bin:$HOME/.pyenv/shims:$HOME/miniconda3/bin:$HOME/anaconda3/bin:$PATH"
export PYTHONIOENCODING="utf-8"
export OPENCV_LOG_LEVEL="OFF"

echo "========================================================================"
echo "  BIOMETRIC TARGET LOCKER - ONE-CLICK LAUNCHER (macOS / Linux)"
echo "========================================================================"
echo ""

VENV_PY="$ROOT_DIR/.venv/bin/python"
SYS_PY=""

check_py_range() {
    local py_bin="$1"
    local min_minor="$2"
    local max_minor="$3"
    "$py_bin" -c "import sys; raise SystemExit(0 if (sys.version_info[0]==3 and ${min_minor}<=sys.version_info[1]<=${max_minor}) else 1)" >/dev/null 2>&1
}

if [ -x "$VENV_PY" ]; then
    if "$VENV_PY" -c "import sys" >/dev/null 2>&1; then
        SYS_PY="$VENV_PY"
    fi
fi

# 1. Prefer Python 3.9..3.13 (full prebuilt wheel support for PyTorch + OpenCV + OpenVINO)
if [ -z "$SYS_PY" ]; then
    for cand in \
        /opt/homebrew/bin/python3.12 \
        /opt/homebrew/bin/python3.11 \
        /opt/homebrew/bin/python3.10 \
        /usr/local/bin/python3.12 \
        /usr/local/bin/python3.11 \
        /usr/local/bin/python3.10 \
        python3.12 python3.11 python3.10 python3.9 \
        /opt/homebrew/bin/python3 \
        /usr/bin/python3 \
        /usr/local/bin/python3 \
        python3; do
        if command -v "$cand" >/dev/null 2>&1; then
            resolved="$(command -v "$cand")"
            if check_py_range "$resolved" 9 13; then
                SYS_PY="$resolved"
                break
            fi
        fi
    done
fi

# 2. Fallback to any Python >= 3.9 (e.g., Python 3.14 / 3.15)
if [ -z "$SYS_PY" ]; then
    for cand in /opt/homebrew/bin/python3 /usr/local/bin/python3 /usr/bin/python3 python3; do
        if command -v "$cand" >/dev/null 2>&1; then
            resolved="$(command -v "$cand")"
            if check_py_range "$resolved" 9 99; then
                SYS_PY="$resolved"
                break
            fi
        fi
    done
fi

# 3. Auto-install Python if missing
if [ -z "$SYS_PY" ]; then
    echo "[1/4] Python 3.9+ not found. Attempting automatic installation..."
    if [ "$(uname -s)" = "Darwin" ] && command -v brew >/dev/null 2>&1; then
        brew install python@3.11
        if [ -x "/opt/homebrew/opt/python@3.11/bin/python3.11" ]; then
            SYS_PY="/opt/homebrew/opt/python@3.11/bin/python3.11"
        elif command -v python3.11 >/dev/null 2>&1; then
            SYS_PY="$(command -v python3.11)"
        fi
    elif command -v apt-get >/dev/null 2>&1; then
        sudo apt-get update && sudo apt-get install -y python3 python3-venv python3-pip
        SYS_PY="$(command -v python3)"
    fi
fi

if [ -z "$SYS_PY" ]; then
    echo "[ERROR] Python 3.9+ is required. Please install Python 3 (https://www.python.org/downloads/)"
    exit 1
fi

echo "[1/4] Using Python interpreter: $SYS_PY ($("$SYS_PY" --version 2>&1))"

# 4. Bootstrap virtual environment (.venv)
if [ ! -x "$VENV_PY" ]; then
    echo "[2/4] Creating virtual environment (.venv)..."
    "$SYS_PY" -m venv "$ROOT_DIR/.venv"
    if [ ! -x "$VENV_PY" ]; then
        echo "[WARN] venv creation failed; using $SYS_PY directly."
        VENV_PY="$SYS_PY"
    fi
else
    echo "[2/4] Virtual environment (.venv) is ready."
fi

# 5. Verify / install dependencies
if ! "$VENV_PY" -c "import torch, timm, cv2, numpy, scipy, yaml" >/dev/null 2>&1; then
    echo "[3/4] Installing required packages (PyTorch, timm, OpenCV, NumPy, SciPy)..."
    "$VENV_PY" -m pip install --upgrade pip setuptools wheel
    "$VENV_PY" -m pip install -r "$ROOT_DIR/requirements.txt"
    if ! "$VENV_PY" -c "import torch, timm, cv2, numpy, scipy" >/dev/null 2>&1; then
        echo "[ERROR] Package installation failed. Check internet connection."
        exit 1
    fi
else
    echo "[3/4] All required packages are installed."
fi

# 6. Verify / download model weights
mkdir -p "$ROOT_DIR/models"
if [ ! -f "$ROOT_DIR/models/face_detection_yunet_2023mar.onnx" ]; then
    echo "[3.5/4] Downloading YuNet face detector model..."
    curl -L --fail -o "$ROOT_DIR/models/face_detection_yunet_2023mar.onnx" \
        "https://github.com/thanhmuefatty07/face-detection-gender-age/raw/main/models/face_detection_yunet_2023mar.onnx"
fi

if [ ! -f "$ROOT_DIR/models/convnextv2_tiny_biometrics.pt" ]; then
    echo "[3.5/4] Downloading ConvNeXt-V2 Tiny biometric model..."
    curl -L --fail -o "$ROOT_DIR/models/convnextv2_tiny_biometrics.pt" \
        "https://github.com/thanhmuefatty07/face-detection-gender-age/raw/main/models/convnextv2_tiny_biometrics.pt"
fi

# 7. Launch HUD
echo "[4/4] Starting Biometric Target Locker HUD..."
echo "      (Controls: 'q' or ESC = Quit | 'c' = Switch Camera Port)"
echo ""

"$VENV_PY" "$ROOT_DIR/run_client.py" --camera auto --device auto "$@"
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo ""
    echo "[WARN] Application exited with code $EXIT_CODE."
    echo "       Tip: run './run.command --demo' to test without a camera."
fi

exit $EXIT_CODE
