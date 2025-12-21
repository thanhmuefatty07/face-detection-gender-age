@echo off
title GPU Accelerated Face Detection (HYBRID Mode)
cd /d "%~dp0"

echo ========================================
echo   HYBRID MODE Face Detection
echo   Combines: CPU + GPU for Max Speed
echo ========================================
echo.
echo Controls: Q = Quit, L = Lock Face, U = Unlock
echo.

python scripts/main_gpu.py --source 0 --track --device HYBRID
pause
