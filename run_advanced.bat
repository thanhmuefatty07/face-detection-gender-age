@echo off
title Advanced Face Detection
cd /d "%~dp0"

echo ========================================
echo   Advanced Face Detection (Tracking)
echo ========================================
echo.
echo Controls: Q = Quit, L = Lock Face, U = Unlock
echo.

REM Use system Python (works with PyTorch)
python scripts/inference.py --source 0 --track
pause
