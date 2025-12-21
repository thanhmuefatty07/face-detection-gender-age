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
python scripts/main.py --source 0 --track --conf 0.6
pause
