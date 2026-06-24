@echo off
REM ================================================
REM 2_convert_dataset.bat
REM Chuyển WIDER FACE sang format YOLO
REM ================================================

call .venv\Scripts\activate

python detection\convert_widerface_to_yolo.py ^
    --widerface-root datasets\widerface ^
    --output datasets\widerface_yolo

if errorlevel 1 (
    echo [LỖI] Convert thất bại.
    pause
    exit /b 1
)

echo.
echo [OK] Đã convert xong. File config:
type datasets\widerface_yolo\widerface.yaml
pause
