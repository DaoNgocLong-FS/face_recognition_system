@echo off
REM ================================================
REM 4_test.bat
REM Test model đã train trên webcam
REM ================================================

call .venv\Scripts\activate

REM Lệnh test webcam (nhấn Q để thoát)
yolo predict ^
    model=runs\detect\face_yolov8n\weights\best.pt ^
    source=0 ^
    show=true ^
    conf=0.4 ^
    device=0

pause
