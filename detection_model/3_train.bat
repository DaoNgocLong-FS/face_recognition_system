@echo off
REM ================================================
REM 3_train.bat
REM Train YOLO face detection trên GPU
REM ================================================
REM
REM Chỉnh các tham số bên dưới cho phù hợp với GPU của bạn:
REM   BATCH:  16 (8GB VRAM) / 32 (12GB) / 64 (24GB) / giảm xuống 8-12 nếu OOM
REM   IMGSZ:  640 (chuẩn) / 512 (tiết kiệm VRAM)
REM   EPOCHS: 80 (đủ) / 100 (tối đa) / 30 (nhanh, để test pipeline)
REM ================================================

call .venv\Scripts\activate

set BATCH=16
set IMGSZ=640
set EPOCHS=80
set MODEL=yolov8n.pt
set NAME=face_yolov8n

python detection\train_yolo_face.py ^
    --data datasets\widerface_yolo\widerface.yaml ^
    --model %MODEL% ^
    --epochs %EPOCHS% ^
    --imgsz %IMGSZ% ^
    --batch %BATCH% ^
    --device 0 ^
    --workers 4 ^
    --amp ^
    --name %NAME% ^
    --export-onnx

if errorlevel 1 (
    echo.
    echo [LỖI] Train thất bại.
    echo Nếu OOM, giảm BATCH xuống (mở file bat sửa).
    echo Nếu lỗi multiprocessing, sửa --workers 4 thành --workers 0.
    pause
    exit /b 1
)

echo.
echo [OK] Train xong. Kết quả:
echo   runs\detect\%NAME%\weights\best.pt
echo   runs\detect\%NAME%\results.png  (chèn vào báo cáo)
pause
