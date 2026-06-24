@echo off
REM ================================================
REM 1_download_dataset.bat
REM Tải WIDER FACE qua torchvision (cần Internet)
REM ================================================

call .venv\Scripts\activate

if not exist datasets mkdir datasets

echo Đang tải WIDER FACE (sẽ mất 10-30 phút tuỳ mạng)...
python -c "from torchvision.datasets import WIDERFace; WIDERFace(root='./datasets', split='train', download=True); WIDERFace(root='./datasets', split='val', download=True)"

if errorlevel 1 (
    echo.
    echo [LỖI] Tải qua torchvision thất bại.
    echo Tải tay từ http://shuoyang1213.me/WIDERFACE/
    echo Cần 3 file: WIDER_train.zip, WIDER_val.zip, wider_face_split.zip
    echo Giải nén tất cả vào: datasets\widerface\
    pause
    exit /b 1
)

echo.
echo [OK] Đã tải xong. Kiểm tra:
dir datasets\widerface
pause
