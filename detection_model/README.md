# Face Detection — YOLOv8 + WIDER FACE (Windows + GPU)

Hướng dẫn build model 1 (Face Detection) cho hệ thống chấm công bằng nhận diện khuôn mặt.

```
WIDER FACE  →  YOLO format  →  Train YOLOv8n  →  best.pt
   ↓                                              ↓
3 zip files                                  Dùng cho pipeline
                                             chấm công sau này
```

---

## Cấu trúc thư mục

```
attendance-detection/
├── detection/
│   ├── check_gpu.py                 # Kiểm tra GPU + gợi ý config
│   ├── convert_widerface_to_yolo.py # Chuyển annotation sang YOLO
│   ├── train_yolo_face.py           # Train chính
│   └── infer_detector.py            # Test trên ảnh
├── datasets/                        # Chỗ chứa dataset (sẽ tải về đây)
├── 1_download_dataset.bat           # 4 batch file giúp chạy nhanh trên Windows
├── 2_convert_dataset.bat
├── 3_train.bat
├── 4_test_webcam.bat
├── requirements.txt
└── README.md  ← bạn đang đọc
```

---

## Bước 0 — Chuẩn bị môi trường Windows + GPU

### 0.1 Cập nhật driver NVIDIA

1. Mở Command Prompt, gõ `nvidia-smi` để xem driver hiện tại.
2. Nếu driver cũ (< 525), cập nhật từ https://www.nvidia.com/Download/index.aspx

`nvidia-smi` phải in ra thông tin GPU + dòng "CUDA Version: 12.x" (đó là CUDA cao nhất driver hỗ trợ, không phải CUDA đã cài).

> **KHÔNG cần cài CUDA Toolkit riêng.** PyTorch wheel `cu121` đã bundle sẵn CUDA runtime. Chỉ cần driver mới.

### 0.2 Cài Python 3.10 hoặc 3.11

Tải từ python.org. Khi cài, **tích chọn "Add Python to PATH"**.

Kiểm tra:
```cmd
python --version
```

### 0.3 Tạo virtual environment

```cmd
cd attendance-detection
python -m venv .venv
.venv\Scripts\activate
```

(Mỗi lần mở terminal mới, phải chạy `.venv\Scripts\activate` lại để vào venv.)

### 0.4 Cài PyTorch CUDA + Ultralytics

Quan trọng: **không pip install torch trước rồi mới cài CUDA build sau** — sẽ bị conflict. Cài đúng 1 lần:

```cmd
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Driver cũ hơn? Đổi `cu121` thành `cu118`.

### 0.5 Verify GPU

```cmd
python detection\check_gpu.py
```

Output kỳ vọng:
```
PyTorch built with CUDA: 12.1
[OK] CUDA available — phát hiện 1 GPU:

  GPU 0: NVIDIA GeForce RTX 3060
    VRAM         : 12.0 GB
    Compute cap. : 8.6
    SM count     : 28

Testing GPU compute...
  matmul 2048x2048 OK

============== KHUYẾN NGHỊ CONFIG TRAIN ==============
GPU tier      : HIGH (12-20GB)
Suggested cmd : python detection\train_yolo_face.py ^
    --data datasets\widerface_yolo\widerface.yaml ^
    --model yolov8n.pt --epochs 80 --imgsz 640 --batch 32 ^
    --device 0 --workers 4 --amp --name face_yolov8n
```

Nếu thấy `[X] torch.cuda.is_available() == False`, script sẽ in chỉ dẫn fix.

---

## Bước 1 — Tải WIDER FACE

WIDER FACE có 3 file cần thiết:
- `WIDER_train.zip` (~1.4 GB) — ảnh train
- `WIDER_val.zip` (~350 MB) — ảnh val
- `wider_face_split.zip` (~3 MB) — annotations (link "Face annotations" trên trang gốc)

(Không cần `WIDER_test.zip` — test không công bố nhãn.)

### Cách 1: Tự động qua torchvision

Chạy:
```cmd
1_download_dataset.bat
```

Hoặc thủ công:
```cmd
.venv\Scripts\activate
mkdir datasets
python -c "from torchvision.datasets import WIDERFace; WIDERFace(root='./datasets', split='train', download=True); WIDERFace(root='./datasets', split='val', download=True)"
```

### Cách 2: Tải tay (nếu cách 1 lỗi SSL)

1. Vào http://shuoyang1213.me/WIDERFACE/
2. Tải `WIDER_train.zip`, `WIDER_val.zip` (link Google Drive)
3. Scroll xuống tìm dòng **"Face annotations"**, tải file zip nhỏ
4. Giải nén cả 3 vào `datasets\widerface\`

### Kết quả mong đợi

```cmd
dir datasets\widerface
```
Phải thấy 3 thư mục:
- `WIDER_train`
- `WIDER_val`
- `wider_face_split`

```cmd
dir datasets\widerface\wider_face_split
```
Phải thấy file `wider_face_train_bbx_gt.txt` và `wider_face_val_bbx_gt.txt`.

---

## Bước 2 — Convert sang YOLO format

```cmd
2_convert_dataset.bat
```

Hoặc thủ công:
```cmd
python detection\convert_widerface_to_yolo.py --widerface-root datasets\widerface --output datasets\widerface_yolo
```

Mất 5-15 phút. Output:
```
datasets\widerface_yolo\
├── images\
│   ├── train\*.jpg
│   └── val\*.jpg
├── labels\
│   ├── train\*.txt   (mỗi dòng: 0 x_center y_center w h)
│   └── val\*.txt
└── widerface.yaml
```

Verify YAML config:
```cmd
type datasets\widerface_yolo\widerface.yaml
```
Phải thấy:
```yaml
path: C:/.../attendance-detection/datasets/widerface_yolo
train: images/train
val: images/val
names:
  0: face
```

---

## Bước 3 — Train

### Cách 1: Dùng batch file

Mở `3_train.bat` bằng Notepad, **chỉnh tham số nếu cần** rồi double-click:

```bat
set BATCH=16          ← chỉnh theo VRAM
set IMGSZ=640
set EPOCHS=80
set MODEL=yolov8n.pt  ← yolov8n (nhẹ) / yolov8s (mạnh hơn)
set NAME=face_yolov8n
```

### Cách 2: Lệnh trực tiếp

```cmd
.venv\Scripts\activate

python detection\train_yolo_face.py ^
    --data datasets\widerface_yolo\widerface.yaml ^
    --model yolov8n.pt ^
    --epochs 80 ^
    --imgsz 640 ^
    --batch 16 ^
    --device 0 ^
    --workers 4 ^
    --amp ^
    --name face_yolov8n ^
    --export-onnx
```

### Cấu hình theo VRAM

| VRAM | BATCH | IMGSZ | EPOCHS | MODEL | Thời gian dự kiến |
|---|---|---|---|---|---|
| 24 GB (RTX 4090/3090) | 64 | 640 | 100 | yolov8s.pt | ~2-3h |
| 12-16 GB (RTX 4070/3060Ti) | 32 | 640 | 80 | yolov8n.pt | ~3-4h |
| 8-10 GB (RTX 3060/3070/4060) | 16 | 640 | 80 | yolov8n.pt | ~4-6h |
| 6 GB (GTX 1660/RTX 2060) | 12 | 640 | 60 | yolov8n.pt | ~6-8h |
| 4 GB (GTX 1650/RTX 3050) | 8 | 512 | 50 | yolov8n.pt | ~8-12h |

### Theo dõi trong khi train

Mỗi epoch in 1 dòng:
```
Epoch  GPU_mem  box_loss  cls_loss  dfl_loss  Instances  Size
1/80   4.2G     1.823     1.456     1.234     127        640
...
Class  Images  Instances  P     R     mAP50  mAP50-95
all    3226    39707      0.85  0.72  0.78   0.45
```

`box_loss` giảm + `mAP50` tăng dần = đang học tốt.

Mở thêm 1 terminal khác để xem GPU usage realtime:
```cmd
nvidia-smi -l 2
```

### Output sau khi train

```
runs\detect\face_yolov8n\weights\best.pt    ← model tốt nhất (dùng cho pipeline)
runs\detect\face_yolov8n\weights\last.pt
runs\detect\face_yolov8n\results.png        ← biểu đồ loss/mAP (chèn báo cáo)
runs\detect\face_yolov8n\confusion_matrix.png
runs\detect\face_yolov8n\val_batch0_pred.jpg  ← visualize prediction
runs\detect\face_yolov8n\weights\best.onnx  ← (nếu --export-onnx)
```

---

## Bước 4 — Test model

### Test webcam

```cmd
4_test_webcam.bat
```

Hoặc:
```cmd
yolo predict model=runs\detect\face_yolov8n\weights\best.pt source=0 show=true conf=0.4 device=0
```

Nhấn `Q` để thoát.

### Test 1 ảnh

```cmd
python detection\infer_detector.py ^
    --weights runs\detect\face_yolov8n\weights\best.pt ^
    --source D:\path\to\your_photo.jpg ^
    --conf 0.4 ^
    --device 0
```

Mở `runs\infer_detector\your_photo.jpg` xem kết quả.

### Test thư mục ảnh

```cmd
python detection\infer_detector.py ^
    --weights runs\detect\face_yolov8n\weights\best.pt ^
    --source D:\path\to\photos_folder ^
    --device 0
```

### Đánh giá val lần cuối (lấy số đẹp cho báo cáo)

```cmd
yolo val model=runs\detect\face_yolov8n\weights\best.pt data=datasets\widerface_yolo\widerface.yaml imgsz=640 device=0
```

In ra mAP@0.5, mAP@0.5:0.95, Precision, Recall, FPS — copy vào báo cáo.

---

## Số liệu kỳ vọng

YOLOv8n trên WIDER FACE (chuẩn nghiên cứu):
- **mAP@0.5**       ≈ 0.88 – 0.92
- **mAP@0.5:0.95**  ≈ 0.55 – 0.62
- **Precision**     ≈ 0.85 – 0.90
- **Recall**        ≈ 0.78 – 0.85
- **FPS** (RTX 3060, imgsz=640): ~120-150 FPS

Nếu mAP@0.5 < 0.7 sau 50+ epoch → có vấn đề (xem Troubleshooting).

---

## Troubleshooting

### `torch.cuda.is_available() == False` mặc dù có GPU

```cmd
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
python detection\check_gpu.py
```

### CUDA out of memory

Giảm theo thứ tự:
1. `--batch 16` → `8` → `4`
2. `--imgsz 640` → `512` → `416`
3. Bỏ `--amp` (đôi khi gây OOM ngược, nhưng hiếm)

### `BrokenPipeError` / `RuntimeError: DataLoader worker exited unexpectedly`

Lỗi multiprocessing trên Windows. Đặt:
```cmd
--workers 0
```

### Train rất chậm dù có GPU

Mở terminal khác chạy `nvidia-smi -l 2` lúc đang train:
- GPU util < 50% → bottleneck data loading. Tăng `--workers` (nếu Windows không lỗi). Hoặc thêm `cache='ram'` trong `train_yolo_face.py` (cần ≥ 32 GB RAM).
- GPU util > 90% → đã max, đành chấp nhận tốc độ này.

### Loss bị NaN

LR quá cao hoặc batch quá nhỏ. Mở `train_yolo_face.py`, trong `model.train(...)` thêm:
```python
lr0=0.001,
```

### mAP rất thấp dù train lâu

1. Mở `runs\detect\face_yolov8n\train_batch0.jpg` — kiểm tra bbox có vẽ đúng lên mặt không. Nếu sai → bug ở convert.
2. Mở vài file label random trong `datasets\widerface_yolo\labels\train\` — số phải trong khoảng 0-1.
3. Nếu cả 2 OK mà mAP vẫn thấp → tăng epochs lên 150-200.

### Resume training khi bị crash

```cmd
python detection\train_yolo_face.py --data datasets\widerface_yolo\widerface.yaml --resume --name face_yolov8n --device 0
```

Sẽ tìm `runs\detect\face_yolov8n\weights\last.pt` để chạy tiếp.

---

## Tổng kết: chạy tuần tự 4 bước

```cmd
:: Bước 0 (1 lần duy nhất): setup env + cài deps
cd attendance-detection
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python detection\check_gpu.py        ← phải thấy [OK] CUDA available

:: Bước 1: download dataset
1_download_dataset.bat

:: Bước 2: convert
2_convert_dataset.bat

:: Bước 3: train (chỉnh batch trong file .bat trước nếu cần)
3_train.bat

:: Bước 4: test
4_test_webcam.bat
```

Sau khi xong, file model duy nhất cần giữ là:
```
runs\detect\face_yolov8n\weights\best.pt
```
Đây là input cho module Recognition và pipeline chấm công ở giai đoạn sau.
