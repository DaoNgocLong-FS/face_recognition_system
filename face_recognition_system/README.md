# Face Recognition Attendance System

Hệ thống điểm danh / nhận diện khuôn mặt qua camera, kiến trúc 6 bước:

| Bước | Module | Công nghệ chính |
|------|--------|-----------------|
| 1. Detection + Alignment | `src/face_processor.py` | InsightFace (RetinaFace) + MediaPipe |
| 2. Embedding 512-D | `src/face_processor.py` | InsightFace ArcFace (buffalo_l) |
| 3. Liveness / Anti-spoofing | `src/liveness.py` | MediaPipe Face Mesh (EAR blink) + texture check |
| 4. Vector Database | `src/vector_store.py` | FAISS (IndexFlatIP, cosine) |
| 5. Backend + Logs | `api/main.py`, `src/database.py` | FastAPI + SQLAlchemy + SQLite/PostgreSQL |
| 6. Optimization | `scripts/export_onnx.py` | ONNX Runtime / TensorRT-ready |

---

## 1. Cài đặt

```bash
# Python 3.10+
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

## 2. Đăng ký khuôn mặt

### Cách A — qua webcam (tương tác)
```bash
python apps/register_face.py --name "Nguyen Van A" --employee-id NV001
```
Nhìn vào camera, nhấn **SPACE** để chụp ~5 ảnh ở các góc khác nhau, **Q** để thoát.

### Cách B — từ thư mục ảnh
```bash
python apps/enroll_from_folder.py --folder ./photos
```
Cấu trúc thư mục:
```
photos/
├── NV001_Nguyen_Van_A/
│   ├── 1.jpg
│   └── 2.jpg
└── NV002_Tran_Thi_B/
    └── 1.jpg
```

## 3. Chạy điểm danh real-time

```bash
python apps/camera_checkin.py
```
Phím tắt: `Q` thoát, `S` chụp screenshot.

## 4. Chạy API server

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
Mở Swagger UI: <http://localhost:8000/docs>

Endpoints chính:
- `POST /register` — đăng ký khuôn mặt mới (multipart upload nhiều ảnh)
- `POST /checkin` — gửi 1 ảnh, trả về `{employee_id, name, similarity, liveness}`
- `GET  /logs` — lịch sử điểm danh, lọc theo ngày / nhân viên
- `GET  /employees` — danh sách nhân viên đã đăng ký
- `DELETE /employees/{id}` — xoá khỏi vector DB và Postgres

## 5. Docker

```bash
docker compose up --build
```
Mặc định: API chạy ở `:8000`, Postgres ở `:5432`, dữ liệu FAISS persist ở `./data/`.

## 6. Tối ưu cho production

| Vấn đề | Cách xử lý |
|--------|------------|
| Latency CPU cao | `python scripts/export_onnx.py` rồi đặt `USE_ONNX=1` |
| Edge device (Jetson) | Convert ONNX → TensorRT INT8 |
| Hàng triệu nhân viên | Đổi FAISS → Milvus / Pinecone (sửa duy nhất `vector_store.py`) |
| Multi-camera tập trung | Đặt RabbitMQ trước endpoint `/checkin`, worker pull frame |
| Chống brute-force ảnh in | Bật `LIVENESS_REQUIRE_BLINK=1` (yêu cầu nháy mắt trong N giây) |

## 7. Cấu hình

Copy `.env.example` → `.env` và chỉnh:
```
SIMILARITY_THRESHOLD=0.45     # ngưỡng cosine, càng thấp càng dễ trùng (false positive)
CHECKIN_COOLDOWN_SECONDS=300  # 5 phút mới cho check-in lại
LIVENESS_MIN_SCORE=0.5
DATABASE_URL=sqlite:///./data/attendance.db
DETECTOR_SIZE=640             # 640 = chính xác, 320 = nhanh
```

## 8. Lưu ý đạo đức & pháp lý

- Phải có **đồng ý rõ ràng** từ người được nhận diện (GDPR / Nghị định 13/2023/NĐ-CP về Bảo vệ dữ liệu cá nhân).
- Dữ liệu sinh trắc học là **dữ liệu nhạy cảm** — mã hoá vector + lưu offline khi có thể.
- Cung cấp cơ chế **xoá** (`DELETE /employees/{id}`) — quyền được lãng quên.
