"""
Cấu hình trung tâm. Tất cả module khác import `settings` từ đây.
Đọc giá trị từ biến môi trường hoặc file .env.
"""
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ---- Detection / Embedding ----
    insightface_model_pack: str = "buffalo_l"      # buffalo_l = ResNet50 ArcFace, buffalo_s = MobileFace
    detector_size: int = 640                        # 640 chính xác, 320 nhanh hơn
    use_gpu: bool = False                           # True nếu có CUDA
    use_onnx: bool = True                           # InsightFace đã dùng ONNX sẵn

    # ---- Recognition ----
    similarity_threshold: float = 0.45              # cosine similarity ngưỡng nhận dạng
    embedding_dim: int = 512

    # ---- Liveness ----
    liveness_min_score: float = 0.5
    liveness_require_blink: bool = False            # bật để bắt buộc nháy mắt
    blink_ear_threshold: float = 0.21               # eye aspect ratio
    blink_consec_frames: int = 2

    # ---- Business logic ----
    checkin_cooldown_seconds: int = 300             # cùng 1 người chỉ check 1 lần / 5 phút

    # ---- Storage ----
    database_url: str = f"sqlite:///{DATA_DIR / 'attendance.db'}"
    faiss_index_path: str = str(DATA_DIR / "faces.index")
    faiss_meta_path: str = str(DATA_DIR / "faces_meta.json")

    # ---- API ----
    api_host: str = "0.0.0.0"
    api_port: int = 8000


settings = Settings()
