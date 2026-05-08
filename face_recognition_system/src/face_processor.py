"""
Bước 1 + 2: Face Detection -> Alignment -> Embedding.

Dùng InsightFace 'buffalo_l' đóng gói sẵn:
  - RetinaFace (detection + 5 landmarks)
  - ArcFace ResNet50 (embedding 512-D đã L2-normalize)
Tất cả đều chạy ONNX Runtime, không cần PyTorch khi inference.

Output chuẩn của module này là `FaceResult`:
  bbox    : (x1, y1, x2, y2)
  kps     : 5x2 landmarks
  embedding: np.float32[512] đã normalize
  det_score: confidence của detector
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
from insightface.app import FaceAnalysis
from loguru import logger

from config import settings


@dataclass
class FaceResult:
    bbox: tuple[int, int, int, int]
    kps: np.ndarray              # (5, 2)
    embedding: np.ndarray        # (512,) L2-normalized
    det_score: float

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]

    @property
    def area(self) -> int:
        return self.width * self.height


class FaceProcessor:
    """Singleton-style wrapper. Khởi tạo 1 lần, dùng nhiều lần."""

    _instance: Optional["FaceProcessor"] = None

    def __new__(cls) -> "FaceProcessor":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_models()
        return cls._instance

    def _init_models(self) -> None:
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if settings.use_gpu
            else ["CPUExecutionProvider"]
        )
        logger.info(
            f"Loading InsightFace pack='{settings.insightface_model_pack}' "
            f"providers={providers} det_size={settings.detector_size}"
        )
        self.app = FaceAnalysis(
            name=settings.insightface_model_pack,
            providers=providers,
            allowed_modules=["detection", "recognition"],
        )
        self.app.prepare(ctx_id=0 if settings.use_gpu else -1,
                         det_size=(settings.detector_size, settings.detector_size))
        logger.success("FaceProcessor ready.")

    # ---------------------------------------------------------------- public

    def detect(self, image_bgr: np.ndarray) -> List[FaceResult]:
        """Trả về list khuôn mặt detect được. InsightFace đã tự align rồi embed."""
        if image_bgr is None or image_bgr.size == 0:
            return []
        faces = self.app.get(image_bgr)
        out: List[FaceResult] = []
        for f in faces:
            x1, y1, x2, y2 = [int(v) for v in f.bbox]
            emb = f.normed_embedding.astype(np.float32)
            out.append(
                FaceResult(
                    bbox=(x1, y1, x2, y2),
                    kps=f.kps,
                    embedding=emb,
                    det_score=float(f.det_score),
                )
            )
        return out

    def detect_largest(self, image_bgr: np.ndarray) -> Optional[FaceResult]:
        """Trả về 1 mặt to nhất — dùng khi enroll / checkin chỉ 1 người."""
        faces = self.detect(image_bgr)
        if not faces:
            return None
        return max(faces, key=lambda f: f.area)
