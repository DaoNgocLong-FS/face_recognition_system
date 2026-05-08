"""
Service layer — gói toàn bộ pipeline thành 2 method nghiệp vụ:

  register_face(employee_id, name, images)   -> tạo Employee + add vectors vào FAISS
  recognize_and_log(image, source)           -> detect, liveness, search FAISS,
                                                 ghi AttendanceLog (có cooldown)

Tất cả script khác (CLI, FastAPI) chỉ cần gọi 2 hàm này.
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Optional, List
import numpy as np
from sqlalchemy import select
from loguru import logger

from config import settings
from src.face_processor import FaceProcessor, FaceResult
from src.liveness import LivenessChecker, LivenessResult
from src.vector_store import FaissVectorStore
from src.database import (
    Employee, AttendanceLog, get_session, init_db,
)


@dataclass
class RecognizeResult:
    matched: bool
    employee_id: Optional[str] = None
    name: Optional[str] = None
    similarity: float = 0.0
    liveness: LivenessResult = None  # type: ignore
    bbox: Optional[tuple[int, int, int, int]] = None
    logged: bool = False
    message: str = ""


class FaceService:
    _instance: Optional["FaceService"] = None

    def __new__(cls) -> "FaceService":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init()
        return cls._instance

    def _init(self) -> None:
        init_db()
        self.processor = FaceProcessor()
        self.liveness = LivenessChecker()
        self.store = FaissVectorStore()
        logger.success(f"FaceService ready. Vectors in store: {len(self.store)}")

    # ====================================================== REGISTER

    def register_face(
        self,
        employee_id: str,
        name: str,
        images_bgr: List[np.ndarray],
        department: Optional[str] = None,
    ) -> dict:
        """images_bgr: list ảnh, mỗi ảnh chỉ chứa 1 khuôn mặt rõ."""
        if not images_bgr:
            raise ValueError("Cần ít nhất 1 ảnh.")

        embeddings: list[np.ndarray] = []
        for img in images_bgr:
            face = self.processor.detect_largest(img)
            if face is None:
                continue
            embeddings.append(face.embedding)

        if not embeddings:
            raise ValueError("Không phát hiện được khuôn mặt nào trong ảnh đã cung cấp.")

        emb_arr = np.stack(embeddings)
        # Lưu DB
        with get_session() as db:
            emp = db.get(Employee, employee_id)
            if emp is None:
                emp = Employee(employee_id=employee_id, name=name,
                               department=department, num_samples=len(embeddings))
                db.add(emp)
            else:
                emp.name = name
                if department:
                    emp.department = department
                emp.num_samples += len(embeddings)
            db.commit()

        # Lưu FAISS
        self.store.add_many(employee_id, emb_arr)
        self.store.save()

        return {
            "employee_id": employee_id,
            "name": name,
            "samples_added": len(embeddings),
            "total_in_store": len(self.store),
        }

    # ====================================================== RECOGNIZE

    def recognize_and_log(
        self,
        image_bgr: np.ndarray,
        source: str = "camera",
        do_log: bool = True,
    ) -> RecognizeResult:
        face = self.processor.detect_largest(image_bgr)
        if face is None:
            return RecognizeResult(matched=False, message="Không thấy khuôn mặt.")

        # Liveness
        live = self.liveness.check(image_bgr, face.bbox)
        if not live.is_live:
            return RecognizeResult(
                matched=False, liveness=live, bbox=face.bbox,
                message=f"Liveness fail (score={live.score:.2f})",
            )

        # Search FAISS
        match = self.store.best_match(face.embedding)
        if match is None:
            return RecognizeResult(
                matched=False, liveness=live, bbox=face.bbox,
                message="Database trống — chưa có ai được đăng ký.",
            )
        emp_id, sim = match
        if sim < settings.similarity_threshold:
            return RecognizeResult(
                matched=False, liveness=live, bbox=face.bbox,
                similarity=sim,
                message=f"Không đủ giống (sim={sim:.2f} < {settings.similarity_threshold})",
            )

        # Lookup tên + check cooldown + ghi log
        logged = False
        name = emp_id
        with get_session() as db:
            emp = db.get(Employee, emp_id)
            if emp:
                name = emp.name

            if do_log:
                cutoff = datetime.now(timezone.utc) - timedelta(
                    seconds=settings.checkin_cooldown_seconds
                )
                last = db.execute(
                    select(AttendanceLog)
                    .where(AttendanceLog.employee_id == emp_id,
                           AttendanceLog.timestamp >= cutoff)
                    .order_by(AttendanceLog.timestamp.desc())
                    .limit(1)
                ).scalar_one_or_none()

                if last is None:
                    db.add(AttendanceLog(
                        employee_id=emp_id,
                        similarity=sim,
                        liveness_score=live.score,
                        source=source,
                    ))
                    db.commit()
                    logged = True

        return RecognizeResult(
            matched=True, employee_id=emp_id, name=name,
            similarity=sim, liveness=live, bbox=face.bbox,
            logged=logged,
            message="OK" if logged else "Đã check-in trong khoảng cooldown.",
        )

    # ====================================================== ADMIN

    def delete_employee(self, employee_id: str) -> dict:
        removed_vec = self.store.remove(employee_id)
        with get_session() as db:
            emp = db.get(Employee, employee_id)
            if emp:
                db.delete(emp)
                db.commit()
        return {"employee_id": employee_id, "vectors_removed": removed_vec}

    def list_employees(self) -> list[dict]:
        with get_session() as db:
            rows = db.execute(select(Employee).order_by(Employee.created_at)).scalars().all()
            return [
                {
                    "employee_id": e.employee_id,
                    "name": e.name,
                    "department": e.department,
                    "num_samples": e.num_samples,
                    "created_at": e.created_at.isoformat(),
                } for e in rows
            ]
