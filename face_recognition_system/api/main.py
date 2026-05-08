"""
FastAPI backend (Bước 5).

Endpoints:
  POST   /register             — đăng ký khuôn mặt (multipart upload)
  POST   /checkin              — nhận dạng + ghi log (1 ảnh / request)
  GET    /employees            — list nhân viên
  DELETE /employees/{emp_id}   — xoá khỏi FAISS + DB
  GET    /logs                 — lịch sử check-in (filter ?employee_id=&date_from=&date_to=)
  GET    /health
"""
from __future__ import annotations
from datetime import datetime, timezone
from typing import Optional
from io import BytesIO

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session
from PIL import Image

from config import settings
from src.service import FaceService
from src.database import get_session_dep, AttendanceLog, Employee


app = FastAPI(
    title="Face Recognition Attendance API",
    version="1.0.0",
    description="Hệ thống điểm danh bằng khuôn mặt — InsightFace + FAISS + FastAPI.",
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

service = FaceService()


# ---------- helpers ---------------------------------------------------------

async def _read_image(file: UploadFile) -> np.ndarray:
    raw = await file.read()
    if not raw:
        raise HTTPException(400, f"File rỗng: {file.filename}")
    try:
        img = Image.open(BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"Không đọc được ảnh {file.filename}: {e}")
    arr = np.array(img)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


# ---------- schemas ---------------------------------------------------------

class RegisterResponse(BaseModel):
    employee_id: str
    name: str
    samples_added: int
    total_in_store: int


class CheckinResponse(BaseModel):
    matched: bool
    employee_id: Optional[str] = None
    name: Optional[str] = None
    similarity: float = 0.0
    liveness_score: float = 0.0
    is_live: bool = False
    logged: bool = False
    message: str


class EmployeeOut(BaseModel):
    employee_id: str
    name: str
    department: Optional[str]
    num_samples: int
    created_at: str


class LogOut(BaseModel):
    id: int
    employee_id: str
    employee_name: Optional[str]
    timestamp: str
    similarity: float
    liveness_score: float
    source: str


# ---------- endpoints -------------------------------------------------------

@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "vectors_in_store": len(service.store),
        "similarity_threshold": settings.similarity_threshold,
    }


@app.post("/register", response_model=RegisterResponse)
async def register(
    employee_id: str = Form(...),
    name: str = Form(...),
    department: Optional[str] = Form(None),
    files: list[UploadFile] = File(..., description="Ít nhất 1 ảnh khuôn mặt."),
):
    images = [await _read_image(f) for f in files]
    try:
        info = service.register_face(employee_id, name, images, department=department)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return RegisterResponse(**info)


@app.post("/checkin", response_model=CheckinResponse)
async def checkin(file: UploadFile = File(...)):
    img = await _read_image(file)
    res = service.recognize_and_log(img, source="api")
    return CheckinResponse(
        matched=res.matched,
        employee_id=res.employee_id,
        name=res.name,
        similarity=round(res.similarity, 4),
        liveness_score=round(res.liveness.score, 4) if res.liveness else 0.0,
        is_live=bool(res.liveness and res.liveness.is_live),
        logged=res.logged,
        message=res.message,
    )


@app.get("/employees", response_model=list[EmployeeOut])
def list_employees():
    return service.list_employees()


@app.delete("/employees/{employee_id}")
def delete_employee(employee_id: str):
    return service.delete_employee(employee_id)


@app.get("/logs", response_model=list[LogOut])
def list_logs(
    employee_id: Optional[str] = Query(None),
    date_from: Optional[datetime] = Query(None),
    date_to: Optional[datetime] = Query(None),
    limit: int = Query(100, le=1000),
    db: Session = Depends(get_session_dep),
):
    stmt = select(AttendanceLog, Employee.name).join(
        Employee, Employee.employee_id == AttendanceLog.employee_id, isouter=True
    ).order_by(AttendanceLog.timestamp.desc())

    if employee_id:
        stmt = stmt.where(AttendanceLog.employee_id == employee_id)
    if date_from:
        stmt = stmt.where(AttendanceLog.timestamp >= date_from)
    if date_to:
        stmt = stmt.where(AttendanceLog.timestamp <= date_to)
    stmt = stmt.limit(limit)

    out: list[LogOut] = []
    for log, ename in db.execute(stmt).all():
        out.append(LogOut(
            id=log.id,
            employee_id=log.employee_id,
            employee_name=ename,
            timestamp=log.timestamp.astimezone(timezone.utc).isoformat(),
            similarity=log.similarity,
            liveness_score=log.liveness_score,
            source=log.source,
        ))
    return out


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host=settings.api_host, port=settings.api_port, reload=False)
