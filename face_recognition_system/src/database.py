"""
Bước 5: ORM models + DB session.

SQLAlchemy 2.0 style. Mặc định SQLite (zero-config), đổi DATABASE_URL trong
.env để dùng PostgreSQL.

Bảng:
  employees      — danh sách nhân viên đã đăng ký
  attendance_log — mỗi lần check-in lưu 1 dòng
"""
from __future__ import annotations
from datetime import datetime, timezone
from typing import Optional
from sqlalchemy import (
    String, DateTime, Float, Integer, ForeignKey, create_engine, Index,
)
from sqlalchemy.orm import (
    DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker, Session,
)

from config import settings


class Base(DeclarativeBase):
    pass


class Employee(Base):
    __tablename__ = "employees"

    employee_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    department: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    num_samples: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    logs: Mapped[list["AttendanceLog"]] = relationship(
        back_populates="employee", cascade="all, delete-orphan"
    )


class AttendanceLog(Base):
    __tablename__ = "attendance_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    employee_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("employees.employee_id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc),
        index=True,
    )
    similarity: Mapped[float] = mapped_column(Float, nullable=False)
    liveness_score: Mapped[float] = mapped_column(Float, nullable=False)
    source: Mapped[str] = mapped_column(String(64), default="camera")  # camera | api | kiosk

    employee: Mapped[Employee] = relationship(back_populates="logs")


Index("ix_attendance_emp_time", AttendanceLog.employee_id, AttendanceLog.timestamp)


# ----- Engine + session factory ----------------------------------------------

_engine = create_engine(
    settings.database_url,
    echo=False,
    future=True,
    connect_args={"check_same_thread": False} if settings.database_url.startswith("sqlite") else {},
)
SessionLocal = sessionmaker(bind=_engine, autoflush=False, autocommit=False, future=True)


def init_db() -> None:
    Base.metadata.create_all(_engine)


def get_session() -> Session:
    """Dùng trong CLI script. Trong FastAPI dùng Depends(get_session_dep)."""
    return SessionLocal()


def get_session_dep():
    """Dependency cho FastAPI."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
