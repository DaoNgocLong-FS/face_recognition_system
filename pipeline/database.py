r"""
database.py
-----------
SQLite database for attendance system.

Tables:
  - employees:  code, name, embedding blob (multiple embeddings per person)
  - attendance: log of recognition events
"""

from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np
from sqlalchemy import (Column, DateTime, Float, ForeignKey, Integer,
                        LargeBinary, String, create_engine, select)
from sqlalchemy.orm import (DeclarativeBase, Mapped, Session, mapped_column,
                            relationship)


class Base(DeclarativeBase):
    pass


class Employee(Base):
    __tablename__ = "employees"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    code: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(128))
    embeddings_blob: Mapped[bytes] = mapped_column(LargeBinary)
    embedding_dim: Mapped[int] = mapped_column(Integer)
    num_embeddings: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    logs = relationship("Attendance", back_populates="employee")

    def load_embeddings(self) -> np.ndarray:
        arr = np.frombuffer(self.embeddings_blob, dtype=np.float32)
        return arr.reshape(self.num_embeddings, self.embedding_dim)


class Attendance(Base):
    __tablename__ = "attendance"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    employee_id: Mapped[int] = mapped_column(ForeignKey("employees.id"), index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    sim_score: Mapped[float] = mapped_column(Float)
    live_score: Mapped[float] = mapped_column(Float)
    employee = relationship("Employee", back_populates="logs")


class AttendanceDB:
    def __init__(self, db_path: str = "attendance.db"):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(f"sqlite:///{db_path}", echo=False, future=True)
        Base.metadata.create_all(self.engine)

    def register_employee(self, code: str, name: str, embeddings: np.ndarray) -> int:
        assert embeddings.ndim == 2
        embeddings = embeddings.astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9
        embeddings = embeddings / norms
        with Session(self.engine) as s:
            emp = Employee(
                code=code, name=name,
                embeddings_blob=embeddings.tobytes(),
                embedding_dim=embeddings.shape[1],
                num_embeddings=embeddings.shape[0],
            )
            s.add(emp)
            s.commit()
            return emp.id

    def get_all_employee_embeddings(self) -> Tuple[List[int], List[str], List[str], np.ndarray, np.ndarray]:
        with Session(self.engine) as s:
            emps = list(s.scalars(select(Employee)))
        if not emps:
            return [], [], [], np.zeros((0, 0), dtype=np.float32), np.zeros((0,), dtype=np.int64)
        ids = [e.id for e in emps]
        codes = [e.code for e in emps]
        names = [e.name for e in emps]
        all_embs = []
        owner = []
        for i, e in enumerate(emps):
            embs = e.load_embeddings()
            all_embs.append(embs)
            owner.extend([i] * embs.shape[0])
        return ids, codes, names, np.concatenate(all_embs, axis=0), np.asarray(owner, dtype=np.int64)

    def log_attendance(self, employee_id: int, sim_score: float, live_score: float) -> int:
        with Session(self.engine) as s:
            row = Attendance(employee_id=employee_id, sim_score=sim_score, live_score=live_score)
            s.add(row); s.commit()
            return row.id

    def list_today(self) -> List[dict]:
        from datetime import date, datetime as dt
        today_start = dt.combine(date.today(), dt.min.time())
        with Session(self.engine) as s:
            q = select(Attendance, Employee).join(Employee).where(Attendance.timestamp >= today_start)
            rows = []
            for att, emp in s.execute(q):
                rows.append({
                    "id": att.id, "employee_code": emp.code, "name": emp.name,
                    "timestamp": att.timestamp.isoformat(),
                    "sim_score": att.sim_score, "live_score": att.live_score,
                })
        return rows

    def last_attendance_today(self, employee_id: int):
        """Return the most recent attendance time today for this employee, or None."""
        from datetime import date, datetime as dt
        today_start = dt.combine(date.today(), dt.min.time())
        with Session(self.engine) as s:
            row = s.scalar(select(Attendance)
                           .where(Attendance.employee_id == employee_id)
                           .where(Attendance.timestamp >= today_start)
                           .order_by(Attendance.timestamp.desc()))
            return row.timestamp if row else None
