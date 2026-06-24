r"""
attendance.py  (anti-spoofing OPTIONAL)
---------------------------------------
Main attendance pipeline. Anti-spoofing can be skipped (antispoof_ckpt=None)
so you can build the full enroll + recognize + database system BEFORE training
the anti-spoofing module. Add it later by passing antispoof_ckpt.

Flow:
    frame -> YOLO detect -> crop+expand -> MediaPipe align 112x112
          -> IResNet50 embedding -> cosine match with DB
          -> [optional] anti-spoof liveness -> decision -> log
"""

import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "recognition"))
sys.path.insert(0, str(ROOT / "antispoofing"))
sys.path.insert(0, str(ROOT / "pipeline"))

from alignment import FaceAligner
from extract_embedding import EmbeddingExtractor
from database import AttendanceDB
from ultralytics import YOLO


@dataclass
class RecognitionResult:
    bbox: tuple
    aligned_face: Optional[np.ndarray]
    employee_code: Optional[str]
    employee_name: Optional[str]
    sim_score: float
    live_score: float
    decision: str
    logged: bool


class AttendancePipeline:
    def __init__(
        self,
        detector_weights: str,
        recognition_ckpt: str,
        antispoof_ckpt: Optional[str] = None,   # <-- now OPTIONAL
        db_path: str = "attendance.db",
        device: str = "cuda",
        det_conf: float = 0.4,
        det_imgsz: int = 512,
        sim_threshold: float = 0.30,
        live_threshold: float = 0.70,
        cooldown_min: int = 5,
        crop_margin: float = 0.3,
    ):
        self.detector = YOLO(detector_weights)
        self.det_conf = det_conf
        self.det_imgsz = det_imgsz
        self.device = device

        self.aligner = FaceAligner(confidence=0.5, output_size=112)
        self.embedder = EmbeddingExtractor(recognition_ckpt, device=device)

        # Anti-spoofing is optional
        self.antispoof = None
        if antispoof_ckpt:
            from infer import AntiSpoofClassifier
            self.antispoof = AntiSpoofClassifier(antispoof_ckpt, device=device)
            print("[Pipeline] Anti-spoofing ENABLED")
        else:
            print("[Pipeline] Anti-spoofing DISABLED (no checkpoint given)")

        self.db = AttendanceDB(db_path)
        self.sim_threshold = sim_threshold
        self.live_threshold = live_threshold
        self.cooldown = timedelta(minutes=cooldown_min)
        self.crop_margin = crop_margin
        self._refresh_gallery()

    def _refresh_gallery(self):
        ids, codes, names, embs, owner = self.db.get_all_employee_embeddings()
        self.gallery_ids = ids
        self.gallery_codes = codes
        self.gallery_names = names
        self.gallery_embs = embs
        self.gallery_owner = owner

    def reload(self):
        self._refresh_gallery()

    def _expand_crop(self, frame, bbox):
        H, W = frame.shape[:2]
        x1, y1, x2, y2 = bbox[:4]
        bw, bh = x2 - x1, y2 - y1
        mx, my = int(bw * self.crop_margin), int(bh * self.crop_margin)
        x1 = max(0, x1 - mx); y1 = max(0, y1 - my)
        x2 = min(W, x2 + mx); y2 = min(H, y2 + my)
        return frame[y1:y2, x1:x2]

    def _detect_faces(self, frame_bgr):
        results = self.detector.predict(
            source=frame_bgr, imgsz=self.det_imgsz, conf=self.det_conf,
            device=self.device, verbose=False)
        boxes = []
        for r in results:
            for b in r.boxes:
                x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().tolist()
                score = float(b.conf[0].cpu().numpy())
                boxes.append((int(x1), int(y1), int(x2), int(y2), score))
        return boxes

    def _match(self, emb):
        if self.gallery_embs.shape[0] == 0:
            return None, None, None, 0.0
        sims = self.gallery_embs @ emb
        best_per_emp = {}
        for s, o in zip(sims, self.gallery_owner):
            o = int(o)
            if o not in best_per_emp or s > best_per_emp[o]:
                best_per_emp[o] = float(s)
        best_owner = max(best_per_emp, key=best_per_emp.get)
        return (self.gallery_ids[best_owner], self.gallery_codes[best_owner],
                self.gallery_names[best_owner], float(best_per_emp[best_owner]))

    def process_frame(self, frame_bgr, log: bool = True) -> List[RecognitionResult]:
        results = []
        boxes = self._detect_faces(frame_bgr)
        if not boxes:
            return results

        for bbox in boxes:
            face_crop = self._expand_crop(frame_bgr, bbox)
            if face_crop.size == 0:
                continue

            aligned = self.aligner.align(face_crop)
            if aligned is None:
                aligned = cv2.resize(face_crop, (112, 112))
                prefix = "no_landmark"
            else:
                prefix = None

            emb = self.embedder.embed_bgr(aligned)
            emp_id, emp_code, emp_name, sim = self._match(emb)

            # Anti-spoof: if disabled, treat as live (score 1.0)
            if self.antispoof is not None:
                live_score = self.antispoof.predict(face_crop)
            else:
                live_score = 1.0

            if emp_id is None or sim < self.sim_threshold:
                decision, logged = "unknown", False
            elif self.antispoof is not None and live_score < self.live_threshold:
                decision, logged = "spoof", False
            else:
                last = self.db.last_attendance_today(emp_id)
                if last is not None and datetime.utcnow() - last < self.cooldown:
                    decision, logged = "cooldown", False
                else:
                    if log:
                        self.db.log_attendance(emp_id, sim_score=sim, live_score=live_score)
                        decision, logged = "logged", True
                    else:
                        decision, logged = "would_log", False

            if prefix:
                decision = f"{prefix}+{decision}"

            results.append(RecognitionResult(
                bbox=bbox, aligned_face=aligned,
                employee_code=emp_code, employee_name=emp_name,
                sim_score=sim, live_score=live_score,
                decision=decision, logged=logged))
        return results

    def enroll(self, code: str, name: str, frames_bgr: List[np.ndarray]) -> int:
        embs = []
        for frame in frames_bgr:
            boxes = self._detect_faces(frame)
            if not boxes:
                continue
            boxes.sort(key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
            face_crop = self._expand_crop(frame, boxes[0])
            aligned = self.aligner.align(face_crop)
            if aligned is None:
                aligned = cv2.resize(face_crop, (112, 112))
            embs.append(self.embedder.embed_bgr(aligned))
        if not embs:
            raise ValueError("No valid faces found in enrollment frames")
        embs = np.stack(embs, axis=0)
        emp_id = self.db.register_employee(code, name, embs)
        self._refresh_gallery()
        return emp_id


def draw_result(frame, results: List[RecognitionResult]):
    for r in results:
        x1, y1, x2, y2, _ = r.bbox
        if r.logged or r.decision == "would_log":
            color = (0, 255, 0)
            txt = f"{r.employee_name} sim={r.sim_score:.2f}"
        elif "spoof" in r.decision:
            color = (0, 0, 255)
            txt = f"SPOOF live={r.live_score:.2f}"
        elif "cooldown" in r.decision:
            color = (255, 200, 0)
            txt = f"{r.employee_name} (cooldown)"
        elif "unknown" in r.decision:
            color = (0, 165, 255)
            txt = f"Unknown sim={r.sim_score:.2f}"
        else:
            color = (180, 180, 180)
            txt = r.decision
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, txt, (x1, max(15, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return frame
