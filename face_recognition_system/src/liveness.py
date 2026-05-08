"""
Bước 3: Liveness / Anti-spoofing.

Phiên bản nhẹ — không cần model thứ 2:
  (a) Texture/laplacian variance: ảnh in / ảnh điện thoại thường có variance thấp
      hơn da thật vì có moiré và mờ chi tiết.
  (b) Color saturation check: màn hình điện thoại thường saturated hơn da thật.
  (c) Blink detection (tuỳ chọn) qua MediaPipe Face Mesh — yêu cầu user nháy mắt
      ít nhất 1 lần trong N frame liên tiếp.

Production: thay class `TextureLiveness` bằng model Silent-Face-Anti-Spoofing
(MiniVision) — chỉ cần đổi method `score_image()`, interface giữ nguyên.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import cv2
import mediapipe as mp
from loguru import logger

from config import settings


# ---------- (a)+(b) Texture-based scoring -----------------------------------

class TextureLiveness:
    """Heuristic nhanh: kết hợp Laplacian variance + saturation."""

    def __init__(self, lap_min: float = 50.0, lap_max: float = 800.0,
                 sat_max: float = 140.0):
        self.lap_min = lap_min
        self.lap_max = lap_max
        self.sat_max = sat_max

    def score_face(self, image_bgr: np.ndarray, bbox: tuple[int, int, int, int]) -> float:
        """Trả về score [0..1]. >= settings.liveness_min_score coi như sống."""
        x1, y1, x2, y2 = bbox
        h, w = image_bgr.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop = image_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return 0.0

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        sat_mean = float(hsv[..., 1].mean())

        # Map sang [0..1]
        # - lap_var quá thấp (mờ) HOẶC quá cao (noise màn hình) đều bị trừ điểm
        if lap_var < self.lap_min:
            lap_score = lap_var / self.lap_min
        elif lap_var > self.lap_max:
            lap_score = max(0.0, 1.0 - (lap_var - self.lap_max) / self.lap_max)
        else:
            lap_score = 1.0

        sat_score = max(0.0, 1.0 - max(0.0, sat_mean - self.sat_max) / 100.0)

        return float(np.clip(0.6 * lap_score + 0.4 * sat_score, 0.0, 1.0))


# ---------- (c) Blink Detection ---------------------------------------------

# Eye landmarks indices trong FaceMesh (468-pt)
LEFT_EYE  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


def _eye_aspect_ratio(landmarks, indices, w: int, h: int) -> float:
    pts = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in indices])
    # EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    a = np.linalg.norm(pts[1] - pts[5])
    b = np.linalg.norm(pts[2] - pts[4])
    c = np.linalg.norm(pts[0] - pts[3])
    return float((a + b) / (2.0 * c + 1e-6))


@dataclass
class BlinkState:
    consec_below: int = 0
    blink_count: int = 0


class BlinkDetector:
    """Stateful — gọi `update(frame)` mỗi frame, đọc `state.blink_count`."""

    def __init__(self):
        self.mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=False,
            min_detection_confidence=0.5, min_tracking_confidence=0.5,
        )
        self.state = BlinkState()

    def reset(self) -> None:
        self.state = BlinkState()

    def update(self, image_bgr: np.ndarray) -> Optional[float]:
        """Trả về EAR hiện tại, None nếu không thấy mặt."""
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w = image_bgr.shape[:2]
        res = self.mesh.process(rgb)
        if not res.multi_face_landmarks:
            return None
        lm = res.multi_face_landmarks[0].landmark
        ear = (_eye_aspect_ratio(lm, LEFT_EYE, w, h)
               + _eye_aspect_ratio(lm, RIGHT_EYE, w, h)) / 2.0

        if ear < settings.blink_ear_threshold:
            self.state.consec_below += 1
        else:
            if self.state.consec_below >= settings.blink_consec_frames:
                self.state.blink_count += 1
            self.state.consec_below = 0
        return ear


# ---------- High-level facade -----------------------------------------------

@dataclass
class LivenessResult:
    is_live: bool
    score: float
    reason: str = ""


class LivenessChecker:
    def __init__(self):
        self.texture = TextureLiveness()

    def check(self, image_bgr: np.ndarray, bbox: tuple[int, int, int, int]) -> LivenessResult:
        score = self.texture.score_face(image_bgr, bbox)
        is_live = score >= settings.liveness_min_score
        return LivenessResult(
            is_live=is_live,
            score=score,
            reason="texture-ok" if is_live else "texture-suspicious",
        )
