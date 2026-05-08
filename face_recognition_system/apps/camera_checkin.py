"""
Điểm danh real-time qua webcam.

  python apps/camera_checkin.py --camera 0

Phím tắt:  Q = thoát   |   S = chụp screenshot   |   R = reset blink counter
"""
import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings
from src.service import FaceService
from src.liveness import BlinkDetector


def draw_panel(frame, lines, x=10, y=30):
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (x, y + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
                    cv2.LINE_AA)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--every-n-frames", type=int, default=2,
                    help="Chạy nhận dạng mỗi N frame để giảm tải CPU.")
    args = ap.parse_args()

    service = FaceService()
    blink = BlinkDetector() if settings.liveness_require_blink else None

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("❌ Không mở được camera.")
        return 1

    frame_idx = 0
    last_result_text = "Đang khởi tạo..."
    last_box = None
    last_color = (200, 200, 200)
    last_inference_ms = 0.0
    fps_t0 = time.time()
    fps_count = 0
    fps = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        fps_count += 1
        if time.time() - fps_t0 >= 1.0:
            fps = fps_count / (time.time() - fps_t0)
            fps_count = 0
            fps_t0 = time.time()

        # Blink (mỗi frame, optional)
        if blink is not None:
            blink.update(frame)

        # Nhận dạng mỗi N frame
        if frame_idx % args.every_n_frames == 0:
            t0 = time.time()
            res = service.recognize_and_log(frame, source="camera")
            last_inference_ms = (time.time() - t0) * 1000
            last_box = res.bbox

            # Bắt buộc nháy mắt nếu bật
            if (settings.liveness_require_blink and blink is not None
                    and blink.state.blink_count == 0 and res.matched):
                last_result_text = f"Vui lòng nháy mắt — {res.name}"
                last_color = (0, 255, 255)
            elif res.matched:
                tag = "✓ LOGGED" if res.logged else "(cooldown)"
                last_result_text = (f"{res.name} ({res.employee_id}) "
                                    f"sim={res.similarity:.2f} {tag}")
                last_color = (0, 255, 0) if res.logged else (0, 200, 200)
                if res.logged and blink is not None:
                    blink.reset()
            else:
                last_result_text = res.message
                last_color = (0, 0, 255)

        # Vẽ
        if last_box:
            x1, y1, x2, y2 = last_box
            cv2.rectangle(frame, (x1, y1), (x2, y2), last_color, 2)

        info_lines = [
            f"FPS: {fps:.1f}    Inference: {last_inference_ms:.0f} ms",
            f"DB size: {len(service.store)} vectors",
            last_result_text,
        ]
        if blink is not None:
            info_lines.append(f"Blinks: {blink.state.blink_count}")
        draw_panel(frame, info_lines)

        cv2.imshow("Face Check-in — Q quit, S screenshot", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if k == ord("s"):
            fname = f"screenshot_{datetime.now():%Y%m%d_%H%M%S}.jpg"
            cv2.imwrite(fname, frame)
            print(f"💾 Saved {fname}")
        if k == ord("r") and blink is not None:
            blink.reset()

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
