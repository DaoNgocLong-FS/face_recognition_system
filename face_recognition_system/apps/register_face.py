"""
Đăng ký khuôn mặt qua webcam.

  python apps/register_face.py --name "Nguyen Van A" --employee-id NV001

Phím tắt:
  SPACE  — chụp 1 sample (cần >=3 sample để xong)
  ENTER  — kết thúc và lưu
  Q      — bỏ qua, không lưu
"""
import argparse
import sys
from pathlib import Path

import cv2

# cho phép chạy script trực tiếp từ root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.service import FaceService


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--employee-id", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--department", default=None)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--min-samples", type=int, default=3)
    args = ap.parse_args()

    service = FaceService()
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("❌ Không mở được camera.")
        return 1

    samples: list = []
    print(f"📷 SPACE = chụp, ENTER = lưu (>= {args.min_samples}), Q = huỷ.")

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        face = service.processor.detect_largest(frame)
        disp = frame.copy()
        if face:
            x1, y1, x2, y2 = face.bbox
            cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(disp, f"det={face.det_score:.2f}",
                        (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.putText(disp,
                    f"Samples: {len(samples)} | Need >= {args.min_samples}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Register face — SPACE/ENTER/Q", disp)

        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            print("Cancelled.")
            cap.release(); cv2.destroyAllWindows()
            return 0
        if k == 32:                       # SPACE
            if face is None:
                print("⚠️  Chưa thấy mặt — bỏ qua.")
                continue
            samples.append(frame.copy())
            print(f"✅ Đã chụp sample #{len(samples)}")
        if k in (10, 13):                 # ENTER
            if len(samples) < args.min_samples:
                print(f"❗ Cần ít nhất {args.min_samples} samples.")
                continue
            break

    cap.release()
    cv2.destroyAllWindows()

    info = service.register_face(
        employee_id=args.employee_id,
        name=args.name,
        images_bgr=samples,
        department=args.department,
    )
    print("🎉 Registered:", info)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
