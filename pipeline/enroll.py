r"""
enroll.py  (anti-spoofing OPTIONAL)
-----------------------------------
Enroll a new employee by capturing photos via webcam / iPhone (iVCam).

Usage WITHOUT anti-spoofing (current stage):
    python pipeline\enroll.py ^
        --detector detection_model\best.pt ^
        --recognition checkpoints\recognition\last.pt ^
        --db attendance.db ^
        --code EMP001 ^
        --name "Nguyen Van A" ^
        --num-photos 5 ^
        --cam 1

Add --antispoof checkpoints\antispoof\best.pt later when you have it.

During capture:
    - Stay in the green box, move head slightly between shots
    - Press SPACE to capture, or wait for auto-capture
    - ESC to abort
Capture 5 photos: front, slight left, slight right, slight up, slight down.
"""

import argparse
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.attendance import AttendancePipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", required=True)
    ap.add_argument("--recognition", required=True)
    ap.add_argument("--antispoof", default=None,
                    help="Optional. Anti-spoof checkpoint. Skip if not trained yet.")
    ap.add_argument("--db", default="attendance.db")
    ap.add_argument("--code", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--num-photos", type=int, default=5)
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--rtsp", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--pause-sec", type=float, default=1.5)
    args = ap.parse_args()

    pipeline = AttendancePipeline(
        detector_weights=args.detector,
        recognition_ckpt=args.recognition,
        antispoof_ckpt=args.antispoof,   # None = skip
        db_path=args.db,
        device=args.device,
    )

    src = args.rtsp if args.rtsp else args.cam
    cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG) if args.rtsp else cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera: {src}")

    print(f"\nEnrolling: {args.code}  {args.name}")
    print(f"Capture {args.num_photos} photos. SPACE=capture, ESC=abort.\n")

    frames_captured = []
    captured = 0
    last_capture = 0.0

    while captured < args.num_photos:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.05)
            continue

        boxes = pipeline._detect_faces(frame)
        for x1, y1, x2, y2, s in boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.putText(frame, f"Capture {captured + 1}/{args.num_photos}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(frame, "SPACE=capture  ESC=abort",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        cv2.imshow("Enroll", frame)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:  # ESC
            print("Aborted.")
            cap.release(); cv2.destroyAllWindows()
            return

        auto = (time.time() - last_capture) > args.pause_sec and boxes and captured > 0
        if (k == ord(" ") and boxes) or auto:
            frames_captured.append(frame.copy())
            captured += 1
            last_capture = time.time()
            print(f"  captured {captured}/{args.num_photos}")
            time.sleep(0.3)

    cap.release()
    cv2.destroyAllWindows()

    if len(frames_captured) < 3:
        print(f"[WARN] Only {len(frames_captured)} photos. Recommend >=3.")

    print(f"\nRegistering {args.code} with {len(frames_captured)} photos...")
    try:
        emp_id = pipeline.enroll(args.code, args.name, frames_captured)
        print(f"[OK] Enrolled. ID={emp_id}, DB={args.db}")
    except ValueError as e:
        print(f"[ERROR] {e}")


if __name__ == "__main__":
    main()
