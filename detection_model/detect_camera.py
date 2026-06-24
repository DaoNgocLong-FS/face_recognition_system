r"""
detect_camera.py
----------------
PHASE 1 VALIDATION: Test detection + alignment on a live camera stream.

This is a STANDALONE test that only needs:
    - Your trained YOLO detection model (best.pt)
    - MediaPipe (for alignment preview — no training needed)

It does NOT need recognition or anti-spoofing models. Use it to verify:
    1. Camera RTSP stream works
    2. Detection model detects faces well on the real feed
    3. Alignment produces clean 112x112 faces (preview)
    4. FPS / performance is acceptable

None of this work is wasted — the same FaceDetector + FaceAligner are reused
in the final pipeline once you train recognition.

Usage (webcam):
    python detect_camera.py --detector detection_best.pt --cam 0

Usage (Imou RTSP):
    python detect_camera.py --detector detection_best.pt --rtsp "rtsp://admin:CODE@192.168.1.105:554/cam/realmonitor?channel=1&subtype=1"

Keys:
    q : quit
    a : toggle alignment preview (shows aligned 112x112 face in corner)
    s : save current detected + aligned faces to ./captured_faces/
        (useful later for enrollment or quality inspection)
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Import detection + alignment (both reusable in final pipeline)
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "recognition"))

from ultralytics import YOLO

try:
    from alignment import FaceAligner
    HAS_ALIGNER = True
except Exception as e:
    print(f"[WARN] alignment not available: {e}")
    print("       Detection will still work; alignment preview disabled.")
    HAS_ALIGNER = False


def expand_crop(frame, bbox, margin=0.3):
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = bbox[:4]
    bw, bh = x2 - x1, y2 - y1
    mx, my = int(bw * margin), int(bh * margin)
    x1 = max(0, x1 - mx); y1 = max(0, y1 - my)
    x2 = min(W, x2 + mx); y2 = min(H, y2 + my)
    return frame[y1:y2, x1:x2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", required=True, help="YOLO best.pt")
    ap.add_argument("--cam", type=int, default=0, help="Webcam index")
    ap.add_argument("--rtsp", default=None, help="RTSP URL (overrides --cam)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--imgsz", type=int, default=512)
    args = ap.parse_args()

    # Load detector
    print(f"Loading detector: {args.detector}")
    detector = YOLO(args.detector)

    # Aligner (optional)
    aligner = FaceAligner(output_size=112) if HAS_ALIGNER else None

    # Open source
    if args.rtsp:
        print(f"Opening RTSP: {args.rtsp}")
        cap = cv2.VideoCapture(args.rtsp, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    else:
        print(f"Opening webcam {args.cam}")
        cap = cv2.VideoCapture(args.cam)

    if not cap.isOpened():
        print("[FAILED] Cannot open video source.")
        print("If RTSP: test URL in VLC first, check IP/password/TLS.")
        return

    save_dir = Path("captured_faces")
    save_dir.mkdir(exist_ok=True)

    show_align = True
    frame_count = 0
    t0 = time.time()
    fps = 0.0

    print("\nControls: q=quit, a=toggle alignment preview, s=save faces")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[WARN] frame read failed, retrying...")
            time.sleep(0.05)
            continue

        frame_count += 1
        if frame_count % 10 == 0:
            fps = 10 / (time.time() - t0)
            t0 = time.time()

        # Detect
        results = detector.predict(
            source=frame, imgsz=args.imgsz, conf=args.conf,
            device=args.device, verbose=False,
        )
        boxes = []
        for r in results:
            for b in r.boxes:
                x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().tolist()
                score = float(b.conf[0].cpu().numpy())
                boxes.append((int(x1), int(y1), int(x2), int(y2), score))

        # Draw boxes
        aligned_faces = []
        for i, (x1, y1, x2, y2, score) in enumerate(boxes):
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{score:.2f}", (x1, max(15, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Alignment preview
            if show_align and aligner is not None:
                crop = expand_crop(frame, (x1, y1, x2, y2), margin=0.3)
                aligned = aligner.align(crop)
                if aligned is not None:
                    aligned_faces.append(aligned)
                    # Mark "aligned OK" in green
                    cv2.putText(frame, "aligned", (x1, y2 + 18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
                else:
                    # Landmark detection failed
                    cv2.putText(frame, "no landmark", (x1, y2 + 18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1)

        # Show aligned faces in top-right corner
        if aligned_faces:
            x_off = frame.shape[1] - 120
            for j, af in enumerate(aligned_faces[:4]):  # show up to 4
                y_off = 10 + j * 120
                if y_off + 112 <= frame.shape[0] and x_off >= 0:
                    frame[y_off:y_off + 112, x_off:x_off + 112] = af
                    cv2.rectangle(frame, (x_off, y_off),
                                  (x_off + 112, y_off + 112), (255, 255, 0), 1)

        # HUD
        hud = f"Faces: {len(boxes)}  FPS: {fps:.1f}  Align: {'ON' if show_align else 'OFF'}"
        cv2.putText(frame, hud, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 0), 2)

        cv2.imshow("Detection + Alignment Test", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        elif k == ord("a"):
            show_align = not show_align
            print(f"Alignment preview: {'ON' if show_align else 'OFF'}")
        elif k == ord("s"):
            ts = int(time.time())
            for i, (x1, y1, x2, y2, score) in enumerate(boxes):
                crop = expand_crop(frame, (x1, y1, x2, y2), margin=0.3)
                cv2.imwrite(str(save_dir / f"face_{ts}_{i}_crop.jpg"), crop)
                if aligner is not None:
                    aligned = aligner.align(crop)
                    if aligned is not None:
                        cv2.imwrite(str(save_dir / f"face_{ts}_{i}_aligned.jpg"), aligned)
            print(f"Saved {len(boxes)} faces to {save_dir}/")

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
