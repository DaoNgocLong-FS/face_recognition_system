r"""
test_rtsp.py
------------
Test RTSP connection to an Imou (Dahua) camera before integrating into pipeline.

Usage (Windows PowerShell, 1 line):
    python pipeline\test_rtsp.py --url "rtsp://admin:SAFETYCODE@192.168.1.105:554/cam/realmonitor?channel=1&subtype=0"

If video window shows up -> RTSP works, you can use this URL in camera_demo.py.

Tips:
    - subtype=0 : main stream (high-res, more lag)
    - subtype=1 : sub stream (low-res, less lag — RECOMMENDED for realtime AI)
    - If password has special chars (@ : / ?), URL-encode them:
        @ -> %40    : -> %3A    / -> %2F
    - Press q to quit.
"""

import argparse
import time

import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="Full RTSP URL")
    ap.add_argument("--show", action="store_true", default=True)
    args = ap.parse_args()

    print(f"Connecting to: {args.url}")
    print("(this may take 5-10 seconds...)")

    # FFMPEG backend handles RTSP best
    cap = cv2.VideoCapture(args.url, cv2.CAP_FFMPEG)

    # Reduce buffering for lower latency
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("\n[FAILED] Cannot open RTSP stream.")
        print("Checklist:")
        print("  1. Camera and PC on same network?")
        print("  2. IP address correct? (check Imou Life app)")
        print("  3. Password = SAFETY CODE on camera body (not Imou account password)?")
        print("  4. Encryption/TLS disabled in Imou Life app settings?")
        print("  5. Try subtype=1 instead of subtype=0")
        print("  6. Test in VLC first (Media -> Open Network Stream)")
        return

    print("[OK] Stream opened. Reading frames...")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Resolution: {W}x{H}  FPS (reported): {fps:.1f}")

    frame_count = 0
    t0 = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            print("[WARN] Frame read failed (network hiccup?). Retrying...")
            time.sleep(0.1)
            continue

        frame_count += 1
        # Measure actual FPS
        if frame_count % 30 == 0:
            actual_fps = frame_count / (time.time() - t0)
            print(f"  frames={frame_count}  actual_fps={actual_fps:.1f}")

        # Resize for display if too big
        disp = frame
        if W > 1280:
            disp = cv2.resize(frame, (1280, int(1280 * H / W)))

        cv2.putText(disp, "RTSP OK - press q to quit", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("RTSP Test", disp)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nDone. Total frames: {frame_count}")


if __name__ == "__main__":
    main()
