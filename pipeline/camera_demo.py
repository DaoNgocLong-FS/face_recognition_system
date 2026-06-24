r"""
camera_demo.py  (anti-spoofing OPTIONAL)
----------------------------------------
Realtime attendance demo.

Usage WITHOUT anti-spoofing (current stage), iPhone via iVCam:
    python pipeline\camera_demo.py ^
        --detector detection_model\best.pt ^
        --recognition checkpoints\recognition\last.pt ^
        --db attendance.db ^
        --device cuda ^
        --cam 1 ^
        --sim 0.30

Add --antispoof checkpoints\antispoof\best.pt later when trained.

Keys:
    q : quit
    s : toggle logging on/off
    l : list today's attendance records
"""

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.attendance import AttendancePipeline, draw_result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", required=True)
    ap.add_argument("--recognition", required=True)
    ap.add_argument("--antispoof", default=None,
                    help="Optional. Skip if not trained yet.")
    ap.add_argument("--db", default="attendance.db")
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--rtsp", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--sim", type=float, default=0.30)
    ap.add_argument("--live", type=float, default=0.70)
    ap.add_argument("--det-conf", type=float, default=0.4)
    ap.add_argument("--cooldown", type=int, default=5)
    args = ap.parse_args()

    print("Initializing pipeline...")
    pipeline = AttendancePipeline(
        detector_weights=args.detector,
        recognition_ckpt=args.recognition,
        antispoof_ckpt=args.antispoof,   # None = skip
        db_path=args.db,
        device=args.device,
        det_conf=args.det_conf,
        sim_threshold=args.sim,
        live_threshold=args.live,
        cooldown_min=args.cooldown,
    )
    print(f"Gallery: {len(pipeline.gallery_ids)} employees, "
          f"{int(pipeline.gallery_embs.shape[0])} embeddings")

    if args.rtsp:
        cap = cv2.VideoCapture(args.rtsp, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    else:
        cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera source")

    logging_on = True
    print("\nControls: q=quit, s=toggle log, l=list today")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        results = pipeline.process_frame(frame, log=logging_on)
        frame = draw_result(frame, results)
        spoof_status = "ON" if pipeline.antispoof else "OFF(not trained)"
        status = f"LOG:{'ON' if logging_on else 'OFF'}  Spoof:{spoof_status}  ppl:{len(pipeline.gallery_ids)}"
        cv2.putText(frame, status, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 0) if logging_on else (180, 180, 180), 2)
        cv2.imshow("Attendance Demo", frame)

        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        elif k == ord("s"):
            logging_on = not logging_on
            print(f"Logging: {'ON' if logging_on else 'OFF'}")
        elif k == ord("l"):
            records = pipeline.db.list_today()
            print(f"\n=== Today's attendance ({len(records)}) ===")
            for r in records:
                print(f"  {r['timestamp'][:19]}  {r['employee_code']}  {r['name']}  sim={r['sim_score']:.3f}")
            print()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
