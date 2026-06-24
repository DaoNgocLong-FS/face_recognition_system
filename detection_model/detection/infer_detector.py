r"""
infer_detector.py
-----------------
Inference face detector — test model trên ảnh hoặc thư mục ảnh.

Cách chạy trên Windows:
    python detection\infer_detector.py ^
        --weights runs\detect\face_yolov8n\weights\best.pt ^
        --source path\to\image_or_folder ^
        --conf 0.4 ^
        --device 0
"""

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


class FaceDetector:
    """Wrapper YOLO face detector. Trả về list (x1,y1,x2,y2,score) trên ảnh BGR."""

    def __init__(self, weights, device="cpu", conf=0.4, imgsz=640):
        self.model = YOLO(weights)
        self.device = device
        self.conf = conf
        self.imgsz = imgsz

    def detect(self, img_bgr):
        results = self.model.predict(
            source=img_bgr, imgsz=self.imgsz,
            conf=self.conf, device=self.device, verbose=False,
        )
        boxes = []
        for r in results:
            for b in r.boxes:
                x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().tolist()
                score = float(b.conf[0].cpu().numpy())
                boxes.append((int(x1), int(y1), int(x2), int(y2), score))
        return boxes


def draw_boxes(img, boxes):
    for x1, y1, x2, y2, s in boxes:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"{s:.2f}", (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--source", required=True, help="ảnh hoặc thư mục")
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--device", default="0",
                    help="GPU id ('0') hoặc 'cpu'")
    ap.add_argument("--out", default="runs/infer_detector")
    args = ap.parse_args()

    det = FaceDetector(args.weights, device=args.device, conf=args.conf)

    src = Path(args.source)
    if src.is_file():
        files = [src]
    else:
        # Tìm các đuôi ảnh thường gặp
        files = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            files.extend(src.glob(ext))
        files = sorted(files)

    if not files:
        print(f"Không tìm thấy ảnh trong {src}")
        return

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            print(f"  [skip] {f.name} — không đọc được")
            continue
        boxes = det.detect(img)
        vis = draw_boxes(img.copy(), boxes)
        out_path = out_dir / f.name
        cv2.imwrite(str(out_path), vis)
        print(f"  {f.name}: phát hiện {len(boxes)} face(s) -> {out_path}")

    print(f"\nKết quả lưu ở: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
