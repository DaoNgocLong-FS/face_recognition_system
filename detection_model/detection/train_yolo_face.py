r"""
train_yolo_face.py
------------------
Train YOLO face detector trên WIDER FACE đã convert sang YOLO format.

GPU (khuyến nghị):
    python detection\train_yolo_face.py ^
        --data datasets\widerface_yolo\widerface.yaml ^
        --model yolov8n.pt ^
        --epochs 80 ^
        --imgsz 640 ^
        --batch 16 ^
        --device 0 ^
        --workers 4 ^
        --amp ^
        --export-onnx

Sau khi train xong:
    runs/detect/face_yolov8n/weights/best.pt   ← model tốt nhất
    runs/detect/face_yolov8n/weights/last.pt
    runs/detect/face_yolov8n/results.png       ← biểu đồ loss/mAP (cho báo cáo)
    runs/detect/face_yolov8n/confusion_matrix.png

Windows note: nếu bị BrokenPipeError hoặc lỗi multiprocessing, đặt --workers 0.
"""

import argparse
import platform
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Đường dẫn widerface.yaml")
    ap.add_argument(
        "--model", default="yolov8n.pt",
        help="Pretrained: yolov8n.pt (nhẹ) / yolov8s.pt / yolov8m.pt"
    )
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0", help="GPU id (0, 1, ...) hoặc 'cpu'")
    ap.add_argument("--workers", type=int, default=4,
                    help="DataLoader workers. Đặt 0 nếu Windows bị lỗi multiprocessing.")
    ap.add_argument("--amp", action="store_true",
                    help="Bật mixed precision (FP16) — tiết kiệm VRAM, train nhanh hơn.")
    ap.add_argument("--name", default="face_yolov8n", help="Tên run trong runs/detect/")
    ap.add_argument("--patience", type=int, default=30,
                    help="Early stopping: dừng nếu mAP không cải thiện sau N epoch")
    ap.add_argument("--resume", action="store_true",
                    help="Resume training từ checkpoint last.pt cùng tên run")
    ap.add_argument("--export-onnx", action="store_true",
                    help="Export ONNX sau khi train")
    args = ap.parse_args()

    # Import sau khi parse args để --help nhanh hơn (ultralytics import lâu)
    import torch
    from ultralytics import YOLO

    # Sanity check GPU
    if args.device != "cpu" and not torch.cuda.is_available():
        print("[CẢNH BÁO] --device {} nhưng torch.cuda.is_available()=False. "
              "Chuyển sang CPU.".format(args.device))
        args.device = "cpu"
    if args.device != "cpu":
        gpu_idx = int(args.device)
        gpu_name = torch.cuda.get_device_properties(gpu_idx).name
        vram_gb = torch.cuda.get_device_properties(gpu_idx).total_memory / (1024**3)
        print(f"Training on GPU {gpu_idx}: {gpu_name} ({vram_gb:.1f} GB VRAM)")
    else:
        print("Training on CPU (sẽ rất chậm)")

    # Windows + workers > 0 hay gặp lỗi. Cảnh báo người dùng.
    if platform.system() == "Windows" and args.workers > 0:
        print(f"[INFO] Windows detected. Nếu bị lỗi multiprocessing/BrokenPipeError, "
              f"chạy lại với --workers 0")

    # Load pretrained (Ultralytics tự tải nếu chưa có)
    if args.resume:
        ckpt = Path("runs/detect") / args.name / "weights" / "last.pt"
        if not ckpt.exists():
            raise FileNotFoundError(f"Không thấy {ckpt} để resume")
        model = YOLO(str(ckpt))
        train_kwargs = {"resume": True}
        print(f"Resume từ {ckpt}")
    else:
        model = YOLO(args.model)
        train_kwargs = {}

    # Train
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        amp=args.amp,
        name=args.name,
        patience=args.patience,
        # Augmentation cho face — không xoay mạnh
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        # Tuỳ chọn cache để tăng tốc nếu dataset vừa RAM
        # cache='ram',   # bật nếu RAM ≥ 32GB
        **train_kwargs,
    )

    # Đánh giá val cuối cùng
    print("\n" + "=" * 50)
    print(" VALIDATION ".center(50, "="))
    print("=" * 50)
    metrics = model.val(data=args.data, imgsz=args.imgsz, device=args.device)
    print(f"mAP@0.5       = {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95  = {metrics.box.map:.4f}")
    print(f"Precision     = {metrics.box.mp:.4f}")
    print(f"Recall        = {metrics.box.mr:.4f}")

    if args.export_onnx:
        best = Path("runs/detect") / args.name / "weights" / "best.pt"
        if best.exists():
            print(f"\nExporting ONNX từ {best} ...")
            model_best = YOLO(str(best))
            onnx_path = model_best.export(
                format="onnx", imgsz=args.imgsz, dynamic=True, simplify=True
            )
            print(f"[OK] ONNX: {onnx_path}")


if __name__ == "__main__":
    # Windows multiprocessing entry-point
    main()
