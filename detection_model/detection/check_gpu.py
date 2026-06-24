"""
check_gpu.py
------------
Kiểm tra xem GPU đã sẵn sàng cho training chưa.

Chạy:
    python detection/check_gpu.py

Sẽ in ra:
  - Python version
  - PyTorch version + có CUDA build hay không
  - GPU name + VRAM tổng + VRAM trống
  - Có chạy được matrix multiplication trên GPU không
  - Khuyến nghị batch_size phù hợp với VRAM của bạn

Nếu thiếu CUDA → hướng dẫn fix.
"""

import sys


def main():
    print("=" * 60)
    print(" GPU CHECK ".center(60, "="))
    print("=" * 60)

    print(f"Python: {sys.version.split()[0]}")

    try:
        import torch
    except ImportError:
        print("[X] PyTorch chưa cài. Chạy: pip install torch torchvision")
        return

    print(f"PyTorch: {torch.__version__}")
    print(f"PyTorch built with CUDA: {torch.version.cuda}")

    if not torch.cuda.is_available():
        print("\n[X] torch.cuda.is_available() == False")
        print("\nNguyên nhân thường gặp:")
        print("  1. Cài bản PyTorch CPU-only (không có CUDA).")
        print("     Fix: gỡ rồi cài lại PyTorch CUDA:")
        print("         pip uninstall torch torchvision -y")
        print("         pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
        print("     (đổi cu121 thành cu118 nếu driver cũ)")
        print("  2. Driver NVIDIA cũ. Cập nhật driver từ nvidia.com/Download/index.aspx")
        print("  3. Máy không có GPU NVIDIA (GPU AMD/Intel không dùng CUDA được).")
        return

    n_gpus = torch.cuda.device_count()
    print(f"\n[OK] CUDA available — phát hiện {n_gpus} GPU:\n")

    for i in range(n_gpus):
        props = torch.cuda.get_device_properties(i)
        total_gb = props.total_memory / (1024**3)
        # Compute capability
        cc = f"{props.major}.{props.minor}"
        print(f"  GPU {i}: {props.name}")
        print(f"    VRAM         : {total_gb:.1f} GB")
        print(f"    Compute cap. : {cc}")
        print(f"    SM count     : {props.multi_processor_count}")

    # Thử 1 phép tính nhỏ trên GPU
    print("\nTesting GPU compute...")
    try:
        x = torch.randn(2048, 2048, device="cuda")
        y = torch.randn(2048, 2048, device="cuda")
        z = (x @ y).sum().item()
        print(f"  matmul 2048x2048 OK, sum={z:.2f}")
    except Exception as e:
        print(f"  [X] Lỗi khi chạy GPU: {e}")
        return

    # Khuyến nghị batch_size cho YOLOv8n imgsz=640
    print("\n" + "=" * 60)
    print(" KHUYẾN NGHỊ CONFIG TRAIN ".center(60, "="))
    print("=" * 60)

    props = torch.cuda.get_device_properties(0)
    total_gb = props.total_memory / (1024**3)

    if total_gb >= 20:
        batch, imgsz, model = 64, 640, "yolov8s.pt"
        epochs = 100
        tier = "HIGH-END (≥20GB)"
    elif total_gb >= 12:
        batch, imgsz, model = 32, 640, "yolov8n.pt"
        epochs = 80
        tier = "HIGH (12-20GB)"
    elif total_gb >= 8:
        batch, imgsz, model = 16, 640, "yolov8n.pt"
        epochs = 80
        tier = "MID (8-12GB)"
    elif total_gb >= 6:
        batch, imgsz, model = 12, 640, "yolov8n.pt"
        epochs = 60
        tier = "MID-LOW (6-8GB)"
    elif total_gb >= 4:
        batch, imgsz, model = 8, 512, "yolov8n.pt"
        epochs = 50
        tier = "LOW (4-6GB)"
    else:
        batch, imgsz, model = 4, 416, "yolov8n.pt"
        epochs = 30
        tier = "VERY LOW (<4GB)"

    print(f"GPU tier      : {tier}")
    print(f"Suggested cmd :")
    print(f"  python detection\\train_yolo_face.py ^")
    print(f"      --data datasets\\widerface_yolo\\widerface.yaml ^")
    print(f"      --model {model} ^")
    print(f"      --epochs {epochs} ^")
    print(f"      --imgsz {imgsz} ^")
    print(f"      --batch {batch} ^")
    print(f"      --device 0 ^")
    print(f"      --workers 4 ^")
    print(f"      --amp ^")
    print(f"      --name face_yolov8n")
    print()
    print("Tip:")
    print("  - Nếu OOM khi train, giảm --batch xuống 1/2.")
    print("  - --amp bật mixed precision (FP16) → tiết kiệm ~40% VRAM, train nhanh hơn.")
    print("  - --workers 4 trên Windows đôi khi lỗi, lúc đó dùng --workers 0.")
    print("=" * 60)


if __name__ == "__main__":
    main()
