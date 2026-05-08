"""
Bước 6: Tối ưu — kiểm tra rằng InsightFace đang dùng ONNX Runtime, in ra
provider đang chạy, và benchmark tốc độ.

InsightFace 'buffalo_l' đã ở dạng ONNX (.onnx) sẵn — không cần convert.
Nếu bạn muốn TensorRT, hãy:
    pip install onnxruntime-gpu
    pip install nvidia-pyindex && pip install nvidia-tensorrt
    đổi providers thành ["TensorrtExecutionProvider", "CUDAExecutionProvider", ...]
"""
import sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import onnxruntime as ort
from src.face_processor import FaceProcessor


def main() -> int:
    print("ONNX Runtime version:", ort.__version__)
    print("Available providers:", ort.get_available_providers())

    proc = FaceProcessor()

    # Random ảnh 640x640 BGR để test thời gian end-to-end
    img = (np.random.rand(640, 640, 3) * 255).astype(np.uint8)
    n_warm, n_run = 5, 30
    for _ in range(n_warm):
        proc.detect(img)
    t0 = time.time()
    for _ in range(n_run):
        proc.detect(img)
    elapsed = time.time() - t0
    print(f"\nBenchmark (det+embed, no real face):")
    print(f"  {n_run} iters in {elapsed:.2f}s  ->  "
          f"{(elapsed/n_run)*1000:.1f} ms/frame  ->  {n_run/elapsed:.1f} FPS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
