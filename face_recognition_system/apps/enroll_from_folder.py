"""
Đăng ký hàng loạt từ thư mục.

Cấu trúc:
    photos/
      NV001_Nguyen_Van_A/      <-- {employee_id}_{name with underscores}
        1.jpg
        2.jpg
      NV002_Tran_Thi_B/
        ...

Chạy:  python apps/enroll_from_folder.py --folder ./photos
"""
import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.service import FaceService

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_dir_name(d: Path) -> tuple[str, str]:
    parts = d.name.split("_", 1)
    if len(parts) == 1:
        return parts[0], parts[0]
    emp_id, name = parts[0], parts[1].replace("_", " ")
    return emp_id, name


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True, type=Path)
    args = ap.parse_args()

    if not args.folder.is_dir():
        print(f"❌ Folder không tồn tại: {args.folder}")
        return 1

    service = FaceService()
    total_emp, total_img = 0, 0
    for emp_dir in sorted(p for p in args.folder.iterdir() if p.is_dir()):
        emp_id, name = parse_dir_name(emp_dir)
        images = []
        for img_path in emp_dir.iterdir():
            if img_path.suffix.lower() not in VALID_EXTS:
                continue
            img = cv2.imread(str(img_path))
            if img is not None:
                images.append(img)

        if not images:
            print(f"⚠️  {emp_dir.name}: không có ảnh hợp lệ.")
            continue

        try:
            info = service.register_face(emp_id, name, images)
            print(f"✅ {emp_id} {name}: +{info['samples_added']} samples")
            total_emp += 1
            total_img += info["samples_added"]
        except ValueError as e:
            print(f"❌ {emp_dir.name}: {e}")

    print(f"\n🎯 Done: {total_emp} employees, {total_img} samples added.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
