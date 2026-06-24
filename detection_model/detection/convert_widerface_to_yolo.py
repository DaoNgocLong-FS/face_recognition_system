r"""
convert_widerface_to_yolo.py
----------------------------
Chuyển annotation của WIDER FACE sang định dạng YOLO.

Cấu trúc WIDER FACE (sau khi tải về và giải nén):
    datasets/widerface/
        WIDER_train/images/<event>/<image>.jpg
        WIDER_val/images/<event>/<image>.jpg
        wider_face_split/
            wider_face_train_bbx_gt.txt
            wider_face_val_bbx_gt.txt

Sau khi chạy script này:
    datasets/widerface_yolo/
        images/train/*.jpg
        images/val/*.jpg
        labels/train/*.txt   (mỗi dòng: 0 x_center y_center w h, đã normalize 0-1)
        labels/val/*.txt
        widerface.yaml       (file cấu hình YOLO)

Chạy trên Windows:
    python detection\convert_widerface_to_yolo.py ^
        --widerface-root datasets\widerface ^
        --output datasets\widerface_yolo
"""

import argparse
import shutil
from pathlib import Path

import cv2
from tqdm import tqdm


def parse_widerface_gt(gt_file: Path):
    """
    Format wider_face_*_bbx_gt.txt:
        <image_path>
        <num_boxes>
        x1 y1 w h blur expression illumination invalid occlusion pose
        ...
    """
    items = []
    with open(gt_file, "r") as f:
        lines = [l.strip() for l in f.readlines()]

    i = 0
    while i < len(lines):
        img_path = lines[i]
        i += 1
        num = int(lines[i])
        i += 1
        boxes = []
        # File gốc đôi khi đặt num=0 nhưng vẫn có 1 dòng box dummy "0 0 0 0 ..."
        count = max(num, 1)
        for _ in range(count):
            parts = lines[i].split()
            i += 1
            x1, y1, w, h = map(int, parts[:4])
            if w <= 0 or h <= 0:
                continue
            boxes.append((x1, y1, w, h))
        items.append((img_path, boxes))
    return items


def convert_split(src_images_dir: Path, gt_file: Path, dst_root: Path, split: str):
    items = parse_widerface_gt(gt_file)
    img_out = dst_root / "images" / split
    lbl_out = dst_root / "labels" / split
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    for img_rel, boxes in tqdm(items, desc=f"Converting {split}"):
        src = src_images_dir / img_rel
        if not src.exists():
            continue

        img = cv2.imread(str(src))
        if img is None:
            continue
        H, W = img.shape[:2]

        # Flatten path: thay '/' và '\\' bằng '__' để không lồng folder
        flat_name = img_rel.replace("/", "__").replace("\\", "__")
        dst_img = img_out / flat_name
        dst_lbl = lbl_out / (Path(flat_name).stem + ".txt")

        shutil.copy(src, dst_img)

        lines = []
        for x, y, w, h in boxes:
            # Filter box quá nhỏ (<10px) — YOLO khó học, lại tốn thời gian
            if w < 10 or h < 10:
                continue
            xc = (x + w / 2) / W
            yc = (y + h / 2) / H
            ww = w / W
            hh = h / H
            lines.append(f"0 {xc:.6f} {yc:.6f} {ww:.6f} {hh:.6f}")

        # Cho phép file label rỗng (ảnh không có face hợp lệ)
        dst_lbl.write_text("\n".join(lines))


def write_yaml(dst_root: Path):
    """
    Ghi YAML config — dùng đường dẫn tuyệt đối với forward slash để
    chạy được cả trên Windows lẫn Linux mà không phải sửa tay.
    """
    yaml_path = dst_root / "widerface.yaml"
    # Convert Windows path C:\ ... sang dạng C:/... cho YOLO yaml
    abs_path = str(dst_root.resolve()).replace("\\", "/")
    content = f"""# Cấu hình dataset cho Ultralytics YOLO
path: {abs_path}
train: images/train
val: images/val
names:
  0: face
"""
    yaml_path.write_text(content)
    print(f"Đã ghi {yaml_path}")
    print(f"path = {abs_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--widerface-root", required=True,
        help="Thư mục chứa WIDER_train, WIDER_val, wider_face_split"
    )
    ap.add_argument(
        "--output", default="datasets/widerface_yolo",
        help="Thư mục output theo định dạng YOLO"
    )
    args = ap.parse_args()

    src = Path(args.widerface_root)
    dst = Path(args.output)
    dst.mkdir(parents=True, exist_ok=True)

    if not (src / "wider_face_split").exists():
        raise FileNotFoundError(
            f"Không thấy {src / 'wider_face_split'}. "
            "Bạn đã giải nén 'Face annotations' (wider_face_split.zip) chưa?"
        )

    convert_split(
        src_images_dir=src / "WIDER_train" / "images",
        gt_file=src / "wider_face_split" / "wider_face_train_bbx_gt.txt",
        dst_root=dst,
        split="train",
    )
    convert_split(
        src_images_dir=src / "WIDER_val" / "images",
        gt_file=src / "wider_face_split" / "wider_face_val_bbx_gt.txt",
        dst_root=dst,
        split="val",
    )
    write_yaml(dst)
    print("\n[OK] Hoàn tất convert.")


if __name__ == "__main__":
    main()
