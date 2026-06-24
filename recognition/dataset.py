r"""
dataset.py
----------
CASIA-WebFace dataset loader for fine-tuning.

Expected structure:
    root/
        0/img_0.jpg
        0/img_1.jpg
        1/...
        ...
"""

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


# AdaFace / InsightFace convention: face normalized to [-1, 1]
FACE_MEAN = (0.5, 0.5, 0.5)
FACE_STD = (0.5, 0.5, 0.5)


def normalize_face(img_rgb_float: np.ndarray) -> np.ndarray:
    """img_rgb in [0, 1] -> [-1, 1]"""
    return (img_rgb_float - np.array(FACE_MEAN, dtype=np.float32)) / np.array(FACE_STD, dtype=np.float32)


class FaceFolderDataset(Dataset):
    def __init__(
        self,
        root: str,
        img_size: int = 112,
        is_train: bool = True,
        exts: Tuple[str, ...] = (".jpg", ".jpeg", ".png"),
    ):
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(self.root)
        self.img_size = img_size
        self.is_train = is_train

        id_dirs = sorted([p for p in self.root.iterdir() if p.is_dir()])
        if not id_dirs:
            raise RuntimeError(f"No identity directories in {self.root}")
        self.class_to_idx = {p.name: i for i, p in enumerate(id_dirs)}

        self.samples: List[Tuple[Path, int]] = []
        for p in id_dirs:
            label = self.class_to_idx[p.name]
            for img in p.iterdir():
                if img.suffix.lower() in exts:
                    self.samples.append((img, label))

        if not self.samples:
            raise RuntimeError(f"No images in {self.root}")

    @property
    def num_classes(self) -> int:
        return len(self.class_to_idx)

    def __len__(self) -> int:
        return len(self.samples)

    def _augment(self, img_rgb: np.ndarray) -> np.ndarray:
        # Light augmentation only — face recognition is sensitive
        if np.random.rand() < 0.5:
            img_rgb = np.ascontiguousarray(img_rgb[:, ::-1])
        if np.random.rand() < 0.3:
            alpha = np.random.uniform(0.9, 1.1)
            beta = np.random.uniform(-10, 10)
            img_rgb = np.clip(alpha * img_rgb.astype(np.float32) + beta,
                              0, 255).astype(np.uint8)
        return img_rgb

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.imread(str(path))
        if img is None:
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        # IMPORTANT: AdaFace expects BGR channel order. cv2 reads BGR, so we
        # keep it as BGR (do NOT convert to RGB) to match the pretrained model.
        if img.shape[:2] != (self.img_size, self.img_size):
            img = cv2.resize(img, (self.img_size, self.img_size))
        if self.is_train:
            img = self._augment(img)
        img_f = normalize_face(img.astype(np.float32) / 255.0)
        return torch.from_numpy(img_f.transpose(2, 0, 1).copy()).float(), label
