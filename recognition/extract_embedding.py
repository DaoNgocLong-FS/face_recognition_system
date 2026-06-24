r"""
extract_embedding.py  (ROBUST version - auto-detects architecture)
------------------------------------------------------------------
Inference wrapper to extract 512-d face embeddings.

This version AUTO-DETECTS the backbone architecture from the saved weights
themselves, instead of trusting the (sometimes wrong) 'backbone_name' metadata.
This prevents the "missing se_block keys" / "ir_50 vs ir_50_se" mismatch.

AdaFace expects BGR channel order (see official inference.py), so we keep BGR.
"""

import argparse
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch

from iresnet import build_iresnet, load_adaface_pretrained

# Defined inline (NOT imported from dataset.py) to avoid name collision with
# antispoofing/dataset.py when both folders are on sys.path.
# AdaFace / InsightFace convention: face normalized to [-1, 1].
FACE_MEAN = (0.5, 0.5, 0.5)
FACE_STD = (0.5, 0.5, 0.5)


def preprocess_bgr(img_bgr: np.ndarray, size: int = 112) -> torch.Tensor:
    """BGR uint8 -> tensor (1,3,H,W) normalized to [-1,1]. Keep BGR (AdaFace convention)."""
    if img_bgr.shape[:2] != (size, size):
        img_bgr = cv2.resize(img_bgr, (size, size))
    img = img_bgr.astype(np.float32) / 255.0  # keep BGR, do NOT convert to RGB
    img = (img - np.array(FACE_MEAN, dtype=np.float32)) / np.array(FACE_STD, dtype=np.float32)
    return torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float()


def _detect_arch_from_state(state: dict) -> str:
    """
    Auto-detect backbone name from state_dict keys.
      - SE present?  -> any key contains 'se_block'
      - num_layers?  -> infer from number of distinct 'body.N' blocks
                        (ir_50 has 24 blocks: 3+4+14+3)
    """
    has_se = any("se_block" in k for k in state.keys())

    # Count distinct body block indices
    block_ids = set()
    for k in state.keys():
        if k.startswith("body."):
            try:
                block_ids.add(int(k.split(".")[1]))
            except (IndexError, ValueError):
                pass
    n_blocks = len(block_ids)

    # Map block count to num_layers
    # ir_18: 8 blocks, ir_34: 16, ir_50: 24, ir_100: 49
    if n_blocks <= 8:
        layers = 18
    elif n_blocks <= 16:
        layers = 34
    elif n_blocks <= 24:
        layers = 50
    else:
        layers = 100

    name = f"ir_{layers}"
    if has_se:
        name += "_se"
    return name


class EmbeddingExtractor:
    def __init__(self, ckpt_path: str, device: str = "cpu", backbone_name: str = None):
        self.device = torch.device(device)
        ck = torch.load(ckpt_path, map_location=self.device, weights_only=False)

        if "backbone_state" in ck:
            state = ck["backbone_state"]
            self.embedding_size = ck.get("embedding_size", 512)
            # AUTO-DETECT architecture from weights (robust to wrong metadata)
            detected = _detect_arch_from_state(state)
            self.backbone_name = backbone_name or detected
            self.model = build_iresnet(self.backbone_name,
                                       embedding_size=self.embedding_size).to(self.device)
            self.model.load_state_dict(state)
        else:
            # Raw AdaFace .ckpt
            self.embedding_size = 512
            # Strip 'model.' prefix then detect
            tmp = {}
            raw = ck["state_dict"] if "state_dict" in ck else ck
            for k, v in raw.items():
                tmp[k[len("model."):]] = v if k.startswith("model.") else v
            detected = _detect_arch_from_state(tmp)
            self.backbone_name = backbone_name or detected
            self.model = build_iresnet(self.backbone_name,
                                       embedding_size=self.embedding_size).to(self.device)
            load_adaface_pretrained(self.model, ckpt_path, strict=False)

        self.model.eval()

    @torch.no_grad()
    def embed_bgr(self, img_bgr: np.ndarray) -> np.ndarray:
        x = preprocess_bgr(img_bgr).to(self.device)
        emb, _ = self.model(x)
        return emb.cpu().numpy().squeeze(0)

    @torch.no_grad()
    def embed_batch(self, imgs_bgr: Sequence[np.ndarray]) -> np.ndarray:
        xs = torch.cat([preprocess_bgr(i) for i in imgs_bgr], dim=0).to(self.device)
        emb, _ = self.model(xs)
        return emb.cpu().numpy()

    @staticmethod
    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ext = EmbeddingExtractor(args.ckpt, device=args.device)
    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(args.image)
    emb = ext.embed_bgr(img)
    print(f"Backbone (auto-detected): {ext.backbone_name}")
    print(f"Embedding dim:  {emb.shape[0]}")
    print(f"L2 norm:        {np.linalg.norm(emb):.4f}")


if __name__ == "__main__":
    main()
