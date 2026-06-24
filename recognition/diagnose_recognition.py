r"""
diagnose_recognition.py
-----------------------
Diagnose why recognition gives similar scores for different people.

Checks:
    1. Does the pretrained AdaFace checkpoint load correctly? (key match stats)
    2. Do embeddings DISCRIMINATE between different faces?

Run with 2-3 DIFFERENT people's face images (can be your captured_faces/*.jpg):

    python diagnose_recognition.py ^
        --pretrained pretrained\adaface_ir50_ms1mv2.ckpt ^
        --finetuned checkpoints\recognition\test_copy.pt ^
        --faces face1.jpg face2.jpg face3.jpg

Where face1, face2 = same person (2 photos), face3 = different person.
Ideal output:
    sim(face1, face2) HIGH  (~0.5-0.8) -- same person
    sim(face1, face3) LOW   (~0.0-0.2) -- different person

If ALL sims are high (>0.5) -> embedding collapse -> weights not loaded properly.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "recognition"))

from iresnet import build_iresnet, load_adaface_pretrained
from dataset import FACE_MEAN, FACE_STD


def preprocess(img_bgr, size=112):
    if img_bgr.shape[:2] != (size, size):
        img_bgr = cv2.resize(img_bgr, (size, size))
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - np.array(FACE_MEAN, dtype=np.float32)) / np.array(FACE_STD, dtype=np.float32)
    return torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float()


@torch.no_grad()
def get_embeddings(model, image_paths, device):
    model.eval()
    embs = []
    for p in image_paths:
        img = cv2.imread(p)
        if img is None:
            print(f"  [WARN] cannot read {p}")
            embs.append(None)
            continue
        x = preprocess(img).to(device)
        emb, norm = model(x)
        embs.append(emb.cpu().numpy().squeeze(0))
    return embs


def check_keys(ckpt_path, backbone_name="ir_50"):
    """Load and report key match statistics."""
    model = build_iresnet(backbone_name, embedding_size=512)
    total_keys = len(model.state_dict())
    result = load_adaface_pretrained(model, ckpt_path, strict=False)
    n_missing = len(result["missing_keys"])
    n_unexpected = len(result["unexpected_keys"])
    n_loaded = total_keys - n_missing
    return model, total_keys, n_loaded, n_missing, n_unexpected, result


def print_sim_matrix(embs, paths):
    n = len(embs)
    print("\n  Pairwise cosine similarity:")
    print("  " + " " * 18 + "  ".join(f"[{i}]" for i in range(n)))
    for i in range(n):
        row = f"  [{i}] {Path(paths[i]).name[:14]:14s}"
        for j in range(n):
            if embs[i] is None or embs[j] is None:
                row += "   N/A"
            else:
                sim = float(np.dot(embs[i], embs[j]))
                row += f"  {sim:.2f}"
        print(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained", required=True, help="AdaFace pretrained .ckpt")
    ap.add_argument("--finetuned", default=None, help="Your fine-tuned checkpoint (optional)")
    ap.add_argument("--faces", nargs="+", required=True,
                    help="2-3 face images (different people)")
    ap.add_argument("--backbone", default="ir_50")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)

    print("=" * 60)
    print(" CHECK 1: Does PRETRAINED load correctly? ".center(60, "="))
    print("=" * 60)
    model, total, loaded, missing, unexpected, result = check_keys(
        args.pretrained, args.backbone)
    print(f"Backbone:        {args.backbone}")
    print(f"Total keys:      {total}")
    print(f"Loaded keys:     {loaded}")
    print(f"Missing keys:    {missing}")
    print(f"Unexpected keys: {unexpected}")

    if missing > 20:
        print("\n  >>> PROBLEM FOUND <<<")
        print(f"  {missing} keys did NOT load. Architecture mismatch!")
        print("  This means the backbone is mostly RANDOM -> embeddings garbage.")
        print(f"  First 10 missing keys:")
        for k in result["missing_keys"][:10]:
            print(f"    {k}")
        print(f"  First 10 unexpected (in ckpt but not model):")
        for k in result["unexpected_keys"][:10]:
            print(f"    {k}")
    else:
        print("\n  [OK] Pretrained loaded well (architecture matches).")

    model = model.to(device)

    # Test discrimination with pretrained
    print("\n" + "=" * 60)
    print(" CHECK 2: Do PRETRAINED embeddings discriminate? ".center(60, "="))
    print("=" * 60)
    embs = get_embeddings(model, args.faces, device)
    # Report embedding norms (should NOT all be identical)
    print("  Embedding norms (raw, before normalize would be ~20):")
    for i, e in enumerate(embs):
        if e is not None:
            print(f"    [{i}] {Path(args.faces[i]).name[:20]:20s} "
                  f"L2={np.linalg.norm(e):.4f}  first3={e[:3]}")
    print_sim_matrix(embs, args.faces)

    # Optionally test fine-tuned
    if args.finetuned and Path(args.finetuned).exists():
        print("\n" + "=" * 60)
        print(" CHECK 3: Do FINE-TUNED embeddings discriminate? ".center(60, "="))
        print("=" * 60)
        ck = torch.load(args.finetuned, map_location=device, weights_only=False)
        ft_model = build_iresnet(ck.get("backbone_name", args.backbone),
                                 embedding_size=ck.get("embedding_size", 512)).to(device)
        ft_model.load_state_dict(ck["backbone_state"])
        ft_embs = get_embeddings(ft_model, args.faces, device)
        print_sim_matrix(ft_embs, args.faces)

    print("\n" + "=" * 60)
    print(" VERDICT ".center(60, "="))
    print("=" * 60)
    print("  Expected for GOOD model:")
    print("    same person pair   -> 0.4 to 0.8")
    print("    different person   -> -0.1 to 0.25")
    print("  If ALL pairs > 0.5 (even different people) -> embedding collapse")
    print("    -> pretrained likely didn't load (see CHECK 1)")


if __name__ == "__main__":
    main()
