r"""
evaluate.py
-----------
Evaluate fine-tuned face recognition model on LFW (or similar pairs benchmark).

Standard LFW pairs.txt format (10 folds, 600 pairs/fold):
    First line: "10 300"  (num_folds, num_pairs_per_fold for each polarity)
    Then for each fold:
        300 matched pairs: "name idx1 idx2"
        300 mismatched pairs: "name1 idx1 name2 idx2"

Usage:
    python recognition\evaluate.py ^
        --ckpt checkpoints\recognition\last.pt ^
        --root datasets\lfw_aligned_112 ^
        --pairs datasets\lfw_pairs.txt
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from sklearn.metrics import roc_curve, auc

from extract_embedding import EmbeddingExtractor


def read_lfw_pairs(pairs_file: Path) -> List[Tuple[str, str, int]]:
    items = []
    with open(pairs_file, "r") as f:
        header = f.readline().strip().split()
        n_folds = int(header[0])
        n_per_fold = int(header[1])
        for _ in range(n_folds):
            for _ in range(n_per_fold):
                parts = f.readline().strip().split()
                name, i1, i2 = parts[0], int(parts[1]), int(parts[2])
                items.append((
                    f"{name}/{name}_{i1:04d}.jpg",
                    f"{name}/{name}_{i2:04d}.jpg",
                    1,
                ))
            for _ in range(n_per_fold):
                parts = f.readline().strip().split()
                n1, i1, n2, i2 = parts[0], int(parts[1]), parts[2], int(parts[3])
                items.append((
                    f"{n1}/{n1}_{i1:04d}.jpg",
                    f"{n2}/{n2}_{i2:04d}.jpg",
                    0,
                ))
    return items


def kfold_accuracy(scores, labels, n_folds=10):
    n = len(scores)
    fold_size = n // n_folds
    accs = []
    best_thrs = []
    thresholds = np.arange(-1.0, 1.0, 0.005)
    for k in range(n_folds):
        test_idx = np.arange(k * fold_size, (k + 1) * fold_size)
        train_idx = np.setdiff1d(np.arange(n), test_idx)
        best_acc, best_thr = 0.0, 0.0
        for t in thresholds:
            pred = (scores[train_idx] >= t).astype(int)
            a = (pred == labels[train_idx]).mean()
            if a > best_acc:
                best_acc, best_thr = a, t
        pred_test = (scores[test_idx] >= best_thr).astype(int)
        accs.append((pred_test == labels[test_idx]).mean())
        best_thrs.append(best_thr)
    return float(np.mean(accs)), float(np.std(accs)), float(np.mean(best_thrs))


def compute_eer(fpr, tpr, thresholds):
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    return float(fpr[idx]), float(thresholds[idx])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    print(f"Loading model from {args.ckpt} ...")
    ext = EmbeddingExtractor(args.ckpt, device=args.device)
    print(f"  Backbone: {ext.backbone_name}  dim={ext.embedding_size}")

    pairs = read_lfw_pairs(Path(args.pairs))
    print(f"Loaded {len(pairs)} pairs")

    scores = []
    labels = []
    cache = {}
    missing = 0

    root = Path(args.root)
    for p1, p2, lbl in pairs:
        f1 = root / p1
        f2 = root / p2
        if not (f1.exists() and f2.exists()):
            missing += 1
            continue
        for f in [f1, f2]:
            if str(f) not in cache:
                img = cv2.imread(str(f))
                if img is None:
                    cache[str(f)] = None
                else:
                    cache[str(f)] = ext.embed_bgr(img)
        e1, e2 = cache[str(f1)], cache[str(f2)]
        if e1 is None or e2 is None:
            missing += 1
            continue
        scores.append(float(np.dot(e1, e2)))
        labels.append(lbl)

    if missing:
        print(f"[warn] missing/unreadable: {missing}")

    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)

    mean_acc, std_acc, mean_thr = kfold_accuracy(scores, labels, n_folds=10)
    fpr, tpr, thr = roc_curve(labels, scores)
    auc_v = auc(fpr, tpr)
    eer, eer_thr = compute_eer(fpr, tpr, thr)

    print("\n" + "=" * 50)
    print(" LFW RESULTS ".center(50, "="))
    print("=" * 50)
    print(f"Pairs:            {len(scores)}")
    print(f"Accuracy 10-fold: {mean_acc*100:.2f} +/- {std_acc*100:.2f}%")
    print(f"Best threshold:   {mean_thr:.4f}")
    print(f"AUC:              {auc_v:.4f}")
    print(f"EER:              {eer*100:.2f}% (at {eer_thr:.4f})")
    print("=" * 50)


if __name__ == "__main__":
    main()
