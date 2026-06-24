r"""
finetune.py
-----------
Fine-tune pretrained AdaFace IR-50 on CASIA-WebFace.

Strategy:
    - Load AdaFace pretrained backbone (MS1MV2, ~99.7% LFW out of the box)
    - Re-init AdaFace head with new num_classes (CASIA has 10572 identities)
    - Train with LOW LR (1e-4) for 5 epochs
    - Use cosine schedule + warmup (1 epoch)
    - AMP enabled

This typically:
    - Preserves pretrained quality (~99.5-99.6% LFW after fine-tune)
    - Adapts head for CASIA identity distribution
    - Provides training metrics/curves for the report
    - Takes 2-3 hours on RTX 3060

Usage on Windows:
    python recognition\finetune.py ^
        --data datasets\casia_webface_aligned ^
        --pretrained pretrained\adaface_ir50_ms1mv2.ckpt ^
        --batch 64 ^
        --epochs 5 ^
        --lr 1e-4 ^
        --workers 4 ^
        --amp ^
        --out checkpoints\recognition

Note: batch 64 fits in 8 GB VRAM. Increase to 128 if you have 12 GB.
"""

import argparse
import math
import platform
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from iresnet import build_iresnet, load_adaface_pretrained
from adaface import AdaFaceHead
from dataset import FaceFolderDataset


def save_ckpt(path, backbone, head, optimizer, epoch, num_classes, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "backbone_name": "ir_50_se",
        "embedding_size": 512,
        "num_classes": num_classes,
        "backbone_state": backbone.state_dict(),
        "head_state": head.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "args": vars(args),
    }, path)


def cosine_lr(step, total_steps, warmup_steps, base_lr, min_lr=0.0):
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--pretrained", required=True,
                    help="Path to AdaFace pretrained .ckpt")
    ap.add_argument("--backbone", default="ir_50",
                    choices=["ir_50", "ir_50_se", "ir_100", "ir_101"])
    ap.add_argument("--embedding-size", type=int, default=512)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-4,
                    help="Low LR for fine-tuning (1e-4 typical)")
    ap.add_argument("--min-lr", type=float, default=0.0)
    ap.add_argument("--warmup-epochs", type=int, default=1)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--adaface-s", type=float, default=64.0)
    ap.add_argument("--adaface-m", type=float, default=0.4)
    ap.add_argument("--adaface-h", type=float, default=0.333)
    ap.add_argument("--freeze-backbone", action="store_true",
                    help="Freeze backbone, only train head (very fast, ~30 min)")
    ap.add_argument("--out", default="checkpoints/recognition")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--log-every", type=int, default=50)
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name} ({props.total_memory / (1024**3):.1f} GB)")

    if platform.system() == "Windows" and args.workers > 0:
        print("[INFO] Windows: if BrokenPipeError occurs, retry with --workers 0")

    # ---- Dataset ----
    train_set = FaceFolderDataset(args.data, img_size=112, is_train=True)
    print(f"Dataset: {len(train_set)} images / {train_set.num_classes} identities")

    loader = DataLoader(
        train_set,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        persistent_workers=(args.workers > 0),
    )
    steps_per_epoch = len(loader)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch * args.warmup_epochs

    # ---- Backbone + load pretrained ----
    backbone = build_iresnet(args.backbone, embedding_size=args.embedding_size)
    if not Path(args.pretrained).exists():
        raise FileNotFoundError(
            f"Pretrained not found: {args.pretrained}. "
            "Download with: python recognition\\download_pretrained.py"
        )
    print(f"Loading pretrained: {args.pretrained}")
    result = load_adaface_pretrained(backbone, args.pretrained, strict=False)
    n_missing = len(result["missing_keys"])
    n_unexpected = len(result["unexpected_keys"])
    print(f"  missing keys:    {n_missing}")
    print(f"  unexpected keys: {n_unexpected}")
    if n_missing > 5:
        print(f"  WARNING: many missing keys. First 5: {result['missing_keys'][:5]}")
        print(f"  Backbone architecture may not match pretrained model.")
    backbone = backbone.to(device)

    # ---- AdaFace head (always fresh — new num_classes) ----
    head = AdaFaceHead(
        embedding_size=args.embedding_size,
        num_classes=train_set.num_classes,
        s=args.adaface_s, m=args.adaface_m, h=args.adaface_h,
    ).to(device)

    # ---- Freeze backbone if requested ----
    if args.freeze_backbone:
        for p in backbone.parameters():
            p.requires_grad = False
        backbone.eval()
        params_to_optimize = list(head.parameters())
        print("Backbone FROZEN. Training only AdaFace head.")
    else:
        params_to_optimize = list(backbone.parameters()) + list(head.parameters())

    n_trainable = sum(p.numel() for p in params_to_optimize if p.requires_grad) / 1e6
    print(f"Trainable params: {n_trainable:.2f}M")

    # ---- Optimizer ----
    optimizer = torch.optim.SGD(
        params_to_optimize,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )
    scaler = torch.cuda.amp.GradScaler() if (args.amp and device.type == "cuda") else None

    # ---- Output dir + log ----
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "log.txt"
    log_file = open(log_path, "a", encoding="utf-8")
    log_file.write(f"\n===== Fine-tune start: {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
    log_file.write(f"Args: {vars(args)}\n")
    log_file.flush()

    # ---- Training loop ----
    global_step = 0
    epoch_times = []
    for epoch in range(1, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")
        if args.freeze_backbone:
            backbone.eval()
        else:
            backbone.train()
        head.train()

        total = 0
        correct = 0
        loss_sum = 0.0
        ep_start = time.time()

        for i, (imgs, labels) in enumerate(loader):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            cur_lr = cosine_lr(global_step, total_steps, warmup_steps,
                               args.lr, args.min_lr)
            for pg in optimizer.param_groups:
                pg["lr"] = cur_lr

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler is not None):
                emb, norm = backbone(imgs)
                logits = head(emb, norm, labels)
                loss = F.cross_entropy(logits, labels)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params_to_optimize, max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params_to_optimize, max_norm=5.0)
                optimizer.step()

            with torch.no_grad():
                pred = logits.argmax(dim=1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)
                loss_sum += loss.item() * labels.size(0)

            global_step += 1

            if (i + 1) % args.log_every == 0:
                speed = total / max(1e-6, time.time() - ep_start)
                msg = (f"  E{epoch} step {i+1}/{steps_per_epoch}  "
                       f"loss={loss_sum/total:.4f}  acc={correct/total:.4f}  "
                       f"lr={cur_lr:.6f}  norm_mean={head.batch_mean.item():.2f}  "
                       f"speed={speed:.0f} img/s")
                print(msg)
                log_file.write(msg + "\n")
                log_file.flush()

        ep_time = time.time() - ep_start
        epoch_times.append(ep_time)
        eta_h = sum(epoch_times) / len(epoch_times) * (args.epochs - epoch) / 3600

        summary = (f"Epoch {epoch}: loss={loss_sum/total:.4f}  acc={correct/total:.4f}  "
                   f"time={ep_time/60:.1f}min  ETA={eta_h:.1f}h")
        print(summary)
        log_file.write(summary + "\n")
        log_file.flush()

        # Save
        save_ckpt(out_dir / f"finetune_epoch_{epoch:02d}.pt",
                  backbone, head, optimizer, epoch, train_set.num_classes, args)
        save_ckpt(out_dir / "last.pt",
                  backbone, head, optimizer, epoch, train_set.num_classes, args)
        print(f"  saved last.pt + finetune_epoch_{epoch:02d}.pt")

    log_file.write(f"===== Fine-tune end: {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
    log_file.close()
    print(f"\nDone. Next step: evaluate on LFW")
    print(f"  python recognition\\evaluate.py --ckpt {out_dir}\\last.pt ...")


if __name__ == "__main__":
    main()
