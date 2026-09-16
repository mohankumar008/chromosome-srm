"""
train.py
--------
Step 4 of the pipeline: train SRCNN or EDSR on paired LR/HR chromosome
patches, tracking training/validation loss and PSNR/SSIM per epoch.

Usage examples:
    python training/train.py --model edsr --epochs 50 --scale 4 \
        --train_hr dataset/train/HR --train_lr dataset/train/LR \
        --val_hr dataset/val/HR --val_lr dataset/val/LR

    python training/train.py --model srcnn --epochs 50 \
        --train_hr dataset/train/HR --train_lr dataset/train/LR \
        --val_hr dataset/val/HR --val_lr dataset/val/LR

Saves:
    checkpoints/<model>_best.pth   (best validation PSNR)
    checkpoints/<model>_last.pth   (final epoch)
    outputs/<model>_training_log.csv
"""

import argparse
import csv
import os
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.srcnn import SRCNN
from models.edsr import EDSR
from training.dataset import SRDataset, SRCNNDataset
from evaluation.metrics import compute_psnr, compute_ssim


def build_model(args):
    if args.model == "srcnn":
        return SRCNN(num_channels=1)
    elif args.model == "edsr":
        return EDSR(
            num_channels=1,
            num_features=args.num_features,
            num_res_blocks=args.num_res_blocks,
            scale=args.scale,
        )
    else:
        raise ValueError(f"Unknown model: {args.model}")


def build_datasets(args):
    if args.model == "srcnn":
        train_ds = SRCNNDataset(args.train_hr, args.train_lr, patch_size=args.patch_size)
        val_ds = SRCNNDataset(args.val_hr, args.val_lr, patch_size=args.patch_size, augment=False)
    else:
        train_ds = SRDataset(args.train_hr, args.train_lr, scale=args.scale, patch_size=args.patch_size)
        val_ds = SRDataset(
            args.val_hr, args.val_lr, scale=args.scale, patch_size=args.patch_size, augment=False
        )
    return train_ds, val_ds


@torch.no_grad()
def validate(model, loader, device, criterion):
    model.eval()
    total_loss, total_psnr, total_ssim, n = 0.0, 0.0, 0.0, 0
    for lr, hr in loader:
        lr, hr = lr.to(device), hr.to(device)
        pred = model(lr)
        loss = criterion(pred, hr)
        total_loss += loss.item() * lr.size(0)

        pred_np = pred.clamp(0, 1).cpu().numpy()
        hr_np = hr.cpu().numpy()
        for i in range(pred_np.shape[0]):
            total_psnr += compute_psnr(hr_np[i, 0], pred_np[i, 0])
            total_ssim += compute_ssim(hr_np[i, 0], pred_np[i, 0])
        n += lr.size(0)

    return total_loss / n, total_psnr / n, total_ssim / n


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = build_model(args).to(device)
    train_ds, val_ds = build_datasets(args)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    criterion = nn.L1Loss() if args.loss == "l1" else nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    log_path = os.path.join("outputs", f"{args.model}_training_log.csv")

    best_psnr = -1.0
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_psnr", "val_ssim", "seconds"])

        for epoch in range(1, args.epochs + 1):
            model.train()
            start = time.time()
            running_loss = 0.0

            for lr_img, hr_img in train_loader:
                lr_img, hr_img = lr_img.to(device), hr_img.to(device)

                optimizer.zero_grad()
                pred = model(lr_img)
                loss = criterion(pred, hr_img)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * lr_img.size(0)

            train_loss = running_loss / len(train_ds)
            val_loss, val_psnr, val_ssim = validate(model, val_loader, device, criterion)
            elapsed = time.time() - start

            print(
                f"Epoch {epoch:3d}/{args.epochs} | "
                f"train_loss {train_loss:.5f} | val_loss {val_loss:.5f} | "
                f"PSNR {val_psnr:.2f} dB | SSIM {val_ssim:.4f} | {elapsed:.1f}s"
            )
            writer.writerow([epoch, train_loss, val_loss, val_psnr, val_ssim, round(elapsed, 1)])
            f.flush()

            if val_psnr > best_psnr:
                best_psnr = val_psnr
                torch.save(model.state_dict(), os.path.join("checkpoints", f"{args.model}_best.pth"))

            torch.save(model.state_dict(), os.path.join("checkpoints", f"{args.model}_last.pth"))

    print(f"Training complete. Best val PSNR: {best_psnr:.2f} dB")
    print(f"Log saved to {log_path}")


def main():
    parser = argparse.ArgumentParser(description="Train SRCNN or EDSR for chromosome super-resolution.")
    parser.add_argument("--model", choices=["srcnn", "edsr"], required=True)
    parser.add_argument("--train_hr", type=str, default="dataset/train/HR")
    parser.add_argument("--train_lr", type=str, default="dataset/train/LR")
    parser.add_argument("--val_hr", type=str, default="dataset/val/HR")
    parser.add_argument("--val_lr", type=str, default="dataset/val/LR")
    parser.add_argument("--scale", type=int, default=4, help="Upsampling factor (EDSR only)")
    parser.add_argument("--patch_size", type=int, default=96, help="HR patch size used during training")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--loss", choices=["l1", "mse"], default="l1")
    parser.add_argument("--num_features", type=int, default=64, help="EDSR width")
    parser.add_argument("--num_res_blocks", type=int, default=16, help="EDSR depth")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
