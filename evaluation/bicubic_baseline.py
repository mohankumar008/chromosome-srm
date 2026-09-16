"""
bicubic_baseline.py
--------------------
Step 4 of the pipeline: a pure classical baseline (no deep learning) that
your SRCNN/EDSR results must be compared against. Run this first, before
touching any neural network, to establish the number to beat.

Usage:
    python evaluation/bicubic_baseline.py --test_hr dataset/test/HR --test_lr dataset/test/LR
"""

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.metrics import compute_psnr, compute_ssim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_hr", type=str, default="dataset/test/HR")
    parser.add_argument("--test_lr", type=str, default="dataset/test/LR")
    args = parser.parse_args()

    valid_ext = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
    files = sorted(f for f in os.listdir(args.test_hr) if f.lower().endswith(valid_ext))
    if not files:
        print(f"No images found in {args.test_hr}")
        return

    psnrs, ssims = [], []
    for fname in files:
        hr = cv2.imread(os.path.join(args.test_hr, fname), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        lr = cv2.imread(os.path.join(args.test_lr, fname), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0

        h, w = hr.shape
        up = cv2.resize(lr, (w, h), interpolation=cv2.INTER_CUBIC)
        up = np.clip(up, 0, 1)

        psnrs.append(compute_psnr(hr, up))
        ssims.append(compute_ssim(hr, up))
        print(f"{fname:25s}  PSNR: {psnrs[-1]:6.2f} dB   SSIM: {ssims[-1]:.4f}")

    print("\n--- Bicubic baseline summary ---")
    print(f"Mean PSNR: {np.mean(psnrs):.2f} dB")
    print(f"Mean SSIM: {np.mean(ssims):.4f}")


if __name__ == "__main__":
    main()
