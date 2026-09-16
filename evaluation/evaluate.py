"""
evaluate.py
-----------
Step 8 + 9 of the pipeline: run the test set through Bicubic, SRCNN and
EDSR, compute PSNR/SSIM for each, save a results table (CSV) and
side-by-side visual comparison images.

Usage:
    python evaluation/evaluate.py --scale 4 \
        --test_hr dataset/test/HR --test_lr dataset/test/LR \
        --srcnn_ckpt checkpoints/srcnn_best.pth \
        --edsr_ckpt checkpoints/edsr_best.pth

Outputs:
    outputs/comparison_results.csv            (per-image + averaged metrics)
    outputs/comparison/<filename>_compare.png (Medium-Res | Bicubic | SRCNN | EDSR | Ground Truth)
"""

import argparse
import csv
import os
import sys

import cv2
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.srcnn import SRCNN
from models.edsr import EDSR
from evaluation.metrics import compute_psnr, compute_ssim


def load_gray01(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img.astype(np.float32) / 255.0


def to_tensor(img01: np.ndarray, device) -> torch.Tensor:
    return torch.from_numpy(img01).unsqueeze(0).unsqueeze(0).to(device)


def to_image(tensor: torch.Tensor) -> np.ndarray:
    arr = tensor.squeeze().detach().cpu().clamp(0, 1).numpy()
    return (arr * 255.0).round().astype(np.uint8)


def make_comparison_strip(images_with_titles, display_size=None):
    """images_with_titles: list of (title, uint8 image), possibly different
    sizes (e.g. the medium-res input is smaller than the SR/HR outputs).
    All panels are resized to `display_size` (defaults to the size of the
    LAST image, i.e. Ground Truth) purely for side-by-side visualization —
    metrics are always computed on native resolutions elsewhere."""
    if display_size is None:
        h, w = images_with_titles[-1][1].shape[:2]
    else:
        h, w = display_size
    pad_top = 30
    canvas = np.full((h + pad_top, w * len(images_with_titles), 3), 255, dtype=np.uint8)
    for i, (title, img) in enumerate(images_with_titles):
        if img.shape[:2] != (h, w):
            interp = cv2.INTER_NEAREST if img.shape[0] < h else cv2.INTER_AREA
            img = cv2.resize(img, (w, h), interpolation=interp)
        if img.ndim == 2:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            img_bgr = img
        x0 = i * w
        canvas[pad_top:, x0 : x0 + w] = img_bgr
        cv2.putText(
            canvas, title, (x0 + 5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA
        )
    return canvas


def main():
    parser = argparse.ArgumentParser(description="Evaluate Bicubic vs SRCNN vs EDSR.")
    parser.add_argument("--test_hr", type=str, default="dataset/test/HR")
    parser.add_argument("--test_lr", type=str, default="dataset/test/LR")
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--srcnn_ckpt", type=str, default="checkpoints/srcnn_best.pth")
    parser.add_argument("--edsr_ckpt", type=str, default="checkpoints/edsr_best.pth")
    parser.add_argument("--num_features", type=int, default=64)
    parser.add_argument("--num_res_blocks", type=int, default=16)
    parser.add_argument("--out_dir", type=str, default="outputs")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    comparison_dir = os.path.join(args.out_dir, "comparison")
    os.makedirs(comparison_dir, exist_ok=True)

    have_srcnn = os.path.exists(args.srcnn_ckpt)
    have_edsr = os.path.exists(args.edsr_ckpt)

    srcnn = edsr = None
    if have_srcnn:
        srcnn = SRCNN(num_channels=1).to(device)
        srcnn.load_state_dict(torch.load(args.srcnn_ckpt, map_location=device))
        srcnn.eval()
    else:
        print(f"[warn] SRCNN checkpoint not found at {args.srcnn_ckpt}; skipping SRCNN.")

    if have_edsr:
        edsr = EDSR(
            num_channels=1,
            num_features=args.num_features,
            num_res_blocks=args.num_res_blocks,
            scale=args.scale,
        ).to(device)
        edsr.load_state_dict(torch.load(args.edsr_ckpt, map_location=device))
        edsr.eval()
    else:
        print(f"[warn] EDSR checkpoint not found at {args.edsr_ckpt}; skipping EDSR.")

    valid_ext = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
    files = sorted(f for f in os.listdir(args.test_hr) if f.lower().endswith(valid_ext))
    if not files:
        print(f"No test images found in {args.test_hr}")
        return

    results = []  # rows for CSV
    running = {"bicubic": [], "srcnn": [], "edsr": []}

    with torch.no_grad():
        for fname in files:
            hr = load_gray01(os.path.join(args.test_hr, fname))
            lr = load_gray01(os.path.join(args.test_lr, fname))
            H, W = hr.shape

            # --- Method 1: Bicubic ---
            bicubic = cv2.resize(lr, (W, H), interpolation=cv2.INTER_CUBIC)
            bicubic = np.clip(bicubic, 0, 1)
            psnr_b, ssim_b = compute_psnr(hr, bicubic), compute_ssim(hr, bicubic)
            running["bicubic"].append((psnr_b, ssim_b))

            strip = [
                ("Medium-Res (input)", (lr * 255).astype(np.uint8)),
                ("Bicubic", (bicubic * 255).astype(np.uint8)),
            ]
            row = {"filename": fname, "bicubic_psnr": psnr_b, "bicubic_ssim": ssim_b}

            # --- Method 2: SRCNN (needs bicubic-upsampled input) ---
            if srcnn is not None:
                inp = to_tensor(bicubic.astype(np.float32), device)
                pred = srcnn(inp)
                pred_img = to_image(pred).astype(np.float32) / 255.0
                psnr_s, ssim_s = compute_psnr(hr, pred_img), compute_ssim(hr, pred_img)
                running["srcnn"].append((psnr_s, ssim_s))
                strip.append(("SRCNN", (pred_img * 255).astype(np.uint8)))
                row["srcnn_psnr"], row["srcnn_ssim"] = psnr_s, ssim_s

            # --- Method 3: EDSR (native LR input, learned upsampling) ---
            if edsr is not None:
                inp = to_tensor(lr.astype(np.float32), device)
                pred = edsr(inp)
                pred_img = to_image(pred).astype(np.float32) / 255.0
                # Guard against off-by-one size mismatch from rounding
                if pred_img.shape != hr.shape:
                    pred_img = cv2.resize(pred_img, (W, H))
                psnr_e, ssim_e = compute_psnr(hr, pred_img), compute_ssim(hr, pred_img)
                running["edsr"].append((psnr_e, ssim_e))
                strip.append(("EDSR", (pred_img * 255).astype(np.uint8)))
                row["edsr_psnr"], row["edsr_ssim"] = psnr_e, ssim_e

            strip.append(("Ground Truth", (hr * 255).astype(np.uint8)))

            canvas = make_comparison_strip(strip)
            out_name = os.path.splitext(fname)[0] + "_compare.png"
            cv2.imwrite(os.path.join(comparison_dir, out_name), canvas)

            results.append(row)

    # --- Write CSV ---
    csv_path = os.path.join(args.out_dir, "comparison_results.csv")
    fieldnames = ["filename", "bicubic_psnr", "bicubic_ssim"]
    if have_srcnn:
        fieldnames += ["srcnn_psnr", "srcnn_ssim"]
    if have_edsr:
        fieldnames += ["edsr_psnr", "edsr_ssim"]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    # --- Print summary table ---
    print("\n=== Average results over test set ===")
    print(f"{'Method':<10}{'PSNR (dB)':>12}{'SSIM':>10}")
    for method in ["bicubic", "srcnn", "edsr"]:
        if running[method]:
            arr = np.array(running[method])
            print(f"{method.upper():<10}{arr[:,0].mean():>12.2f}{arr[:,1].mean():>10.4f}")
    print(f"\nPer-image results saved to: {csv_path}")
    print(f"Visual comparisons saved to: {comparison_dir}/")


if __name__ == "__main__":
    main()
