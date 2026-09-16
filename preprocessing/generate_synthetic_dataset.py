"""
generate_synthetic_dataset.py
------------------------------
NOT a replacement for a real chromosome dataset — this exists purely so you
can smoke-test the ENTIRE pipeline (dataset -> LR generation -> training ->
evaluation -> Streamlit demo) end-to-end before you have real microscopy
images, or before you've cleared dataset licensing.

It generates simple synthetic "chromosome-like" X-shapes with banding
patterns on a noisy background — enough structure for SRCNN/EDSR to have
something non-trivial to learn, purely for pipeline verification.

Usage:
    python preprocessing/generate_synthetic_dataset.py --out_dir dataset --n_train 40 --n_val 8 --n_test 8 --hr_size 256
"""

import argparse
import os

import cv2
import numpy as np


def make_synthetic_chromosome(size: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    img = np.zeros((size, size), dtype=np.float32)

    cx, cy = size // 2 + rng.randint(-10, 10), size // 2 + rng.randint(-10, 10)
    length = size * rng.uniform(0.5, 0.7)
    thickness = max(3, int(size * rng.uniform(0.06, 0.1)))
    angle = rng.uniform(0, 180)

    # Draw two overlapping arms (chromatids) to form an X-like chromosome shape
    for arm_angle in [angle, angle + 90 + rng.uniform(-15, 15)]:
        theta = np.deg2rad(arm_angle)
        dx, dy = np.cos(theta), np.sin(theta)
        x1 = int(cx - dx * length / 2)
        y1 = int(cy - dy * length / 2)
        x2 = int(cx + dx * length / 2)
        y2 = int(cy + dy * length / 2)
        cv2.line(img, (x1, y1), (x2, y2), color=1.0, thickness=thickness, lineType=cv2.LINE_AA)

    # Slight constriction at the centromere
    cv2.circle(img, (cx, cy), max(2, thickness // 3), color=0.6, thickness=-1)

    # Banding pattern: modulate intensity with stripes along one axis
    band_freq = rng.uniform(0.15, 0.3)
    xx, yy = np.meshgrid(np.arange(size), np.arange(size))
    bands = 0.15 * np.sin(2 * np.pi * band_freq * (xx * np.cos(theta) + yy * np.sin(theta)) / size)
    img = np.clip(img + bands * (img > 0), 0, 1)

    # Gaussian smoothing to emulate microscope PSF, then background noise
    img = cv2.GaussianBlur(img, (0, 0), sigmaX=1.2)
    background = rng.normal(0.03, 0.01, (size, size)).astype(np.float32)
    img = np.clip(img + background, 0, 1)

    return (img * 255).astype(np.uint8)


def generate_split(out_hr_dir: str, n: int, hr_size: int, seed_offset: int):
    os.makedirs(out_hr_dir, exist_ok=True)
    for i in range(n):
        img = make_synthetic_chromosome(hr_size, seed=seed_offset + i)
        cv2.imwrite(os.path.join(out_hr_dir, f"chr{i+1:03d}.png"), img)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="dataset")
    parser.add_argument("--hr_size", type=int, default=256)
    parser.add_argument("--n_train", type=int, default=40)
    parser.add_argument("--n_val", type=int, default=8)
    parser.add_argument("--n_test", type=int, default=8)
    args = parser.parse_args()

    generate_split(os.path.join(args.out_dir, "train", "HR"), args.n_train, args.hr_size, seed_offset=0)
    generate_split(os.path.join(args.out_dir, "val", "HR"), args.n_val, args.hr_size, seed_offset=10_000)
    generate_split(os.path.join(args.out_dir, "test", "HR"), args.n_test, args.hr_size, seed_offset=20_000)

    print(f"Synthetic HR images written under {args.out_dir}/{{train,val,test}}/HR/")
    print("Next: run preprocessing/create_lr_images.py on each split to generate LR pairs.")


if __name__ == "__main__":
    main()
