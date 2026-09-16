"""
create_lr_images.py
--------------------
Step 2 of the pipeline: generate medium/low-resolution training images from
your high-resolution chromosome images.

    HR image (e.g. 1024x1024)
        |
        v
    Downsample (bicubic) -> e.g. 256x256
        |
        v
    Optional degradation: Gaussian blur, Gaussian noise, JPEG compression
        |
        v
    LR / medium-resolution image (saved to disk)

Usage:
    python create_lr_images.py --hr_dir dataset/train/HR --lr_dir dataset/train/LR \
        --scale 4 --blur --noise --jpeg

Run this once per split (train / val / test). The HR and LR folders will
contain files with matching filenames, e.g.:

    train/HR/chr001.png   (1024x1024, ground truth)
    train/LR/chr001.png   (256x256,  degraded input)
"""

import argparse
import os

import cv2
import numpy as np


def degrade_image(
    hr_image: np.ndarray,
    scale: int = 4,
    add_blur: bool = False,
    add_noise: bool = False,
    add_jpeg: bool = False,
    blur_sigma: float = 1.0,
    noise_sigma: float = 5.0,
    jpeg_quality: int = 70,
) -> np.ndarray:
    """Turn a high-resolution image into a realistic medium/low-resolution
    version. Order matters: blur -> downsample -> noise -> JPEG, which
    mimics how a real microscope/camera pipeline degrades detail before
    compression artifacts are introduced.
    """
    img = hr_image.copy()

    # Optional blur to emulate optical/motion blur before downsampling
    if add_blur:
        k = max(3, int(blur_sigma * 3) | 1)  # ensure odd kernel size
        img = cv2.GaussianBlur(img, (k, k), blur_sigma)

    # Downsample using bicubic interpolation (creates the medium-res image)
    h, w = img.shape[:2]
    lr = cv2.resize(
        img, (w // scale, h // scale), interpolation=cv2.INTER_CUBIC
    )

    # Optional sensor/scan noise
    if add_noise:
        noise = np.random.normal(0, noise_sigma, lr.shape).astype(np.float32)
        lr = np.clip(lr.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # Optional JPEG compression artifacts
    if add_jpeg:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
        success, encimg = cv2.imencode(".jpg", lr, encode_param)
        if success:
            lr = cv2.imdecode(encimg, cv2.IMREAD_UNCHANGED)

    return lr


def process_directory(hr_dir: str, lr_dir: str, scale: int, args) -> None:
    os.makedirs(lr_dir, exist_ok=True)
    valid_ext = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
    files = [f for f in sorted(os.listdir(hr_dir)) if f.lower().endswith(valid_ext)]

    if not files:
        print(f"No images found in {hr_dir}. Nothing to do.")
        return

    for fname in files:
        hr_path = os.path.join(hr_dir, fname)
        hr_img = cv2.imread(hr_path, cv2.IMREAD_GRAYSCALE)
        if hr_img is None:
            print(f"  [skip] could not read {hr_path}")
            continue

        lr_img = degrade_image(
            hr_img,
            scale=scale,
            add_blur=args.blur,
            add_noise=args.noise,
            add_jpeg=args.jpeg,
            blur_sigma=args.blur_sigma,
            noise_sigma=args.noise_sigma,
            jpeg_quality=args.jpeg_quality,
        )

        out_path = os.path.join(lr_dir, fname)
        cv2.imwrite(out_path, lr_img)

    print(f"Done: {len(files)} images processed -> {lr_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate LR images from HR chromosome images.")
    parser.add_argument("--hr_dir", type=str, required=True, help="Folder of high-resolution images")
    parser.add_argument("--lr_dir", type=str, required=True, help="Output folder for LR images")
    parser.add_argument("--scale", type=int, default=4, help="Downsampling factor (e.g. 4 = 1024->256)")
    parser.add_argument("--blur", action="store_true", help="Apply Gaussian blur before downsampling")
    parser.add_argument("--noise", action="store_true", help="Add Gaussian noise after downsampling")
    parser.add_argument("--jpeg", action="store_true", help="Apply JPEG compression artifacts")
    parser.add_argument("--blur_sigma", type=float, default=1.0)
    parser.add_argument("--noise_sigma", type=float, default=5.0)
    parser.add_argument("--jpeg_quality", type=int, default=70)
    args = parser.parse_args()

    process_directory(args.hr_dir, args.lr_dir, args.scale, args)


if __name__ == "__main__":
    main()
