"""
dataset.py
----------
PyTorch Dataset for paired LR/HR chromosome images, with optional random
patch cropping (Step 3 of the pipeline — this is what keeps GPU memory
requirements sane and multiplies the effective size of a small dataset).

Two dataset classes are provided:
    - SRDataset          : for EDSR-style models (LR stays small, network
                            learns the upsampling itself).
    - SRCNNDataset        : for SRCNN, which expects the LR image to already
                            be bicubic-upsampled to the HR image's size.
"""

import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def _list_images(folder: str):
    valid_ext = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
    return sorted(f for f in os.listdir(folder) if f.lower().endswith(valid_ext))


def _load_gray(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def _to_tensor(img: np.ndarray) -> torch.Tensor:
    """uint8 HxW image -> float32 tensor [1, H, W] normalized to [0, 1]."""
    img = img.astype(np.float32) / 255.0
    return torch.from_numpy(img).unsqueeze(0)


def _random_crop_pair(lr: np.ndarray, hr: np.ndarray, patch_size: int, scale: int):
    """Crop a random patch from LR and the corresponding patch from HR.
    Assumes hr.shape == scale * lr.shape."""
    lr_h, lr_w = lr.shape[:2]
    lr_patch = patch_size // scale
    if lr_h < lr_patch or lr_w < lr_patch:
        raise ValueError("Image smaller than requested patch size; reduce --patch_size.")

    x = random.randint(0, lr_w - lr_patch)
    y = random.randint(0, lr_h - lr_patch)

    lr_crop = lr[y : y + lr_patch, x : x + lr_patch]
    hr_crop = hr[y * scale : y * scale + patch_size, x * scale : x * scale + patch_size]
    return lr_crop, hr_crop


class SRDataset(Dataset):
    """For EDSR: returns (LR patch, HR patch) at native LR resolution."""

    def __init__(self, hr_dir: str, lr_dir: str, scale: int = 4, patch_size: int = 96, augment: bool = True):
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.scale = scale
        self.patch_size = patch_size
        self.augment = augment
        self.filenames = _list_images(hr_dir)
        if not self.filenames:
            raise RuntimeError(f"No images found in {hr_dir}")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        hr = _load_gray(os.path.join(self.hr_dir, fname))
        lr = _load_gray(os.path.join(self.lr_dir, fname))

        lr_crop, hr_crop = _random_crop_pair(lr, hr, self.patch_size, self.scale)

        if self.augment:
            if random.random() < 0.5:
                lr_crop = np.fliplr(lr_crop).copy()
                hr_crop = np.fliplr(hr_crop).copy()
            if random.random() < 0.5:
                lr_crop = np.flipud(lr_crop).copy()
                hr_crop = np.flipud(hr_crop).copy()
            k = random.choice([0, 1, 2, 3])
            lr_crop = np.rot90(lr_crop, k).copy()
            hr_crop = np.rot90(hr_crop, k).copy()

        return _to_tensor(lr_crop), _to_tensor(hr_crop)


class SRCNNDataset(Dataset):
    """For SRCNN: LR is bicubic-upsampled to HR size before being returned,
    since SRCNN does not learn upsampling itself — only refinement."""

    def __init__(self, hr_dir: str, lr_dir: str, patch_size: int = 96, augment: bool = True):
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.patch_size = patch_size
        self.augment = augment
        self.filenames = _list_images(hr_dir)
        if not self.filenames:
            raise RuntimeError(f"No images found in {hr_dir}")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        hr = _load_gray(os.path.join(self.hr_dir, fname))
        lr = _load_gray(os.path.join(self.lr_dir, fname))

        # Bicubic-upsample LR to HR's spatial size first
        lr_up = cv2.resize(lr, (hr.shape[1], hr.shape[0]), interpolation=cv2.INTER_CUBIC)

        h, w = hr.shape[:2]
        ps = self.patch_size
        if h < ps or w < ps:
            raise ValueError("Image smaller than requested patch size; reduce --patch_size.")
        x = random.randint(0, w - ps)
        y = random.randint(0, h - ps)

        hr_crop = hr[y : y + ps, x : x + ps]
        lr_crop = lr_up[y : y + ps, x : x + ps]

        if self.augment:
            if random.random() < 0.5:
                lr_crop = np.fliplr(lr_crop).copy()
                hr_crop = np.fliplr(hr_crop).copy()
            if random.random() < 0.5:
                lr_crop = np.flipud(lr_crop).copy()
                hr_crop = np.flipud(hr_crop).copy()

        return _to_tensor(lr_crop), _to_tensor(hr_crop)
