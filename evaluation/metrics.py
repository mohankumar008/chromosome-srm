"""
metrics.py
----------
PSNR and SSIM computation, shared by training (validation loop) and
evaluation (Step 8 — Bicubic vs SRCNN vs EDSR comparison table).

Both functions expect float arrays in [0, 1], single-channel (grayscale).
"""

import numpy as np
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim


def compute_psnr(img_true: np.ndarray, img_pred: np.ndarray) -> float:
    img_true = np.clip(img_true, 0, 1)
    img_pred = np.clip(img_pred, 0, 1)
    return float(sk_psnr(img_true, img_pred, data_range=1.0))


def compute_ssim(img_true: np.ndarray, img_pred: np.ndarray) -> float:
    img_true = np.clip(img_true, 0, 1)
    img_pred = np.clip(img_pred, 0, 1)
    return float(sk_ssim(img_true, img_pred, data_range=1.0))
