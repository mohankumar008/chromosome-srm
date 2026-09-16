"""
EDSR — Enhanced Deep Super-Resolution network (simplified, single-scale)
-------------------------------------------------------------------------
Reference: Lim et al., "Enhanced Deep Residual Networks for Single Image
Super-Resolution" (CVPRW 2017).

Unlike SRCNN, EDSR takes a LOW-resolution image at its NATIVE (small) size
and learns the upsampling itself via a PixelShuffle sub-pixel convolution
layer at the end. This is the main model for this project.

Architecture:

    Input LR
       |
       v
    Initial Conv (head)
       |
       v
    [ Residual Block ] x N   (Conv -> ReLU -> Conv, scaled skip connection)
       |
       v
    Conv + long skip connection (add head features back in)
       |
       v
    Upsampling (PixelShuffle, x2 / x4)
       |
       v
    Final reconstruction Conv
       |
       v
    SR image
"""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Standard EDSR residual block: Conv -> ReLU -> Conv, no batch-norm
    (EDSR intentionally removes BatchNorm — it hurts SR quality), with a
    residual scaling factor for training stability at greater depth."""

    def __init__(self, num_features: int, res_scale: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.res_scale = res_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.conv2(self.relu(self.conv1(x)))
        return x + residual * self.res_scale


class UpsampleBlock(nn.Module):
    """Sub-pixel convolution upsampling block, supports scale 2, 3, 4."""

    def __init__(self, num_features: int, scale: int):
        super().__init__()
        layers = []
        if scale in (2, 4, 8) and (scale & (scale - 1)) == 0:
            # Repeated x2 upsampling (works for 2, 4, 8)
            for _ in range(int(scale.bit_length() - 1)):
                layers.append(nn.Conv2d(num_features, num_features * 4, 3, padding=1))
                layers.append(nn.PixelShuffle(2))
        elif scale == 3:
            layers.append(nn.Conv2d(num_features, num_features * 9, 3, padding=1))
            layers.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f"Unsupported upsampling scale: {scale}")
        self.body = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class EDSR(nn.Module):
    def __init__(
        self,
        num_channels: int = 1,
        num_features: int = 64,
        num_res_blocks: int = 16,
        scale: int = 4,
        res_scale: float = 0.1,
    ):
        """
        Args:
            num_channels:   1 (grayscale) or 3 (RGB)
            num_features:   width of the network (paper default 256, we use
                             64 by default — lighter, easier to train on a
                             single GPU / limited chromosome dataset)
            num_res_blocks: depth of the network (paper default 32; 16 is a
                             sensible starting point for a final-year project)
            scale:          upsampling factor, e.g. 4 for 256x256 -> 1024x1024
            res_scale:      residual scaling inside each residual block
        """
        super().__init__()

        self.head = nn.Conv2d(num_channels, num_features, kernel_size=3, padding=1)

        self.body = nn.Sequential(
            *[ResidualBlock(num_features, res_scale) for _ in range(num_res_blocks)]
        )
        self.body_conv = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)

        self.upsample = UpsampleBlock(num_features, scale)

        self.tail = nn.Conv2d(num_features, num_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        head_feat = self.head(x)
        body_feat = self.body(head_feat)
        body_feat = self.body_conv(body_feat)
        feat = head_feat + body_feat  # long skip connection
        feat = self.upsample(feat)
        out = self.tail(feat)
        return out


if __name__ == "__main__":
    # Quick sanity check: 256x256 input at scale=4 should produce 1024x1024.
    model = EDSR(num_channels=1, num_features=64, num_res_blocks=8, scale=4)
    dummy = torch.randn(1, 1, 256, 256)
    out = model(dummy)
    print("EDSR input :", dummy.shape)
    print("EDSR output:", out.shape)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")
