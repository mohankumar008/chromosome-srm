"""
SRCNN — Super-Resolution Convolutional Neural Network
------------------------------------------------------
Reference: Dong et al., "Image Super-Resolution Using Deep Convolutional
Networks" (2014/2016).

Pipeline (conceptually):
    LR (already upsampled to target size) -> Feature Extraction
                                           -> Non-linear Mapping
                                           -> Reconstruction
                                           -> SR image

SRCNN is intentionally the *baseline deep model* in this project — simple,
fast to train, easy to reason about — before moving to EDSR.

IMPORTANT: SRCNN expects the LR input to already be resized (e.g. via
bicubic interpolation) to the SAME spatial size as the HR target. It does
not do learned upsampling; it only refines a pre-upsampled image. Keep this
in mind in the dataset / training pipeline.
"""

import torch
import torch.nn as nn


class SRCNN(nn.Module):
    def __init__(self, num_channels: int = 1):
        """
        Args:
            num_channels: 1 for grayscale chromosome microscopy images,
                          3 if you decide to keep RGB.
        """
        super().__init__()

        # Patch extraction and representation
        self.feature_extraction = nn.Conv2d(
            num_channels, 64, kernel_size=9, padding=9 // 2
        )

        # Non-linear mapping (feature space -> feature space)
        self.mapping = nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2)

        # Reconstruction
        self.reconstruction = nn.Conv2d(
            32, num_channels, kernel_size=5, padding=5 // 2
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.feature_extraction(x))
        x = self.relu(self.mapping(x))
        x = self.reconstruction(x)
        return x


if __name__ == "__main__":
    # Quick sanity check: shapes should match (SRCNN does not change size).
    model = SRCNN(num_channels=1)
    dummy = torch.randn(2, 1, 256, 256)
    out = model(dummy)
    print("SRCNN input :", dummy.shape)
    print("SRCNN output:", out.shape)
