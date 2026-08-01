"""Medical-safe weak/strong augmentations for FixMatch / Mean Teacher.

Weak view = existing dataloader train transform (normalize + ResizedRandomCrop).
Strong view = additional photometric / blur / small rotation on the tensor.
No horizontal flip (preserves laterality for hip radiographs).
"""
from __future__ import annotations

import torch
import torchvision.transforms.functional as TF


class StrongAugment:
    """Apply a mild strong augmentation to a BCHW float tensor in [0, 1]-ish range."""

    def __init__(
        self,
        brightness: float = 0.2,
        contrast: float = 0.2,
        blur_sigma: tuple[float, float] = (0.1, 1.5),
        rotation_deg: float = 5.0,
        p_blur: float = 0.5,
        p_rotate: float = 0.5,
    ):
        self.brightness = brightness
        self.contrast = contrast
        self.blur_sigma = blur_sigma
        self.rotation_deg = rotation_deg
        self.p_blur = p_blur
        self.p_rotate = p_rotate

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (B, C, H, W) float tensor already weakly augmented / normalized.
        Returns:
            Strongly augmented copy of the same shape.
        """
        out = []
        for img in images:
            x = img
            # Brightness / contrast (per-image, independent of batch)
            b = 1.0 + (torch.rand(1).item() * 2 - 1) * self.brightness
            c = 1.0 + (torch.rand(1).item() * 2 - 1) * self.contrast
            x = TF.adjust_brightness(x, b)
            x = TF.adjust_contrast(x, c)
            if torch.rand(1).item() < self.p_blur:
                sigma = (
                    self.blur_sigma[0]
                    + torch.rand(1).item() * (self.blur_sigma[1] - self.blur_sigma[0])
                )
                x = TF.gaussian_blur(x, kernel_size=5, sigma=[sigma, sigma])
            if torch.rand(1).item() < self.p_rotate:
                angle = (torch.rand(1).item() * 2 - 1) * self.rotation_deg
                x = TF.rotate(x, angle)
            out.append(x)
        return torch.stack(out, dim=0)
