"""
data/transforms.py
Image augmentation pipelines for training and validation/test sets.
Uses torchvision transforms. Designed for 224×224 retinal fundus images.

Ben Graham's circle-crop preprocessing (used in benchmark solutions) is
included as an optional callable that can be applied before augmentation.
"""

import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import IMAGE_SIZE


# ──────────────────────────────────────────────
# ImageNet normalization stats
# ──────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ──────────────────────────────────────────────
# Optional: Ben Graham circle crop
# Removes non-retinal black border for cleaner features
# ──────────────────────────────────────────────
class CircleCrop:
    """
    Crops the circular retinal field from a raw fundus image.
    Applies a Gaussian blur and scales brightness to a standard level.
    This is a known preprocessing step that boosts DR grading performance.
    """
    def __init__(self, sigmaX: float = 10.0):
        self.sigmaX = sigmaX

    def __call__(self, img: Image.Image) -> Image.Image:
        img_np = np.array(img)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # Blur and subtract to normalise illumination
        img_np = cv2.addWeighted(
            img_np, 4,
            cv2.GaussianBlur(img_np, (0, 0), self.sigmaX), -4,
            128
        )

        # Mask outside circle
        h, w = img_np.shape[:2]
        mask = np.zeros_like(img_np)
        radius = int(min(h, w) * 0.45)
        cy, cx = h // 2, w // 2
        cv2.circle(mask, (cx, cy), radius, (1, 1, 1), -1)
        img_np = img_np * mask + 128 * (1 - mask)
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)

        img_pil = Image.fromarray(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))
        return img_pil


# ──────────────────────────────────────────────
# Training transforms (aggressive augmentation)
# ──────────────────────────────────────────────
def get_train_transforms(use_circle_crop: bool = False) -> transforms.Compose:
    """
    Augmentation pipeline for training. Includes:
    - Random horizontal/vertical flips (retinas are symmetric)
    - Random rotation ±15°
    - Color jitter (simulate lighting variation across devices)
    - Random grid distortion via elastic transforms via RandomAffine
    - Normalize to ImageNet stats
    """
    transform_list = []

    if use_circle_crop:
        transform_list.append(CircleCrop(sigmaX=10.0))

    transform_list += [
        transforms.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),
        transforms.RandomCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.2,
            hue=0.05
        ),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.05, 0.05),
            scale=(0.95, 1.05),
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        # Random erasing simulates low-quality patches / artefacts
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.08)),
    ]

    return transforms.Compose(transform_list)


# ──────────────────────────────────────────────
# Validation / Test transforms (deterministic)
# ──────────────────────────────────────────────
def get_val_transforms(use_circle_crop: bool = False) -> transforms.Compose:
    """
    Deterministic validation/test pipeline. Only resize + centre crop + normalize.
    """
    transform_list = []

    if use_circle_crop:
        transform_list.append(CircleCrop(sigmaX=10.0))

    transform_list += [
        transforms.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]

    return transforms.Compose(transform_list)


# ──────────────────────────────────────────────
# Convenience: inverse transform (for Grad-CAM display)
# ──────────────────────────────────────────────
def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """
    Reverses ImageNet normalisation so a tensor can be displayed as an image.
    tensor: (C, H, W) or (B, C, H, W)
    """
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    if tensor.dim() == 4:
        mean = mean.unsqueeze(0)
        std  = std.unsqueeze(0)
    return (tensor * std + mean).clamp(0, 1)
