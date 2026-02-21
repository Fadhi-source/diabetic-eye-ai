"""
data/transforms.py
Image augmentation pipelines for 224×224 retinal fundus images.
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

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


class CircleCrop:
    """
    Crops the circular retinal field from a raw fundus image.
    Applies local contrast normalisation (Ben Graham preprocessing).
    """
    def __init__(self, sigmaX: float = 10.0):
        self.sigmaX = sigmaX

    def __call__(self, img: Image.Image) -> Image.Image:
        img_np = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img_np = cv2.addWeighted(
            img_np, 4,
            cv2.GaussianBlur(img_np, (0, 0), self.sigmaX), -4,
            128
        )
        h, w   = img_np.shape[:2]
        mask   = np.zeros_like(img_np)
        radius = int(min(h, w) * 0.45)
        cv2.circle(mask, (w // 2, h // 2), radius, (1, 1, 1), -1)
        img_np = np.clip(img_np * mask + 128 * (1 - mask), 0, 255).astype(np.uint8)
        return Image.fromarray(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))


def get_train_transforms(use_circle_crop: bool = False) -> transforms.Compose:
    pipeline = [CircleCrop()] if use_circle_crop else []
    pipeline += [
        transforms.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),
        transforms.RandomCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.08)),
    ]
    return transforms.Compose(pipeline)


def get_val_transforms(use_circle_crop: bool = False) -> transforms.Compose:
    pipeline = [CircleCrop()] if use_circle_crop else []
    pipeline += [
        transforms.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    return transforms.Compose(pipeline)


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Reverses ImageNet normalisation for display. Accepts (C,H,W) or (B,C,H,W)."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    if tensor.dim() == 4:
        mean = mean.unsqueeze(0)
        std  = std.unsqueeze(0)
    return (tensor * std + mean).clamp(0, 1)
