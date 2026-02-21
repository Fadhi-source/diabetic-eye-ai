"""
explainability/gradcam.py
Grad-CAM++ heatmap generation for the EfficientNet image branch.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
import cv2
from PIL import Image
from typing import Optional

from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from models.multimodal_model import MultiModalModel
from data.transforms import get_val_transforms
from config import TABULAR_INPUT_DIM


class ImageOnlyWrapper(nn.Module):
    """
    Wraps the full multi-modal model with a fixed tabular input so Grad-CAM
    can compute image gradients via a single-input forward pass.
    """

    def __init__(self, full_model: MultiModalModel, tabular_vec: torch.Tensor):
        super().__init__()
        self.full_model  = full_model
        self.tabular_vec = tabular_vec

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        tab = self.tabular_vec.expand(image.shape[0], -1).to(image.device)
        return self.full_model(image, tab)["logits"]


def generate_gradcam(
    model: MultiModalModel,
    image_tensor: torch.Tensor,
    tabular_tensor: torch.Tensor,
    target_class: int = 0,
    device: str       = "cpu",
) -> np.ndarray:
    """
    Generate a Grad-CAM++ heatmap.

    Args:
        model          : Trained MultiModalModel
        image_tensor   : (1, 3, H, W) normalised image tensor
        tabular_tensor : (1, TABULAR_INPUT_DIM) tabular feature tensor
        target_class   : Class index for gradient computation
        device         : "cpu" or "cuda"

    Returns:
        cam : (H, W) float32 ndarray in [0, 1]
    """
    model.eval().to(device)
    image_tensor   = image_tensor.to(device)
    tabular_tensor = tabular_tensor.to(device)

    wrapper       = ImageOnlyWrapper(model, tabular_tensor)
    target_layers = [model.image_branch.backbone.blocks[-1]]

    with GradCAMPlusPlus(model=wrapper, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=image_tensor, targets=[ClassifierOutputTarget(target_class)])

    return grayscale_cam[0]


def overlay_gradcam_on_image(
    original_image_np: np.ndarray,
    cam: np.ndarray,
) -> np.ndarray:
    """Overlay Grad-CAM heatmap on the original image (RGB uint8)."""
    return show_cam_on_image(original_image_np, cam, use_rgb=True)


def explain_image(
    model: MultiModalModel,
    pil_image: Image.Image,
    tabular_tensor: torch.Tensor,
    device: str = "cpu",
) -> tuple:
    """
    End-to-end Grad-CAM pipeline from a PIL image.

    Returns:
        (overlay_pil, cam) — PIL image with heatmap overlaid, raw (H,W) heatmap
    """
    transform    = get_val_transforms(use_circle_crop=False)
    image_tensor = transform(pil_image.convert("RGB")).unsqueeze(0)
    orig_np      = np.array(pil_image.convert("RGB").resize((224, 224))) / 255.0

    cam     = generate_gradcam(model, image_tensor, tabular_tensor, device=device)
    overlay = overlay_gradcam_on_image(orig_np.astype(np.float32), cam)

    return Image.fromarray(overlay), cam
