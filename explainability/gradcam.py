"""
explainability/gradcam.py
Grad-CAM++ visualisation for the EfficientNet image branch.

Grad-CAM (Gradient-weighted Class Activation Mapping) computes the gradient of
the class score with respect to the last convolutional feature map, then
weighted-averages those feature maps to produce a localised heatmap.
The result visually highlights WHICH retinal regions (e.g., optic disc,
macula, haemorrhages) caused the model to predict a high complication risk.

We use the `pytorch-grad-cam` library (grad-cam PyPI package) which handles
all the hook registration and backward pass management.
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
from pytorch_grad_cam.utils.image import (
    show_cam_on_image,
    preprocess_image,
)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from models.multimodal_model import MultiModalModel
from data.transforms import get_val_transforms, denormalize
from config import TABULAR_INPUT_DIM


# ──────────────────────────────────────────────────────────────────────────────
# Wrapper: expose only the image branch for Grad-CAM
# (Grad-CAM needs a model whose forward() accepts a single image tensor)
# ──────────────────────────────────────────────────────────────────────────────

class ImageOnlyWrapper(nn.Module):
    """
    Wraps the full multi-modal model so that Grad-CAM can interact with
    only the image branch + classifier, keeping tabular input fixed.

    Args:
        full_model  : The trained MultiModalModel
        tabular_vec : Fixed tabular feature tensor (1, TABULAR_INPUT_DIM)
    """

    def __init__(self, full_model: MultiModalModel, tabular_vec: torch.Tensor):
        super().__init__()
        self.full_model  = full_model
        self.tabular_vec = tabular_vec

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        B = image.shape[0]
        tab = self.tabular_vec.expand(B, -1).to(image.device)
        out = self.full_model(image, tab)
        return out["logits"]    # Return logits for Grad-CAM


# ──────────────────────────────────────────────────────────────────────────────
# Main Grad-CAM function
# ──────────────────────────────────────────────────────────────────────────────

def generate_gradcam(
    model: MultiModalModel,
    image_tensor: torch.Tensor,
    tabular_tensor: torch.Tensor,
    target_class: int           = 0,
    device: str                 = "cpu",
) -> np.ndarray:
    """
    Generate a Grad-CAM++ heatmap for the given image + tabular input.

    Args:
        model          : Trained MultiModalModel
        image_tensor   : (1, 3, H, W) normalised image tensor (on CPU/GPU)
        tabular_tensor : (1, TABULAR_INPUT_DIM) tabular feature tensor
        target_class   : Class index to compute gradients for (0 for binary)
        device         : "cpu" or "cuda"

    Returns:
        cam : (H, W) float32 ndarray in [0, 1], the Grad-CAM heatmap
    """
    model.eval()
    model.to(device)
    image_tensor   = image_tensor.to(device)
    tabular_tensor = tabular_tensor.to(device)

    # Create the image-only wrapper
    wrapper = ImageOnlyWrapper(model, tabular_tensor)

    # Target: the last EfficientNet convolutional block's activation
    target_layers = [model.image_branch.backbone.blocks[-1]]

    with GradCAMPlusPlus(model=wrapper, target_layers=target_layers) as cam:
        targets = [ClassifierOutputTarget(target_class)]
        grayscale_cam = cam(
            input_tensor=image_tensor,
            targets=targets,
        )

    return grayscale_cam[0]   # (H, W)


def overlay_gradcam_on_image(
    original_image_np: np.ndarray,
    cam: np.ndarray,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Overlay the Grad-CAM heatmap on the original image.

    Args:
        original_image_np : (H, W, 3) float32 image in [0, 1]
        cam               : (H, W) float32 heatmap in [0, 1]
        colormap          : OpenCV colormap (default COLORMAP_JET)

    Returns:
        overlay : (H, W, 3) uint8 visualisation
    """
    return show_cam_on_image(original_image_np, cam, use_rgb=True)


# ──────────────────────────────────────────────────────────────────────────────
# High-level convenience function (used by Streamlit app)
# ──────────────────────────────────────────────────────────────────────────────

def explain_image(
    model: MultiModalModel,
    pil_image: Image.Image,
    tabular_tensor: torch.Tensor,
    device: str = "cpu",
) -> tuple:
    """
    End-to-end Grad-CAM pipeline starting from a raw PIL image.

    Returns:
        overlay_pil : PIL.Image — original image with heatmap overlaid
        cam         : (H, W) ndarray — raw heatmap (for further processing)
    """
    transform     = get_val_transforms(use_circle_crop=False)
    image_tensor  = transform(pil_image.convert("RGB")).unsqueeze(0)   # (1,3,H,W)

    # Original image as float32 [0,1] for overlay
    import torchvision.transforms.functional as TF
    orig_np = np.array(pil_image.convert("RGB").resize((224, 224))) / 255.0

    cam     = generate_gradcam(model, image_tensor, tabular_tensor, device=device)
    overlay = overlay_gradcam_on_image(orig_np.astype(np.float32), cam)

    overlay_pil = Image.fromarray(overlay)
    return overlay_pil, cam
