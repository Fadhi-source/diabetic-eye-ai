"""
models/image_branch.py
EfficientNet-B0 image encoder for the multi-modal diabetic predictor.

Architecture:
    EfficientNet-B0 (pretrained, ImageNet) → Global Average Pool
    → Linear(1280, IMAGE_EMBEDDING_DIM) → BatchNorm → ReLU → Dropout

Freezing strategy:
    The first FREEZE_RATIO of parameter groups are frozen.
    This preserves low-level features (edges, textures) learned on ImageNet
    while allowing the top layers to specialise in retinal features
    (exudates, haemorrhages, neovascularisation).
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import timm
import torch
import torch.nn as nn
from config import (
    IMAGE_BACKBONE, FREEZE_RATIO,
    IMAGE_EMBEDDING_DIM, DROPOUT_IMG
)


class ImageBranch(nn.Module):
    """
    EfficientNet-B0 backbone with a custom projection head.

    Args:
        pretrained        : Load ImageNet pretrained weights (default True)
        freeze_ratio      : Fraction of backbone params to freeze (0.0 – 1.0)
        embedding_dim     : Output dimension of the image embedding
        dropout           : Dropout rate for the projection head
    """

    def __init__(
        self,
        pretrained: bool        = True,
        freeze_ratio: float     = FREEZE_RATIO,
        embedding_dim: int      = IMAGE_EMBEDDING_DIM,
        dropout: float          = DROPOUT_IMG,
    ):
        super().__init__()

        # ── Load backbone ─────────────────────────────────────────────────────
        # num_classes=0  → removes the default classifier, returns 1280-d features
        self.backbone = timm.create_model(
            IMAGE_BACKBONE,
            pretrained=pretrained,
            num_classes=0,          # strip classifier
            global_pool="avg",      # Global Average Pooling built-in
        )

        # ── Freeze first freeze_ratio of parameter groups ─────────────────────
        self._apply_freezing(freeze_ratio)

        # ── Projection / embedding head ───────────────────────────────────────
        in_features = self.backbone.num_features   # 1280 for EfficientNet-B0
        self.head = nn.Sequential(
            nn.Linear(in_features, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )

        self.embedding_dim = embedding_dim

    def _apply_freezing(self, freeze_ratio: float) -> None:
        """Freeze the first `freeze_ratio` fraction of backbone parameters."""
        all_params = list(self.backbone.parameters())
        n_freeze   = int(len(all_params) * freeze_ratio)
        for i, param in enumerate(all_params):
            param.requires_grad = (i >= n_freeze)

        frozen = sum(1 for p in self.backbone.parameters() if not p.requires_grad)
        total  = sum(1 for p in self.backbone.parameters())
        print(f"[ImageBranch] Frozen {frozen}/{total} backbone params "
              f"({freeze_ratio * 100:.0f}%)")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, 3, H, W) normalised image tensor
        Returns:
            embedding : (B, embedding_dim)
        """
        features  = self.backbone(x)    # (B, 1280)
        embedding = self.head(features) # (B, embedding_dim)
        return embedding

    def unfreeze_top(self, top_ratio: float = 0.30) -> None:
        """
        Un-freeze the top `top_ratio` fraction of backbone parameters.
        Called during Stage 4 (end-to-end fine-tuning).
        """
        all_params = list(self.backbone.parameters())
        n_unfreeze = int(len(all_params) * top_ratio)
        for param in all_params[-n_unfreeze:]:
            param.requires_grad = True
        print(f"[ImageBranch] Unfrozen top {top_ratio * 100:.0f}% of backbone")


# ──────────────────────────────────────────────────────────────────────────────
# Quick sanity check
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = ImageBranch(pretrained=False)
    dummy = torch.randn(4, 3, 224, 224)
    out   = model(dummy)
    print(f"Output shape: {out.shape}")   # Expected: (4, 256)
    assert out.shape == (4, IMAGE_EMBEDDING_DIM), "Shape mismatch!"
    print("ImageBranch test PASSED ✓")
