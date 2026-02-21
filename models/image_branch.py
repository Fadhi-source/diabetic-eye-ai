import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import timm
import torch
import torch.nn as nn
from config import IMAGE_BACKBONE, FREEZE_RATIO, IMAGE_EMBEDDING_DIM, DROPOUT_IMG


class ImageBranch(nn.Module):
    """
    EfficientNet-B0 encoder with a projection head.

    Args:
        pretrained    : Load ImageNet pretrained weights
        freeze_ratio  : Fraction of backbone params to freeze initially
        embedding_dim : Output embedding dimension
        dropout       : Dropout rate on the projection head
    """

    def __init__(
        self,
        pretrained: bool    = True,
        freeze_ratio: float = FREEZE_RATIO,
        embedding_dim: int  = IMAGE_EMBEDDING_DIM,
        dropout: float      = DROPOUT_IMG,
    ):
        super().__init__()

        self.backbone = timm.create_model(
            IMAGE_BACKBONE,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        self._apply_freezing(freeze_ratio)

        in_features = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Linear(in_features, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )
        self.embedding_dim = embedding_dim

    def _apply_freezing(self, freeze_ratio: float) -> None:
        all_params = list(self.backbone.parameters())
        n_freeze = int(len(all_params) * freeze_ratio)
        for i, param in enumerate(all_params):
            param.requires_grad = (i >= n_freeze)

        frozen = sum(1 for p in self.backbone.parameters() if not p.requires_grad)
        total  = sum(1 for p in self.backbone.parameters())
        print(f"[ImageBranch] Frozen {frozen}/{total} backbone params ({freeze_ratio*100:.0f}%)")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))

    def unfreeze_top(self, top_ratio: float = 0.30) -> None:
        """Unfreeze the top `top_ratio` fraction of backbone params for fine-tuning."""
        all_params = list(self.backbone.parameters())
        n_unfreeze = int(len(all_params) * top_ratio)
        for param in all_params[-n_unfreeze:]:
            param.requires_grad = True
        print(f"[ImageBranch] Unfrozen top {top_ratio*100:.0f}% of backbone")


if __name__ == "__main__":
    model = ImageBranch(pretrained=False)
    dummy = torch.randn(4, 3, 224, 224)
    out   = model(dummy)
    assert out.shape == (4, IMAGE_EMBEDDING_DIM)
    print(f"Output shape: {out.shape}  ✓")
