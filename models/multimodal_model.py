import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from config import FUSION_DIM, NUM_CLASSES

from models.image_branch   import ImageBranch
from models.tabular_branch import TabularBranch
from models.fusion         import GatedFusion


class MultiModalModel(nn.Module):
    """
    End-to-end multi-modal model for diabetic complication risk prediction.

    Fuses an EfficientNet-B0 image branch (256-d) with a tabular MLP branch (128-d)
    via gated attention fusion, then classifies into a binary risk probability.

    Args:
        pretrained   : Load pretrained ImageNet weights for image branch
        num_classes  : 1 for binary, >1 for multi-label
        freeze_ratio : Fraction of image backbone params to freeze initially
    """

    def __init__(
        self,
        pretrained: bool    = True,
        num_classes: int    = NUM_CLASSES,
        freeze_ratio: float = 0.70,
    ):
        super().__init__()

        self.image_branch   = ImageBranch(pretrained=pretrained, freeze_ratio=freeze_ratio)
        self.tabular_branch = TabularBranch()
        self.fusion         = GatedFusion()
        self.classifier     = nn.Linear(FUSION_DIM, num_classes)
        self.num_classes    = num_classes

        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, image: torch.Tensor, tabular: torch.Tensor) -> dict:
        """
        Args:
            image   : (B, 3, H, W)
            tabular : (B, TABULAR_INPUT_DIM)
        Returns:
            dict with keys: logits, probs, img_emb, tab_emb, gate_weights
        """
        img_emb      = self.image_branch(image)
        tab_emb      = self.tabular_branch(tabular)
        gate_weights = self.fusion.get_gate_weights(img_emb, tab_emb)
        fused        = self.fusion(img_emb, tab_emb)
        logits       = self.classifier(fused)
        probs        = torch.sigmoid(logits)

        return {
            "logits":       logits,
            "probs":        probs,
            "img_emb":      img_emb,
            "tab_emb":      tab_emb,
            "gate_weights": gate_weights,
        }

    def predict_proba(self, image: torch.Tensor, tabular: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.forward(image, tabular)["probs"]

    def fine_tune_mode(self) -> None:
        """Unfreeze top 30% of image backbone for end-to-end fine-tuning."""
        self.image_branch.unfreeze_top(top_ratio=0.30)
        print("[MultiModalModel] Entered fine-tuning mode")

    def count_parameters(self) -> dict:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen    = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        return {"trainable": trainable, "frozen": frozen, "total": trainable + frozen}


if __name__ == "__main__":
    from config import TABULAR_INPUT_DIM

    model  = MultiModalModel(pretrained=False)
    params = model.count_parameters()
    print(f"Trainable: {params['trainable']:,}  |  Frozen: {params['frozen']:,}")

    imgs = torch.randn(4, 3, 224, 224)
    tabs = torch.randn(4, TABULAR_INPUT_DIM)
    out  = model(imgs, tabs)

    for k, v in out.items():
        print(f"  {k:15s}: {v.shape}")

    assert out["probs"].shape == (4, NUM_CLASSES)
    print("MultiModalModel  ✓")
