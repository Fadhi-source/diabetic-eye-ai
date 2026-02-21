"""
models/multimodal_model.py
Top-level multi-modal model that assembles all branches.

          ┌─────────────┐   ┌───────────────┐
          │ Fundus Image │   │  EHR Tabular  │
          └──────┬───────┘   └───────┬───────┘
                 │                   │
          ┌──────▼───────┐   ┌───────▼───────┐
          │ ImageBranch  │   │ TabularBranch  │
          │ (EfficientNet)│   │    (MLP)       │
          └──────┬───────┘   └───────┬───────┘
                 │  (256)             │ (128)
                 └────────┬──────────┘
                          │ concat (384)
                   ┌──────▼───────┐
                   │  GatedFusion  │
                   └──────┬───────┘
                          │ (192)
                   ┌──────▼───────┐
                   │  Classifier  │
                   │  Linear→σ   │
                   └──────┬───────┘
                          │ scalar ∈ [0,1]
                    risk probability
"""

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

    Args:
        pretrained      : Load pretrained ImageNet weights for image branch
        num_classes     : 1 for binary, >1 for multi-label
        freeze_ratio    : Fraction of image backbone params to freeze initially
    """

    def __init__(
        self,
        pretrained: bool    = True,
        num_classes: int    = NUM_CLASSES,
        freeze_ratio: float = 0.70,
    ):
        super().__init__()

        self.image_branch   = ImageBranch(pretrained=pretrained,
                                          freeze_ratio=freeze_ratio)
        self.tabular_branch = TabularBranch()
        self.fusion         = GatedFusion()

        # ── Classification head ──────────────────────────────────────────────
        # Binary: single sigmoid neuron
        # Multi-label: k sigmoid neurons (independent probabilities)
        self.classifier = nn.Linear(FUSION_DIM, num_classes)
        self.num_classes = num_classes

        # Initialise classifier
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(
        self,
        image: torch.Tensor,
        tabular: torch.Tensor,
    ) -> dict:
        """
        Args:
            image   : (B, 3, H, W) normalised fundus image
            tabular : (B, TABULAR_INPUT_DIM) scaled clinical feature vector
        Returns:
            dict with keys:
              - "logits"     : raw logits (B, num_classes)
              - "probs"      : sigmoid probabilities (B, num_classes)
              - "img_emb"    : image embedding (B, img_dim) — for Grad-CAM
              - "tab_emb"    : tabular embedding (B, tab_dim) — for SHAP
              - "gate_weights": fusion gate activations (B, fusion_dim)
        """
        img_emb = self.image_branch(image)     # (B, 256)
        tab_emb = self.tabular_branch(tabular) # (B, 128)

        gate_weights = self.fusion.get_gate_weights(img_emb, tab_emb)
        fused        = self.fusion(img_emb, tab_emb)   # (B, 192)

        logits = self.classifier(fused)                # (B, num_classes)
        probs  = torch.sigmoid(logits)                 # (B, num_classes)

        return {
            "logits":       logits,
            "probs":        probs,
            "img_emb":      img_emb,
            "tab_emb":      tab_emb,
            "gate_weights": gate_weights,
        }

    def predict_proba(
        self,
        image: torch.Tensor,
        tabular: torch.Tensor,
    ) -> torch.Tensor:
        """Convenience method: returns just the probability tensor."""
        with torch.no_grad():
            return self.forward(image, tabular)["probs"]

    def fine_tune_mode(self) -> None:
        """
        Switch to end-to-end fine-tuning (Stage 4):
        Unfreeze the top 30% of the image backbone
        while leaving everything else trainable.
        """
        self.image_branch.unfreeze_top(top_ratio=0.30)
        print("[MultiModalModel] Entered fine-tuning mode")

    def count_parameters(self) -> dict:
        """Handy summary of trainable / frozen parameter counts."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen    = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        return {"trainable": trainable, "frozen": frozen, "total": trainable + frozen}


# ──────────────────────────────────────────────────────────────────────────────
# Quick sanity check
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from config import TABULAR_INPUT_DIM

    model = MultiModalModel(pretrained=False)
    params = model.count_parameters()
    print(f"Trainable params : {params['trainable']:,}")
    print(f"Frozen params    : {params['frozen']:,}")
    print(f"Total params     : {params['total']:,}")

    # Dummy forward pass
    imgs = torch.randn(4, 3, 224, 224)
    tabs = torch.randn(4, TABULAR_INPUT_DIM)
    out  = model(imgs, tabs)

    print(f"\nForward pass output shapes:")
    for k, v in out.items():
        print(f"  {k:15s}: {v.shape}")

    assert out["probs"].shape == (4, NUM_CLASSES), "Probs shape mismatch!"
    print("\nMultiModalModel test PASSED ✓")
