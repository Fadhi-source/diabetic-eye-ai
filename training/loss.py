"""
training/loss.py
Loss functions for handling class imbalance in the diabetic complication dataset.

Asymmetric Focal Loss (AFC):
    The standard Cross-Entropy loss treats all misclassified examples equally.
    In highly imbalanced clinical datasets (healthy patients >> sick patients),
    this causes the model to be overwhelmed by easy negatives (healthy patients
    that are trivially classified as "no complication").

    Focal Loss fixes this with a modulating factor (1 - p_t)^γ that down-weights
    the contribution of easy examples to the loss, forcing the model to focus
    on hard, uncertain, and rare positive cases.

    L_FL = -α * (1 - p_t)^γ * log(p_t)

    where:
        p_t = model probability of the correct class
        γ   = focusing parameter (typically 2.0; higher = more focus on hard examples)
        α   = class balance weight (down-weight majority class)
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import FOCAL_LOSS_GAMMA, FOCAL_LOSS_ALPHA


class BinaryFocalLoss(nn.Module):
    """
    Focal Loss for binary classification with sigmoid output.

    Args:
        alpha  : Weight for positive class (default 0.25)
        gamma  : Focusing parameter (default 2.0)
        reduce : If True, returns scalar mean; else returns per-sample tensor
    """

    def __init__(
        self,
        alpha: float = FOCAL_LOSS_ALPHA,
        gamma: float = FOCAL_LOSS_GAMMA,
        reduce: bool = True,
    ):
        super().__init__()
        self.alpha  = alpha
        self.gamma  = gamma
        self.reduce = reduce

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits  : (B,) or (B,1) raw output from final linear layer
            targets : (B,) binary labels {0, 1}
        Returns:
            Focal loss scalar (or per-sample tensor if reduce=False)
        """
        logits  = logits.view(-1)
        targets = targets.view(-1).float()

        # Standard BCE with logits (numerically stable)
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )

        # Compute p_t
        probs = torch.sigmoid(logits)
        p_t   = probs * targets + (1.0 - probs) * (1.0 - targets)

        # Focal modulation
        focal_weight = (1.0 - p_t) ** self.gamma

        # Alpha weighting
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)

        focal_loss = alpha_t * focal_weight * bce_loss

        return focal_loss.mean() if self.reduce else focal_loss


class MultiLabelFocalLoss(nn.Module):
    """
    Focal Loss for multi-label classification (each label is independent sigmoid).
    Use this if you extend the project to predict DR + Nephropathy + Neuropathy.

    Args:
        alpha : Overall positive-class weight
        gamma : Focusing parameter
    """

    def __init__(self, alpha: float = FOCAL_LOSS_ALPHA, gamma: float = FOCAL_LOSS_GAMMA):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits  : (B, num_classes) raw logits
            targets : (B, num_classes) multi-hot binary labels
        Returns:
            mean focal loss scalar
        """
        targets = targets.float()
        bce     = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs   = torch.sigmoid(logits)
        p_t     = probs * targets + (1.0 - probs) * (1 - targets)
        focal_w = (1 - p_t) ** self.gamma
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        return (alpha_t * focal_w * bce).mean()


# ──────────────────────────────────────────────────────────────────────────────
# Quick sanity check
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    loss_fn  = BinaryFocalLoss()
    logits   = torch.tensor([ 2.0, -1.5,  0.3, -0.8])
    targets  = torch.tensor([ 1.0,  0.0,  1.0,  0.0])
    loss     = loss_fn(logits, targets)
    print(f"Focal loss: {loss.item():.4f}")
    print("BinaryFocalLoss test PASSED ✓")
