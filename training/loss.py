import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import FOCAL_LOSS_GAMMA, FOCAL_LOSS_ALPHA


class BinaryFocalLoss(nn.Module):
    """
    Focal Loss for binary classification.

    Down-weights easy negatives via (1 - p_t)^γ, forcing the model to focus
    on hard positives — critical for class-imbalanced clinical datasets.

    L = -α · (1 - p_t)^γ · log(p_t)

    Args:
        alpha  : Weight for the positive class (default 0.25)
        gamma  : Focusing parameter — higher values = more focus on hard examples
        reduce : Return mean scalar if True, per-sample tensor if False
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

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits  = logits.view(-1)
        targets = targets.view(-1).float()

        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs    = torch.sigmoid(logits)
        p_t      = probs * targets + (1.0 - probs) * (1.0 - targets)

        focal_weight = (1.0 - p_t) ** self.gamma
        alpha_t      = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)

        loss = alpha_t * focal_weight * bce_loss
        return loss.mean() if self.reduce else loss


class MultiLabelFocalLoss(nn.Module):
    """
    Focal Loss for multi-label classification (independent sigmoid per label).
    Extend this if predicting DR + Nephropathy + Neuropathy simultaneously.
    """

    def __init__(self, alpha: float = FOCAL_LOSS_ALPHA, gamma: float = FOCAL_LOSS_GAMMA):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        bce     = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs   = torch.sigmoid(logits)
        p_t     = probs * targets + (1.0 - probs) * (1 - targets)
        focal_w = (1 - p_t) ** self.gamma
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (alpha_t * focal_w * bce).mean()


if __name__ == "__main__":
    loss_fn = BinaryFocalLoss()
    logits  = torch.tensor([ 2.0, -1.5,  0.3, -0.8])
    targets = torch.tensor([ 1.0,  0.0,  1.0,  0.0])
    loss    = loss_fn(logits, targets)
    print(f"Focal loss: {loss.item():.4f}  ✓")
