"""
models/tabular_branch.py
3-layer MLP encoder for clinical (EHR) tabular features.

Architecture:
    Input (17-d) → Linear → BN → ReLU → Dropout
                 → Linear → BN → ReLU → Dropout
                 → Linear → BN → ReLU → Dropout
                 → Output embedding (128-d)

Design rationale:
  - BatchNorm before activation reduces sensitivity to feature scaling
    (important since clinical features span vastly different ranges)
  - Three hidden layers capture non-linear interactions (e.g., the joint
    effect of high HbA1c + long disease duration on complication risk)
  - Dropout at each layer provides regularisation on small clinical cohorts
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from typing import List
from config import (
    TABULAR_INPUT_DIM, TABULAR_HIDDEN_DIMS,
    TABULAR_EMBEDDING_DIM, DROPOUT_TAB
)


class TabularBranch(nn.Module):
    """
    Multi-layer perceptron encoder for clinical features.

    Args:
        input_dim     : Number of input features (default 17 from config)
        hidden_dims   : List of hidden layer sizes (default [128, 256, 128])
        embedding_dim : Output embedding dimension (default 128)
        dropout       : Dropout probability at each layer
    """

    def __init__(
        self,
        input_dim: int       = TABULAR_INPUT_DIM,
        hidden_dims: List[int] = TABULAR_HIDDEN_DIMS,
        embedding_dim: int   = TABULAR_EMBEDDING_DIM,
        dropout: float       = DROPOUT_TAB,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim

        # ── Build MLP as a sequence of (Linear → BN → ReLU → Dropout) blocks ─
        all_dims = [input_dim] + hidden_dims + [embedding_dim]
        layers   = []

        for in_d, out_d in zip(all_dims[:-1], all_dims[1:]):
            layers += [
                nn.Linear(in_d, out_d),
                nn.BatchNorm1d(out_d),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
            ]

        self.mlp = nn.Sequential(*layers)

        # Weight initialisation: Kaiming He for ReLU networks
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, input_dim) scaled tabular feature tensor
        Returns:
            embedding : (B, embedding_dim)
        """
        return self.mlp(x)


# ──────────────────────────────────────────────────────────────────────────────
# Quick sanity check
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = TabularBranch()
    dummy = torch.randn(8, TABULAR_INPUT_DIM)
    out   = model(dummy)
    print(f"Output shape: {out.shape}")   # Expected: (8, 128)
    assert out.shape == (8, TABULAR_EMBEDDING_DIM), "Shape mismatch!"
    print("TabularBranch test PASSED ✓")
