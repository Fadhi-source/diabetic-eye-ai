"""
models/fusion.py
Gated attention fusion layer combining image and tabular embeddings.

Mathematical formulation:
    concat = [e_img ; e_tab]                     (B, img_dim + tab_dim)
    gate   = sigmoid(W_g @ concat + b_g)          (B, fusion_dim)
    feat   = tanh(W_f @ concat + b_f)             (B, fusion_dim)
    h      = gate ⊙ feat                           (B, fusion_dim)

Intuition:
    The sigmoid gate learns to up/down-weight each dimension of the fused
    representation. If the fundus image is blurry (low quality), the gate
    can suppress the image-driven dimensions and rely more on clinical data.
    This makes the model robust to missing or low-quality modalities —
    a critical requirement in rural Indian hospital settings.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from config import (
    IMAGE_EMBEDDING_DIM, TABULAR_EMBEDDING_DIM, FUSION_DIM
)


class GatedFusion(nn.Module):
    """
    Gated multi-modal fusion of image and tabular embeddings.

    Args:
        img_dim    : Image embedding dimension (default 256)
        tab_dim    : Tabular embedding dimension (default 128)
        fusion_dim : Output fusion dimension (default 192)
        dropout    : Dropout on the fused representation
    """

    def __init__(
        self,
        img_dim: int    = IMAGE_EMBEDDING_DIM,
        tab_dim: int    = TABULAR_EMBEDDING_DIM,
        fusion_dim: int = FUSION_DIM,
        dropout: float  = 0.30,
    ):
        super().__init__()

        concat_dim = img_dim + tab_dim   # 256 + 128 = 384

        # The two linear transformations that form the gated unit
        self.gate_linear = nn.Linear(concat_dim, fusion_dim)
        self.feat_linear = nn.Linear(concat_dim, fusion_dim)

        self.gate_bn = nn.BatchNorm1d(fusion_dim)
        self.feat_bn = nn.BatchNorm1d(fusion_dim)

        self.dropout = nn.Dropout(p=dropout)

        self.fusion_dim = fusion_dim

        # Weight init
        nn.init.xavier_uniform_(self.gate_linear.weight)
        nn.init.xavier_uniform_(self.feat_linear.weight)

    def forward(
        self,
        img_emb: torch.Tensor,
        tab_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            img_emb : (B, img_dim)   — image branch embedding
            tab_emb : (B, tab_dim)   — tabular branch embedding
        Returns:
            fused   : (B, fusion_dim)
        """
        concat = torch.cat([img_emb, tab_emb], dim=1)   # (B, 384)

        gate  = torch.sigmoid(self.gate_bn(self.gate_linear(concat)))   # (B, 192)
        feat  = torch.tanh(  self.feat_bn(self.feat_linear(concat)))     # (B, 192)
        fused = gate * feat                                               # (B, 192)

        return self.dropout(fused)

    def get_gate_weights(
        self,
        img_emb: torch.Tensor,
        tab_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Returns the raw gate activations (per-dimension weights).
        Useful for explaining which modality the model is relying on.
        """
        concat = torch.cat([img_emb, tab_emb], dim=1)
        return torch.sigmoid(self.gate_bn(self.gate_linear(concat)))


# ──────────────────────────────────────────────────────────────────────────────
# Quick sanity check
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    fusion  = GatedFusion()
    img_emb = torch.randn(4, IMAGE_EMBEDDING_DIM)
    tab_emb = torch.randn(4, TABULAR_EMBEDDING_DIM)

    out = fusion(img_emb, tab_emb)
    print(f"Fused embedding shape: {out.shape}")    # Expected: (4, 192)
    assert out.shape == (4, FUSION_DIM), "Shape mismatch!"
    print("GatedFusion test PASSED ✓")
