import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from config import IMAGE_EMBEDDING_DIM, TABULAR_EMBEDDING_DIM, FUSION_DIM


class GatedFusion(nn.Module):
    """
    Gated multi-modal fusion of image and tabular embeddings.

    Learns a per-dimension gate to selectively weight the fused representation,
    making the model robust to missing or low-quality modalities.

        concat = [img_emb ; tab_emb]
        gate   = sigmoid(W_g · concat)
        feat   = tanh(W_f · concat)
        h      = gate ⊙ feat

    Args:
        img_dim    : Image embedding dimension
        tab_dim    : Tabular embedding dimension
        fusion_dim : Output fusion dimension
        dropout    : Dropout on the fused output
    """

    def __init__(
        self,
        img_dim: int    = IMAGE_EMBEDDING_DIM,
        tab_dim: int    = TABULAR_EMBEDDING_DIM,
        fusion_dim: int = FUSION_DIM,
        dropout: float  = 0.30,
    ):
        super().__init__()
        concat_dim = img_dim + tab_dim

        self.gate_linear = nn.Linear(concat_dim, fusion_dim)
        self.feat_linear = nn.Linear(concat_dim, fusion_dim)
        self.gate_bn     = nn.BatchNorm1d(fusion_dim)
        self.feat_bn     = nn.BatchNorm1d(fusion_dim)
        self.dropout     = nn.Dropout(p=dropout)
        self.fusion_dim  = fusion_dim

        nn.init.xavier_uniform_(self.gate_linear.weight)
        nn.init.xavier_uniform_(self.feat_linear.weight)

    def forward(self, img_emb: torch.Tensor, tab_emb: torch.Tensor) -> torch.Tensor:
        concat = torch.cat([img_emb, tab_emb], dim=1)
        gate   = torch.sigmoid(self.gate_bn(self.gate_linear(concat)))
        feat   = torch.tanh(self.feat_bn(self.feat_linear(concat)))
        return self.dropout(gate * feat)

    def get_gate_weights(self, img_emb: torch.Tensor, tab_emb: torch.Tensor) -> torch.Tensor:
        """Returns raw gate activations — useful for modality reliance analysis."""
        concat = torch.cat([img_emb, tab_emb], dim=1)
        return torch.sigmoid(self.gate_bn(self.gate_linear(concat)))


if __name__ == "__main__":
    fusion  = GatedFusion()
    img_emb = torch.randn(4, IMAGE_EMBEDDING_DIM)
    tab_emb = torch.randn(4, TABULAR_EMBEDDING_DIM)
    out     = fusion(img_emb, tab_emb)
    assert out.shape == (4, FUSION_DIM)
    print(f"Fused shape: {out.shape}  ✓")
