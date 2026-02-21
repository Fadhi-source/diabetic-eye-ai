import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from typing import List
from config import TABULAR_INPUT_DIM, TABULAR_HIDDEN_DIMS, TABULAR_EMBEDDING_DIM, DROPOUT_TAB


class TabularBranch(nn.Module):
    """
    MLP encoder for clinical EHR features.

    Args:
        input_dim     : Number of input features (17 from config)
        hidden_dims   : Hidden layer sizes
        embedding_dim : Output embedding dimension
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

        all_dims = [input_dim] + hidden_dims + [embedding_dim]
        layers = []
        for in_d, out_d in zip(all_dims[:-1], all_dims[1:]):
            layers += [
                nn.Linear(in_d, out_d),
                nn.BatchNorm1d(out_d),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
            ]
        self.mlp = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


if __name__ == "__main__":
    model = TabularBranch()
    dummy = torch.randn(8, TABULAR_INPUT_DIM)
    out   = model(dummy)
    assert out.shape == (8, TABULAR_EMBEDDING_DIM)
    print(f"Output shape: {out.shape}  ✓")
