"""
explainability/shap_explainer.py
SHAP (GradientExplainer) for tabular feature attribution.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Optional, List

from models.multimodal_model import MultiModalModel
from config import ALL_FEATURES, TABULAR_INPUT_DIM


class TabularOnlyWrapper(nn.Module):
    """
    Wraps the full model with a fixed image input so SHAP can attribute
    contributions from tabular features only.
    """

    def __init__(self, full_model: MultiModalModel, image_tensor: torch.Tensor):
        super().__init__()
        self.full_model   = full_model
        self.image_tensor = image_tensor

    def forward(self, tabular: torch.Tensor) -> torch.Tensor:
        img = self.image_tensor.expand(tabular.shape[0], -1, -1, -1).to(tabular.device)
        return torch.sigmoid(self.full_model(img, tabular)["logits"])


class SHAPExplainer:
    """
    Computes and visualises SHAP values for tabular features.

    Args:
        model        : Trained MultiModalModel
        background   : Background dataset tensor (N_bg, TABULAR_INPUT_DIM)
        image_tensor : Fixed mean/zero image tensor (1, 3, 224, 224)
        device       : "cpu" or "cuda"
    """

    def __init__(
        self,
        model: MultiModalModel,
        background: torch.Tensor,
        image_tensor: Optional[torch.Tensor] = None,
        device: str = "cpu",
    ):
        self.device     = device
        self.model      = model.to(device).eval()
        self.background = background.to(device)

        if image_tensor is None:
            image_tensor = torch.zeros(1, 3, 224, 224)
        self.image_tensor = image_tensor.to(device)

        self.wrapper   = TabularOnlyWrapper(self.model, self.image_tensor)
        self.explainer = shap.GradientExplainer(self.wrapper, self.background)

    def compute_shap_values(self, tabular_tensor: torch.Tensor, n_samples: int = 50) -> np.ndarray:
        """
        Compute SHAP values for one or more patients.

        Returns:
            (B, TABULAR_INPUT_DIM) float32 ndarray
        """
        tabular_tensor = tabular_tensor.to(self.device)
        with torch.no_grad():
            shap_vals = self.explainer.shap_values(tabular_tensor, nsamples=n_samples)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]
        return shap_vals if isinstance(shap_vals, np.ndarray) else shap_vals.numpy()

    def waterfall_plot(
        self,
        tabular_tensor: torch.Tensor,
        feature_names: List[str] = ALL_FEATURES,
        patient_idx: int = 0,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """SHAP waterfall chart for a single patient (top 15 features by impact)."""
        shap_vals = self.compute_shap_values(tabular_tensor)
        vals      = shap_vals[patient_idx]
        raw_feats = tabular_tensor[patient_idx].cpu().numpy()

        order    = np.argsort(np.abs(vals))[::-1][:15]
        features = [feature_names[i] for i in order]
        s_vals   = vals[order]
        f_vals   = raw_feats[order]
        colors   = ["#FF6B6B" if v > 0 else "#4ECDC4" for v in s_vals]

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(range(len(features))[::-1], s_vals, color=colors, edgecolor="white", linewidth=0.6)
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels([f"{f}  ({v:.2f})" for f, v in zip(features, f_vals)], fontsize=9)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("SHAP Value (impact on risk score)")
        ax.set_title("Feature Contribution to Complication Risk\n(Red = increases risk,  Teal = decreases risk)", fontsize=11)
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150)
        return fig

    def summary_plot(
        self,
        tabular_tensor: torch.Tensor,
        feature_names: List[str] = ALL_FEATURES,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """SHAP beeswarm summary plot showing global feature importance."""
        shap_vals = self.compute_shap_values(tabular_tensor)

        fig, _ = plt.subplots(figsize=(10, 7))
        shap.summary_plot(shap_vals, tabular_tensor.cpu().numpy(), feature_names=feature_names, show=False, plot_size=None)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150)
        return fig
