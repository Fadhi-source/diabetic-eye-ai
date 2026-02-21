"""
explainability/shap_explainer.py
SHAP (SHapley Additive exPlanations) for the tabular branch.

Uses shap.GradientExplainer which works natively with PyTorch models
and computes Shapley values via a smooth-gradient approximation.

SHAP values answer: "How much did each clinical feature contribute to
this patient's risk score, relative to the background average patient?"

Positive SHAP → feature pushed risk UP
Negative SHAP → feature pushed risk DOWN
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


# ──────────────────────────────────────────────────────────────────────────────
# Wrapper: expose only the tabular branch for SHAP
# (SHAP needs a callable that accepts the tabular input tensor and returns logits)
# ──────────────────────────────────────────────────────────────────────────────

class TabularOnlyWrapper(nn.Module):
    """
    Wraps the full multi-modal model, keeping image input fixed,
    so SHAP can attribute contributions from the tabular features only.

    Args:
        full_model : Trained MultiModalModel
        image_tensor : Fixed dummy/mean image tensor (1, 3, H, W)
    """

    def __init__(self, full_model: MultiModalModel, image_tensor: torch.Tensor):
        super().__init__()
        self.full_model   = full_model
        self.image_tensor = image_tensor

    def forward(self, tabular: torch.Tensor) -> torch.Tensor:
        B   = tabular.shape[0]
        img = self.image_tensor.expand(B, -1, -1, -1).to(tabular.device)
        out = self.full_model(img, tabular)
        return torch.sigmoid(out["logits"])   # Return probabilities for SHAP


# ──────────────────────────────────────────────────────────────────────────────
# Main SHAP wrapper class
# ──────────────────────────────────────────────────────────────────────────────

class SHAPExplainer:
    """
    Computes and visualises SHAP values for tabular features.

    Args:
        model        : Trained MultiModalModel
        background   : Background dataset tensor (N_bg, TABULAR_INPUT_DIM)
                       Used to compute the SHAP baseline (expected value).
                       Typically 50–200 representative training samples.
        image_tensor : A fixed mean image tensor (1, 3, 224, 224)
        device       : "cpu" or "cuda"
    """

    def __init__(
        self,
        model: MultiModalModel,
        background: torch.Tensor,
        image_tensor: Optional[torch.Tensor] = None,
        device: str = "cpu",
    ):
        self.model      = model.to(device).eval()
        self.device     = device
        self.background = background.to(device)

        if image_tensor is None:
            # Use a zero (black) image as neutral baseline
            image_tensor = torch.zeros(1, 3, 224, 224)
        self.image_tensor = image_tensor.to(device)

        self.wrapper   = TabularOnlyWrapper(self.model, self.image_tensor)
        self.explainer = shap.GradientExplainer(self.wrapper, self.background)

    def compute_shap_values(
        self,
        tabular_tensor: torch.Tensor,
        n_samples: int = 50,
    ) -> np.ndarray:
        """
        Compute SHAP values for one or more patients.

        Args:
            tabular_tensor : (B, TABULAR_INPUT_DIM) input features
            n_samples      : Number of integration samples (higher = more accurate)

        Returns:
            shap_values : (B, TABULAR_INPUT_DIM) float32 ndarray
        """
        tabular_tensor = tabular_tensor.to(self.device)
        with torch.no_grad():
            shap_vals = self.explainer.shap_values(
                tabular_tensor,
                nsamples=n_samples,
            )
        # shap_vals may be a list [neg_class, pos_class] for binary
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]   # take positive class
        return shap_vals if isinstance(shap_vals, np.ndarray) else shap_vals.numpy()

    # ── Visualisation helpers ─────────────────────────────────────────────────

    def waterfall_plot(
        self,
        tabular_tensor: torch.Tensor,
        feature_names: List[str] = ALL_FEATURES,
        patient_idx: int = 0,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        SHAP waterfall chart for a single patient.
        Bars extend right (positive contribution to risk) or left (negative).
        """
        shap_vals = self.compute_shap_values(tabular_tensor)
        vals      = shap_vals[patient_idx]      # (TABULAR_INPUT_DIM,)
        raw_feats = tabular_tensor[patient_idx].cpu().numpy()

        # Sort by absolute SHAP value descending
        order    = np.argsort(np.abs(vals))[::-1][:15]   # top 15 features
        features = [feature_names[i] for i in order]
        s_vals   = vals[order]
        f_vals   = raw_feats[order]

        colors = ["#FF6B6B" if v > 0 else "#4ECDC4" for v in s_vals]

        fig, ax = plt.subplots(figsize=(8, 6))
        bars = ax.barh(
            range(len(features))[::-1],
            s_vals,
            color=colors,
            edgecolor="white",
            linewidth=0.6,
        )
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(
            [f"{f}  ({v:.2f})" for f, v in zip(features, f_vals)],
            fontsize=9
        )
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("SHAP Value (impact on risk score)")
        ax.set_title("Feature Contribution to Complication Risk\n"
                     "(Red = increases risk,  Teal = decreases risk)", fontsize=11)
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
        """
        SHAP beeswarm / summary plot across multiple patients.
        Shows global feature importance.
        """
        shap_vals = self.compute_shap_values(tabular_tensor)

        fig, ax = plt.subplots(figsize=(10, 7))
        shap.summary_plot(
            shap_vals,
            tabular_tensor.cpu().numpy(),
            feature_names=feature_names,
            show=False,
            plot_size=None,
        )
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150)

        return fig
