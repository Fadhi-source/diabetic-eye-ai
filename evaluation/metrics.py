"""
evaluation/metrics.py
Classification metrics with bootstrapped CIs, calibration, ROC/PR curves, and subgroup analysis.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, brier_score_loss,
    roc_curve, precision_recall_curve,
    confusion_matrix,
)
from sklearn.calibration import calibration_curve
from typing import Dict, Tuple, Optional


def compute_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float  = 0.50,
    n_bootstrap: int  = 1000,
    ci_alpha: float   = 0.05,
) -> Dict[str, float]:
    """
    Comprehensive binary classification metrics with bootstrapped 95% CIs.

    Args:
        y_true      : Ground-truth binary labels
        y_prob      : Predicted probabilities for the positive class
        threshold   : Decision threshold
        n_bootstrap : Bootstrap iterations for CI estimation
        ci_alpha    : Significance level (0.05 = 95% CI)

    Returns:
        Dict of metric names → values (with _lo/_hi CI bounds for AUC)
    """
    y_pred = (y_prob >= threshold).astype(int)

    auc_roc = roc_auc_score(y_true, y_prob)
    pr_auc  = average_precision_score(y_true, y_prob)
    f1      = f1_score(y_true, y_pred, zero_division=0)
    brier   = brier_score_loss(y_true, y_prob)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    ppv         = tp / max(tp + fp, 1)
    npv         = tn / max(tn + fn, 1)

    boot_auc, boot_pr = [], []
    rng = np.random.default_rng(42)
    n   = len(y_true)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        if y_true[idx].sum() in (0, n):
            continue
        boot_auc.append(roc_auc_score(y_true[idx], y_prob[idx]))
        boot_pr.append(average_precision_score(y_true[idx], y_prob[idx]))

    lo, hi = ci_alpha / 2, 1 - ci_alpha / 2
    return {
        "auc_roc":     float(auc_roc),
        "auc_roc_lo":  float(np.quantile(boot_auc, lo)),
        "auc_roc_hi":  float(np.quantile(boot_auc, hi)),
        "pr_auc":      float(pr_auc),
        "pr_auc_lo":   float(np.quantile(boot_pr, lo)),
        "pr_auc_hi":   float(np.quantile(boot_pr, hi)),
        "f1":          float(f1),
        "brier_score": float(brier),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "ppv":         float(ppv),
        "npv":         float(npv),
        "threshold":   threshold,
        "n_positive":  int(y_true.sum()),
        "n_negative":  int((y_true == 0).sum()),
    }


def optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    method: str = "youden",
) -> Tuple[float, float]:
    """
    Find the optimal decision threshold.

    Args:
        method : "youden" (Sensitivity + Specificity - 1) or "f1" (maximise F1)

    Returns:
        (threshold, score) tuple
    """
    if method == "youden":
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        j_scores  = tpr - fpr
        best_idx  = np.argmax(j_scores)
        return float(thresholds[best_idx]), float(j_scores[best_idx])
    elif method == "f1":
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
        best_idx  = np.argmax(f1_scores[:-1])
        return float(thresholds[best_idx]), float(f1_scores[best_idx])
    else:
        raise ValueError(f"Unknown method: {method}. Use 'youden' or 'f1'.")


def plot_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plots reliability diagram. A well-calibrated model lies on the diagonal."""
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated")
    ax.plot(prob_pred, prob_true, marker="o", color="#FF6B6B", label="Model")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_roc_pr_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plots ROC and Precision-Recall curves side by side."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_roc      = roc_auc_score(y_true, y_prob)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc       = average_precision_score(y_true, y_prob)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(fpr, tpr, color="#4ECDC4", lw=2, label=f"AUC-ROC = {auc_roc:.3f}")
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray")
    axes[0].set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC Curve")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(recall, precision, color="#FF6B6B", lw=2, label=f"PR-AUC = {pr_auc:.3f}")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-Recall Curve")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Model Performance Curves", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def subgroup_analysis(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    groups: pd.DataFrame,
) -> pd.DataFrame:
    """
    AUC-ROC, PR-AUC, and F1 broken down by subgroup (fairness audit).

    Args:
        y_true : Ground-truth labels
        y_prob : Predicted probabilities
        groups : DataFrame with subgroup columns (e.g. gender, age_group)
    """
    records   = []
    threshold = 0.50

    for col in groups.columns:
        for val in groups[col].unique():
            mask = groups[col] == val
            gt, gp = y_true[mask], y_prob[mask]

            if gt.sum() == 0 or len(gt) < 10:
                continue

            auc = roc_auc_score(gt, gp) if gt.nunique() > 1 else float("nan")
            pr  = average_precision_score(gt, gp) if gt.nunique() > 1 else float("nan")
            f1  = f1_score(gt, (gp >= threshold).astype(int), zero_division=0)

            records.append({
                "group_col":  col,
                "group_val":  val,
                "n_samples":  int(mask.sum()),
                "n_positive": int(gt.sum()),
                "auc_roc":    round(auc, 4),
                "pr_auc":     round(pr, 4),
                "f1":         round(f1, 4),
            })

    return pd.DataFrame(records)


def print_metrics(metrics: dict) -> None:
    print(f"\n{'='*55}")
    print("  Classification Metrics")
    print(f"{'='*55}")
    print(f"  AUC-ROC : {metrics['auc_roc']:.4f} [{metrics['auc_roc_lo']:.4f}, {metrics['auc_roc_hi']:.4f}]")
    print(f"  PR-AUC  : {metrics['pr_auc']:.4f} [{metrics['pr_auc_lo']:.4f}, {metrics['pr_auc_hi']:.4f}]")
    print(f"  F1      : {metrics['f1']:.4f}")
    print(f"  Brier   : {metrics['brier_score']:.4f}")
    print(f"  Sens.   : {metrics['sensitivity']:.4f}  |  Spec. : {metrics['specificity']:.4f}")
    print(f"  PPV     : {metrics['ppv']:.4f}       |  NPV  : {metrics['npv']:.4f}")
    print(f"  Threshold: {metrics['threshold']}")
    print(f"{'='*55}\n")
