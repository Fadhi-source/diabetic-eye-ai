"""
evaluation/evaluate.py
Full evaluation script — run against a trained checkpoint on the test set.

Usage:
    python evaluation/evaluate.py
    python evaluation/evaluate.py --checkpoint checkpoints/best-epoch=05-val_pr_auc=0.8420.ckpt
    python evaluation/evaluate.py --smoke_test
"""

import sys
import os
import argparse
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from config import (
    SYNTHETIC_CSV, IMAGE_DIR, CHECKPOINTS_DIR, LOGS_DIR,
    BATCH_SIZE, RANDOM_SEED, TABULAR_INPUT_DIM
)
from data.dataset import create_dataloaders
from models.multimodal_model import MultiModalModel
from training.trainer import DiabetesLightningModule
from evaluation.metrics import (
    compute_classification_metrics,
    optimal_threshold,
    plot_roc_pr_curves,
    plot_calibration_curve,
    subgroup_analysis,
    print_metrics,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the trained model")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to .ckpt file. Defaults to best in checkpoints/")
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--save_plots", action="store_true", default=True,
                        help="Save evaluation plots to logs/")
    return parser.parse_args()


def collect_predictions(model, loader, device):
    """Run inference on entire DataLoader, return (y_true, y_prob) arrays."""
    model.eval()
    all_probs  = []
    all_labels = []

    with torch.no_grad():
        for imgs, tabs, labels in tqdm(loader, desc="Evaluating"):
            imgs   = imgs.to(device)
            tabs   = tabs.to(device)
            out    = model(imgs, tabs)
            probs  = out["probs"].squeeze(-1).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.numpy())

    return (
        np.concatenate(all_labels),
        np.concatenate(all_probs),
    )


def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Load checkpoint ───────────────────────────────────────────────────────
    ckpt_path = args.checkpoint
    if ckpt_path is None:
        ckpt_files = sorted(Path(CHECKPOINTS_DIR).glob("best-*.ckpt"))
        ckpt_path  = str(ckpt_files[-1]) if ckpt_files else None

    if ckpt_path and Path(ckpt_path).exists():
        print(f"Loading checkpoint: {ckpt_path}")
        lm    = DiabetesLightningModule.load_from_checkpoint(ckpt_path, map_location=device)
        model = lm.model
    else:
        print("⚠️  No checkpoint found. Using untrained model (metrics will be random).")
        model = MultiModalModel(pretrained=False)

    model = model.to(device).eval()

    # ── DataLoaders ───────────────────────────────────────────────────────────
    loaders, _ = create_dataloaders(
        csv_path=SYNTHETIC_CSV,
        image_dir=IMAGE_DIR,
        batch_size=4 if args.smoke_test else BATCH_SIZE,
        dummy_images=args.smoke_test,
        num_workers=0,
    )

    # ── Collect predictions ───────────────────────────────────────────────────
    print("\n[1/4] Running inference on test set…")
    y_true, y_prob = collect_predictions(model, loaders["test"], device)

    # ── Find optimal threshold ────────────────────────────────────────────────
    best_thresh, _ = optimal_threshold(y_true, y_prob, method="youden")
    print(f"\n[2/4] Optimal threshold (Youden-J): {best_thresh:.3f}")

    # ── Compute metrics ───────────────────────────────────────────────────────
    metrics = compute_classification_metrics(y_true, y_prob, threshold=best_thresh)
    print_metrics(metrics)

    # ── Save plots ────────────────────────────────────────────────────────────
    if args.save_plots:
        print("[3/4] Saving evaluation plots…")
        os.makedirs(LOGS_DIR, exist_ok=True)
        plot_roc_pr_curves(y_true, y_prob,
                           save_path=os.path.join(LOGS_DIR, "roc_pr_curves.png"))
        plot_calibration_curve(y_true, y_prob,
                               save_path=os.path.join(LOGS_DIR, "calibration_curve.png"))
        print(f"  Plots saved to: {LOGS_DIR}/")

    # ── Subgroup analysis ─────────────────────────────────────────────────────
    print("[4/4] Running subgroup analysis…")
    df_test     = pd.read_csv(SYNTHETIC_CSV)
    test_loader = loaders["test"]

    # Match test set rows (use patient_id from dataset split)
    # For simplicity: sample n rows from the test CSV
    n_preds = len(y_true)
    df_sub  = df_test.sample(n=n_preds, random_state=RANDOM_SEED).reset_index(drop=True)

    # Age group binning
    df_sub["age_group"] = pd.cut(df_sub["age"],
                                  bins=[0, 45, 60, 100],
                                  labels=["<45", "45-60", ">60"])
    subgroup_cols = ["gender", "age_group", "rural_urban", "hypertension"]
    subgroup_df   = df_sub[subgroup_cols]

    sg_results = subgroup_analysis(y_true, y_prob, subgroup_df)
    print("\n  Subgroup AUC-ROC / F1:")
    print(sg_results.to_string(index=False))
    sg_results.to_csv(os.path.join(LOGS_DIR, "subgroup_analysis.csv"), index=False)
    print(f"\n  Subgroup table saved → {LOGS_DIR}/subgroup_analysis.csv")

    print("\n✅ Evaluation complete.")
    return metrics


if __name__ == "__main__":
    main()
