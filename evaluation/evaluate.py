"""
evaluation/evaluate.py
Full evaluation script against a trained checkpoint on the test set.

Usage:
    python evaluation/evaluate.py
    python evaluation/evaluate.py --checkpoint checkpoints/best-epoch=14-val-pr_auc=0.9737.ckpt
    python evaluation/evaluate.py --smoke_test
"""

import os
import argparse

import numpy as np
import pandas as pd
import torch
from loguru import logger
from tqdm import tqdm

from config import SYNTHETIC_CSV, IMAGE_DIR, CHECKPOINTS_DIR, LOGS_DIR, BATCH_SIZE, RANDOM_SEED
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
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument("--checkpoint", type=str,  default=None)
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--save_plots", action="store_true", default=True)
    return parser.parse_args()


def collect_predictions(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for imgs, tabs, labels in tqdm(loader, desc="Evaluating"):
            probs = model(imgs.to(device), tabs.to(device))["probs"].squeeze(-1).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.numpy())
    return np.concatenate(all_labels), np.concatenate(all_probs)


def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_path = args.checkpoint
    if ckpt_path is None:
        ckpt_files = sorted(Path(CHECKPOINTS_DIR).glob("best-*.ckpt"))
        ckpt_path  = str(ckpt_files[-1]) if ckpt_files else None

    if ckpt_path and os.path.exists(ckpt_path):
        logger.info(f"Loading checkpoint: {ckpt_path}")
        model = DiabetesLightningModule.load_from_checkpoint(ckpt_path, map_location=device).model
    else:
        logger.warning("No checkpoint found. Using untrained model.")
        model = MultiModalModel(pretrained=False)

    model = model.to(device).eval()

    loaders, _ = create_dataloaders(
        csv_path=SYNTHETIC_CSV,
        image_dir=IMAGE_DIR,
        batch_size=4 if args.smoke_test else BATCH_SIZE,
        dummy_images=args.smoke_test,
        num_workers=0,
    )

    logger.info("Running inference on test set")
    y_true, y_prob = collect_predictions(model, loaders["test"], device)

    best_thresh, _ = optimal_threshold(y_true, y_prob, method="youden")
    logger.info(f"Optimal threshold (Youden-J): {best_thresh:.3f}")

    metrics = compute_classification_metrics(y_true, y_prob, threshold=best_thresh)
    print_metrics(metrics)

    if args.save_plots:
        logger.info("Saving evaluation plots")
        os.makedirs(LOGS_DIR, exist_ok=True)
        plot_roc_pr_curves(y_true, y_prob, save_path=os.path.join(LOGS_DIR, "roc_pr_curves.png"))
        plot_calibration_curve(y_true, y_prob, save_path=os.path.join(LOGS_DIR, "calibration_curve.png"))
        logger.info(f"Plots saved to {LOGS_DIR}/")

    logger.info("Running subgroup analysis")
    df_test  = pd.read_csv(SYNTHETIC_CSV).sample(n=len(y_true), random_state=RANDOM_SEED).reset_index(drop=True)
    df_test["age_group"] = pd.cut(df_test["age"], bins=[0, 45, 60, 100], labels=["<45", "45-60", ">60"])

    sg_results = subgroup_analysis(y_true, y_prob, df_test[["gender", "age_group", "rural_urban", "hypertension"]])
    logger.info("Subgroup AUC-ROC / F1:\n" + sg_results.to_string(index=False))
    sg_results.to_csv(os.path.join(LOGS_DIR, "subgroup_analysis.csv"), index=False)
    logger.info(f"Subgroup table saved to {LOGS_DIR}/subgroup_analysis.csv")
    logger.info("Evaluation complete")

    return metrics


if __name__ == "__main__":
    main()
