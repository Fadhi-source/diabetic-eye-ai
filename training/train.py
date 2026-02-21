"""
training/train.py
Main training entry point for the Multi-Modal Diabetic Complication Predictor.

Supports two modes:
  1. Standard training:
       python training/train.py --epochs 30 --batch_size 32 --lr 1e-4

  2. Hyperparameter Optimisation (Optuna):
       python training/train.py --hpo --n_trials 20

  3. Smoke test (1 epoch, 4 samples — CI / architecture verification):
       python training/train.py --smoke_test
"""

import sys
import os
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
import optuna

from config import (
    SYNTHETIC_CSV, IMAGE_DIR, CHECKPOINTS_DIR, LOGS_DIR,
    BATCH_SIZE, MAX_EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    WANDB_PROJECT, WANDB_ENTITY, HPO_N_TRIALS, HPO_METRIC, HPO_DIRECTION,
    RANDOM_SEED
)
from data.dataset import create_dataloaders
from training.trainer import DiabetesLightningModule
from training.callbacks import get_standard_callbacks


# ──────────────────────────────────────────────────────────────────────────────
# Parse CLI args
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the Multi-Modal Diabetic Complication Predictor"
    )
    parser.add_argument("--epochs",      type=int,   default=MAX_EPOCHS)
    parser.add_argument("--batch_size",  type=int,   default=BATCH_SIZE)
    parser.add_argument("--lr",          type=float, default=LEARNING_RATE)
    parser.add_argument("--weight_decay",type=float, default=WEIGHT_DECAY)
    parser.add_argument("--freeze_ratio",type=float, default=0.70)
    parser.add_argument("--fine_tune_epoch", type=int, default=10,
                        help="Epoch to unfreeze top 30% of image backbone")
    parser.add_argument("--use_wandb",   action="store_true",
                        help="Enable Weights & Biases logging")
    parser.add_argument("--hpo",         action="store_true",
                        help="Run Optuna hyperparameter optimisation")
    parser.add_argument("--n_trials",    type=int,   default=HPO_N_TRIALS)
    parser.add_argument("--smoke_test",  action="store_true",
                        help="Run 1 epoch with 4 dummy samples (no real data)")
    parser.add_argument("--csv",         type=str,   default=SYNTHETIC_CSV,
                        help="Path to training CSV (use data/merged.csv for real APTOS data)")
    parser.add_argument("--num_workers", type=int,   default=0,
                        help="DataLoader num_workers (0 = safe on Windows)")
    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Single training run
# ──────────────────────────────────────────────────────────────────────────────

def train(
    lr: float           = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    batch_size: int     = BATCH_SIZE,
    epochs: int         = MAX_EPOCHS,
    freeze_ratio: float = 0.70,
    fine_tune_epoch: int = 10,
    use_wandb: bool     = False,
    smoke_test: bool    = False,
    num_workers: int    = 0,
    csv_path: str       = SYNTHETIC_CSV,   # Pass data/merged.csv for real APTOS data
    trial: optuna.Trial = None,    # Only set during HPO
) -> float:
    """
    Runs one full training run and returns the best val/pr_auc.

    Returns:
        best_val_pr_auc : float (used by Optuna to optimise)
    """
    pl.seed_everything(RANDOM_SEED, workers=True)

    # ── DataLoaders ───────────────────────────────────────────────────────────
    loaders, _ = create_dataloaders(
        csv_path=SYNTHETIC_CSV if smoke_test else csv_path,
        image_dir=IMAGE_DIR,
        batch_size=4 if smoke_test else batch_size,
        num_workers=num_workers,
        dummy_images=smoke_test,    # Use random tensors if smoke_test
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = DiabetesLightningModule(
        lr=lr,
        weight_decay=weight_decay,
        max_epochs=epochs,
        pretrained=(not smoke_test),
        freeze_ratio=freeze_ratio,
    )

    # ── Logger ────────────────────────────────────────────────────────────────
    if use_wandb:
        logger = WandbLogger(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            log_model="all",
        )
    else:
        logger = TensorBoardLogger(save_dir=LOGS_DIR, name="multimodal_predictor")

    # ── Callbacks ─────────────────────────────────────────────────────────────
    callbacks = get_standard_callbacks(fine_tune_epoch=fine_tune_epoch)

    # Add Optuna pruning callback if running HPO
    if trial is not None:
        from optuna.integration import PyTorchLightningPruningCallback
        callbacks.append(
            PyTorchLightningPruningCallback(trial, monitor="val/pr_auc")
        )

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = pl.Trainer(
        max_epochs=1 if smoke_test else epochs,
        accelerator="auto",          # GPU if available, else CPU
        devices=1,
        precision="16-mixed",        # Mixed precision for speed
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=5,
        deterministic=True,
        enable_progress_bar=True,
        limit_train_batches=2 if smoke_test else 1.0,
        limit_val_batches=2 if smoke_test else 1.0,
    )

    trainer.fit(
        model,
        train_dataloaders=loaders["train"],
        val_dataloaders=loaders["val"],
    )

    # Retrieve best PR-AUC from checkpoint callback
    ckpt_cb      = next(cb for cb in callbacks
                        if isinstance(cb, pl.callbacks.ModelCheckpoint))
    best_model_path = ckpt_cb.best_model_path
    best_val_score  = ckpt_cb.best_model_score or 0.0

    # ── Run test set evaluation ───────────────────────────────────────────────
    if not smoke_test:
        trainer.test(
            ckpt_path=best_model_path if best_model_path else "best",
            dataloaders=loaders["test"],
        )

    print(f"\n{'='*55}")
    print(f"  Best val/pr_auc : {float(best_val_score):.4f}")
    if best_model_path:
        print(f"  Best checkpoint : {best_model_path}")
    print(f"{'='*55}\n")

    return float(best_val_score)


# ──────────────────────────────────────────────────────────────────────────────
# Optuna HPO objective
# ──────────────────────────────────────────────────────────────────────────────

def hpo_objective(trial: optuna.Trial, args: argparse.Namespace) -> float:
    """
    Optuna objective function.
    Suggests a hyperparameter configuration and returns validation PR-AUC.
    """
    lr           = trial.suggest_float("lr",           1e-5, 5e-4, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    batch_size   = trial.suggest_categorical("batch_size", [16, 32, 64])
    freeze_ratio = trial.suggest_float("freeze_ratio", 0.50, 0.90)
    fine_tune_epoch = trial.suggest_int("fine_tune_epoch", 5, 20)

    return train(
        lr=lr,
        weight_decay=weight_decay,
        batch_size=batch_size,
        epochs=args.epochs,
        freeze_ratio=freeze_ratio,
        fine_tune_epoch=fine_tune_epoch,
        use_wandb=False,   # Disable W&B per trial during HPO
        trial=trial,
        num_workers=args.num_workers,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if args.smoke_test:
        print("\n[SMOKE TEST MODE] — Running 1 epoch with 4 dummy samples\n")
        score = train(smoke_test=True, num_workers=args.num_workers)
        print(f"Smoke test complete. val/pr_auc = {score:.4f}")
        return

    if args.hpo:
        print(f"\n[HPO MODE] — Running {args.n_trials} Optuna trials\n")
        study = optuna.create_study(
            direction=HPO_DIRECTION,
            study_name="multimodal_diabetic_hpo",
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
            storage=f"sqlite:///{LOGS_DIR}/optuna_study.db",
            load_if_exists=True,
        )
        study.optimize(
            lambda trial: hpo_objective(trial, args),
            n_trials=args.n_trials,
            show_progress_bar=True,
        )

        print("\n[HPO COMPLETE]")
        print(f"  Best value       : {study.best_value:.4f}")
        print(f"  Best params      : {study.best_params}")
        return

    # Standard training
    train(
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        epochs=args.epochs,
        freeze_ratio=args.freeze_ratio,
        fine_tune_epoch=args.fine_tune_epoch,
        use_wandb=args.use_wandb,
        num_workers=args.num_workers,
        csv_path=args.csv,
    )


if __name__ == "__main__":
    main()
