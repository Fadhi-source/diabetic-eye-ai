"""
training/train.py
Main training entry point.

Usage:
    python training/train.py --epochs 30 --batch_size 32        # standard
    python training/train.py --hpo --n_trials 20                # Optuna HPO
    python training/train.py --smoke_test                       # 1 epoch, 4 samples
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
    RANDOM_SEED,
)
from data.dataset import create_dataloaders
from training.trainer import DiabetesLightningModule
from training.callbacks import get_standard_callbacks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Multi-Modal Diabetic Complication Predictor")
    parser.add_argument("--epochs",          type=int,   default=MAX_EPOCHS)
    parser.add_argument("--batch_size",      type=int,   default=BATCH_SIZE)
    parser.add_argument("--lr",              type=float, default=LEARNING_RATE)
    parser.add_argument("--weight_decay",    type=float, default=WEIGHT_DECAY)
    parser.add_argument("--freeze_ratio",    type=float, default=0.70)
    parser.add_argument("--fine_tune_epoch", type=int,   default=10)
    parser.add_argument("--use_wandb",       action="store_true")
    parser.add_argument("--hpo",             action="store_true")
    parser.add_argument("--n_trials",        type=int,   default=HPO_N_TRIALS)
    parser.add_argument("--smoke_test",      action="store_true")
    parser.add_argument("--csv",             type=str,   default=SYNTHETIC_CSV)
    parser.add_argument("--num_workers",     type=int,   default=0)
    return parser.parse_args()


def train(
    lr: float            = LEARNING_RATE,
    weight_decay: float  = WEIGHT_DECAY,
    batch_size: int      = BATCH_SIZE,
    epochs: int          = MAX_EPOCHS,
    freeze_ratio: float  = 0.70,
    fine_tune_epoch: int = 10,
    use_wandb: bool      = False,
    smoke_test: bool     = False,
    num_workers: int     = 0,
    csv_path: str        = SYNTHETIC_CSV,
    trial: optuna.Trial  = None,
) -> float:
    """Runs one full training run and returns best val/pr_auc (used by Optuna)."""
    pl.seed_everything(RANDOM_SEED, workers=True)

    loaders, _ = create_dataloaders(
        csv_path=SYNTHETIC_CSV if smoke_test else csv_path,
        image_dir=IMAGE_DIR,
        batch_size=4 if smoke_test else batch_size,
        num_workers=num_workers,
        dummy_images=smoke_test,
    )

    model = DiabetesLightningModule(
        lr=lr,
        weight_decay=weight_decay,
        max_epochs=epochs,
        pretrained=(not smoke_test),
        freeze_ratio=freeze_ratio,
    )

    logger = (
        WandbLogger(project=WANDB_PROJECT, entity=WANDB_ENTITY, log_model="all")
        if use_wandb
        else TensorBoardLogger(save_dir=LOGS_DIR, name="multimodal_predictor")
    )

    callbacks = get_standard_callbacks(fine_tune_epoch=fine_tune_epoch)
    if trial is not None:
        from optuna.integration import PyTorchLightningPruningCallback
        callbacks.append(PyTorchLightningPruningCallback(trial, monitor="val/pr_auc"))

    trainer = pl.Trainer(
        max_epochs=1 if smoke_test else epochs,
        accelerator="auto",
        devices=1,
        precision="16-mixed",
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=5,
        deterministic=True,
        enable_progress_bar=True,
        limit_train_batches=2 if smoke_test else 1.0,
        limit_val_batches=2 if smoke_test else 1.0,
    )

    trainer.fit(model, train_dataloaders=loaders["train"], val_dataloaders=loaders["val"])

    ckpt_cb         = next(cb for cb in callbacks if isinstance(cb, pl.callbacks.ModelCheckpoint))
    best_model_path = ckpt_cb.best_model_path
    best_val_score  = ckpt_cb.best_model_score or 0.0

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


def hpo_objective(trial: optuna.Trial, args: argparse.Namespace) -> float:
    return train(
        lr=trial.suggest_float("lr", 1e-5, 5e-4, log=True),
        weight_decay=trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
        batch_size=trial.suggest_categorical("batch_size", [16, 32, 64]),
        freeze_ratio=trial.suggest_float("freeze_ratio", 0.50, 0.90),
        fine_tune_epoch=trial.suggest_int("fine_tune_epoch", 5, 20),
        epochs=args.epochs,
        use_wandb=False,
        trial=trial,
        num_workers=args.num_workers,
    )


def main():
    args = parse_args()

    if args.smoke_test:
        print("\n[SMOKE TEST] 1 epoch, 4 dummy samples\n")
        score = train(smoke_test=True, num_workers=args.num_workers)
        print(f"Smoke test val/pr_auc = {score:.4f}")
        return

    if args.hpo:
        print(f"\n[HPO] {args.n_trials} Optuna trials\n")
        study = optuna.create_study(
            direction=HPO_DIRECTION,
            study_name="multimodal_diabetic_hpo",
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
            storage=f"sqlite:///{LOGS_DIR}/optuna_study.db",
            load_if_exists=True,
        )
        study.optimize(lambda trial: hpo_objective(trial, args), n_trials=args.n_trials, show_progress_bar=True)
        print(f"\n[HPO COMPLETE]  Best value: {study.best_value:.4f}  |  Best params: {study.best_params}")
        return

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
