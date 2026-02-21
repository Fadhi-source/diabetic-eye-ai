"""
training/callbacks.py
PyTorch Lightning callbacks for training quality-of-life and portfolio polish.

Includes:
  - ModelCheckpoint  : Save best model by val/pr_auc
  - EarlyStopping    : Halt when val/pr_auc stops improving
  - LearningRateMonitor : Log LR to W&B / TensorBoard
  - RichProgressBar  : Beautiful rich terminal output
  - FineTuneCallback : Automatically unfreeze backbone at epoch N for Stage 4
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    RichProgressBar,
)
from config import (
    CHECKPOINTS_DIR,
    EARLY_STOPPING_PATIENCE,
)


def get_standard_callbacks(fine_tune_epoch: int = 10) -> list:
    """
    Returns a list of standard callbacks for the training run.

    Args:
        fine_tune_epoch : Epoch at which to unfreeze the top 30% of image backbone
    """
    checkpoint_callback = ModelCheckpoint(
        dirpath=CHECKPOINTS_DIR,
        filename="best-{epoch:02d}-{val/pr_auc:.4f}",
        monitor="val/pr_auc",
        mode="max",
        save_top_k=2,
        save_last=True,
        verbose=True,
    )

    early_stop_callback = EarlyStopping(
        monitor="val/pr_auc",
        patience=EARLY_STOPPING_PATIENCE,
        mode="max",
        verbose=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    progress_bar = RichProgressBar()

    fine_tune_cb = BackboneFineTuneCallback(unfreeze_epoch=fine_tune_epoch)

    return [
        checkpoint_callback,
        early_stop_callback,
        lr_monitor,
        progress_bar,
        fine_tune_cb,
    ]


class BackboneFineTuneCallback(pl.Callback):
    """
    At `unfreeze_epoch`, calls model.fine_tune_mode() to unfreeze the top 30%
    of the EfficientNet backbone layers and reduce the learning rate by 10x.
    This simulates the manual Stage 4 fine-tuning step from the implementation plan.

    Args:
        unfreeze_epoch : Epoch after which backbone layers are unfrozen
        lr_scale       : Amount to divide current LR by after unfreezing
    """

    def __init__(self, unfreeze_epoch: int = 10, lr_scale: float = 10.0):
        super().__init__()
        self.unfreeze_epoch = unfreeze_epoch
        self.lr_scale       = lr_scale
        self._unfrozen      = False

    def on_train_epoch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        if trainer.current_epoch == self.unfreeze_epoch and not self._unfrozen:
            print(f"\n[BackboneFineTuneCallback] Epoch {self.unfreeze_epoch}: "
                  f"Unfreezing top 30% of image backbone…")

            pl_module.model.fine_tune_mode()

            # Reduce LR for fine-tuning stage
            for g in trainer.optimizers[0].param_groups:
                g["lr"] /= self.lr_scale

            print(f"[BackboneFineTuneCallback] LR reduced by {self.lr_scale}x")
            self._unfrozen = True
