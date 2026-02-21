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
from config import CHECKPOINTS_DIR, EARLY_STOPPING_PATIENCE


def get_standard_callbacks(fine_tune_epoch: int = 10) -> list:
    """
    Returns training callbacks.

    Args:
        fine_tune_epoch : Epoch at which the top 30% of image backbone is unfrozen
    """
    return [
        ModelCheckpoint(
            dirpath=CHECKPOINTS_DIR,
            filename="best-{epoch:02d}-{val/pr_auc:.4f}",
            monitor="val/pr_auc",
            mode="max",
            save_top_k=2,
            save_last=True,
            verbose=True,
        ),
        EarlyStopping(
            monitor="val/pr_auc",
            patience=EARLY_STOPPING_PATIENCE,
            mode="max",
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
        RichProgressBar(),
        BackboneFineTuneCallback(unfreeze_epoch=fine_tune_epoch),
    ]


class BackboneFineTuneCallback(pl.Callback):
    """
    At `unfreeze_epoch`, unfreezes the top 30% of the EfficientNet backbone
    and reduces the learning rate by `lr_scale` for stable fine-tuning.

    Args:
        unfreeze_epoch : Epoch to trigger fine-tuning
        lr_scale       : Factor to divide current LR by after unfreezing
    """

    def __init__(self, unfreeze_epoch: int = 10, lr_scale: float = 10.0):
        super().__init__()
        self.unfreeze_epoch = unfreeze_epoch
        self.lr_scale       = lr_scale
        self._unfrozen      = False

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.current_epoch == self.unfreeze_epoch and not self._unfrozen:
            print(f"\n[FineTune] Epoch {self.unfreeze_epoch}: unfreezing top 30% of backbone")
            pl_module.model.fine_tune_mode()

            for g in trainer.optimizers[0].param_groups:
                g["lr"] /= self.lr_scale
            print(f"[FineTune] LR reduced by {self.lr_scale}x")
            self._unfrozen = True
