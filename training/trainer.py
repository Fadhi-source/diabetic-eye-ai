import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import AveragePrecision, AUROC, F1Score
from torchmetrics.classification import BinaryCalibrationError

from models.multimodal_model import MultiModalModel
from training.loss import BinaryFocalLoss
from config import (
    LEARNING_RATE, WEIGHT_DECAY, MAX_EPOCHS,
    FOCAL_LOSS_GAMMA, FOCAL_LOSS_ALPHA,
    LR_SCHEDULER, INFERENCE_THRESHOLD,
)


class DiabetesLightningModule(pl.LightningModule):
    """
    Lightning wrapper for the MultiModalModel.

    Handles forward pass, loss, metric accumulation (PR-AUC, ROC-AUC, F1, Brier),
    AdamW optimizer, and CosineAnnealingLR scheduler.

    Args:
        lr           : Learning rate
        weight_decay : AdamW weight decay
        max_epochs   : Total epochs (for CosineAnnealingLR T_max)
        pretrained   : Use ImageNet pretrained weights
        freeze_ratio : Fraction of image backbone to freeze initially
    """

    def __init__(
        self,
        lr: float           = LEARNING_RATE,
        weight_decay: float = WEIGHT_DECAY,
        max_epochs: int     = MAX_EPOCHS,
        pretrained: bool    = True,
        freeze_ratio: float = 0.70,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model   = MultiModalModel(pretrained=pretrained, freeze_ratio=freeze_ratio)
        self.loss_fn = BinaryFocalLoss(alpha=FOCAL_LOSS_ALPHA, gamma=FOCAL_LOSS_GAMMA)

        self.train_pr_auc = AveragePrecision(task="binary")
        self.val_pr_auc   = AveragePrecision(task="binary")
        self.val_roc_auc  = AUROC(task="binary")
        self.val_f1       = F1Score(task="binary", threshold=INFERENCE_THRESHOLD)
        self.val_brier    = BinaryCalibrationError(n_bins=10)
        self.test_pr_auc  = AveragePrecision(task="binary")
        self.test_roc_auc = AUROC(task="binary")
        self.test_f1      = F1Score(task="binary", threshold=INFERENCE_THRESHOLD)

    def forward(self, image, tabular):
        return self.model(image, tabular)

    def training_step(self, batch, batch_idx):
        images, tabular, labels = batch
        out    = self(images, tabular)
        logits = out["logits"].squeeze(-1)
        probs  = out["probs"].squeeze(-1)
        loss   = self.loss_fn(logits, labels)
        self.train_pr_auc.update(probs, labels.int())
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def on_train_epoch_end(self):
        self.log("train/pr_auc", self.train_pr_auc.compute(), prog_bar=True)
        self.train_pr_auc.reset()

    def validation_step(self, batch, batch_idx):
        images, tabular, labels = batch
        out    = self(images, tabular)
        logits = out["logits"].squeeze(-1)
        probs  = out["probs"].squeeze(-1)
        loss   = self.loss_fn(logits, labels)
        self.val_pr_auc.update(probs, labels.int())
        self.val_roc_auc.update(probs, labels.int())
        self.val_f1.update(probs, labels.int())
        self.val_brier.update(probs, labels.int())
        self.log("val/loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def on_validation_epoch_end(self):
        self.log("val/pr_auc",  self.val_pr_auc.compute(),  prog_bar=True)
        self.log("val/roc_auc", self.val_roc_auc.compute(), prog_bar=True)
        self.log("val/f1",      self.val_f1.compute())
        self.log("val/brier",   self.val_brier.compute())
        self.val_pr_auc.reset()
        self.val_roc_auc.reset()
        self.val_f1.reset()
        self.val_brier.reset()

    def test_step(self, batch, batch_idx):
        images, tabular, labels = batch
        out   = self(images, tabular)
        probs = out["probs"].squeeze(-1)
        self.test_pr_auc.update(probs, labels.int())
        self.test_roc_auc.update(probs, labels.int())
        self.test_f1.update(probs, labels.int())

    def on_test_epoch_end(self):
        self.log("test/pr_auc",  self.test_pr_auc.compute())
        self.log("test/roc_auc", self.test_roc_auc.compute())
        self.log("test/f1",      self.test_f1.compute())
        self.test_pr_auc.reset()
        self.test_roc_auc.reset()
        self.test_f1.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.max_epochs,
            eta_min=1e-6,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "monitor": "val/pr_auc"},
        }
