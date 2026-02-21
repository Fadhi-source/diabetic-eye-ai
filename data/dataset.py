"""
data/dataset.py
PyTorch Dataset for paired (image, tabular, label) samples.

The dataset supports two modes:
  1. Real mode   — reads from APTOS image directory + merged CSV
  2. Dummy mode  — generates random tensors so the model can be smoke-tested
     without any downloaded images (useful for CI / architecture verification)

Patient-level split logic (train / val / test) is handled here so a patient's
left AND right eye images always land in the same split, preventing leakage.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, Dict, List
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    SYNTHETIC_CSV, IMAGE_DIR,
    CONTINUOUS_FEATURES, CATEGORICAL_FEATURES, ALL_FEATURES,
    TABULAR_INPUT_DIM,
    TRAIN_RATIO, VAL_RATIO, RANDOM_SEED, BATCH_SIZE, NUM_WORKERS
)
from data.transforms import get_train_transforms, get_val_transforms


# ──────────────────────────────────────────────────────────────────────────────
# Helper: build a scaler fit on the training split only
# ──────────────────────────────────────────────────────────────────────────────

def fit_scaler(df_train: pd.DataFrame) -> StandardScaler:
    """Fit StandardScaler on continuous training features only."""
    scaler = StandardScaler()
    scaler.fit(df_train[CONTINUOUS_FEATURES].fillna(df_train[CONTINUOUS_FEATURES].median()))
    return scaler


# ──────────────────────────────────────────────────────────────────────────────
# Main Dataset
# ──────────────────────────────────────────────────────────────────────────────

class DiabeticDataset(Dataset):
    """
    Multi-modal dataset returning (image_tensor, tabular_tensor, label_tensor).

    Args:
        df            : DataFrame slice (train / val / test)
        image_dir     : Directory where images reside (filename = <patient_id>.jpg/png)
        scaler        : Fitted StandardScaler for continuous features
        transform     : torchvision transform callable
        dummy_images  : If True, generate random image tensors (no disk reads)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        image_dir: str,
        scaler: StandardScaler,
        transform=None,
        dummy_images: bool = False,
    ):
        self.df           = df.reset_index(drop=True)
        self.image_dir    = image_dir
        self.scaler       = scaler
        self.transform    = transform
        self.dummy_images = dummy_images

        # Pre-compute scaled tabular matrix (N × TABULAR_INPUT_DIM)
        self._tabular_matrix = self._preprocess_tabular(df)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _preprocess_tabular(self, df: pd.DataFrame) -> np.ndarray:
        """
        1. Fill missing continuous values with the per-column median.
        2. Scale continuous features with the pre-fit scaler.
        3. Concatenate scaled continuous + raw categorical columns.
        Returns float32 ndarray of shape (N, TABULAR_INPUT_DIM).
        """
        cont = df[CONTINUOUS_FEATURES].copy()
        cont = cont.fillna(cont.median())
        cont_scaled = self.scaler.transform(cont)

        cat = df[CATEGORICAL_FEATURES].fillna(0).values.astype(np.float32)

        return np.concatenate([cont_scaled, cat], axis=1).astype(np.float32)

    def _load_image(self, patient_id: int) -> torch.Tensor:
        """
        Try common image extensions. Fall back to dummy tensor if file not found.
        This graceful fallback allows the codebase to run even before the APTOS
        images are downloaded.
        """
        for ext in [".jpg", ".jpeg", ".png"]:
            path = os.path.join(self.image_dir, f"{patient_id}{ext}")
            if os.path.exists(path):
                img = Image.open(path).convert("RGB")
                if self.transform:
                    img = self.transform(img)
                return img

        # Image not found — return a zero tensor so the pipeline doesn't crash
        return torch.zeros(3, 224, 224)

    # ── Dataset API ───────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]

        # ── Image ─────────────────────────────────────────────────────────
        if self.dummy_images:
            image_tensor = torch.randn(3, 224, 224)
        else:
            image_tensor = self._load_image(int(row["patient_id"]))

        # ── Tabular ───────────────────────────────────────────────────────
        tabular_tensor = torch.tensor(self._tabular_matrix[idx], dtype=torch.float32)

        # ── Label ─────────────────────────────────────────────────────────
        label = torch.tensor(float(row["complication_label"]), dtype=torch.float32)

        return image_tensor, tabular_tensor, label


# ──────────────────────────────────────────────────────────────────────────────
# Factory: create train / val / test DataLoaders in one call
# ──────────────────────────────────────────────────────────────────────────────

def create_dataloaders(
    csv_path: str = SYNTHETIC_CSV,
    image_dir: str = IMAGE_DIR,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    dummy_images: bool = False,
    use_circle_crop: bool = False,
    val_ratio: float = VAL_RATIO,
    seed: int = RANDOM_SEED,
) -> Dict[str, DataLoader]:
    """
    Performs patient-level stratified split and returns a dict of DataLoaders:
        {"train": ..., "val": ..., "test": ...}

    Patient-level splitting: each unique patient_id goes to exactly one split.
    This prevents data leakage if a patient has multiple images (left/right eye).
    """
    df = pd.read_csv(csv_path)

    # ── Patient-level stratified split ────────────────────────────────────
    unique_patients = df[["patient_id", "complication_label"]].drop_duplicates(
        subset="patient_id"
    )

    # First: split off test set (15%)
    train_val_ids, test_ids = train_test_split(
        unique_patients["patient_id"].values,
        test_size=0.15,
        random_state=seed,
        stratify=unique_patients["complication_label"].values,
    )
    # Second: split train_val into train / val
    val_size_adjusted = val_ratio / (1 - 0.15)
    train_ids, val_ids = train_test_split(
        train_val_ids,
        test_size=val_size_adjusted,
        random_state=seed,
    )

    df_train = df[df["patient_id"].isin(train_ids)].copy()
    df_val   = df[df["patient_id"].isin(val_ids)].copy()
    df_test  = df[df["patient_id"].isin(test_ids)].copy()

    # ── Fit scaler on training data only ──────────────────────────────────
    scaler = fit_scaler(df_train)

    # ── Transforms ────────────────────────────────────────────────────────
    train_tfm = get_train_transforms(use_circle_crop)
    val_tfm   = get_val_transforms(use_circle_crop)

    # ── Instantiate datasets ───────────────────────────────────────────────
    train_ds = DiabeticDataset(df_train, image_dir, scaler, train_tfm, dummy_images)
    val_ds   = DiabeticDataset(df_val,   image_dir, scaler, val_tfm,   dummy_images)
    test_ds  = DiabeticDataset(df_test,  image_dir, scaler, val_tfm,   dummy_images)

    # ── Class weights for optional weighted sampler (handles imbalance) ───
    pos_count = df_train["complication_label"].sum()
    neg_count = len(df_train) - pos_count
    class_weights = {0: 1.0, 1: neg_count / max(pos_count, 1)}
    sample_weights = df_train["complication_label"].map(class_weights).values
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    loaders = {
        "train": DataLoader(
            train_ds, batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        ),
        "val": DataLoader(
            val_ds, batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
        "test": DataLoader(
            test_ds, batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
    }

    print(f"[DataLoader] Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    return loaders, scaler


# ──────────────────────────────────────────────────────────────────────────────
# Quick sanity check
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    loaders, scaler = create_dataloaders(dummy_images=True)
    imgs, tabs, labels = next(iter(loaders["train"]))
    print(f"Image tensor  : {imgs.shape}")         # (B, 3, 224, 224)
    print(f"Tabular tensor: {tabs.shape}")          # (B, 17)
    print(f"Labels        : {labels.shape}")        # (B,)
    print(f"Label sample  : {labels[:8].tolist()}")
