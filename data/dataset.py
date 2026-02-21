"""
data/dataset.py
PyTorch Dataset returning (image, tabular, label) triplets.

Patient-level splitting ensures left and right eye images of the same patient
always land in the same split, preventing data leakage.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, Dict
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    SYNTHETIC_CSV, IMAGE_DIR,
    CONTINUOUS_FEATURES, CATEGORICAL_FEATURES, ALL_FEATURES,
    TABULAR_INPUT_DIM, TRAIN_RATIO, VAL_RATIO, RANDOM_SEED, BATCH_SIZE, NUM_WORKERS
)
from data.transforms import get_train_transforms, get_val_transforms


def fit_scaler(df_train: pd.DataFrame) -> StandardScaler:
    """Fit StandardScaler on continuous training features only (prevents leakage)."""
    scaler = StandardScaler()
    scaler.fit(df_train[CONTINUOUS_FEATURES].fillna(df_train[CONTINUOUS_FEATURES].median()))
    return scaler


class DiabeticDataset(Dataset):
    """
    Multi-modal dataset for diabetic complication prediction.

    Args:
        df           : DataFrame slice (train / val / test)
        image_dir    : Directory containing images named <patient_id>.jpg/png
        scaler       : Fitted StandardScaler for continuous features
        transform    : torchvision transform callable
        dummy_images : Return random tensors instead of loading images (for smoke tests)
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
        self._tabular_matrix = self._preprocess_tabular(df)

    def _preprocess_tabular(self, df: pd.DataFrame) -> np.ndarray:
        cont = df[CONTINUOUS_FEATURES].copy().fillna(df[CONTINUOUS_FEATURES].median())
        cat  = df[CATEGORICAL_FEATURES].fillna(0).values.astype(np.float32)
        return np.concatenate([self.scaler.transform(cont), cat], axis=1).astype(np.float32)

    def _load_image(self, patient_id: int) -> torch.Tensor:
        for ext in [".jpg", ".jpeg", ".png"]:
            path = os.path.join(self.image_dir, f"{patient_id}{ext}")
            if os.path.exists(path):
                img = Image.open(path).convert("RGB")
                return self.transform(img) if self.transform else img
        return torch.zeros(3, 224, 224)  # graceful fallback if image missing

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row            = self.df.iloc[idx]
        image_tensor   = torch.randn(3, 224, 224) if self.dummy_images else self._load_image(int(row["patient_id"]))
        tabular_tensor = torch.tensor(self._tabular_matrix[idx], dtype=torch.float32)
        label          = torch.tensor(float(row["complication_label"]), dtype=torch.float32)
        return image_tensor, tabular_tensor, label


def create_dataloaders(
    csv_path: str       = SYNTHETIC_CSV,
    image_dir: str      = IMAGE_DIR,
    batch_size: int     = BATCH_SIZE,
    num_workers: int    = NUM_WORKERS,
    dummy_images: bool  = False,
    use_circle_crop: bool = False,
    val_ratio: float    = VAL_RATIO,
    seed: int           = RANDOM_SEED,
) -> Dict[str, DataLoader]:
    """
    Patient-level stratified split into train/val/test, returns DataLoader dict.

    Returns:
        (loaders, scaler) — scaler needed for inference preprocessing
    """
    df = pd.read_csv(csv_path)

    unique_patients = df[["patient_id", "complication_label"]].drop_duplicates(subset="patient_id")
    train_val_ids, test_ids = train_test_split(
        unique_patients["patient_id"].values,
        test_size=0.15,
        random_state=seed,
        stratify=unique_patients["complication_label"].values,
    )
    train_ids, val_ids = train_test_split(
        train_val_ids,
        test_size=val_ratio / (1 - 0.15),
        random_state=seed,
    )

    df_train = df[df["patient_id"].isin(train_ids)].copy()
    df_val   = df[df["patient_id"].isin(val_ids)].copy()
    df_test  = df[df["patient_id"].isin(test_ids)].copy()

    scaler    = fit_scaler(df_train)
    train_tfm = get_train_transforms(use_circle_crop)
    val_tfm   = get_val_transforms(use_circle_crop)

    train_ds = DiabeticDataset(df_train, image_dir, scaler, train_tfm, dummy_images)
    val_ds   = DiabeticDataset(df_val,   image_dir, scaler, val_tfm,   dummy_images)
    test_ds  = DiabeticDataset(df_test,  image_dir, scaler, val_tfm,   dummy_images)

    pos_count      = df_train["complication_label"].sum()
    neg_count      = len(df_train) - pos_count
    class_weights  = {0: 1.0, 1: neg_count / max(pos_count, 1)}
    sample_weights = df_train["complication_label"].map(class_weights).values
    sampler        = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )

    loaders = {
        "train": DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                            num_workers=num_workers, pin_memory=True, drop_last=True),
        "val":   DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True),
        "test":  DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True),
    }

    print(f"[DataLoader] Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    return loaders, scaler


if __name__ == "__main__":
    loaders, _ = create_dataloaders(dummy_images=True)
    imgs, tabs, labels = next(iter(loaders["train"]))
    print(f"Image: {imgs.shape}  Tabular: {tabs.shape}  Labels: {labels.shape}  ✓")
