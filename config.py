"""
config.py — Central configuration for the Multi-Modal Diabetic Complication Predictor.
All hyperparameters, paths, and feature lists live here. Import from this file across
all modules to maintain a single source of truth.
"""

import os
from dataclasses import dataclass, field
from typing import List


# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
ROOT_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR        = os.path.join(ROOT_DIR, "data")
IMAGE_DIR       = os.path.join(DATA_DIR, "images")          # APTOS fundus images go here
SYNTHETIC_CSV   = os.path.join(DATA_DIR, "synthetic_ehr.csv")
CHECKPOINTS_DIR = os.path.join(ROOT_DIR, "checkpoints")
LOGS_DIR        = os.path.join(ROOT_DIR, "logs")
SAMPLE_DIR      = os.path.join(ROOT_DIR, "app", "sample_images")

# Create directories if they don't exist
for _dir in [DATA_DIR, IMAGE_DIR, CHECKPOINTS_DIR, LOGS_DIR, SAMPLE_DIR]:
    os.makedirs(_dir, exist_ok=True)


# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────
NUM_CLASSES     = 1          # Binary: 0 = No complication, 1 = Complication risk
IMAGE_SIZE      = 224        # Input resolution for EfficientNet-B0
NUM_WORKERS     = 4          # DataLoader workers (set to 0 on Windows if issues)
TRAIN_RATIO     = 0.70
VAL_RATIO       = 0.15
TEST_RATIO      = 0.15
RANDOM_SEED     = 42

# Synthetic EHR generation
NUM_SYNTHETIC_PATIENTS = 2000

# ──────────────────────────────────────────────
# Tabular Feature Configuration
# ──────────────────────────────────────────────
CONTINUOUS_FEATURES: List[str] = [
    "age",
    "diabetes_duration_years",
    "hba1c",
    "fasting_blood_sugar",
    "systolic_bp",
    "diastolic_bp",
    "bmi",
    "serum_creatinine",
    "ldl_cholesterol",
    "hdl_cholesterol",
    "triglycerides",
]

CATEGORICAL_FEATURES: List[str] = [
    "gender",           # 0 = Female, 1 = Male
    "smoker",           # 0 = No, 1 = Yes
    "hypertension",     # 0 = No, 1 = Yes
    "on_insulin",       # 0 = No, 1 = Yes
    "family_history",   # 0 = No, 1 = Yes
    "rural_urban",      # 0 = Rural, 1 = Urban
]

ALL_FEATURES: List[str] = CONTINUOUS_FEATURES + CATEGORICAL_FEATURES
TABULAR_INPUT_DIM: int  = len(ALL_FEATURES)   # 17


# ──────────────────────────────────────────────
# Model Architecture
# ──────────────────────────────────────────────
IMAGE_BACKBONE      = "efficientnet_b0"   # timm model name
FREEZE_RATIO        = 0.70               # Freeze first 70% of backbone layers
IMAGE_EMBEDDING_DIM = 256                # Output dim of image branch
TABULAR_HIDDEN_DIMS = [128, 256, 128]    # MLP layer sizes for tabular branch
TABULAR_EMBEDDING_DIM = 128             # Output dim of tabular branch
FUSION_DIM          = 192               # Output dim of gated fusion layer
DROPOUT_IMG         = 0.40
DROPOUT_TAB         = 0.30


# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────
BATCH_SIZE      = 32
MAX_EPOCHS      = 30
LEARNING_RATE   = 1e-4
WEIGHT_DECAY    = 1e-4
FOCAL_LOSS_GAMMA = 2.0
FOCAL_LOSS_ALPHA = 0.25   # Down-weight easy negatives
EARLY_STOPPING_PATIENCE = 7
LR_SCHEDULER    = "cosine"   # "cosine" | "step"

# Optuna HPO
HPO_N_TRIALS    = 20
HPO_METRIC      = "val_pr_auc"
HPO_DIRECTION   = "maximize"


# ──────────────────────────────────────────────
# Logging / Tracking
# ──────────────────────────────────────────────
WANDB_PROJECT   = "multimodal-diabetic-predictor"
WANDB_ENTITY    = None          # Set to your W&B username


# ──────────────────────────────────────────────
# Inference / App
# ──────────────────────────────────────────────
INFERENCE_THRESHOLD = 0.50   # Default classification threshold
GRADCAM_TARGET_LAYER = "blocks[-1]"   # EfficientNet's last conv block
