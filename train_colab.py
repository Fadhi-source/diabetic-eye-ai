# ─────────────────────────────────────────────────────────────────────────────
# colab_train.py  —  Run on Google Colab T4 GPU
#
# Workflow:
#   1. Clone your GitHub repo (code only, ~10MB)
#   2. Download APTOS dataset directly from Kaggle (no manual upload needed)
#   3. Install dependencies
#   4. Merge datasets & train
# ─────────────────────────────────────────────────────────────────────────────

# ── CELL 1: Clone your repo ───────────────────────────────────────────────────
# Replace with your actual GitHub URL after pushing
GITHUB_URL = "https://github.com/YOUR_USERNAME/HC2.git"

import os, sys
!git clone {GITHUB_URL} /content/HC2
os.chdir('/content/HC2')
sys.path.insert(0, '/content/HC2')


# ── CELL 2: Install dependencies ─────────────────────────────────────────────
!pip install -q timm pytorch-lightning torchmetrics \
    grad-cam shap optuna wandb plotly rich tensorboard tensorboardX kaggle


# ── CELL 3: Download APTOS dataset via Kaggle API ────────────────────────────
# First: upload your kaggle.json API key (download from kaggle.com/account)
from google.colab import files
print("Upload your kaggle.json file (from kaggle.com → Account → API)")
uploaded = files.upload()   # select kaggle.json

# Set up Kaggle credentials
os.makedirs('/root/.kaggle', exist_ok=True)
os.rename('kaggle.json', '/root/.kaggle/kaggle.json')
os.chmod('/root/.kaggle/kaggle.json', 0o600)

# Download and extract APTOS dataset (~8GB) directly to Colab
os.makedirs('data/images', exist_ok=True)
!kaggle competitions download -c aptos2019-blindness-detection -p data/
!unzip -q data/aptos2019-blindness-detection.zip -d data/aptos_raw/
!mv data/aptos_raw/train_images/* data/images/
!cp data/aptos_raw/train.csv data/aptos_train.csv
print("APTOS dataset ready!")


# ── CELL 4: Verify GPU ────────────────────────────────────────────────────────
import torch
print("CUDA:", torch.cuda.is_available())
print("GPU: ", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")


# ── CELL 5: Merge datasets ────────────────────────────────────────────────────
!python data/merge_aptos.py


# ── CELL 6: Train 🚀 ─────────────────────────────────────────────────────────
!python training/train.py \
    --csv data/merged.csv \
    --num_workers 2 \
    --epochs 30 \
    --batch_size 32


# ── CELL 7: Download best checkpoint to your PC ───────────────────────────────
import glob
from google.colab import files

ckpts = sorted(glob.glob('checkpoints/best-*.ckpt'))
if ckpts:
    print(f"Downloading: {ckpts[-1]}")
    files.download(ckpts[-1])
else:
    print("No checkpoint found.")
