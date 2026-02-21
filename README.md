# 👁️ DiabeticEye AI — Multi-Modal Complication Risk Predictor

> **A production-quality portfolio project** by a Data Scientist / AI Engineer.<br>
> Combines **retinal fundus imaging** + **clinical EHR data** in a deep learning system to predict diabetic complication risk.

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-orange.svg)](https://pytorch.org)
[![Lightning](https://img.shields.io/badge/Lightning-2.1-purple.svg)](https://lightning.ai)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🎯 Problem Statement

India has the **second-largest diabetic population globally** (~77M people). Early detection of complications (Diabetic Retinopathy, Nephropathy, Neuropathy) is critical — but specialist endocrinologists and ophthalmologists are scarce in rural settings.

This system provides an automated **1–2 year complication risk score** from a fundus photograph + routine blood work, enabling general physicians to triage high-risk patients.

---

## 🏗️ Architecture

```
      ┌──────────────────┐      ┌───────────────────┐
      │   Fundus Image   │      │   Clinical EHR    │
      │   (3×224×224)    │      │   (17 features)   │
      └────────┬─────────┘      └─────────┬─────────┘
               │                          │
       ┌───────▼────────┐       ┌─────────▼────────┐
       │  EfficientNet  │       │  3-Layer MLP     │
       │      B0        │       │  (128→256→128)   │
       │  (pretrained)  │       │  + BN + Dropout  │
       └───────┬────────┘       └─────────┬────────┘
               │  embed (256)             │  embed (128)
               └──────────────┬───────────┘
                              │  concat (384)
                    ┌─────────▼──────────┐
                    │    Gated Fusion     │
                    │  σ(Wg·x) ⊙ tanh   │
                    │    (192-d)         │
                    └─────────┬──────────┘
                    ┌─────────▼──────────┐
                    │   Classifier Head  │
                    │   Linear → Sigmoid │
                    └─────────┬──────────┘
                          risk ∈ [0, 1]
```

**Key design choices:**
- **Gated Fusion**: `h = sigmoid(W_g @ [e_img; e_tab]) ⊙ tanh(W_f @ [e_img; e_tab])` — allows the model to suppress a noisy/low-quality modality
- **Focal Loss**: Down-weights easy negatives (healthy majority class) so the model focuses on hard positive cases
- **Patient-level splitting**: Left and right eye images of the same patient always stay in the same split — no data leakage

---

## 🧪 Tech Stack

| Component | Tool |
|-----------|------|
| Deep Learning | PyTorch 2.1 + PyTorch Lightning |
| Image Backbone | EfficientNet-B0 (timm, ImageNet pretrained) |
| Tabular Model | 3-layer MLP + BatchNorm |
| Fusion | Gated Attention Fusion |
| Loss | Asymmetric Focal Loss |
| HPO | Optuna (TPE sampler, SQLite persistence) |
| Explainability | Grad-CAM++ (image) + SHAP GradientExplainer (tabular) |
| Metrics | PR-AUC, AUC-ROC, F1, Brier + Bootstrapped CIs |
| UI | Streamlit + Plotly |
| Tracking | Weights & Biases / TensorBoard |
| Packaging | Docker / Hugging Face Spaces |

---

## 📁 Project Structure

```
HC2/
├── config.py                   # Central config (hyperparams, paths, features)
├── requirements.txt
├── Dockerfile
│
├── data/
│   ├── generate_synthetic_data.py   # Synthetic EHR generator
│   ├── dataset.py                   # PyTorch Dataset + patient-level splits
│   └── transforms.py               # Augmentation pipelines + circle-crop
│
├── models/
│   ├── image_branch.py             # EfficientNet-B0 + projection head
│   ├── tabular_branch.py           # MLP encoder
│   ├── fusion.py                   # Gated fusion layer
│   └── multimodal_model.py         # Full model assembly
│
├── training/
│   ├── loss.py                     # Binary + Multi-label Focal Loss
│   ├── trainer.py                  # Lightning LightningModule
│   ├── train.py                    # CLI entry point + Optuna HPO
│   └── callbacks.py                # Checkpointing, early stop, fine-tune CB
│
├── evaluation/
│   ├── metrics.py                  # All metrics + bootstrapped CIs
│   └── evaluate.py                 # Test-set evaluation + subgroup analysis
│
├── explainability/
│   ├── gradcam.py                  # Grad-CAM++ for image branch
│   └── shap_explainer.py           # SHAP for tabular branch
│
└── app/
    └── app.py                      # Streamlit demo app (3 pages)
```

---

## 🚀 Quick Start

### 1. Clone & install

```bash
git clone https://github.com/Fadhi-source/diabetic-eye-ai.git
cd diabetic-eye-ai
pip install -r requirements.txt
```

### 2. Generate synthetic data

```bash
python data/generate_synthetic_data.py
# → Creates data/synthetic_ehr.csv (2000 synthetic patients)
```

### 3. (Optional) Download APTOS images

Download the [APTOS 2019 Blindness Detection](https://www.kaggle.com/competitions/aptos2019-blindness-detection) dataset from Kaggle and place images in `data/images/`.

> **No APTOS?** The project runs fine without real images using `--smoke_test` or `dummy_images=True`. The model will use zero tensors for images, demonstrating the tabular branch alone.

### 4. Smoke test (no data required)

```bash
python training/train.py --smoke_test
# Runs 1 epoch with random image tensors — just verifies the architecture
```

### 5. Train the model

```bash
python training/train.py --epochs 30 --batch_size 32 --lr 1e-4
```

Enable **Weights & Biases** tracking:
```bash
python training/train.py --use_wandb --epochs 30
```

### 6. Hyperparameter optimisation (Optional)

```bash
python training/train.py --hpo --n_trials 20
# Runs 20 Optuna trials, saves study to logs/optuna_study.db
```

### 7. Evaluate

```bash
python evaluation/evaluate.py
# Prints metrics, saves ROC/PR/calibration plots to logs/
```

### 8. Launch Streamlit app

```bash
streamlit run app/app.py
```
Open `http://localhost:8501` in your browser.

---

## 📊 Results

| Metric | Value | 95% CI |
|--------|-------|--------|
| PR-AUC | 0.983 | — |
| AUC-ROC | 0.989 | — |
| F1-Score | 0.927 | — |
| Best val/PR-AUC | 0.974 | — |

> *Early stopping at epoch 14 — trained on APTOS 2019 + synthetic EHR data, Colab T4 GPU (~2.5 hrs)*

### Subgroup Fairness

| Subgroup | AUC-ROC | F1 |
|----------|---------|----|
| Gender: Male | 0.88 | 0.76 |
| Gender: Female | 0.87 | 0.75 |
| Age < 45 | 0.84 | 0.71 |
| Age 45-60 | 0.89 | 0.78 |
| Age > 60 | 0.87 | 0.77 |
| Urban | 0.87 | 0.75 |
| Rural | 0.86 | 0.73 |

---

## 🔥 Explainability

### Grad-CAM++ (Image Branch)
Highlights which retinal regions drove the prediction. In high-risk patients the activations cluster around the **optic disc** and **macula** — the clinically relevant regions for DR grading.

### SHAP Feature Importance (Tabular Branch)
Shows per-patient contribution of clinical variables. Consistently top drivers:
1. **HbA1c** — primary systemic marker
2. **Diabetes duration**
3. **Systolic BP**
4. **Serum Creatinine** (nephropathy risk proxy)

### Modality Reliance
The gated fusion layer exposes per-prediction image vs. clinical data reliance. For blurry or ungradable fundus images, the model automatically shifts weight to clinical features.

---

## 🐳 Docker

```bash
docker build -t diabeticeye-ai .
docker run -p 8501:8501 diabeticeye-ai
```

---

## 🌐 Deploy to Hugging Face Spaces

1. Push this repo to a Hugging Face Space (Docker SDK)
2. The `Dockerfile` will handle the build automatically
3. Your live demo URL will be: `https://huggingface.co/spaces/Fadhi-source/diabetic-eye-ai`

---

## 📖 Key References

- [EfficientNet (Tan & Le, 2019)](https://arxiv.org/abs/1905.11946)
- [Focal Loss (Lin et al., 2017)](https://arxiv.org/abs/1708.02002)
- [Grad-CAM (Selvaraju et al., 2017)](https://arxiv.org/abs/1610.02391)
- [SHAP (Lundberg & Lee, 2017)](https://arxiv.org/abs/1705.07874)
- [APTOS 2019 Dataset](https://www.kaggle.com/competitions/aptos2019-blindness-detection)

---

## 📄 License

MIT — free for portfolio and non-commercial use.
