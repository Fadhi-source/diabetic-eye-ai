# 🎓 DiabeticEye AI — Interview Preparation Guide

> **Your project in one sentence:**
> *"A multi-modal deep learning system that fuses retinal fundus images and clinical EHR data using a gated attention mechanism to predict diabetic complication risk, trained end-to-end with PyTorch Lightning, achieving a PR-AUC of 0.9737."*

---

## 1. Project Overview

### What problem does it solve?
Diabetic Retinopathy (DR) and related complications affect **537 million diabetics worldwide**. Early detection requires a specialist examining fundus images — a resource that doesn't exist in rural or developing settings. This system automates that screening by combining **two data sources** that doctors themselves use: the eye image AND the patient's blood work / vitals.

### Why is multi-modal better than just the image?
- A blurry or low-quality image alone is unreliable
- Clinical data alone misses retinal-specific pathology
- Together they provide complementary signals — the gated fusion layer **learns** when to trust the image more vs. the clinical data

### Dataset
| Source | What it contains | Size |
|---|---|---|
| **APTOS 2019** (Kaggle) | Retinal fundus images + diabetic retinopathy grade (0–4) | 3,662 images |
| **Synthetic EHR** (generated) | 17 clinical features correlated with DR grade | 2,000 patients |

The synthetic EHR is statistically engineered so that sicker patients (grade 3–4 DR) have higher HbA1c, longer disease duration, higher BP etc. This makes the multi-modal fusion meaningful.

---

## 2. Architecture — Full Deep Dive

```
[Fundus Image]          [Clinical EHR (17 features)]
      │                           │
┌─────▼──────────┐       ┌────────▼────────────┐
│  ImageBranch   │       │   TabularBranch      │
│ EfficientNet-B0│       │   3-Layer MLP        │
│ (pretrained)   │       │   [128 → 256 → 128]  │
│ → 256-d embed  │       │   → 128-d embed      │
└─────┬──────────┘       └────────┬────────────┘
      │  (256-d)                  │ (128-d)
      └──────────────┬────────────┘
                     │ concat → 384-d
              ┌──────▼──────────┐
              │   GatedFusion   │
              │ gate = σ(W·x)   │
              │ feat = tanh(V·x)│
              │ h = gate ⊙ feat │
              │ → 192-d output  │
              └──────┬──────────┘
              ┌──────▼──────────┐
              │  Linear(192→1)  │
              │   + Sigmoid     │
              └──────┬──────────┘
                     │
              Risk probability ∈ [0,1]
```

### 2.1 Image Branch (`models/image_branch.py`)

**Backbone:** EfficientNet-B0 from `timm` library
- `num_classes=0` strips the original classifier → outputs 1280-d feature vector
- `global_pool="avg"` applies Global Average Pooling

**Projection head:**
```
Linear(1280 → 256) → BatchNorm → ReLU → Dropout(0.4)
```

**Why EfficientNet-B0?**
- Best accuracy/parameter tradeoff (scales width, depth, resolution together)
- Pretrained on ImageNet — transfers well to retinal images (edges, textures)
- Small enough to train on a free T4 GPU in <3 hours

**Freezing strategy:**
- First **70%** of backbone layers are frozen initially
- At epoch 10, top **30%** are unfrozen for fine-tuning
- Why? Low-level features (edges, textures) are universal — don't need retraining. High-level features (DR-specific: exudates, haemorrhages) need to adapt.

### 2.2 Tabular Branch (`models/tabular_branch.py`)

**Input:** 17 clinical features = 11 continuous + 6 categorical

| Feature type | Features |
|---|---|
| Continuous (scaled) | age, diabetes_duration, HbA1c, fasting_blood_sugar, systolic_bp, diastolic_bp, BMI, serum_creatinine, LDL, HDL, triglycerides |
| Categorical (0/1) | gender, smoker, hypertension, on_insulin, family_history, rural_urban |

**Architecture:** `17 → 128 → 256 → 128 → 128` (with BN + ReLU + Dropout after each)

**Why BatchNorm before ReLU?**
Clinical features span wildly different ranges (age: 30–80 vs. triglycerides: 50–600). BN normalises each layer's activations so the network isn't dominated by large-valued features.

**Weight init:** Kaiming He (`kaiming_normal_`) — optimal for ReLU networks, prevents vanishing/exploding gradients.

### 2.3 Gated Fusion (`models/fusion.py`)

**The key innovation of this project.**

```python
concat = [img_emb ; tab_emb]        # (B, 384)
gate   = sigmoid(W_g @ concat + b)  # (B, 192) — values in [0,1]
feat   = tanh(W_f @ concat + b)     # (B, 192) — values in [-1,1]
h      = gate ⊙ feat                # (B, 192) — element-wise product
```

**Why gated fusion?**
- Simple concatenation treats both modalities equally — bad when one is missing or low-quality
- The gate **learns** to suppress dimensions of the fused vector when a modality is unreliable
- If the image is blurry → gate suppresses image-driven dimensions → model relies more on clinical data
- This makes the model robust in real-world clinical settings (poor image quality is common)

**Interpretability bonus:** `get_gate_weights()` returns the gate activations. In the Streamlit app, this shows users "the model relied 65% on the image and 35% on clinical data" for this patient.

### 2.4 Classifier Head (`models/multimodal_model.py`)

```python
self.classifier = nn.Linear(192, 1)   # Binary: one output neuron
# Output → sigmoid → probability in [0,1]
```

Xavier uniform init on the classifier — appropriate for sigmoid output.

---

## 3. Data Pipeline

### 3.1 Preprocessing (`data/dataset.py`)

**Train/Val/Test split: Patient-level stratified split**
- 70% train / 15% val / 15% test
- Split is done on **unique patient IDs**, not individual images
- Why? If a patient has left AND right eye images, they must both be in the same split — otherwise you get **data leakage** (model sees a patient's right eye in training, then predicts on their left eye in test)

**Tabular preprocessing:**
1. Fill missing continuous values with **column median** (not mean — robust to outliers)
2. StandardScaler fit **only on training data**, applied to val/test (prevents leakage)
3. Categorical features kept as raw 0/1 (already encoded)

**Class imbalance handling:** `WeightedRandomSampler`
- Computes: `weight[i] = neg_count / pos_count` for positive class
- Oversamples minority (sick patients) in training
- Val/Test left unsampled for honest evaluation

### 3.2 Transforms (`data/transforms.py`)

**Training augmentations:**
- Random horizontal/vertical flip
- Random rotation (±10°)
- Colour jitter (brightness, contrast, saturation)
- Gaussian blur
- Normalise with ImageNet mean/std

**Validation/Test:** Only resize + centre crop + normalise (no random augmentation)

**Why ImageNet mean/std for retinal images?**
The EfficientNet backbone was pretrained on ImageNet with these normalisation values. Using the same values ensures the backbone's learned feature detectors work correctly — the input distribution matches what those neurons expect.

### 3.3 Merge Script (`data/merge_aptos.py`)
Joins APTOS retinal images (grade 0–4) with synthetic EHR records, creating a unified `merged.csv` with columns: `patient_id`, all 17 features, `dr_grade`, `complication_label` (binary: grade ≥ 2 → 1, else 0).

---

## 4. Training Strategy

### 4.1 Loss Function — Binary Focal Loss (`training/loss.py`)

**Formula:**
```
L = -α · (1 - p_t)^γ · log(p_t)
```
Where:
- `p_t` = model's probability of the correct class
- `γ = 2.0` — focusing parameter. High γ = more focus on hard examples
- `α = 0.25` — down-weights easy negatives (healthy patients)

**Why not standard BCE?**
In an imbalanced dataset, most samples are easy negatives (healthy patients = trivially predicted as "no risk"). Standard BCE gets dominated by these easy examples — the model learns to just predict "low risk" for everyone and still gets low loss. Focal Loss solves this by multiplying the loss of easy examples (high `p_t`) by a tiny number → forces the model to pay attention to hard positives.

**Why BCE with logits vs. sigmoid + BCE separately?**
`F.binary_cross_entropy_with_logits` uses the log-sum-exp trick for numerical stability — avoids exploding/vanishing gradients near probabilities of 0 or 1.

### 4.2 Optimizer & Scheduler (`training/trainer.py`)

- **AdamW** (Adam + decoupled weight decay)
  - L2 penalty applied separately from gradient update (more principled than standard Adam)
  - Only **trainable parameters** are passed (frozen backbone layers excluded)
- **CosineAnnealingLR:** Learning rate anneals smoothly from `1e-4` to `1e-6` over 30 epochs
  - Avoids sharp LR drops that can destabilise training late in training

### 4.3 Training Callbacks (`training/callbacks.py`)

| Callback | What it does |
|---|---|
| `ModelCheckpoint` | Saves top-2 checkpoints by `val/pr_auc` |
| `EarlyStopping` | Stops training if `val/pr_auc` doesn't improve for 7 epochs |
| `BackboneFineTuneCallback` | At epoch 10, unfreezes top 30% of EfficientNet + divides LR by 10 |
| `LearningRateMonitor` | Logs LR to W&B / TensorBoard every epoch |
| `RichProgressBar` | Pretty terminal progress bar |

**Two-stage training:**
- **Epochs 0–9:** Only new layers (heads, fusion, tabular branch) are trained. Backbone frozen.
- **Epoch 10+:** Top 30% of backbone unfrozen, LR reduced 10x. End-to-end fine-tuning.

### 4.4 Metrics Tracked

| Metric | Why it matters |
|---|---|
| **PR-AUC** (primary) | Best for imbalanced datasets — doesn't get inflated by true negatives |
| **ROC-AUC** | Overall discrimination ability |
| **F1-Score** | Harmonic mean of precision and recall |
| **Brier Score** | Calibration — are probabilities well-calibrated? |

**Why PR-AUC over ROC-AUC for imbalanced data?**
ROC-AUC is optimistic when negatives heavily outnumber positives — a model predicting random scores can still get high ROC-AUC because it correctly ranks most true negatives. PR-AUC is anchored to the positive class, so it only looks good when you actually catch sick patients.

### 4.5 Mixed Precision Training (`precision="16-mixed"`)
FP16 for forward/backward passes, FP32 for weight updates. Doubles throughput on GPU, halves memory usage — essential for training on free Colab/Kaggle GPUs.

### 4.6 HPO with Optuna (optional mode)
- TPE (Tree-structured Parzen Estimator) sampler — smarter than random search
- MedianPruner: kills underperforming trials early
- Results stored in SQLite DB → can resume across sessions
- Hyperparameters searched: `lr`, `weight_decay`, `batch_size`, `freeze_ratio`, `fine_tune_epoch`

---

## 5. Explainability

### 5.1 Grad-CAM++ (`explainability/gradcam.py`)

**What it does:** Highlights WHICH pixels in the retinal image caused the high risk score.

**How it works:**
1. Forward pass → compute class score
2. Backpropagate gradients to the **last convolutional layer** of EfficientNet (`blocks[-1]`)
3. Average the gradients across spatial dimensions → per-channel importance weights
4. Weighted sum of feature maps → heatmap
5. Overlay on original image (red = important regions)

**ImageOnlyWrapper trick:** Grad-CAM needs a model that accepts only an image. We wrap the full model, fixing the tabular input as a constant, so Grad-CAM can compute image gradients normally.

**Clinical meaning:** Red regions might show optic disc neovascularisation, macular oedema, haemorrhages — the actual pathological features a retina specialist would look at.

### 5.2 SHAP (`explainability/shap_explainer.py`)

**What it does:** Explains WHICH clinical features pushed the risk score up or down for a specific patient.

**Method used:** `shap.GradientExplainer` — computes Shapley values via smooth-gradient approximation through the model's backpropagation.

**Output:** A waterfall chart showing:
- Red bars: features that **increased** risk (e.g., HbA1c = 11.2 → +0.34 SHAP)
- Teal bars: features that **decreased** risk (e.g., HDL = 62 → -0.18 SHAP)

**TabularOnlyWrapper trick:** Same pattern as Grad-CAM — fix the image to a constant (black/mean image) so SHAP only attributes to tabular features.

**Background dataset:** 50-sample random tensor used as SHAP baseline (represents "average patient"). SHAP values = contribution relative to this baseline.

---

## 6. Results

| Metric | Value |
|---|---|
| **PR-AUC** | **0.9737** (best checkpoint, epoch 14) |
| Training time | ~2–3 hours on Colab T4 GPU |
| Checkpoint size | ~45 MB |
| Total parameters | ~4.5M (EfficientNet-B0: ~4M + heads) |

**What PR-AUC of 0.9737 means:** If you ranked all patients from highest to lowest predicted risk, the model is **97.4% accurate** at putting sick patients above healthy ones — across all possible thresholds.

---

## 7. Streamlit App (`app/app.py`)

Three pages:

| Page | What it shows |
|---|---|
| **🔬 Predict** | Upload fundus image + enter 17 clinical vitals → Plotly gauge chart showing risk % |
| **🧠 Explain** | Grad-CAM++ heatmap side-by-side original + SHAP waterfall chart + modality gate balance |
| **📊 About** | Architecture diagram, tech stack, dataset info, real metrics |

**Model loading:** `@st.cache_resource` — loads checkpoint once, reuses across all user interactions. Falls back to random weights if no checkpoint found (for demo purposes).

---

## 8. Tech Stack Summary

| Component | Technology | Why |
|---|---|---|
| Deep learning | PyTorch | Industry standard, flexible |
| Training framework | PyTorch Lightning | Removes boilerplate, adds best practices automatically |
| Image backbone | EfficientNet-B0 (timm) | State-of-art efficiency, pretrained |
| Metrics | torchmetrics | Handles distributed training, no manual accumulation |
| Experiment tracking | W&B / TensorBoard | W&B for cloud dashboards, TensorBoard as fallback |
| HPO | Optuna | Modern Bayesian HPO, built-in pruning |
| Explainability (image) | pytorch-grad-cam | GradCAM++ in 3 lines of code |
| Explainability (tabular) | SHAP | Industry-standard feature attribution |
| UI | Streamlit | Fastest way to ship an ML demo |
| Visualisations | Plotly | Interactive charts in Streamlit |

---

## 9. Interview Q&A

### 9.1 Why did you build this project?
> "I wanted a portfolio project that demonstrates real-world ML engineering — not just training a model, but the full pipeline: data engineering, multi-modal architecture design, training at scale on free GPU resources, explainability for clinical trust, and a deployable UI. The healthcare domain adds real stakes — a wrong prediction could affect a patient's care — so I had to think carefully about every design decision."

---

### 9.2 ML Theory Questions

**Q: Why use Focal Loss over standard BCE?**
> The dataset is class-imbalanced — most patients don't have complications. Standard BCE gets dominated by easy negatives. Focal Loss applies `(1-p_t)^γ` — a factor near zero for easy examples — so they barely contribute to the gradient. The model is forced to focus on hard, uncertain positive cases.

**Q: Why PR-AUC instead of accuracy or ROC-AUC?**
> Accuracy is meaningless for imbalanced data — a model that always predicts "no complication" gets 80% accuracy but is useless. ROC-AUC is inflated by the large number of true negatives. PR-AUC only cares about precision and recall for the positive class, making it the honest metric for rare-event detection.

**Q: Explain the two-stage training.**
> Stage 1 (epochs 0–9): Freeze 70% of EfficientNet. Only train the new layers — tabular branch, gated fusion, classifier, top image layers. The pretrained backbone already knows edges and textures; we don't want to destroy that with a large learning rate.
> Stage 2 (epoch 10+): Unfreeze top 30% of backbone at 10x lower LR. Now the high-level backbone layers can specialise in retinal features (exudates, haemorrhages) while preserving the low-level ImageNet features.

**Q: Why StandardScaler on only the training set?**
> If you fit the scaler on the full dataset, the mean and variance computed include information from the test set. When the model is deployed, it will see patients with different distributions. The scaler should only know what the average patient looks like in the training data — the same knowledge the model has.

**Q: What is data leakage and how did you prevent it?**
> Data leakage is when information from the test set influences the model during training. I prevented it two ways: (1) Patient-level splitting — if a patient has multiple images, all must be in the same split. (2) Scaler fitted only on training data, applied to val/test without refitting.

**Q: Why Cosine Annealing LR scheduler?**
> Step-based schedulers cause abrupt drops in LR that can destabilise training. Cosine annealing smoothly reduces LR, allowing the model to settle into flat minima (which generalise better) in later epochs.

**Q: Why WeightedRandomSampler on training set but not val/test?**
> During training, we want the model to see an approximately balanced number of positive and negative examples per batch, otherwise it will rarely see sick patients and won't learn their signatures. During evaluation, we want honest metrics on the real-world class distribution — oversampling here would give an artificially optimistic view of performance.

**Q: What is EfficientNet and why is it efficient?**
> EfficientNet uses a compound scaling rule — instead of scaling only depth, width, or resolution separately, it scales all three in a fixed ratio found by neural architecture search. B0 is the smallest variant. It achieves ImageNet state-of-art accuracy with 5.3M parameters, while ResNet-50 needs 25M for similar accuracy.

**Q: What is BatchNorm and when would you NOT use it?**
> BatchNorm normalises each layer's activations to zero mean and unit variance using statistics computed over the current mini-batch. It accelerates training and acts as implicit regularisation. You should avoid it when: (1) batch size is very small (statistics become noisy) — use GroupNorm/LayerNorm instead; (2) during inference with a single sample (batch statistics are undefined).

**Q: Explain the Gated Fusion mathematically.**
> `gate = σ(W_g · [img; tab])` — a per-dimension value in (0,1) learned from both modalities.
> `feat = tanh(W_f · [img; tab])` — the content to be gated, in (-1,1).
> `h = gate ⊙ feat` — element-wise product. Gate near 0 → suppress that dimension. Gate near 1 → pass it through.
> This is analogous to an LSTM gate — it learns to selectively retain information.

---

### 9.3 System Design Questions

**Q: How would you scale this to production?**
> (1) Model serving: Export to ONNX or TorchScript, serve via FastAPI + Uvicorn. (2) Batching: Use dynamic batching to serve multiple patients per GPU call. (3) Preprocessing: Move image transforms to a GPU pipeline (NVIDIA DALI). (4) Monitoring: Log prediction distributions over time to detect distribution shift. (5) Fallback: If image quality is poor (BRISQUE score too high), rely only on tabular branch.

**Q: How would you handle missing clinical features at inference?**
> The current model uses median imputation — missing values are filled with the training-set median. In production, for completely missing features, you could: (1) impute with population medians, (2) create a "missingness" indicator feature, (3) train a version with random feature masking so the model learns to be robust to missing inputs (a technique called MCAR training).

**Q: How would you deploy this to Hugging Face Spaces?**
> Create a `requirements.txt`, add `app.py` with the Streamlit entry point, push to a HF Spaces repo. The Space auto-builds the Docker container and serves it. The checkpoint (~45 MB) is included in the repo. For larger models, use `huggingface_hub` to download the checkpoint on first startup from the HF Model Hub.

**Q: The model achieves 0.97 PR-AUC — is it trustworthy?**
> High PR-AUC on the evaluation set is promising, but I would want to: (1) Validate on a completely held-out external dataset from a different hospital (distribution shift). (2) Run subgroup analysis — does it perform equally well for males vs. females, rural vs. urban, different age groups? (3) Get clinical validation from an ophthalmologist reviewing the Grad-CAM heatmaps. (4) Calibration check — are the predicted probabilities meaningful (e.g., patients predicted at 70% risk actually have ~70% incidence)?

---

### 9.4 Code-Level Questions

**Q: Why `filter(lambda p: p.requires_grad, self.model.parameters())` in the optimizer?**
> Frozen parameters have `requires_grad=False`. Passing them to AdamW would waste memory (optimizer state allocated for parameters that will never update) and slow training. This filter passes only trainable parameters.

**Q: Why `drop_last=True` in the training DataLoader?**
> BatchNorm requires at least 2 samples per batch to compute meaningful statistics. If the last batch has only 1 sample, BN will throw an error (variance of a single sample is 0). `drop_last=True` discards the final incomplete batch.

**Q: Why `pin_memory=True` in the DataLoader?**
> Pinned memory (page-locked RAM) enables faster CPU→GPU memory transfer via DMA (Direct Memory Access), bypassing the OS paging mechanism. This reduces data loading bottlenecks on GPU.

**Q: Why use `F.binary_cross_entropy_with_logits` instead of `sigmoid` + `F.binary_cross_entropy`?**
> Numerical stability. `log(sigmoid(x))` for very large positive `x` results in `log(1.0)` = `0` in float32 (underflow). `binary_cross_entropy_with_logits` uses the log-sum-exp reformulation: `max(x, 0) - x*y + log(1+exp(-|x|))`, which is numerically stable for all values.

**Q: What does `save_hyperparameters()` do in Lightning?**
> It automatically saves all `__init__` arguments to `self.hparams` and stores them in the checkpoint. When you load a checkpoint with `load_from_checkpoint`, Lightning automatically passes the saved hyperparameters to `__init__`, so you don't need to re-specify the architecture. Critical for reproducibility.

---

### 9.5 "Tell Me About" Questions

**Q: What was the hardest problem you solved?**
> Connecting the two modalities meaningfully. A naive approach (just concatenate image and tabular embeddings) works, but doesn't learn which modality to trust. I implemented gated fusion — the same principle as LSTM gates — to let the model dynamically weight modalities. This also gave us a free interpretability signal: we can show users exactly how much the model relied on the image vs. clinical data.

**Q: What would you improve if you had more time?**
> (1) Replace synthetic EHR with real patient records from a hospital dataset like APTOS clinical metadata or UK Biobank. (2) Add temporal modelling — patients have multiple visits, and HbA1c trend over time is more predictive than a single value. (3) Uncertainty quantification with Monte Carlo Dropout — predicting not just risk probability but a confidence interval, so doctors know when the model is uncertain. (4) Multi-task learning: predict DR grade (0–4) simultaneously as an auxiliary task to improve the shared representations.

**Q: How is your model interpretable/trustworthy for clinical use?**
> Three layers of interpretability: (1) **Grad-CAM++** shows exactly which retinal regions drove the prediction — a clinician can verify these are medically meaningful (optic disc, macula area). (2) **SHAP** shows which clinical features pushed the risk up or down — matches clinical knowledge (high HbA1c, long duration = higher risk). (3) **Gate weights** show the modality balance — if the model over-relies on the image for a blurry scan, we can flag it for human review.

---

## 10. Key Terms Glossary

| Term | Simple explanation |
|---|---|
| **PR-AUC** | Area under the Precision-Recall curve. 1.0 = perfect, 0.5 = random (for balanced) |
| **Focal Loss** | Loss function that down-weights easy examples, focuses training on hard ones |
| **Gated Fusion** | Learned per-dimension weighting to combine two information sources |
| **EfficientNet-B0** | Compact CNN backbone with 5.3M params, trained on ImageNet |
| **Grad-CAM++** | Gradient-based method to highlight important image regions |
| **SHAP** | Game-theory-based method to attribute predictions to input features |
| **Mixed Precision** | Using FP16 for speed and FP32 for precision selectively during training |
| **WeightedRandomSampler** | Oversamples minority class during training to handle class imbalance |
| **DataLeakage** | When test set information contaminates training → overly optimistic metrics |
| **StandardScaler** | Transforms features to zero mean / unit variance |
| **Kaiming Init** | Weight initialisation designed for ReLU — preserves gradient magnitude |
| **AdamW** | Adam optimizer with decoupled weight decay (better regularisation) |
| **CosineAnnealingLR** | Smoothly reduces LR from max to min following a cosine curve |
| **BatchNorm** | Normalises layer activations per mini-batch — stabilises training |
| **Optuna** | Bayesian HPO framework using TPE algorithm with trial pruning |
| **PyTorch Lightning** | High-level PyTorch wrapper — removes training boilerplate |
| **torchmetrics** | Library for computing ML metrics correctly (handles batching & devices) |
| **timm** | Library of pretrained vision models (1000+ architectures) |
