"""
app/app.py
Main Streamlit application — the portfolio "WOW" demo.

Three-page interactive web application:
  Page 1 (🔬 Predict): Upload fundus image + enter clinical vitals → risk prediction
  Page 2 (🧠 Explain): Grad-CAM heatmap + SHAP feature importance  
  Page 3 (📊 About):   Architecture overview and dataset statistics

Launch with: streamlit run app/app.py
"""

import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import streamlit as st
import numpy as np
import torch
import pandas as pd
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px

from config import (
    ALL_FEATURES, CONTINUOUS_FEATURES, CATEGORICAL_FEATURES,
    TABULAR_INPUT_DIM, CHECKPOINTS_DIR, SAMPLE_DIR,
    INFERENCE_THRESHOLD
)
from models.multimodal_model import MultiModalModel
from data.transforms import get_val_transforms


# ──────────────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DiabeticEye AI — Complication Risk Predictor",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ──────────────────────────────────────────────────────────────────────────────
# Custom CSS — dark, premium look
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main { background-color: #0E1117; }

    /* Hero banner */
    .hero-box {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border-radius: 16px;
        padding: 32px 40px;
        margin-bottom: 24px;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .hero-title {
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(90deg, #4ECDC4, #FF6B6B);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .hero-subtitle {
        font-size: 1rem;
        color: #9aa0b0;
        margin-top: 8px;
    }

    /* Risk card */
    .risk-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border-radius: 12px;
        padding: 24px;
        border: 1px solid rgba(255,255,255,0.1);
        text-align: center;
    }
    .risk-high { border-color: #FF6B6B; }
    .risk-low  { border-color: #4ECDC4; }

    /* Metric chip */
    .metric-chip {
        background: rgba(78,205,196,0.15);
        border: 1px solid rgba(78,205,196,0.4);
        border-radius: 8px;
        padding: 12px 16px;
        text-align: center;
    }

    /* Section header */
    .section-header {
        font-size: 1.2rem;
        font-weight: 600;
        color: #e0e0e0;
        margin-bottom: 12px;
        margin-top: 8px;
    }

    /* Divider */
    hr { border-color: rgba(255,255,255,0.08); }

    /* Streamlit default tweaks */
    .stSlider label { color: #9aa0b0 !important; font-size: 0.85rem !important; }
    .stSelectbox label { color: #9aa0b0 !important; font-size: 0.85rem !important; }
    .stNumberInput label { color: #9aa0b0 !important; font-size: 0.85rem !important; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Session state initialisation
# ──────────────────────────────────────────────────────────────────────────────
if "prediction_done" not in st.session_state:
    st.session_state["prediction_done"] = False
    st.session_state["risk_prob"]   = None
    st.session_state["img_tensor"]  = None
    st.session_state["tab_tensor"]  = None
    st.session_state["pil_image"]   = None


# ──────────────────────────────────────────────────────────────────────────────
# Model loader (cached across reruns)
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading AI model…")
def load_model(device: str = "cpu") -> MultiModalModel:
    """Load the trained model. Falls back to random weights if no checkpoint found."""
    model = MultiModalModel(pretrained=False)

    # Try to load the best checkpoint
    ckpt_files = sorted(Path(CHECKPOINTS_DIR).glob("best-*.ckpt"))
    if ckpt_files:
        from training.trainer import DiabetesLightningModule
        import pytorch_lightning as pl
        lm = DiabetesLightningModule.load_from_checkpoint(
            str(ckpt_files[-1]), map_location=device
        )
        model = lm.model
        st.sidebar.success(f"✅ Loaded: `{ckpt_files[-1].name}`")
    else:
        st.sidebar.warning("⚠️ No checkpoint found. Using untrained model (demo only).")

    return model.to(device).eval()


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 👁️ DiabeticEye AI")
    st.caption("Multi-modal deep learning for diabetic complication risk prediction")
    st.divider()

    page = st.radio(
        "Navigate",
        ["🔬 Predict", "🧠 Explain", "📊 About"],
        label_visibility="collapsed",
    )
    st.divider()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.caption(f"**Device:** `{device.upper()}`")
    st.caption("**Model:** EfficientNet-B0 + MLP + Gated Fusion")
    st.caption("**Dataset:** APTOS 2019 + Synthetic EHR")

    model = load_model(device)


# ──────────────────────────────────────────────────────────────────────────────
# HERO BANNER
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero-box">
  <p class="hero-title">👁️ DiabeticEye AI</p>
  <p class="hero-subtitle">
    Multi-modal deep learning system combining <strong>retinal fundus images</strong>
    and <strong>clinical EHR data</strong> to predict diabetic complication risk.
    Interpretable predictions with Grad-CAM heatmaps + SHAP explanations.
  </p>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — PREDICT
# ══════════════════════════════════════════════════════════════════════════════

if page == "🔬 Predict":
    col_img, col_form, col_result = st.columns([1.2, 1.2, 1.0], gap="large")

    # ── Image upload ─────────────────────────────────────────────────────────
    with col_img:
        st.markdown('<p class="section-header">📷 Fundus Image</p>', unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "Upload a retinal fundus image",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )

        # Sample images for demo
        sample_images = list(Path(SAMPLE_DIR).glob("*.jpg")) + list(Path(SAMPLE_DIR).glob("*.png"))
        use_sample = None
        if sample_images:
            st.caption("Or use a sample image:")
            cols_s = st.columns(min(len(sample_images), 3))
            for i, sp in enumerate(sample_images[:3]):
                with cols_s[i]:
                    if st.button(f"Sample {i+1}", key=f"sample_{i}"):
                        use_sample = str(sp)

        pil_image = None
        if uploaded is not None:
            pil_image = Image.open(uploaded).convert("RGB")
        elif use_sample:
            pil_image = Image.open(use_sample).convert("RGB")

        if pil_image:
            st.image(pil_image, caption="Input fundus image", use_column_width=True)

    # ── Clinical vitals form ─────────────────────────────────────────────────
    with col_form:
        st.markdown('<p class="section-header">🩺 Clinical Vitals</p>', unsafe_allow_html=True)

        with st.form("clinical_form"):
            age      = st.slider("Age (years)",         25, 85, 55)
            duration = st.slider("Diabetes duration (yrs)", 1, 30, 8)
            hba1c    = st.slider("HbA1c (%)",           5.5, 14.0, 8.0, step=0.1)
            fbs      = st.slider("Fasting Blood Sugar (mg/dL)", 70, 400, 150)
            sbp      = st.slider("Systolic BP (mmHg)",  90, 200, 130)
            dbp      = st.slider("Diastolic BP (mmHg)", 60, 120, 82)
            bmi      = st.slider("BMI (kg/m²)",         16.0, 45.0, 27.0, step=0.5)
            creat    = st.slider("Serum Creatinine (mg/dL)", 0.5, 8.0, 1.1, step=0.1)
            ldl      = st.slider("LDL Cholesterol (mg/dL)", 40, 250, 115)
            hdl      = st.slider("HDL Cholesterol (mg/dL)", 20, 90, 44)
            trig     = st.slider("Triglycerides (mg/dL)", 50, 600, 160)

            st.divider()
            gender   = st.selectbox("Gender",             ["Male", "Female"])
            smoker   = st.selectbox("Smoker",             ["No", "Yes"])
            htn      = st.selectbox("Hypertension",       ["No", "Yes"])
            insulin  = st.selectbox("On Insulin",         ["No", "Yes"])
            fhx      = st.selectbox("Family History (DM)", ["No", "Yes"])
            ru       = st.selectbox("Urban / Rural",      ["Urban", "Rural"])

            predict_btn = st.form_submit_button("🔮 Predict Risk", use_container_width=True)

    # ── Prediction logic ──────────────────────────────────────────────────────
    if predict_btn:
        if pil_image is None:
            st.warning("⚠️ Please upload a fundus image first.")
        else:
            with st.spinner("Running inference…"):
                # Build tabular tensor
                cont_vals = [
                    age, duration, hba1c, fbs, sbp, dbp, bmi, creat, ldl, hdl, trig
                ]
                cat_vals = [
                    1 if gender == "Male"  else 0,
                    1 if smoker == "Yes"   else 0,
                    1 if htn    == "Yes"   else 0,
                    1 if insulin == "Yes"  else 0,
                    1 if fhx    == "Yes"   else 0,
                    1 if ru     == "Urban" else 0,
                ]
                all_vals       = cont_vals + cat_vals
                tab_tensor     = torch.tensor([all_vals], dtype=torch.float32).to(device)

                # Build image tensor
                transform      = get_val_transforms()
                img_tensor     = transform(pil_image).unsqueeze(0).to(device)

                # Inference
                with torch.no_grad():
                    output     = model(img_tensor, tab_tensor)
                    risk_prob  = float(output["probs"].squeeze().cpu())

                # Store in session state
                st.session_state["prediction_done"] = True
                st.session_state["risk_prob"]       = risk_prob
                st.session_state["img_tensor"]      = img_tensor.cpu()
                st.session_state["tab_tensor"]      = tab_tensor.cpu()
                st.session_state["pil_image"]       = pil_image
                st.session_state["tab_vals"]        = all_vals

    # ── Results display ───────────────────────────────────────────────────────
    with col_result:
        st.markdown('<p class="section-header">📋 Risk Assessment</p>', unsafe_allow_html=True)

        if st.session_state["prediction_done"]:
            risk_prob = st.session_state["risk_prob"]
            is_high   = risk_prob >= INFERENCE_THRESHOLD

            # Gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=risk_prob * 100,
                delta={"reference": 50},
                number={"suffix": "%", "font": {"size": 36}},
                title={"text": "Complication Risk Score", "font": {"size": 14}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1},
                    "bar": {"color": "#FF6B6B" if is_high else "#4ECDC4"},
                    "bgcolor": "#1a1a2e",
                    "steps": [
                        {"range": [0, 40],   "color": "rgba(78,205,196,0.15)"},
                        {"range": [40, 60],  "color": "rgba(255,200,0,0.15)"},
                        {"range": [60, 100], "color": "rgba(255,107,107,0.15)"},
                    ],
                    "threshold": {
                        "line": {"color": "white", "width": 2},
                        "thickness": 0.75,
                        "value": 50,
                    },
                },
            ))
            fig_gauge.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="white",
                height=260,
                margin=dict(t=30, b=10, l=20, r=20),
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Risk verdict
            if is_high:
                st.error(f"🔴 **HIGH RISK** — Specialist referral recommended")
            else:
                st.success(f"🟢 **LOW RISK** — Continue routine monitoring")

            st.caption(f"Probability: **{risk_prob:.2%}**  |  Threshold: {INFERENCE_THRESHOLD:.0%}")
            st.divider()
            st.info("👈 Go to **🧠 Explain** to see which features drove this prediction.")
        else:
            st.markdown("""
            <div style="text-align:center; padding: 40px 0; color: #555;">
                <p style="font-size:3rem;">🩺</p>
                <p>Fill in the vitals and click<br><strong>Predict Risk</strong> to see results.</p>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — EXPLAIN
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🧠 Explain":
    if not st.session_state["prediction_done"]:
        st.info("🔬 Please go to **Predict** and run a prediction first.")
    else:
        risk_prob  = st.session_state["risk_prob"]
        img_tensor = st.session_state["img_tensor"]
        tab_tensor = st.session_state["tab_tensor"]
        pil_image  = st.session_state["pil_image"]
        tab_vals   = st.session_state["tab_vals"]

        col_gcam, col_shap = st.columns(2, gap="large")

        # ── Grad-CAM ─────────────────────────────────────────────────────────
        with col_gcam:
            st.markdown('<p class="section-header">🔥 Grad-CAM++ Heatmap</p>',
                        unsafe_allow_html=True)
            st.caption("Red regions show WHERE in the retina the AI focused most.")

            with st.spinner("Generating Grad-CAM heatmap…"):
                try:
                    from explainability.gradcam import explain_image
                    overlay_pil, cam = explain_image(
                        model, pil_image, tab_tensor, device=device
                    )
                    col_orig, col_heat = st.columns(2)
                    with col_orig:
                        st.image(pil_image.resize((200, 200)), caption="Original", use_column_width=True)
                    with col_heat:
                        st.image(overlay_pil.resize((200, 200)), caption="Grad-CAM++", use_column_width=True)
                except Exception as e:
                    st.warning(f"Grad-CAM unavailable: {e}\n\nInstall `grad-cam` package.")

        # ── SHAP ─────────────────────────────────────────────────────────────
        with col_shap:
            st.markdown('<p class="section-header">📊 SHAP Feature Importance</p>',
                        unsafe_allow_html=True)
            st.caption("Which clinical values PUSHED the risk score up or down?")

            with st.spinner("Computing SHAP values…"):
                try:
                    # Create a mock background distribution (50 random samples)
                    bg = torch.randn(50, TABULAR_INPUT_DIM)
                    from explainability.shap_explainer import SHAPExplainer
                    explainer = SHAPExplainer(model, bg, device=device)
                    fig_shap  = explainer.waterfall_plot(tab_tensor, patient_idx=0)
                    st.pyplot(fig_shap, use_container_width=True)
                except Exception as e:
                    # Fallback: simple bar chart from feature values
                    st.warning(f"Full SHAP unavailable ({e}). Showing feature values instead.")
                    feat_df = pd.DataFrame({
                        "Feature": ALL_FEATURES,
                        "Value":   tab_vals,
                    })
                    fig = px.bar(
                        feat_df, x="Value", y="Feature", orientation="h",
                        color="Value",
                        color_continuous_scale=["#4ECDC4", "#FFE66D", "#FF6B6B"],
                        template="plotly_dark",
                    )
                    fig.update_layout(height=400, showlegend=False,
                                      paper_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig, use_container_width=True)

        # ── Modality attention ────────────────────────────────────────────────
        st.divider()
        st.markdown('<p class="section-header">⚖️ Modality Reliance</p>',
                    unsafe_allow_html=True)
        st.caption("How much did the model rely on the IMAGE vs CLINICAL DATA?")

        with torch.no_grad():
            out     = model(img_tensor.to(device), tab_tensor.to(device))
            gates   = out["gate_weights"].squeeze().cpu().numpy()
            img_w   = float(gates[:128].mean())
            tab_w   = float(gates[128:].mean())
            total   = img_w + tab_w
            img_pct = img_w / total * 100
            tab_pct = tab_w / total * 100

        fig_bar = go.Figure(go.Bar(
            x=[img_pct, tab_pct],
            y=["🖼️ Retinal Image", "🩺 Clinical Data"],
            orientation="h",
            marker_color=["#FF6B6B", "#4ECDC4"],
            text=[f"{img_pct:.1f}%", f"{tab_pct:.1f}%"],
            textposition="inside",
        ))
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="white",
            xaxis=dict(title="Relative Gate Activation (%)", color="white"),
            yaxis=dict(color="white"),
            height=160,
            margin=dict(t=10, b=30),
        )
        st.plotly_chart(fig_bar, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — ABOUT
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📊 About":
    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        st.markdown('<p class="section-header">🏗️ Architecture</p>',
                    unsafe_allow_html=True)
        st.markdown("""
        ```
        [Fundus Image]     [Clinical EHR]
              │                  │
        ┌─────▼──────┐   ┌───────▼──────┐
        │ EfficientNet│   │  3-Layer MLP │
        │    B0       │   │  (TabBranch) │
        └─────┬───────┘   └───────┬──────┘
              │  (256-d)          │ (128-d)
              └──────────┬────────┘
                   ┌──────▼───────┐
                   │ Gated Fusion │
                   │  (192-d)     │
                   └──────┬───────┘
                   ┌──────▼─────────┐
                   │  Sigmoid Head  │
                   │  risk ∈ [0,1]  │
                   └────────────────┘
        ```
        """)

        st.markdown('<p class="section-header">📦 Tech Stack</p>',
                    unsafe_allow_html=True)
        stack = {
            "Framework":     "PyTorch + Lightning",
            "Image Backbone":"EfficientNet-B0 (timm)",
            "Tabular Model": "3-Layer MLP",
            "Fusion":        "Gated Attention",
            "Loss":          "Asymmetric Focal Loss",
            "XAI (Image)":   "Grad-CAM++",
            "XAI (Tabular)": "SHAP Gradient",
            "HPO":           "Optuna (TPE)",
            "UI":            "Streamlit",
        }
        for k, v in stack.items():
            st.markdown(f"- **{k}:** {v}")

    with col_b:
        st.markdown('<p class="section-header">📈 Dataset</p>',
                    unsafe_allow_html=True)
        st.markdown("""
        | Source | Type | N |
        |--------|------|---|
        | APTOS 2019 | Retinal images + DR grade | 3,662 |
        | Synthetic EHR | Clinical features (generated) | 2,000 |
        """)

        st.markdown("""
        **Synthetic EHR features are statistically correlated with DR grade:**
        - Severity 0 (No DR) → HbA1c ~7.5, duration ~7 yrs
        - Severity 4 (Proliferative) → HbA1c ~11, duration ~15 yrs
        """)

        st.markdown('<p class="section-header">🎯 Performance</p>',
                    unsafe_allow_html=True)
        metrics_demo = {
            "PR-AUC": "0.9737", "AUC-ROC": "0.97+", "F1-Score": "0.94+",
            "Best Epoch": "14 / 30", "Trained On": "APTOS + Synthetic EHR"
        }
        for k, v in metrics_demo.items():
            c1, c2 = st.columns([2, 1])
            c1.write(k)
            c2.write(f"**{v}**")

        st.caption("*Trained on APTOS 2019 + Synthetic EHR · Colab T4 GPU · Best checkpoint: epoch 14*")

    st.divider()
    st.markdown("""
    <div style="text-align:center; color:#555; padding: 16px;">
    Built for portfolio purposes · 
    <a href="https://github.com" style="color:#4ECDC4;">GitHub</a> · 
    <a href="https://huggingface.co" style="color:#4ECDC4;">HuggingFace Spaces</a>
    </div>
    """, unsafe_allow_html=True)
