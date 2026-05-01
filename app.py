"""
Parkinson's Disease Multimodal Screening — Interactive Research Demo
====================================================================
Research demonstration of the attention-based multimodal fusion framework from:

  "Attention-Based Multimodal Fusion of Voice and Gait for Parkinson's Disease Detection"
   Aakriti Jain, Ujjawal Gaur, Pragya Singh | ICESAIA 2026 (Under Review, IEEE)

⚠️  NOT a clinical tool. For research and educational purposes only.

Run locally:  streamlit run app.py
Deploy:       Push to Hugging Face Spaces (see README.md)
"""

import numpy as np
import pandas as pd
import streamlit as st

from utils.model_loader     import load_full_model, load_scalers, is_demo_mode
from utils.preprocessing    import (
    VOICE_FEATURE_NAMES, VOICE_FEATURE_DISPLAY, VOICE_FEATURE_RANGES,
    dict_to_voice_tensor, preprocess_voice_csv,
    preprocess_gait_file, preprocess_gait_npy,
)
from utils.inference        import (
    run_voice_inference, run_gait_inference,
    run_multimodal_inference, mock_inference,
)
from utils.ui_components    import (
    show_result_card, attention_gauge,
    window_probability_chart, window_attention_chart,
    perf_comparison_chart, PERF_TABLE,
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PD Screening Demo",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global disclaimer banner ───────────────────────────────────────────────────
st.markdown(
    """
    <div style="background:#fff3cd;border-left:5px solid #ffc107;
                padding:0.6rem 1rem;border-radius:6px;margin-bottom:1rem;
                font-size:0.88rem;color:#856404">
        ⚠️ <strong>Research Demo Only.</strong>
        This tool is a demonstration of academic research and is
        <strong>not intended for clinical diagnosis</strong>.
        Results must not be used for any medical decision-making.
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar navigation ─────────────────────────────────────────────────────────
PAGES = {
    "🧠  About the Research":      "about",
    "🎙️  Voice-Only Demo":         "voice",
    "🚶  Gait-Only Demo":          "gait",
    "🔀  Multimodal Fusion Demo":  "multimodal",
}

st.sidebar.title("PD Screening Demo")
st.sidebar.caption("Attention-Based Multimodal Fusion")
page_label = st.sidebar.radio("Navigate", list(PAGES.keys()), label_visibility="collapsed")
page = PAGES[page_label]

st.sidebar.markdown("---")
st.sidebar.caption(
    "**Paper:** Jain, Gaur, Singh — ICESAIA 2026 (Under Review, IEEE)\n\n"
    "**GitHub:** [aacritea](https://github.com/aacritea) · "
    "**LinkedIn:** [Aakriti Jain](https://linkedin.com/in/aakriti-jain-a76386250)"
)

# ── Load model & scalers (cached) ─────────────────────────────────────────────
DEMO_MODE    = is_demo_mode()
model        = load_full_model()
v_scaler, g_scaler = load_scalers()

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — About the Research
# ═══════════════════════════════════════════════════════════════════════════════
if page == "about":
    st.title("🧠 Attention-Based Multimodal Fusion for Parkinson's Detection")

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("""
        ### The Problem
        Parkinson's disease (PD) is a progressive neurodegenerative disorder
        affecting motor function. Early, non-invasive screening is critical to
        improving patient outcomes — but traditional diagnosis relies on clinical
        assessment by specialists.

        ### Our Approach
        We developed an **attention-based multimodal fusion framework** combining
        two complementary non-invasive biomarker streams:
        - **Voice signals** — acoustic features capturing vocal tremor, breathiness,
          and phonation irregularities from sustained /a/ phonations
        - **Gait signals** — vertical ground reaction force (VGRF) recordings
          capturing stride irregularities and altered motor dynamics

        The key innovation is an **attention mechanism** that learns *per-sample*
        weights for each modality, rather than treating them equally. This allows
        the model to down-weight a noisier or less informative modality for each
        individual patient.

        ### Modality Dropout
        During training, we randomly zero one modality's embedding with probability
        p=0.2. This forces the model to build robust single-modality representations,
        improving graceful degradation when a sensor is unavailable — relevant for
        real-world telehealth deployment.
        """)

    with col2:
        st.markdown("### Architecture Summary")
        st.markdown("""
        | Component | Details |
        |---|---|
        | Voice Encoder | 3-layer MLP [22→384→128→128] |
        | Gait Encoder | 4-block 1D-CNN, GAP |
        | Fusion | Attention (d_a = 16) |
        | Head | 128→64→1, Sigmoid |
        | Datasets | UCI PD Voice + PhysioNet VGRF |
        | Training | PyTorch, Adam, 100 epochs |
        """)

        st.markdown("### Key Results")
        st.metric("Accuracy",  "92.26%", "+5.59pp vs voice-only")
        st.metric("AUC-ROC",   "98.64%", "+2.96pp vs concatenation")
        st.metric("F1-Score",  "0.942",  "Best balanced classifier")

    st.markdown("---")
    st.markdown("### Performance Comparison")
    st.plotly_chart(perf_comparison_chart(), use_container_width=True)

    st.markdown("### Full Results Table")
    df_perf = pd.DataFrame(PERF_TABLE)
    st.dataframe(df_perf.set_index("Method"), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Voice-Only Demo
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "voice":
    st.title("🎙️ Voice-Only Inference")
    st.caption(
        "Enter acoustic features from a sustained /a/ phonation recording "
        "(UCI Parkinson's dataset format, 22 features)."
    )

    input_method = st.radio(
        "Input method", ["📊 Upload CSV", "🎛️ Manual sliders"], horizontal=True
    )

    voice_tensor = None

    if input_method == "📊 Upload CSV":
        st.markdown(
            "Upload a CSV with UCI Parkinson's column names. "
            "Columns `name` and `status` are ignored if present."
        )
        f = st.file_uploader("Upload voice features CSV", type=["csv"])
        if f:
            df = pd.read_csv(f)
            st.dataframe(df.head(3), use_container_width=True)
            voice_tensor = preprocess_voice_csv(df, scaler=v_scaler)
            st.success(f"Loaded {len(df)} sample(s).")

    else:  # sliders
        st.markdown("Adjust feature values — defaults are training-set means.")
        feature_dict = {}
        col_pairs = st.columns(2)
        for i, feat in enumerate(VOICE_FEATURE_NAMES):
            lo, hi, default = VOICE_FEATURE_RANGES[feat]
            with col_pairs[i % 2]:
                feature_dict[feat] = st.slider(
                    VOICE_FEATURE_DISPLAY[i],
                    min_value=float(lo),
                    max_value=float(hi),
                    value=float(default),
                    format="%.4f",
                    key=f"voice_{feat}",
                )
        if st.button("▶ Run Voice Inference", type="primary"):
            voice_tensor = dict_to_voice_tensor(feature_dict, scaler=v_scaler)

    if voice_tensor is not None:
        st.markdown("---")
        if DEMO_MODE or model is None:
            result = mock_inference(seed=int(voice_tensor.sum().item() * 100) % 999)
        else:
            result = run_voice_inference(model, voice_tensor)

        show_result_card(result, demo_mode=DEMO_MODE)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### Attention Weights")
            st.caption(
                "In voice-only mode, gait is zeroed — attention naturally collapses to voice."
            )
            st.plotly_chart(
                attention_gauge(result["alpha_v"], result["alpha_g"]),
                use_container_width=True,
            )
        with col_b:
            st.markdown("#### What does this mean?")
            prob = result["prob"]
            if prob >= 0.5:
                st.warning(
                    f"The model assigns a **{prob*100:.1f}% PD probability** based on "
                    f"acoustic features alone. Voice-only accuracy in this framework is "
                    f"**86.67%** on held-out subjects."
                )
            else:
                st.success(
                    f"The model assigns a **{prob*100:.1f}% PD probability** based on "
                    f"acoustic features alone. Adding gait data may refine this estimate."
                )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Gait-Only Demo
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "gait":
    st.title("🚶 Gait-Only Inference")
    st.caption(
        "Upload a PhysioNet VGRF .txt file or load a pre-saved sample "
        "to run inference through the 1D-CNN gait encoder."
    )

    from pathlib import Path
    SAMPLES_DIR = Path("samples")

    gait_tensor = None
    input_method = st.radio(
        "Input method",
        ["📁 Upload PhysioNet .txt", "🗂️ Load sample file"],
        horizontal=True,
    )

    if input_method == "📁 Upload PhysioNet .txt":
        st.markdown(
            "Upload a VGRF `.txt` file from the "
            "[PhysioNet Gait in PD dataset](https://physionet.org/content/gaitpdb/1.0.0/). "
            "Tab-separated, 19 columns (time + 16 sensor channels + total)."
        )
        f = st.file_uploader("Upload VGRF file", type=["txt"])
        if f:
            import tempfile, os
            with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
                tmp.write(f.read())
                tmp_path = tmp.name
            gait_tensor = preprocess_gait_file(tmp_path, scaler=g_scaler)
            os.unlink(tmp_path)
            st.success(f"Segmented into {gait_tensor.shape[0]} windows of size 256.")

    else:
        pd_path  = SAMPLES_DIR / "sample_pd_gait.npy"
        hc_path  = SAMPLES_DIR / "sample_healthy_gait.npy"
        col1, col2 = st.columns(2)
        with col1:
            if st.button("⚠️ Load PD Sample", use_container_width=True):
                if pd_path.exists():
                    arr = np.load(pd_path)
                    gait_tensor = preprocess_gait_npy(arr, scaler=g_scaler)
                    st.success(f"Loaded PD sample — {gait_tensor.shape[0]} windows.")
                else:
                    st.error("sample_pd_gait.npy not found in samples/. See README.")
        with col2:
            if st.button("✅ Load Healthy Sample", use_container_width=True):
                if hc_path.exists():
                    arr = np.load(hc_path)
                    gait_tensor = preprocess_gait_npy(arr, scaler=g_scaler)
                    st.success(f"Loaded healthy sample — {gait_tensor.shape[0]} windows.")
                else:
                    st.error("sample_healthy_gait.npy not found in samples/. See README.")

    if gait_tensor is not None:
        st.markdown("---")
        if DEMO_MODE or model is None:
            result = mock_inference(seed=gait_tensor.shape[0])
        else:
            result = run_gait_inference(model, gait_tensor)

        show_result_card(result, demo_mode=DEMO_MODE)

        st.plotly_chart(
            window_probability_chart(
                result.get("window_probs", np.array([result["prob"]]))
            ),
            use_container_width=True,
        )

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### Attention (gait-only mode)")
            st.plotly_chart(
                attention_gauge(result["alpha_v"], result["alpha_g"]),
                use_container_width=True,
            )
        with col_b:
            st.metric("Gait-Only Benchmark Accuracy", "89.13%")
            st.metric("Gait-Only AUC-ROC",            "99.1%")
            st.caption(
                "Gait signals outperform voice-only across most metrics in our paper, "
                "suggesting temporal motor patterns are highly discriminative for PD."
            )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Multimodal Fusion Demo  ⭐
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "multimodal":
    st.title("🔀 Multimodal Fusion Demo")
    st.markdown(
        "Provide **both** voice features and a gait recording. "
        "The attention mechanism will dynamically weight each modality "
        "and produce a fused prediction — the paper's primary contribution."
    )

    # ── Voice input ──────────────────────────────────────────────────────────
    st.markdown("### Step 1 — Voice Features")
    v_input = st.radio(
        "Voice input", ["📊 Upload CSV", "🎛️ Use training-set means"],
        horizontal=True, key="mm_voice_method"
    )

    voice_tensor = None

    if v_input == "📊 Upload CSV":
        f = st.file_uploader("Upload voice features CSV", type=["csv"], key="mm_v_csv")
        if f:
            df = pd.read_csv(f)
            voice_tensor = preprocess_voice_csv(df, scaler=v_scaler)[:1]  # one sample
            st.success("Voice features loaded.")
    else:
        # Use training-set means as default voice features
        defaults = {f: VOICE_FEATURE_RANGES[f][2] for f in VOICE_FEATURE_NAMES}
        voice_tensor = dict_to_voice_tensor(defaults, scaler=v_scaler)
        st.info("Using training-set mean values as voice input.")

    # ── Gait input ───────────────────────────────────────────────────────────
    st.markdown("### Step 2 — Gait Signal")
    from pathlib import Path
    SAMPLES_DIR = Path("samples")

    g_input = st.radio(
        "Gait input", ["📁 Upload PhysioNet .txt", "🗂️ Load sample"],
        horizontal=True, key="mm_gait_method"
    )

    gait_tensor = None

    if g_input == "📁 Upload PhysioNet .txt":
        f = st.file_uploader("Upload VGRF .txt file", type=["txt"], key="mm_g_txt")
        if f:
            import tempfile, os
            with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
                tmp.write(f.read())
                tmp_path = tmp.name
            gait_tensor = preprocess_gait_file(tmp_path, scaler=g_scaler)
            os.unlink(tmp_path)
            st.success(f"Gait signal loaded — {gait_tensor.shape[0]} windows.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("⚠️ PD Gait Sample", use_container_width=True, key="mm_pd"):
                p = SAMPLES_DIR / "sample_pd_gait.npy"
                if p.exists():
                    gait_tensor = preprocess_gait_npy(np.load(p), scaler=g_scaler)
                    st.session_state["mm_gait"] = gait_tensor
                    st.success(f"PD sample loaded — {gait_tensor.shape[0]} windows.")
                else:
                    st.error("sample_pd_gait.npy not found. See README.")
        with col2:
            if st.button("✅ Healthy Gait Sample", use_container_width=True, key="mm_hc"):
                p = SAMPLES_DIR / "sample_healthy_gait.npy"
                if p.exists():
                    gait_tensor = preprocess_gait_npy(np.load(p), scaler=g_scaler)
                    st.session_state["mm_gait"] = gait_tensor
                    st.success(f"Healthy sample loaded — {gait_tensor.shape[0]} windows.")
                else:
                    st.error("sample_healthy_gait.npy not found. See README.")

        # Persist gait tensor across reruns
        if gait_tensor is None and "mm_gait" in st.session_state:
            gait_tensor = st.session_state["mm_gait"]

    # ── Modality dropout simulation ───────────────────────────────────────────
    st.markdown("### Step 3 — Run Inference")
    st.markdown(
        "**Modality Dropout Simulation** — toggle off a modality to reproduce "
        "the paper's robustness experiment (Section V.C, Table II)."
    )
    col_v, col_g = st.columns(2)
    use_voice = col_v.toggle("Use Voice Modality", value=True)
    use_gait  = col_g.toggle("Use Gait Modality",  value=True)

    if not use_voice and not use_gait:
        st.error("At least one modality must be enabled.")
    elif voice_tensor is not None or gait_tensor is not None:
        if st.button("▶ Run Multimodal Inference", type="primary", use_container_width=True):

            mode = "both"
            if not use_voice: mode = "gait"
            if not use_gait:  mode = "voice"

            # Fallback tensors if one modality disabled
            v_in = voice_tensor if voice_tensor is not None else None
            g_in = gait_tensor  if gait_tensor  is not None else None

            if DEMO_MODE or model is None:
                result = mock_inference(seed=42)
            else:
                result = run_multimodal_inference(model, v_in, g_in, mode=mode)

            st.markdown("---")
            show_result_card(result, demo_mode=DEMO_MODE)

            # ── Attention visualization (the signature feature) ────────────
            st.markdown("#### 🔍 Attention Weight Analysis")
            st.caption(
                "The attention mechanism assigns sample-specific weights to each modality. "
                "In the paper, voice weights ranged from 0.21–0.78 across test samples, "
                "confirming per-sample adaptive fusion rather than fixed weighting."
            )
            st.plotly_chart(
                attention_gauge(result["alpha_v"], result["alpha_g"]),
                use_container_width=True,
            )

            if "window_probs" in result:
                col_a, col_b = st.columns(2)
                with col_a:
                    st.plotly_chart(
                        window_probability_chart(result["window_probs"]),
                        use_container_width=True,
                    )
                with col_b:
                    st.plotly_chart(
                        window_attention_chart(
                            result["window_alpha_v"],
                            result["window_alpha_g"],
                        ),
                        use_container_width=True,
                    )

            # ── Mode-specific insight ──────────────────────────────────────
            if mode != "both":
                dropped = "voice" if mode == "gait" else "gait"
                st.info(
                    f"**Dropout simulation active** — {dropped} modality zeroed. "
                    f"Models trained with dropout achieved {88.3 if dropped=='gait' else 89.8}% "
                    f"accuracy in this mode vs "
                    f"{84.2 if dropped=='gait' else 87.1}% without dropout training "
                    f"(paper Table II).",
                    icon="🔬",
                )

            # ── Benchmark comparison ───────────────────────────────────────
            st.markdown("#### Benchmark vs Paper Results")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("This model (Accuracy)", "92.26%")
            col2.metric("AUC-ROC",               "98.64%")
            col3.metric("vs Voice-Only",          "+5.59pp")
            col4.metric("vs Concatenation",       "+1.98pp")
    else:
        st.info("Provide at least one input above to run inference.")
