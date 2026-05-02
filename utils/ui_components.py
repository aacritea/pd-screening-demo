"""
Reusable Streamlit UI components.
All chart functions return Plotly figures — call st.plotly_chart(fig) on them.
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st


# ── Color palette ──────────────────────────────────────────────────────────────
PD_COLOR      = "#E05C5C"
HEALTHY_COLOR = "#4CAF82"
VOICE_COLOR   = "#5C9BE0"
GAIT_COLOR    = "#E0A45C"
BG_COLOR      = "rgba(0,0,0,0)"


# ── Result card ────────────────────────────────────────────────────────────────

def show_result_card(result: dict, demo_mode: bool = False):
    """Big result card shown at the top of each prediction section."""
    color  = PD_COLOR if result["is_pd"] else HEALTHY_COLOR
    icon   = "⚠️" if result["is_pd"] else "✅"
    conf   = result["confidence"] * 100

    if demo_mode:
        st.info("**Demo mode** — model weights not loaded. Showing placeholder outputs.", icon="ℹ️")

    st.markdown(
        f"""
        <div style="
            background: {color}22;
            border-left: 5px solid {color};
            border-radius: 8px;
            padding: 1.2rem 1.5rem;
            margin-bottom: 1rem;
        ">
            <h2 style="color:{color}; margin:0">{icon} {result['label']}</h2>
            <p style="margin:0.4rem 0 0; font-size:1.05rem; color:#555">
                Model confidence: <strong>{conf:.1f}%</strong> &nbsp;|&nbsp;
                PD probability: <strong>{result['prob']*100:.1f}%</strong>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Attention weight gauge ─────────────────────────────────────────────────────

def attention_gauge(alpha_v: float, alpha_g: float) -> go.Figure:
    """
    Horizontal stacked bar showing the voice/gait attention split.
    This is the signature visualisation of the paper.
    """
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Voice", x=[alpha_v * 100], y=["Attention"],
        orientation="h", marker_color=VOICE_COLOR,
        text=[f"Voice {alpha_v*100:.1f}%"], textposition="inside",
    ))
    fig.add_trace(go.Bar(
        name="Gait", x=[alpha_g * 100], y=["Attention"],
        orientation="h", marker_color=GAIT_COLOR,
        text=[f"Gait {alpha_g*100:.1f}%"], textposition="inside",
    ))
    fig.update_layout(
        barmode="stack",
        xaxis=dict(title="Attention Weight (%)", range=[0, 100]),
        yaxis=dict(showticklabels=False),
        title="Modality Attention Weights (α_v  vs  α_g)",
        height=160,
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        margin=dict(l=10, r=10, t=40, b=30),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


# ── Per-window probability trace ───────────────────────────────────────────────

def window_probability_chart(window_probs: np.ndarray) -> go.Figure:
    """Line chart of per-window PD probabilities across a gait recording."""
    x = np.arange(len(window_probs))
    fig = go.Figure()
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray",
                  annotation_text="Decision threshold (0.5)")
    fig.add_trace(go.Scatter(
        x=x, y=window_probs,
        mode="lines+markers",
        line=dict(color=PD_COLOR, width=2),
        marker=dict(size=4),
        name="PD Probability",
        fill="tozeroy",
        fillcolor="rgba(224,92,92,0.13)",
    ))
    fig.update_layout(
        title="Per-Window PD Probability (Gait Signal)",
        xaxis_title="Gait Window Index",
        yaxis_title="PD Probability",
        yaxis=dict(range=[0, 1]),
        height=280,
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        margin=dict(l=10, r=10, t=40, b=30),
    )
    return fig


# ── Attention weight trace across windows ─────────────────────────────────────

def window_attention_chart(alpha_v: np.ndarray, alpha_g: np.ndarray) -> go.Figure:
    """Shows how the model's attention shifts between modalities window-by-window."""
    x = np.arange(len(alpha_v))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=alpha_v * 100, name="Voice weight α_v",
        line=dict(color=VOICE_COLOR, width=2),
    ))
    fig.add_trace(go.Scatter(
        x=x, y=alpha_g * 100, name="Gait weight α_g",
        line=dict(color=GAIT_COLOR, width=2),
    ))
    fig.update_layout(
        title="Per-Window Attention Weight Distribution",
        xaxis_title="Gait Window Index",
        yaxis_title="Attention Weight (%)",
        yaxis=dict(range=[0, 100]),
        height=260,
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        margin=dict(l=10, r=10, t=40, b=30),
    )
    return fig


# ── Performance comparison table ───────────────────────────────────────────────

PERF_TABLE = {
    "Method":    ["Voice-Only",  "Gait-Only",  "Concatenation", "Attention Fusion (Ours)"],
    "Accuracy":  ["86.67%",      "89.13%",     "90.28%",        "92.26%"],
    "Precision": ["0.846",       "0.865",      "0.899",         "0.969"],
    "Recall":    ["1.000",       "1.000",      "0.967",         "0.917"],
    "F1-Score":  ["0.967",       "0.927",      "0.932",         "0.942"],
    "AUC-ROC":   ["0.977",       "0.991",      "0.957",         "0.986"],
}


def perf_comparison_chart() -> go.Figure:
    """Grouped bar chart of all four methods × accuracy + AUC-ROC."""
    methods   = PERF_TABLE["Method"]
    accuracy  = [86.67, 89.13, 90.28, 92.26]
    auc       = [97.7,  99.1,  95.7,  98.6]
    colors    = [VOICE_COLOR, GAIT_COLOR, "#A0A0A0", PD_COLOR]

    fig = go.Figure()
    for i, (m, acc, a, c) in enumerate(zip(methods, accuracy, auc, colors)):
        fig.add_trace(go.Bar(
            name=m, x=["Accuracy (%)", "AUC-ROC (%)"],
            y=[acc, a], marker_color=c,
            text=[f"{acc:.2f}%", f"{a:.1f}%"],
            textposition="outside",
        ))
    fig.update_layout(
        barmode="group",
        title="Model Performance Comparison (held-out test set)",
        yaxis=dict(range=[80, 102], title="Score (%)"),
        height=360,
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        margin=dict(l=10, r=10, t=50, b=30),
    )
    return fig